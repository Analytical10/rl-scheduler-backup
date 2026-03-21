# LoHi_v202.py (DGE + OCC-LoRA)
import os
import math
import json
from typing import List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoConfig

# =================================================================================
# 1. NVFP4 精确模拟量化函数 (已优化)
# =================================================================================

def quantize_to_fp_format(tensor, num_exponent_bits, num_mantissa_bits):
    max_exponent = 2**(num_exponent_bits - 1) - 1
    max_value = (2 - 2**-num_mantissa_bits) * (2**max_exponent)
    tensor_clamped = torch.clamp(tensor, -max_value, max_value)
    mantissa, exponent = torch.frexp(tensor_clamped)
    mantissa_quantized = torch.round(mantissa * (2**num_mantissa_bits)) / (2**num_mantissa_bits)
    tensor_quantized = torch.ldexp(mantissa_quantized, exponent)
    return tensor_quantized

def generate_e2m1_lut(dtype=torch.bfloat16, device='cpu'):
    lut = []
    for i in range(16):
        sign = -1 if (i >> 3) & 1 else 1
        exponent_raw = (i >> 1) & 0b11
        mantissa_bit = i & 1
        if exponent_raw == 0: val = sign * (2**-1) * mantissa_bit * (2**-1)
        elif exponent_raw == 3: val = sign * 1.5 * 2**1
        else:
            exponent = exponent_raw - 1
            mantissa = 1 + mantissa_bit * 0.5
            val = sign * mantissa * (2**exponent)
        lut.append(val)
    unique_sorted_lut = sorted(list(set(lut)))
    return torch.tensor(unique_sorted_lut, dtype=dtype, device=device)

E2M1_CACHE = {}
def get_e2m1_lut_and_boundaries(device, dtype):
    """获取或生成并缓存E2M1查找表和边界"""
    if device not in E2M1_CACHE or E2M1_CACHE[device]['lut'].dtype != dtype:
        lut = generate_e2m1_lut(dtype=dtype, device=device)
        boundaries = (lut[:-1] + lut[1:]) / 2.0
        E2M1_CACHE[device] = {'lut': lut, 'boundaries': boundaries}
    return E2M1_CACHE[device]['lut'], E2M1_CACHE[device]['boundaries']

def quantize_nvfp4_optimized(w: Tensor, block_size=16, dtype=torch.bfloat16):
    """ ★★★ 使用二分查找优化的NVFP4量化函数 ★★★ """
    _, e2m1_boundaries = get_e2m1_lut_and_boundaries(w.device, dtype)
    
    orig_shape = w.shape
    w_flat = w.flatten()
    pad_len = (block_size - (w_flat.numel() % block_size)) % block_size
    if pad_len > 0: w_flat = F.pad(w_flat, (0, pad_len))
    w_blocks = w_flat.view(-1, block_size)

    scales = w_blocks.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
    scales_q = quantize_to_fp_format(scales, num_exponent_bits=4, num_mantissa_bits=3)
    
    w_normalized = w_blocks / scales_q

    # ★★★ 核心修改：用 searchsorted 替换暴力搜索 ★★★
    indices = torch.searchsorted(e2m1_boundaries, w_normalized)

    indices_byte = indices.to(torch.uint8)
    packed_w = (indices_byte[:, ::2] << 4) | indices_byte[:, 1::2]
    return packed_w, scales_q, orig_shape

def dequantize_nvfp4(quantized_w: Tensor, scales: Tensor, shape: torch.Size, block_size=16, dtype=torch.bfloat16):
    e2m1_lut, _ = get_e2m1_lut_and_boundaries(quantized_w.device, dtype)
    first_indices = (quantized_w >> 4).to(torch.long)
    second_indices = (quantized_w & 0x0F).to(torch.long)
    unpacked_indices = torch.stack([first_indices, second_indices], dim=-1).view(-1, block_size)
    dequantized_normalized_blocks = e2m1_lut[unpacked_indices]
    dequantized_blocks = dequantized_normalized_blocks * scales
    dequantized_flat = dequantized_blocks.flatten()
    dequantized = dequantized_flat[:shape.numel()].view(shape)
    return dequantized

# 将所有对旧函数的调用替换为优化版
quantize_nvfp4 = quantize_nvfp4_optimized

# =================================================================================
# 2. ★★★ 新增: 可微分梯度估计器 (DGE) ★★★
# =================================================================================

class DifferentiableGradientEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W0_fp, W0_low, W0_scales, k, quant_block_size, dtype):
        """前向传播使用反量化值，但梯度计算将由DGE处理。"""
        ctx.save_for_backward(W0_fp, W0_scales)
        ctx.k = k
        ctx.quant_block_size = quant_block_size
        ctx.dtype = dtype
        
        w_dequant = dequantize_nvfp4(W0_low, W0_scales, W0_fp.shape, block_size=quant_block_size, dtype=dtype)
        # 返回一个与STE类似的值，确保前向计算正确
        return W0_fp + (w_dequant - W0_fp).detach()

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播的核心：应用DGE修正梯度。"""
        W0_fp, W0_scales = ctx.saved_tensors
        k = ctx.k

        # 论文公式(8): f'(x) = 1/k * |2x/τ - 1|^(1/k - 1)
        # 这里的 x 是归一化后的权重，τ 是量化区间的宽度，这里为1.0，因为我们已经归一化了
        # 注意：论文的 τ 指的是量化区间的宽度，而不是我们OCC中的激活值阈值
        
        # 1. 计算归一化权重 W_norm = W / scales
        # 由于scales是块级的，我们需要广播它以匹配W0_fp的形状
        num_blocks = W0_scales.shape[0]
        w_reshaped = W0_fp.view(num_blocks, -1)
        w_norm = w_reshaped / W0_scales
        
        # 2. 计算DGE修正项
        # 论文中提到，这个函数应用于每个量化区间。
        # 为简化，我们对整个 [-1, 1] 范围应用一个平滑的DGE近似
        # |2x - 1|^(1/k-1) 在 x=0.5 处奇异，我们用 |2x|^(1/k-1) 近似
        # 更好的方法是直接用论文公式，处理每个区间，但这里先用一个全局近似
        
        # 论文公式(8)的直接实现
        # τ_quant_interval 是每个量化步长，对于absmax是动态的，但DGE的推导是基于固定步长
        # 我们假设在归一化后，主要区间是[-1, 1]，DGE函数主要作用于此
        # f'(x) = 1/k * |x|^(1/k - 1) 是一个合理的简化
        exponent = (1.0 / k) - 1.0
        # 添加一个小的epsilon防止在0点梯度爆炸
        dge_correction = (1.0 / k) * (w_norm.abs() + 1e-8).pow(exponent)
        
        # 论文提到将修正项裁剪到3.0以保证稳定
        dge_correction = torch.clamp(dge_correction, max=3.0)
        
        # 恢复原始形状
        dge_correction = dge_correction.view_as(W0_fp)
        
        # 3. 应用修正项
        grad_input = grad_output * dge_correction
        
        return grad_input, None, None, None, None, None, None

# =================================================================================
# 3. ★★★ 新增: v202 模型和线性层 (DGE + OCC-LoRA) ★★★
# =================================================================================

@dataclass
class QuantPretrainConfig_v202:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    quant_block_size: int = 16
    trainable_scaling: bool = False
    occ_quantile: float = 0.99  # 离群点裁剪分位数
    dge_k: float = 5.0           # DGE的超参数k

class QuantPretrainLinear_v202(nn.Module):
    """
    实现了 DGE + OCC-LoRA 的线性层
    - 权重(W): 使用NVFP4量化, DGE进行梯度更新
    - 激活(X): 使用OCC机制，主体部分NVFP4量化，离群点由LoRA补偿
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        lora_alpha: float = 32,
        lora_dropout: float = 0.0,
        quant_block_size: int = 16,
        trainable_scaling: bool = False,
        occ_quantile: float = 0.999,
        dge_k: float = 5.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features, self.out_features, self.r = in_features, out_features, r
        self.lora_alpha, self.lora_dropout = lora_alpha, nn.Dropout(p=lora_dropout)
        self.quant_block_size, self.dtype = quant_block_size, dtype
        self.occ_quantile, self.dge_k = occ_quantile, dge_k

        # 1. 基础权重 W0 (可训练的全精度影子 + 低精度存储)
        # W0_fp 现在是可训练的，因为DGE会计算它的梯度
        self.W0_fp = Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.W0_fp, a=math.sqrt(5))
        self.W0_fp.requires_grad = True # ★★★ 与v1不同，W0是可训练的

        packed_w, scales, _ = quantize_nvfp4(self.W0_fp.data, block_size=self.quant_block_size, dtype=self.dtype)
        self.register_buffer('W0_low', packed_w)
        self.register_buffer('W0_scales', scales)

        # 2. 偏置项
        if bias:
            self.bias = Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # 3. LoRA 适配器 (BA) - 现在用于补偿离群点
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False, device=device, dtype=dtype)
            self.lora_B = nn.Linear(r, out_features, bias=False, device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight) # B初始化为0，让训练初期补偿为0
        
        self.trainable_scaling = trainable_scaling
        if trainable_scaling:
            self.scaling = Parameter(torch.tensor([1.0], device=device, dtype=dtype))
        else:
            self.scaling = self.lora_alpha / self.r if r > 0 else 0.0

    def update_quantized_weight(self):
        """在优化器 step 之后调用，用更新后的 W0_fp 来更新 W0_low。"""
        packed_w, scales, _ = quantize_nvfp4(self.W0_fp.data, block_size=self.quant_block_size, dtype=self.dtype)
        self.W0_low.data = packed_w
        self.W0_scales.data = scales

    def forward(self, x: Tensor):
        # ★★★ 1. 离群点裁剪与补偿 (OCC) for 激活值 x ★★★
        # 计算动态阈值 tau
        
        tau = torch.quantile(x.abs().float(), self.occ_quantile, dim=-1, keepdim=True).to(x.dtype)
        
        # 裁剪 x
        x_clamped = torch.clamp(x, -tau, tau)
        
        # 计算离群点矩阵
        x_outlier = x - x_clamped
        
        # ★★★ 2. 主路径计算 ★★★
        # 2a. 处理 W0：使用DGE进行反向传播
        weight_for_forward = DifferentiableGradientEstimator.apply(
            self.W0_fp, self.W0_low, self.W0_scales, self.dge_k, self.quant_block_size, self.dtype
        )

        # 2b. 处理 X：只量化被裁剪过的主体部分
        x_clamped_quant, x_clamped_scales, x_shape = quantize_nvfp4(x_clamped, block_size=self.quant_block_size, dtype=self.dtype)
        x_clamped_dequant = dequantize_nvfp4(x_clamped_quant, x_clamped_scales, x_shape, block_size=self.quant_block_size, dtype=self.dtype)

        # 2c. 计算主路径输出
        path1_out = F.linear(x_clamped_dequant, weight_for_forward)

        # ★★★ 3. 补偿路径计算 (LoRA处理离群点) ★★★
        path2_out = 0
        if self.r > 0 and x_outlier.abs().sum() > 0: # 只有存在离群点时才计算
            lora_scaling = self.scaling.tanh() if self.trainable_scaling else self.scaling
            # 注意：LoRA路径现在使用离群点矩阵 x_outlier 作为输入
            path2_out = self.lora_B(self.lora_A(self.lora_dropout(x_outlier))) * lora_scaling

        # ★★★ 4. 合并路径和偏置 ★★★
        out = path1_out + path2_out
        if self.bias is not None:
            out += self.bias
        return out

class QuantPretrainModel_v202(nn.Module):
    """模型包装器，用于将目标 nn.Linear 替换为 QuantPretrainLinear_v202"""
    def __init__(self, model, config: QuantPretrainConfig_v202):
        super().__init__()
        self.wrapped_model = model
        self._config = config

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear) or not any(k in module_name for k in config.target_modules):
                continue
            
            print(f"Applying Quantized Pre-training (v202 DGE+OCC) to module: {module_name}")
            
            parent_module = self.wrapped_model.get_submodule(".".join(module_name.split('.')[:-1]))
            module_suffix = module_name.split('.')[-1]

            new_module = QuantPretrainLinear_v202(
                in_features=module.in_features,
                out_features=module.out_features,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                quant_block_size=config.quant_block_size,
                trainable_scaling=config.trainable_scaling,
                occ_quantile=config.occ_quantile,
                dge_k=config.dge_k,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            setattr(parent_module, module_suffix, new_module)

        self.forward = self.wrapped_model.forward

    
    ## 训练中在反向传播后需要调用这个函数来对所有量化权重进行更新
    def update_all_quantized_weights(self):
        """在每个训练步骤后，更新所有层的量化权重"""
        for module in self.wrapped_model.modules():
            if isinstance(module, QuantPretrainLinear_v202):
                module.update_quantized_weight()