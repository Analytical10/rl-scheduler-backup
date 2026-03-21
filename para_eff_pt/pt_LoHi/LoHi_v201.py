# LoHi_v201.py (Optimized & Fixed)
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
# 1. NVFP4 精确模拟量化函数 (已优化并修复)
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
    if device not in E2M1_CACHE or E2M1_CACHE[device]['lut'].dtype != dtype:
        lut = generate_e2m1_lut(dtype=dtype, device=device)
        boundaries = (lut[:-1] + lut[1:]) / 2.0
        E2M1_CACHE[device] = {'lut': lut, 'boundaries': boundaries}
    return E2M1_CACHE[device]['lut'], E2M1_CACHE[device]['boundaries']

def quantize_nvfp4_optimized(w: Tensor, block_size=16, dtype=torch.bfloat16, use_stochastic_rounding=False):
    """ ★★★ 修复：同时支持确定性舍入和随机舍入的优化版函数 ★★★ """
    e2m1_lut, e2m1_boundaries = get_e2m1_lut_and_boundaries(w.device, dtype)
    
    orig_shape = w.shape
    w_flat = w.flatten()
    pad_len = (block_size - (w_flat.numel() % block_size)) % block_size
    if pad_len > 0: w_flat = F.pad(w_flat, (0, pad_len))
    w_blocks = w_flat.view(-1, block_size)

    scales = w_blocks.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
    scales_q = quantize_to_fp_format(scales, num_exponent_bits=4, num_mantissa_bits=3)
    
    w_normalized = w_blocks / scales_q 

    # ★★★ 核心修复：根据 use_stochastic_rounding 选择不同路径 ★★★
    if not use_stochastic_rounding:
        # 路径A: 确定性舍入 (使用高效的二分查找)
        indices = torch.searchsorted(e2m1_boundaries, w_normalized)
    else:
        # 路径B: 随机舍入 (Stochastic Rounding)
        # 这里的暴力搜索是不可避免的，因为它需要找到最近的两个点
        dist = (w_normalized.unsqueeze(-1) - e2m1_lut).abs()
        _, two_closest_indices = torch.topk(dist, 2, dim=-1, largest=False)
        q1_indices, q2_indices = two_closest_indices[..., 0], two_closest_indices[..., 1]
        
        q1 = e2m1_lut[q1_indices]
        q2 = e2m1_lut[q2_indices]
        
        # 计算插值概率
        p = (w_normalized - q1) / (q2 - q1 + 1e-8)
        rand_val = torch.rand_like(p)
        indices = torch.where(rand_val < p, q2_indices, q1_indices)

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

# ★★★ 将旧函数名指向新的优化版函数，确保代码其他部分无缝调用 ★★★
quantize_nvfp4 = quantize_nvfp4_optimized

# =================================================================================
# 2. 自定义Autograd函数，用于分离式舍入
# =================================================================================
class DequantizeSTE_SR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W0_fp, W0_low, W0_scales, qaf_enabled, quant_block_size, dtype):
        ctx.qaf_enabled = qaf_enabled
        ctx.quant_block_size = quant_block_size
        ctx.dtype = dtype
        w_dequant = dequantize_nvfp4(W0_low, W0_scales, W0_fp.shape, block_size=quant_block_size, dtype=dtype)
        return W0_fp + (w_dequant - W0_fp).detach()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.qaf_enabled:
            return grad_output, None, None, None, None, None
        else:
            grad_quant, grad_scales, _ = quantize_nvfp4(grad_output, block_size=ctx.quant_block_size, dtype=ctx.dtype, use_stochastic_rounding=True)
            grad_dequant = dequantize_nvfp4(grad_quant, grad_scales, grad_output.shape, block_size=ctx.quant_block_size, dtype=ctx.dtype)
            return grad_dequant, None, None, None, None, None

# =================================================================================
# 3. 模型和线性层 (v201)
# =================================================================================
@dataclass
class QuantPretrainConfig_v201:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    quant_bits: int = 4
    quant_block_size: int = 16
    trainable_scaling: bool = False

class QuantPretrainLinear_v201(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        lora_alpha: float = 32,
        lora_dropout: float = 0.0,
        quant_bits: int = 4,
        quant_block_size: int = 16,
        trainable_scaling: bool = False,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features, self.out_features, self.r = in_features, out_features, r
        self.lora_alpha, self.lora_dropout = lora_alpha, nn.Dropout(p=lora_dropout)
        self.trainable_scaling, self.quant_bits, self.quant_block_size, self.dtype = \
            trainable_scaling, quant_bits, quant_block_size, dtype
        
        self.W0_fp = Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.W0_fp, a=math.sqrt(5))
        self.W0_fp.requires_grad = True

        packed_w, scales, _ = quantize_nvfp4(self.W0_fp.data, block_size=self.quant_block_size, dtype=self.dtype)
        self.register_buffer('W0_low', packed_w)
        self.register_buffer('W0_scales', scales)
        
        if bias:
            self.bias = Parameter(torch.zeros(out_features, device=device, dtype=dtype))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W0_fp)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False, device=device, dtype=dtype)
            self.lora_B = nn.Linear(r, out_features, bias=False, device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(500))
        if trainable_scaling:
            self.scaling = Parameter(torch.tensor([1.0], device=device, dtype=dtype))
        else:
            self.scaling = self.lora_alpha / self.r if r > 0 else 0.0

        self.qaf_enabled = False

    def update_quantized_weight(self):
        packed_w, scales, _ = quantize_nvfp4(self.W0_fp.data, block_size=self.quant_block_size, dtype=self.dtype)
        self.W0_low.data = packed_w
        self.W0_scales.data = scales

    def forward(self, x: Tensor):
        weight_for_forward = DequantizeSTE_SR.apply(
            self.W0_fp, self.W0_low, self.W0_scales, self.qaf_enabled, self.quant_block_size, self.dtype
        )
        x_quant, x_scales, x_shape = quantize_nvfp4(x, block_size=self.quant_block_size, dtype=self.dtype)
        x_dequant = dequantize_nvfp4(x_quant, x_scales, x_shape, block_size=self.quant_block_size, dtype=self.dtype)
        path1_out = F.linear(x_dequant, weight_for_forward)

        path2_out = 0
        if self.r > 0:
            lora_scaling = self.scaling.tanh() if self.trainable_scaling else self.scaling
            path2_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * lora_scaling

        out = path1_out + path2_out
        if self.bias is not None:
            out += self.bias
        return out

class QuantPretrainModel_v201(nn.Module):
    """模型包装器，用于将目标 nn.Linear 替换为 QuantPretrainLinear_v201"""
    def __init__(self, model, config: QuantPretrainConfig_v201):
        super().__init__()
        self.wrapped_model = model
        self._config = config
        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear) or not any(k in module_name for k in config.target_modules):
                continue
            print(f"Applying Quantized Pre-training (v201) to module: {module_name}")
            parent_module = self.wrapped_model.get_submodule(".".join(module_name.split('.')[:-1]))
            module_suffix = module_name.split('.')[-1]
            new_module = QuantPretrainLinear_v201(
                in_features=module.in_features, out_features=module.out_features,
                r=config.r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                quant_bits=config.quant_bits, quant_block_size=config.quant_block_size,
                trainable_scaling=config.trainable_scaling, bias=module.bias is not None,
                device=module.weight.device, dtype=module.weight.dtype,
            )
            setattr(parent_module, module_suffix, new_module)
        self.forward = self.wrapped_model.forward

    
    ## 训练中在反向传播后需要调用这个函数来对所有量化权重进行更新
    def update_all_quantized_weights(self):
        """在每个训练步骤后，更新所有层的量化权重"""
        for module in self.wrapped_model.modules():
            if isinstance(module, QuantPretrainLinear_v201):
                module.update_quantized_weight()