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
# 1. 4-bit 伪量化函数库
# =================================================================================

# ------------------------- NF4 (NormalFloat 4-bit) -----------------------------

def get_nf4_stats(dtype=torch.bfloat16):
    """获取NF4数据类型的预计算分位数。"""
    nf4_values = torch.tensor([
        -1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, -0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0000
    ], dtype=dtype)
    nf4_boundaries = (nf4_values[:-1] + nf4_values[1:]) / 2.0
    return nf4_values, nf4_boundaries

def quantize_nf4_fake(w: Tensor, block_size=64):
    """对输入张量w执行NF4伪量化 (Fake Quantization)。"""
    if not w.is_floating_point():
        raise ValueError(f"Input tensor must be a floating point type, but got {w.dtype}")
    
    device, orig_dtype = w.device, w.dtype
    nf4_values, nf4_boundaries = get_nf4_stats(dtype=orig_dtype)
    nf4_values, nf4_boundaries = nf4_values.to(device), nf4_boundaries.to(device)

    orig_shape = w.shape
    w_flat = w.flatten()
    pad_len = (block_size - (w_flat.numel() % block_size)) % block_size
    if pad_len > 0:
        w_flat = F.pad(w_flat, (0, pad_len), 'replicate')
    w_blocks = w_flat.view(-1, block_size)

    scales = w_blocks.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
    w_normalized = w_blocks / scales
    w_quant_indices = torch.searchsorted(nf4_boundaries, w_normalized)
    dequantized_blocks = nf4_values[w_quant_indices] * scales
    
    dequantized_flat = dequantized_blocks.flatten()
    dequantized = dequantized_flat[:orig_shape.numel()].view(orig_shape)
    
    return dequantized.to(orig_dtype)

# ------------------------- 标准 MXFP4 (E2M1) -----------------------------------

def quantize_mxfp4_fake(w: Tensor, block_size=32):
    """
    对输入张量w执行标准的、非随机化的MXFP4 (E2M1) 伪量化。
    实现遵循 OCP Microscaling Formats 规范。
    """
    if not w.is_floating_point():
        raise ValueError(f"Input tensor must be a floating point type, but got {w.dtype}")

    device, orig_dtype = w.device, w.dtype
    
    # FP4 (E2M1) 的正数量化值集合 (包括0和subnormal)
    # E2M1: 1 sign, 2 exponent, 1 mantissa. bias=1.
    # Values: 0, 0.5 (subnormal), 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    mxfp4_pos_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device, dtype=orig_dtype)
    emax_elem = 2 # FP4 E2M1最大正规数的指数 floor(log2(6.0))

    orig_shape = w.shape
    w_flat = w.flatten()
    pad_len = (block_size - (w_flat.numel() % block_size)) % block_size
    if pad_len > 0:
        w_flat = F.pad(w_flat, (0, pad_len), 'replicate')
    w_blocks = w_flat.view(-1, block_size)

    # 1. 计算每个块的共享缩放因子
    max_abs_vals = w_blocks.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
    shared_exp = torch.floor(torch.log2(max_abs_vals)) - emax_elem
    scales = torch.pow(2.0, shared_exp)

    # 2. 归一化
    w_normalized = w_blocks / scales

    # 3. 量化到FP4 (最近邻舍入)
    # a. 取绝对值，准备与正数集合比较
    w_abs_normalized = w_normalized.abs()
    
    # b. 找到每个值与哪个FP4值的距离最近
    #    (N, B, 1) - (V) -> (N, B, V) -> (N, B)
    indices = torch.abs(w_abs_normalized.unsqueeze(-1) - mxfp4_pos_values).argmin(dim=-1)
    
    # c. 获取量化后的绝对值，并恢复原始符号
    quantized_abs = mxfp4_pos_values[indices]
    quantized_normalized = torch.copysign(quantized_abs, w_normalized)

    # 4. 反量化：应用缩放因子，恢复原始数值范围
    dequantized_blocks = quantized_normalized * scales
    
    dequantized_flat = dequantized_blocks.flatten()
    dequantized = dequantized_flat[:orig_shape.numel()].view(orig_shape)
    
    return dequantized.to(orig_dtype)


# =================================================================================
# 2. 修改后的模型和线性层 (干净、简洁)
# =================================================================================

@dataclass
class QuantPretrainConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    quant_format: str = 'nf4'  # 超参: 'nf4' 或 'mxfp4'
    quant_bits: int = 4
    quant_block_size: int = 64
    trainable_scaling: bool = False

class QuantPretrainLinear(nn.Module):
    """
    实现了LoHi算法的线性层，支持NF4和标准的MXFP4两种量化格式。
    已将 lora_A/lora_B 换成显式 Parameter 形式。
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        lora_alpha: float = 32,
        lora_dropout: float = 0.0,
        quant_format: str = 'nf4',
        quant_bits: int = 4,
        quant_block_size: int = 32,
        trainable_scaling: bool = False,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if r < 0:
            raise ValueError("LoRA rank 'r' must be non-negative.")
        if quant_bits != 4:
            raise NotImplementedError("Currently only 4-bit quantization is supported.")

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.trainable_scaling = trainable_scaling
        self.quant_block_size = quant_block_size
        self.dtype = dtype
        self.device = device

        # 根据超参选择量化函数
        self.quant_format = quant_format
        if self.quant_format == 'nf4':
            self.quantize_fake_fn = quantize_nf4_fake
        elif self.quant_format == 'mxfp4':
            if quant_block_size != 32:
                print(f"Warning: MXFP4 typically uses a block size of 32, but got {quant_block_size}.")
            self.quantize_fake_fn = quantize_mxfp4_fake
        else:
            raise ValueError(f"Unsupported quant_format: '{self.quant_format}'. Must be 'nf4' or 'mxfp4'.")

        # 1. 基础权重 W0 (可训练的全精度影子参数)
        self.W0_fp = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.W0_fp, a=math.sqrt(5))

        # 2. 偏置项
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W0_fp)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # 3. LoRA 适配器 (B @ A) —— 显式 Parameter
        if r > 0:
            # A: (r, in_features)
            self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
            # B: (out_features, r)
            self.lora_B = nn.Parameter(torch.empty(out_features, r, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        # 4. LoRA 缩放因子
        if r > 0:
            scaling_val = lora_alpha / r
            self.scaling = (
                nn.Parameter(torch.tensor([scaling_val], device=device, dtype=dtype))
                if trainable_scaling
                else scaling_val
            )
        else:
            self.scaling = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- 路径1: 高秩低精路径 (W_low @ X_low) ---
        w_dequant = self.quantize_fake_fn(self.W0_fp, block_size=self.quant_block_size)
        weight_for_forward = self.W0_fp + (w_dequant - self.W0_fp).detach()

        x_dequant = self.quantize_fake_fn(x, block_size=self.quant_block_size)
        x_for_forward = x + (x_dequant - x).detach()

        path1_out = F.linear(x_for_forward, weight_for_forward)

        # --- 路径2: 低秩高精路径 (B @ A @ X) ---
        path2_out = 0
        if self.r > 0:
            # 手动实现两级线性： x_drop = dropout(x)  ->  z = A @ x_drop.T  ->  out = B @ z
            x_drop = self.lora_dropout(x)                       # (batch, in_features)
            z = F.linear(x_drop, self.lora_A)                   # (batch, r)
            lora_out = F.linear(z, self.lora_B)                 # (batch, out_features)
            path2_out = lora_out * self.scaling

        # --- 合并路径和偏置 ---
        out = path1_out + path2_out
        if self.bias is not None:
            out += self.bias
        return out
    
    def freeze_ba(self):
        if self.r > 0: self.lora_A.weight.requires_grad = False; self.lora_B.weight.requires_grad = False
    
    def unfreeze_ba(self):
        if self.r > 0: self.lora_A.weight.requires_grad = True; self.lora_B.weight.requires_grad = True
        self.W0_fp.requires_grad = False
        
        
    @torch.no_grad()
    def merge_and_reinit(self):

        dtype = self.lora_A.dtype  
        
        lora_A = self.lora_A.detach()
        lora_B = self.lora_B.detach()
        
        ## ---------------------------------------------------------------
        # Step 1: Compute W = BA
        W_target = lora_B @ lora_A
        
        W_target_float = W_target.float()
        
        # Step 2: Perform SVD decomposition: W = UΣV^T
        U, S, Vh = torch.linalg.svd(W_target_float, full_matrices=False)  # U, S (singular values), Vh (V^T)

        # Step 3: Keep only the top-r singular values and vectors
        U_r = U[:, :self.r].to(dtype)  # Top-r left singular vectors
        S_r = S[:self.r].to(dtype)     # Top-r singular values
        Vh_r = Vh[:self.r, :].to(dtype) # Top-r right singular vectors

        # Step 4: Reinitialize B and A
        # B = U_r * sqrt(Σ_r)
        # A = sqrt(Σ_r) * Vh_r
        sqrt_S_r = torch.sqrt(S_r)
        new_B = torch.matmul(U_r, torch.diag(sqrt_S_r))  # U_r * sqrt(Σ_r)
        new_A = torch.matmul(torch.diag(sqrt_S_r), Vh_r)  # sqrt(Σ_r) * Vh_r

        # Step 5: Assign new values to lora_A and lora_B
        self.lora_B.copy_(new_B)
        self.lora_A.copy_(new_A)
        ## ---------------------------------------------------------------


class QuantPretrainModel(nn.Module):
    """模型包装器，用于将目标 nn.Linear 替换为 QuantPretrainLinear。"""
    def __init__(self, model, config: QuantPretrainConfig):
        super().__init__()
        self.wrapped_model = model
        self._config = config

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear): continue
            if not any(target_key in module_name for target_key in config.target_modules): continue
            
            print(f"Applying LoHi Quantized Pre-training (format: {config.quant_format}) to module: {module_name}")
            
            parent_module = self.wrapped_model.get_submodule(".".join(module_name.split('.')[:-1]))
            module_suffix = module_name.split('.')[-1]

            new_module = QuantPretrainLinear(
                in_features=module.in_features, out_features=module.out_features,
                r=config.r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                quant_format=config.quant_format, quant_bits=config.quant_bits,
                quant_block_size=config.quant_block_size, trainable_scaling=config.trainable_scaling,
                bias=module.bias is not None, device=module.weight.device, dtype=module.weight.dtype,
            )
            setattr(parent_module, module_suffix, new_module)

        self.forward = self.wrapped_model.forward