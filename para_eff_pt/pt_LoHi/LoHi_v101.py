# LoHi_v101.py (Corrected Version)
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
# 1. NF4 (4-bit) 量化函数 (保持不变)
# =================================================================================
# ... (get_nf4_stats, quantize_nf4, dequantize_nf4 代码与之前相同，此处省略) ...
def get_nf4_stats(dtype=torch.bfloat16):
    nf4_values = torch.tensor([
        -1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, -0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0000
    ], dtype=dtype)
    nf4_boundaries = (nf4_values[:-1] + nf4_values[1:]) / 2.0
    return nf4_values, nf4_boundaries

def quantize_nf4(w: Tensor, block_size=64, dtype=torch.bfloat16):
    assert w.dtype in [torch.float32, torch.bfloat16]
    device = w.device
    nf4_values, nf4_boundaries = get_nf4_stats(dtype=dtype)
    nf4_values = nf4_values.to(device)
    nf4_boundaries = nf4_boundaries.to(device)
    orig_shape = w.shape
    w_flat = w.flatten()
    pad_len = (block_size - (w_flat.numel() % block_size)) % block_size
    if pad_len > 0:
        w_flat = F.pad(w_flat, (0, pad_len))
    w_blocks = w_flat.view(-1, block_size)
    scales = w_blocks.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
    w_normalized = w_blocks / scales
    w_quant_indices = torch.searchsorted(nf4_boundaries, w_normalized)
    w_quant_indices_byte = w_quant_indices.to(torch.uint8)
    packed_w = (w_quant_indices_byte[:, ::2] << 4) | w_quant_indices_byte[:, 1::2]
    return packed_w, scales, orig_shape

def dequantize_nf4(quantized_w: Tensor, scales: Tensor, shape: torch.Size, block_size=64, dtype=torch.bfloat16):
    device = quantized_w.device
    nf4_values, _ = get_nf4_stats(dtype=dtype)
    nf4_values = nf4_values.to(device)
    first_indices = (quantized_w >> 4).to(torch.long)
    second_indices = (quantized_w & 0x0F).to(torch.long)
    unpacked_indices = torch.stack([first_indices, second_indices], dim=-1).view(-1, block_size)
    dequantized_blocks = nf4_values[unpacked_indices] * scales
    dequantized_flat = dequantized_blocks.flatten()
    dequantized = dequantized_flat[:shape.numel()].view(shape)
    return dequantized

# =================================================================================
# 2. ★★★ 修正后的 FP8 量化和反量化函数 ★★★
# =================================================================================

def quantize_fp8(w: Tensor, block_size=64, dtype=torch.bfloat16):
    assert w.dtype in [torch.float32, torch.bfloat16]
    orig_shape = w.shape
    w_flat = w.flatten()
    pad_len = (block_size - (w_flat.numel() % block_size)) % block_size
    if pad_len > 0:
        w_flat = F.pad(w_flat, (0, pad_len))
    w_blocks = w_flat.view(-1, block_size)
    scales = w_blocks.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
    w_normalized = w_blocks / scales
    quantized_w_fp8 = w_normalized.to(torch.float8_e4m3fn)
    
    # ★★★ 关键修改: 将FP8的底层比特重新解释为uint8进行存储 ★★★
    quantized_w_uint8 = quantized_w_fp8.view(torch.uint8)
    
    return quantized_w_uint8, scales, orig_shape

def dequantize_fp8(quantized_w: Tensor, scales: Tensor, shape: torch.Size, block_size=64, dtype=torch.bfloat16):
    # ★★★ 关键修改: 期望的输入是uint8容器 ★★★
    assert quantized_w.dtype == torch.uint8, f"Expected uint8 but got {quantized_w.dtype}"
    
    # ★★★ 关键修改: 将uint8的底层比特重新解释回FP8 ★★★
    quantized_w_fp8 = quantized_w.view(torch.float8_e4m3fn)
    
    dequantized_blocks = quantized_w_fp8.to(dtype) * scales
    dequantized_flat = dequantized_blocks.flatten()
    dequantized = dequantized_flat[:shape.numel()].view(shape)
    return dequantized

# =================================================================================
# 3. 模型和线性层 (其余部分无需修改)
# =================================================================================

@dataclass
class QuantPretrainConfig_v101:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    quant_bits: int = 8
    quant_block_size: int = 64
    trainable_scaling: bool = False

class QuantPretrainLinear_v101(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        lora_alpha: float = 32,
        lora_dropout: float = 0.0,
        quant_bits: int = 8,
        quant_block_size: int = 64,
        trainable_scaling: bool = False,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if r < 0:
            raise ValueError("r must be non-negative.")
        assert quant_bits in [4, 8], "quant_bits must be 4 or 8."

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.trainable_scaling = trainable_scaling
        self.quant_bits = quant_bits
        self.quant_block_size = quant_block_size
        self.dtype = dtype

        self.W0_fp = Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.W0_fp, a=math.sqrt(5))
        self.W0_fp.requires_grad = True

        if self.quant_bits == 4:
            packed_w, scales, _ = quantize_nf4(self.W0_fp.data, block_size=self.quant_block_size, dtype=self.dtype)
        else:
            packed_w, scales, _ = quantize_fp8(self.W0_fp.data, block_size=self.quant_block_size, dtype=self.dtype)
        
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

    def update_quantized_weight(self):
        if self.quant_bits == 4:
            packed_w, scales, _ = quantize_nf4(self.W0_fp.data, block_size=self.quant_block_size, dtype=self.dtype)
        else:
            packed_w, scales, _ = quantize_fp8(self.W0_fp.data, block_size=self.quant_block_size, dtype=self.dtype)
        self.W0_low.data = packed_w
        self.W0_scales.data = scales

    def forward(self, x: Tensor):
        if self.quant_bits == 4:
            w_dequant = dequantize_nf4(self.W0_low, self.W0_scales, self.W0_fp.shape, block_size=self.quant_block_size, dtype=self.dtype)
            x_quant, x_scales, x_shape = quantize_nf4(x, block_size=self.quant_block_size, dtype=self.dtype)
            x_dequant = dequantize_nf4(x_quant, x_scales, x_shape, block_size=self.quant_block_size, dtype=self.dtype)
        else:
            w_dequant = dequantize_fp8(self.W0_low, self.W0_scales, self.W0_fp.shape, block_size=self.quant_block_size, dtype=self.dtype)
            x_quant, x_scales, x_shape = quantize_fp8(x, block_size=self.quant_block_size, dtype=self.dtype)
            x_dequant = dequantize_fp8(x_quant, x_scales, x_shape, block_size=self.quant_block_size, dtype=self.dtype)

        weight_for_forward = self.W0_fp + (w_dequant - self.W0_fp).detach()
        path1_out = F.linear(x_dequant, weight_for_forward)

        path2_out = 0
        if self.r > 0:
            lora_scaling = self.scaling.tanh() if self.trainable_scaling else self.scaling
            path2_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * lora_scaling

        out = path1_out + path2_out
        if self.bias is not None:
            out += self.bias
        return out

# QuantPretrainModel_v101 定义保持不变
class QuantPretrainModel_v101(nn.Module):
    def __init__(self, model, config: QuantPretrainConfig_v101):
        super().__init__()
        self.wrapped_model = model
        self._config = config

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(target_key in module_name for target_key in config.target_modules):
                continue
            
            print(f"Applying Quantized Pre-training (v101) to module: {module_name}")
            
            parent_module = self.wrapped_model.get_submodule(".".join(module_name.split('.')[:-1]))
            module_suffix = module_name.split('.')[-1]

            new_module = QuantPretrainLinear_v101(
                in_features=module.in_features,
                out_features=module.out_features,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                quant_bits=config.quant_bits,
                quant_block_size=config.quant_block_size,
                trainable_scaling=config.trainable_scaling,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            setattr(parent_module, module_suffix, new_module)

        self.forward = self.wrapped_model.forward

    ## 训练中在反向传播后需要调用这个函数来对所有量化权重进行更新
    def update_all_quantized_weights(self):
        for module in self.wrapped_model.modules():
            if isinstance(module, QuantPretrainLinear_v101):
                module.update_quantized_weight()