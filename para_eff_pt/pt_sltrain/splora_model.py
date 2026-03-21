import os
import json
import math

from typing import List
from dataclasses import dataclass

import torch
from torch import nn
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig


@dataclass
class SpLoRaConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    sp_ratio: float
    sp_type: str
    target_modules: List[str]
    trainable_scaling: bool = False
    random_subspace: bool = False
    # --- 新增配置参数 ---
    use_hybrid_basis: bool = False  # 是否使用混合基分解
    r_fft: int = 16                 # 混合基分解中傅里叶部分的秩
    use_dynamic_restart: bool = False # 是否使用因子化EMA重启
    ema_decay: float = 0.999        # EMA更新的衰减率


class SpLoRaModel(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        target_modules,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        sp_ratio=0.01,
        sp_type="random",
        trainable_scaling=False,
        random_subspace=False,
        # --- 新增初始化参数 ---
        use_hybrid_basis: bool = False,
        r_fft: int = 16,
        use_dynamic_restart: bool = False,
        ema_decay: float = 0.999,
    ):
        if r < 0:
            raise ValueError("r must be nonnegative.")
        if sp_ratio <= 0 or sp_ratio >= 1:
            raise ValueError("sp_ratio must be between 0 and 1.")

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.sp_ratio = sp_ratio
        self.sp_type = sp_type
        self.trainable_scaling = trainable_scaling
        self.parameterized_modules = []
        
        # --- 新增属性 ---
        self.use_hybrid_basis = use_hybrid_basis
        self.r_fft = r_fft
        self.use_dynamic_restart = use_dynamic_restart
        self.ema_decay = ema_decay

        self._config = SpLoRaConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            sp_ratio=sp_ratio,
            sp_type=sp_type,
            target_modules=target_modules,
            random_subspace=random_subspace,
            trainable_scaling=trainable_scaling,
            # --- 保存新增配置 ---
            use_hybrid_basis=use_hybrid_basis,
            r_fft=r_fft,
            use_dynamic_restart=use_dynamic_restart,
            ema_decay=ema_decay,
        )

        # patch methods
        self.forward = self.wrapped_model.forward

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print(f"Reparameterized module: {module_name}")
            new_module = SpLoRaLinear(
                module.in_features,
                module.out_features,
                r=self.r,
                sp_ratio=sp_ratio,
                sp_type=sp_type,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                trainable_scaling=self.trainable_scaling,
                random_subspace=random_subspace,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
                # --- 传递新增参数 ---
                use_hybrid_basis=self.use_hybrid_basis,
                r_fft=self.r_fft,
                use_dynamic_restart=self.use_dynamic_restart,
                ema_decay=self.ema_decay,
            )

            # 释放原权重内存
            module.weight = None
            del module

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

        torch.cuda.empty_cache()

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def save_pretrained(self, path, max_shard_size="100GB"):
        self.wrapped_model.save_pretrained(path, safe_serialization=False)
        with open(os.path.join(path, "splora_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, "splora_config.json"), "r") as f:
            splora_config = json.load(f)

        config = AutoConfig.from_pretrained(path)

        base_model = AutoModelForCausalLM.from_config(config)
        
        # 向后兼容旧配置
        if "trainable_scaling" not in splora_config:
            splora_config["trainable_scaling"] = False
        if "use_hybrid_basis" not in splora_config:
            splora_config["use_hybrid_basis"] = False
        if "r_fft" not in splora_config:
            splora_config["r_fft"] = 16
        if "use_dynamic_restart" not in splora_config:
            splora_config["use_dynamic_restart"] = False
        if "ema_decay" not in splora_config:
            splora_config["ema_decay"] = 0.999

        model = cls(base_model, **splora_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        model.wrapped_model.load_state_dict(state_dict, strict=True)
        return model

# =================================================================================
# 原始的 autograd.Function (用于标准 W = BA + S)
# =================================================================================
class lora_sparse_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lora_B, lora_A, dv, di, bias):
        ctx.save_for_backward(input, lora_B, lora_A, dv, di, bias)
        return sparse_linear_forward(input, lora_B, lora_A, dv, di, bias)

    @staticmethod
    def backward(ctx, output_grad):
        input, lora_B, lora_A, dv, di, bias = ctx.saved_tensors
        grads = sparse_linear_backward(
            output_grad, input, lora_B, lora_A, dv, di,
            ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2],
            ctx.needs_input_grad[3], ctx.needs_input_grad[5], bias
        )
        return grads

# =================================================================================
# 新增的 autograd.Function (用于混合基分解 W = BA_std + BA_fft + S)
# =================================================================================
class lora_hybrid_sparse_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lora_B, lora_A, lora_B_fft, lora_A_fft, dv, di, bias):
        ctx.save_for_backward(input, lora_B, lora_A, lora_B_fft, lora_A_fft, dv, di, bias)
        return hybrid_sparse_linear_forward(input, lora_B, lora_A, lora_B_fft, lora_A_fft, dv, di, bias)

    @staticmethod
    def backward(ctx, output_grad):
        (input, lora_B, lora_A, lora_B_fft, lora_A_fft, dv, di, bias) = ctx.saved_tensors
        grads = hybrid_sparse_linear_backward(
            output_grad, input, lora_B, lora_A, lora_B_fft, lora_A_fft, dv, di,
            ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2],
            ctx.needs_input_grad[3], ctx.needs_input_grad[4], ctx.needs_input_grad[5],
            ctx.needs_input_grad[7], bias
        )
        return grads


class SpLoRaLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        sp_ratio: float = 0.01,
        sp_type: str = "random",
        *,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        trainable_scaling: bool = False,
        random_subspace: bool = False,
        bias=True,
        device=None,
        dtype=None,
        # --- 新增参数 ---
        use_hybrid_basis: bool = False,
        r_fft: int = 64,
        use_dynamic_restart: bool = True,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive.")
        if sp_ratio <= 0 or sp_ratio >= 1:
            raise ValueError("sp_ratio must be between 0 and 1.")

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.random_subspace = random_subspace
        self.trainable_scaling = trainable_scaling
        self.sp_ratio = sp_ratio
        self.sp_type = sp_type
        self.device = device
        self.dtype = dtype
        
        # --- 新增属性 ---
        self.use_hybrid_basis = use_hybrid_basis
        self.r_fft = r_fft
        self.use_dynamic_restart = use_dynamic_restart
        self.ema_decay = ema_decay

        # --- 标准低秩部分 (空间域) ---
        lora_A_requires_grad = not random_subspace
        self.lora_A = nn.Parameter(
            torch.empty(r, in_features, dtype=dtype, device=device),
            requires_grad=lora_A_requires_grad,
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(
            torch.empty(out_features, r, dtype=dtype, device=device)
        )
        nn.init.zeros_(self.lora_B)

        # --- 混合基分解的傅里叶部分 (频率域) ---
        if self.use_hybrid_basis:
            self.lora_A_fft = nn.Parameter(
                torch.empty(r_fft, in_features, dtype=torch.cfloat, device=device),
                requires_grad=True,
            )
            # Kaiming init不直接支持复数，手动实现
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lora_A_fft)
            std = 1 / math.sqrt(fan_in)
            with torch.no_grad():
                self.lora_A_fft.uniform_(-std, std)

            self.lora_B_fft = nn.Parameter(
                torch.empty(out_features, r_fft, dtype=torch.cfloat, device=device),
                requires_grad=True,
            )
            nn.init.zeros_(self.lora_B_fft)

        # --- 稀疏部分 ---
        if sp_type.lower() == "random":
            indices, values, shape = self._init_sparse_parameters()
            self.shape = shape
            self.register_buffer("sparse_index", indices.to(device))
            self.sparse_value = Parameter(values.to(device), requires_grad=True)

        # --- 偏置和缩放 ---
        if bias:
            self.bias = Parameter(
                torch.zeros(out_features, device=device, dtype=dtype, requires_grad=True)
            )
            a = 1 / math.sqrt(out_features)
            nn.init.uniform_(self.bias, -a, a)
        else:
            self.register_parameter("bias", None)

        if trainable_scaling:
            self.scaling = nn.Parameter(
                torch.tensor([1.0], device=device, dtype=dtype), requires_grad=True
            )
        else:
            self.scaling = self.lora_alpha / self.r
            
        # --- 因子化EMA重启机制的EMA Buffer ---
        if self.use_dynamic_restart:
            self.register_buffer('lora_A_ema', self.lora_A.detach().clone())
            self.register_buffer('lora_B_ema', self.lora_B.detach().clone())

    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()
        return self.scaling

    def _init_sparse_parameters(self):
        shape = [self.out_features, self.in_features]
        total_elements = self.in_features * self.out_features
        num_nonzeros = int(self.sp_ratio * total_elements)
        indices = torch.randperm(total_elements)[:num_nonzeros]
        indices, _ = torch.sort(indices)
        values = torch.empty(size=(num_nonzeros,), device=self.device, dtype=self.dtype)
        a = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(values, -a, a)
        return indices, values, shape

    def forward(self, x: Tensor):
        scaled_lora_A = self.lora_A * self._post_lora_scale()
        
        if self.use_hybrid_basis:
            # 使用混合基分解
            return lora_hybrid_sparse_linear.apply(
                x, self.lora_B, scaled_lora_A,
                self.lora_B_fft, self.lora_A_fft, # 傅里叶部分
                self.sparse_value, self.sparse_index, self.bias
            )
        else:
            # 使用标准分解
            return lora_sparse_linear.apply(
                x, self.lora_B, scaled_lora_A,
                self.sparse_value, self.sparse_index, self.bias
            )

    def extra_repr(self) -> str:
        repr_str = (
            f"in_features={self.in_features}, out_features={self.out_features}, rank={self.r}, "
            f"sparsity={self.sp_ratio}, bias={self.bias is not None}"
        )
        if self.use_hybrid_basis:
            repr_str += f", fft_rank={self.r_fft}"
        return repr_str

    @torch.no_grad()
    def update_ema(self):
        """
        更新低秩因子的指数移动平均值(EMA)。
        
        重要提示: 此方法应在训练循环中的 `optimizer.step()` 之后调用。
        例如:
            loss.backward()
            optimizer.step()
            model.update_ema() # 或者对每个模块调用
        
        这样可以确保EMA跟踪的是已更新的参数，并能正确处理梯度累积等情况。
        """
        if not self.use_dynamic_restart:
            return
        
        self.lora_A_ema.mul_(self.ema_decay).add_(self.lora_A.detach(), alpha=1 - self.ema_decay)
        self.lora_B_ema.mul_(self.ema_decay).add_(self.lora_B.detach(), alpha=1 - self.ema_decay)

    @torch.no_grad()
    def merge_and_reinit(self):
        """根据配置选择合适的重启方法"""
        if self.use_dynamic_restart:
            self._merge_and_reinit_dynamic()
        else:
            self._merge_and_reinit_simple()

    @torch.no_grad()
    def _merge_and_reinit_simple(self):
        """原始的重启方法：仅对BA做SVD分解"""
        print("Performing simple restart...")
        lora_A = self.lora_A.detach()
        lora_B = self.lora_B.detach()
        
        W_target = lora_B @ lora_A
        W_target_float = W_target.float()
        
        U, S, Vh = torch.linalg.svd(W_target_float, full_matrices=False)

        U_r = U[:, :self.r].to(self.dtype)
        S_r = S[:self.r].to(self.dtype)
        Vh_r = Vh[:self.r, :].to(self.dtype)

        sqrt_S_r = torch.sqrt(S_r)
        new_B = torch.matmul(U_r, torch.diag(sqrt_S_r))
        new_A = torch.matmul(torch.diag(sqrt_S_r), Vh_r)

        self.lora_B.copy_(new_B)
        self.lora_A.copy_(new_A)
        print("Simple restart finished.")

    @torch.no_grad()
    def _merge_and_reinit_dynamic(self):
        """新的重启方法：因子化EMA与残差再分解"""
        print("Performing dynamic restart with factorized EMA and residual re-decomposition...")
        # 步骤 1: 结合平滑的低秩历史(EMA)和最新的稀疏修正S，形成高质量目标权重
        W_ema = self.lora_B_ema @ self.lora_A_ema
        
        # 将稀疏部分加到W_ema上得到W_target
        W_target = W_ema.clone()
        W_target.view(-1).scatter_add_(0, self.sparse_index.to(torch.int64), self.sparse_value)
        
        # 步骤 2: 对平滑的 B_ema*A_ema 进行SVD分解，得到新的低秩基底
        W_ema_float = W_ema.float()
        U, S, Vh = torch.linalg.svd(W_ema_float, full_matrices=False)
        
        U_r = U[:, :self.r].to(self.dtype)
        S_r = S[:self.r].to(self.dtype)
        Vh_r = Vh[:self.r, :].to(self.dtype)

        # 步骤 3: 重新初始化 B_new, A_new
        sqrt_S_r = torch.sqrt(S_r)
        new_B = torch.matmul(U_r, torch.diag(sqrt_S_r))
        new_A = torch.matmul(torch.diag(sqrt_S_r), Vh_r)
        
        # 步骤 4: 计算残差 Residual = W_target - B_new*A_new
        Residual = W_target - (new_B @ new_A)
        
        # 步骤 5: 用新的稀疏矩阵 S_new 近似残差 (Top-K幅值剪枝)
        num_nonzeros = self.sparse_value.numel()
        residual_flat = Residual.view(-1)
        
        # 寻找残差中绝对值最大的K个值及其索引
        _, top_k_indices = torch.topk(residual_flat.abs(), k=num_nonzeros)
        
        new_sparse_values = residual_flat[top_k_indices]
        new_sparse_indices = top_k_indices.to(self.sparse_index.device)
        # 保持索引有序，这对于某些稀疏操作库可能是必要的，也是一个好习惯
        new_sparse_indices, sort_perm = torch.sort(new_sparse_indices)
        new_sparse_values = new_sparse_values[sort_perm]

        # 步骤 6: 用新参数替换旧参数
        self.lora_B.copy_(new_B)
        self.lora_A.copy_(new_A)
        self.sparse_value.data.copy_(new_sparse_values)
        # sparse_index是buffer，可以直接用新张量替换
        self.sparse_index.data.copy_(new_sparse_indices)
        
        # 重启后，将EMA重置为当前值，以开始新的累积周期
        self.lora_A_ema.copy_(self.lora_A.detach())
        self.lora_B_ema.copy_(self.lora_B.detach())
        
        print("Dynamic restart finished.")


# =================================================================================
# 辅助计算函数
# =================================================================================

def sparse_linear_forward(input, lora_B, lora_A, dv, di, bias=None):
    device = input.device
    dtype = input.dtype
    W = lora_B.to(device, dtype) @ lora_A.to(device, dtype)
    W.view(-1).scatter_add_(0, di.to(device, torch.int64), dv.to(device, dtype))
    return torch.nn.functional.linear(
        input, W, None if bias is None else bias.to(device, dtype)
    )

def sparse_linear_backward(output_grad, input, lora_B, lora_A, dv, di,
                           input_needs_grad, lora_B_needs_grad, lora_A_needs_grad,
                           dv_needs_grad, bias_needs_grad, bias=None):
    device = input.device
    dtype = input.dtype
    
    W = lora_B.to(device, dtype) @ lora_A.to(device, dtype)
    W.view(-1).scatter_add_(0, di.to(device, torch.int64), dv.to(device, dtype))

    output_grad_2d = output_grad.reshape(-1, output_grad.size(-1)).to(device, dtype)
    input_2d = input.view(-1, input.size(-1)).to(device, dtype)

    input_grad = output_grad_2d @ W if input_needs_grad else None
    if input_grad is not None:
        input_grad = input_grad.view_as(input)

    weight_grad = output_grad_2d.t() @ input_2d if (lora_A_needs_grad or lora_B_needs_grad or dv_needs_grad) else None
    
    lora_A_grad = lora_B.t() @ weight_grad if lora_A_needs_grad else None
    lora_B_grad = weight_grad @ lora_A.t() if lora_B_needs_grad else None
    
    dv_grad = weight_grad.view(-1).gather(0, di.to(torch.int64)) if dv_needs_grad else None
    bias_grad = output_grad_2d.sum(0) if bias is not None and bias_needs_grad else None

    return input_grad, lora_B_grad, lora_A_grad, dv_grad, None, bias_grad

def hybrid_sparse_linear_forward(input, lora_B, lora_A, lora_B_fft, lora_A_fft, dv, di, bias=None):
    device = input.device
    dtype = input.dtype
    
    # 空间域 + 稀疏部分
    W_spatial = lora_B.to(device, dtype) @ lora_A.to(device, dtype)
    W_spatial.view(-1).scatter_add_(0, di.to(device, torch.int64), dv.to(device, dtype))
    output_spatial = F.linear(input, W_spatial, bias)
    
    # 频率域部分
    x_fft = torch.fft.fft(input)
    W_fft = lora_B_fft @ lora_A_fft
    output_fft_freq = F.linear(x_fft, W_fft)
    output_fft_spatial = torch.fft.ifft(output_fft_freq).real # 取实部
    
    return output_spatial + output_fft_spatial

def hybrid_sparse_linear_backward(output_grad, input, lora_B, lora_A, lora_B_fft, lora_A_fft, dv, di,
                                  input_needs_grad, lora_B_needs_grad, lora_A_needs_grad,
                                  lora_B_fft_needs_grad, lora_A_fft_needs_grad,
                                  dv_needs_grad, bias_needs_grad, bias=None):
    device = input.device
    dtype = input.dtype
    
    # --- 空间域 + 稀疏部分的反向传播 ---
    W_spatial = lora_B.to(device, dtype) @ lora_A.to(device, dtype)
    W_spatial.view(-1).scatter_add_(0, di.to(device, torch.int64), dv.to(device, dtype))
    
    output_grad_2d = output_grad.reshape(-1, output_grad.size(-1)).to(device, dtype)
    input_2d = input.view(-1, input.size(-1)).to(device, dtype)
    
    input_grad_spatial = output_grad_2d @ W_spatial if input_needs_grad else None
    
    weight_grad_spatial = output_grad_2d.t() @ input_2d if (lora_A_needs_grad or lora_B_needs_grad or dv_needs_grad) else None
    
    lora_A_grad = lora_B.t() @ weight_grad_spatial if lora_A_needs_grad else None
    lora_B_grad = weight_grad_spatial @ lora_A.t() if lora_B_needs_grad else None
    dv_grad = weight_grad_spatial.view(-1).gather(0, di.to(torch.int64)) if dv_needs_grad else None
    bias_grad = output_grad_2d.sum(0) if bias is not None and bias_needs_grad else None

    # --- 频率域部分的反向传播 ---
    lora_A_fft_grad, lora_B_fft_grad, input_grad_fft = (None, None, None)
    
    grad_output_fft_spatial = output_grad
    grad_output_fft_freq = torch.fft.fft(grad_output_fft_spatial)
    
    x_fft = torch.fft.fft(input)
    x_fft_2d = x_fft.view(-1, x_fft.size(-1))
    grad_output_fft_freq_2d = grad_output_fft_freq.view(-1, grad_output_fft_freq.size(-1))

    if input_needs_grad:
        W_fft = lora_B_fft @ lora_A_fft
        grad_x_fft = grad_output_fft_freq_2d @ W_fft
        input_grad_fft = torch.fft.ifft(grad_x_fft.view_as(x_fft)).real

    if lora_A_fft_needs_grad or lora_B_fft_needs_grad:
        grad_W_fft = grad_output_fft_freq_2d.conj().t() @ x_fft_2d
        if lora_A_fft_needs_grad:
            lora_A_fft_grad = lora_B_fft.conj().t() @ grad_W_fft
        if lora_B_fft_needs_grad:
            lora_B_fft_grad = grad_W_fft @ lora_A_fft.conj().t()

    # --- 合并梯度 ---
    input_grad = None
    if input_needs_grad:
        input_grad = input_grad_spatial.view_as(input) + input_grad_fft

    # 返回所有梯度，顺序必须与forward的输入严格对应
    return (input_grad, lora_B_grad, lora_A_grad, lora_B_fft_grad, lora_A_fft_grad, dv_grad, None, bias_grad)