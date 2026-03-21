# DTR_AdamW_final.py

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import math

class SA_GAM_AdamW(Optimizer):
    """
    The final, production-ready implementation of DTR-AdamW.

    Features:
    - EFFICIENT global agreement calculation suitable for large-scale LLMs.
    - Correct, resettable bias correction for the first moment.
    - Wandb-friendly: Exposes the `self.agreement_ema` attribute directly
      on the optimizer instance for easy logging after each step.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        beta_agreement=0.9,
        agreement_threshold=-0.5,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta_agreement < 1.0:
            raise ValueError(f"Invalid beta_agreement: {beta_agreement}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            beta_agreement=beta_agreement,
            agreement_threshold=agreement_threshold,
        )
        super(SA_GAM_AdamW, self).__init__(params, defaults)
        
        # Public attribute for easy logging (e.g., with wandb)
        self.agreement_ema = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # --- Part 1: EFFICIENT Global Agreement Calculation & Reset Decision ---
        beta_agreement = self.defaults['beta_agreement']
        agreement_threshold = self.defaults['agreement_threshold']
        
        global_dot_product = 0.0
        global_grad_norm_sq = 0.0
        global_momentum_norm_sq = 0.0
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    continue
                
                global_dot_product += torch.sum(p.grad * state['exp_avg'])
                global_grad_norm_sq += torch.sum(p.grad * p.grad)
                global_momentum_norm_sq += torch.sum(state['exp_avg'] * state['exp_avg'])

        reset_needed = False
        if global_grad_norm_sq > 0 and global_momentum_norm_sq > 0:
            agreement = global_dot_product / (torch.sqrt(global_grad_norm_sq) * torch.sqrt(global_momentum_norm_sq) + 1e-10)
            agreement = agreement.item()
            
            self.agreement_ema = beta_agreement * self.agreement_ema + (1 - beta_agreement) * agreement
            
            if self.agreement_ema < agreement_threshold:
                reset_needed = True
                self.agreement_ema = 0.0

        # --- Part 2: Parameter-wise Update ---
        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m1_step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                state["step"] += 1
                state["m1_step"] += 1

                if reset_needed:
                    exp_avg.zero_()
                    state["m1_step"] = 0

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad.conj(), value=1.0 - beta2)
                
                bias_correction1 = 1.0 - beta1 ** state["m1_step"] if state["m1_step"] > 0 else 1.0
                bias_correction2 = 1.0 - beta2 ** state["step"]

                denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1
                
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])
                
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss