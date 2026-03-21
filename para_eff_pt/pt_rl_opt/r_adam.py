# r_adam.py
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import Optimizer


@dataclass
class _RL_LRS_LogStats:
    step: int
    schedule_type: str
    base_lr: float
    warmup_damp: float
    action_scale: float
    action_mean: float
    action_std: float
    action_clip: float

    a_raw_mean: float
    a_raw_std: float
    a_clip_mean: float
    a_clip_std: float
    clip_frac: float
    clip_hi_frac: float
    clip_lo_frac: float

    lr_scale_mean: float
    lr_scale_p50: float
    lr_scale_p95: float
    lr_scale_p99: float
    lr_scale_min: float
    lr_scale_max: float

    lr_mean: float
    lr_p50: float
    lr_p95: float
    lr_p99: float
    lr_min: float
    lr_max: float

    lr_cap_factor: Optional[float]
    cap_frac_lr: float

    # Useful theoretical ceiling (if not capped): exp(k_t * action_clip)
    lr_scale_ceiling: float


class R_AdamW(Optimizer):
    """
    RL-LRS compatible AdamW wrapper (execution-side strategy):

    For each step and per-tensor param group i:
      1) sample action: a_i ~ Normal(action_mean, action_std)
      2) clip: a_i <- clip(a_i, [-action_clip, +action_clip])
      3) warmup damping: k_t = action_scale * warmup_damp(t)
      4) multiplicative scaling:
            lr_i = base_lr(t) * exp(k_t * a_i)

    Base LR schedules supported:
      - "wsd": warmup -> stable -> cosine decay (last wsd_final_ratio)
      - "cosine": warmup -> cosine decay (no stable plateau)

    Weight decay:
      - NOT perturbed.
      - Each step sets group['weight_decay'] = group['initial_weight_decay'].

    DDP:
      - optional rank0 sampling + broadcast.

    step():
      - supports closure and optional `loss` passthrough like your RL wrapper.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], List[Dict[str, Any]]],
        base_optimizer_cls=torch.optim.AdamW,

        # ---- base LR schedule ----
        schedule_type: str = "cosine",   # "wsd" or "cosine"
        max_lr: float = 3e-3,
        min_lr: float = 3e-4,
        warmup_steps: int = 1100,
        max_steps: int = 11000,
        # WSD-only:
        wsd_final_ratio: float = 0.10,  # last 10% steps do cosine decay

        # ---- RL-LRS action->lr hyperparams ----
        action_scale: float = 1.8,   # set to match your RL log
        action_mean: float = 0.0,
        action_std: float = 35.0,
        action_clip: float = 1.0,
        noise_warmup: bool = True,

        # ---- DDP sync ----
        ddp_broadcast_action: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        rank: int = 0,

        # ---- optional safety cap (NOT RL-LRS default) ----
        lr_cap_factor: Optional[float] = None,  # e.g., 7.0; None disables

        # ---- logging ----
        log_interval: int = 500,
        log_path: Optional[str] = None,  # JSONL on rank0
        log_param_count: bool = True,

        # ---- optimizer defaults (can be overridden by param groups) ----
        weight_decay: float = 0.1,
        **optimizer_kwargs,
    ):
        if "lr" in optimizer_kwargs:
            optimizer_kwargs.pop("lr")

        self.rank = int(rank)
        self.log_interval = int(log_interval)
        self.log_path = log_path
        self.log_param_count = bool(log_param_count)

        # Schedule params
        self.schedule_type = str(schedule_type).lower().strip()
        if self.schedule_type not in ("wsd", "cosine"):
            raise ValueError(f"schedule_type must be 'wsd' or 'cosine', got {schedule_type!r}")

        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.wsd_final_ratio = float(wsd_final_ratio)

        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {self.max_steps}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.min_lr < 0 or self.max_lr < 0:
            raise ValueError("min_lr and max_lr must be >= 0.")
        if self.min_lr > self.max_lr:
            raise ValueError(f"min_lr must be <= max_lr, got {self.min_lr} > {self.max_lr}")

        if self.schedule_type == "wsd":
            if not (0.0 < self.wsd_final_ratio < 1.0):
                raise ValueError(f"wsd_final_ratio must be in (0,1), got {self.wsd_final_ratio}")

        # Action params
        self.action_scale = float(action_scale)
        self.action_mean = float(action_mean)
        self.action_std = float(action_std)
        self.action_clip = float(action_clip)
        self.noise_warmup = bool(noise_warmup)

        if self.action_std < 0.0:
            raise ValueError(f"action_std must be >= 0, got {self.action_std}")
        if self.action_clip <= 0.0:
            raise ValueError(f"action_clip must be > 0, got {self.action_clip}")

        # DDP
        self.ddp_broadcast_action = bool(ddp_broadcast_action)

        # Safety
        self.lr_cap_factor = None if lr_cap_factor is None else float(lr_cap_factor)
        if self.lr_cap_factor is not None and self.lr_cap_factor <= 0.0:
            raise ValueError(f"lr_cap_factor must be > 0, got {self.lr_cap_factor}")

        # State
        self.current_step = 0
        self.last_action_raw: Optional[np.ndarray] = None
        self.last_action_clipped: Optional[np.ndarray] = None
        self.last_lr_scales: Optional[np.ndarray] = None

        # Device inference
        self._explicit_device = device is not None
        self.device = device  # may be None until we see params

        # Build per-tensor groups
        exploded_groups, all_params = self._explode_to_per_tensor_groups(
            params=params,
            default_weight_decay=float(weight_decay),
        )
        self.all_params = all_params
        self.num_params = len(self.all_params)

        if self.device is None:
            self.device = self.all_params[0].device if self.num_params > 0 else torch.device("cpu")
        else:
            self.device = torch.device(self.device)

        # base optimizer; lr placeholder (overwritten each step)
        self.base_optimizer = base_optimizer_cls(
            exploded_groups,
            lr=float(self.min_lr),
            **optimizer_kwargs,
        )

        # log file
        self._log_fh = None
        if (self.rank == 0) and (self.log_path is not None):
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            self._log_fh = open(self.log_path, "a", encoding="utf-8")

        if self.rank == 0:
            print(
                "R_AdamW (RL-LRS compat): per-tensor LR perturbations via "
                "lr_i = base_lr * exp(action_scale * warmup_damp * clip(N(action_mean,action_std),[-action_clip,action_clip]))"
            )
            print(
                f"Schedule: type={self.schedule_type} warmup_steps={self.warmup_steps} max_steps={self.max_steps} "
                f"min_lr={self.min_lr:.3e} max_lr={self.max_lr:.3e}"
                + (f" wsd_final_ratio={self.wsd_final_ratio:.3f}" if self.schedule_type == "wsd" else "")
            )
            print(
                f"Action: action_scale={self.action_scale:.4f} action_mean={self.action_mean:.4f} "
                f"action_std={self.action_std:.4f} action_clip={self.action_clip:.2f} "
                f"noise_warmup={self.noise_warmup} ddp_broadcast_action={self.ddp_broadcast_action}"
            )
            print("WeightDecay: NOT perturbed (each step resets to initial_weight_decay per group).")
            if self.lr_cap_factor is not None:
                print(f"Safety: lr_cap_factor={self.lr_cap_factor} (cap at base_lr * lr_cap_factor).")
            if self.log_param_count:
                print(f"R_AdamW: controlling {self.num_params} tensors individually.")
            if self._log_fh is not None:
                print(f"Logging: JSONL append to {self.log_path}")

    # passthrough
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @property
    def state(self):
        return self.base_optimizer.state

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        d = self.base_optimizer.state_dict()
        d["_r_adamw_wrapper"] = {
            "current_step": self.current_step,
            "schedule_type": self.schedule_type,
            "action_scale": self.action_scale,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
            "action_clip": self.action_clip,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "wsd_final_ratio": self.wsd_final_ratio,
            "noise_warmup": self.noise_warmup,
            "ddp_broadcast_action": self.ddp_broadcast_action,
            "lr_cap_factor": self.lr_cap_factor,
        }
        return d

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.base_optimizer.load_state_dict(state_dict)
        wrap = state_dict.get("_r_adamw_wrapper", None)
        if isinstance(wrap, dict):
            self.current_step = int(wrap.get("current_step", self.current_step))
            self.schedule_type = str(wrap.get("schedule_type", self.schedule_type)).lower().strip()
            self.action_scale = float(wrap.get("action_scale", self.action_scale))
            self.action_mean = float(wrap.get("action_mean", self.action_mean))
            self.action_std = float(wrap.get("action_std", self.action_std))
            self.action_clip = float(wrap.get("action_clip", self.action_clip))
            self.max_lr = float(wrap.get("max_lr", self.max_lr))
            self.min_lr = float(wrap.get("min_lr", self.min_lr))
            self.warmup_steps = int(wrap.get("warmup_steps", self.warmup_steps))
            self.max_steps = int(wrap.get("max_steps", self.max_steps))
            self.wsd_final_ratio = float(wrap.get("wsd_final_ratio", self.wsd_final_ratio))
            self.noise_warmup = bool(wrap.get("noise_warmup", self.noise_warmup))
            self.ddp_broadcast_action = bool(wrap.get("ddp_broadcast_action", self.ddp_broadcast_action))
            self.lr_cap_factor = wrap.get("lr_cap_factor", self.lr_cap_factor)

    # --- helpers ---
    def _explode_to_per_tensor_groups(
        self,
        params: Union[Iterable[torch.nn.Parameter], List[Dict[str, Any]]],
        default_weight_decay: float,
    ) -> Tuple[List[Dict[str, Any]], List[torch.nn.Parameter]]:
        if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
            in_groups = params  # type: ignore[assignment]
        else:
            in_params = list(params)
            in_groups = [{"params": in_params, "weight_decay": default_weight_decay}]

        exploded: List[Dict[str, Any]] = []
        all_params: List[torch.nn.Parameter] = []

        for g in in_groups:
            if "params" not in g:
                raise ValueError("Param group dict must contain key 'params'.")
            g_params = list(g["params"])
            g_wd = float(g.get("weight_decay", default_weight_decay) or 0.0)

            g_base = {k: v for k, v in g.items() if k != "params"}

            for p in g_params:
                if p is None:
                    continue
                if not isinstance(p, torch.Tensor):
                    raise TypeError(f"Expected torch.Tensor/Parameter, got {type(p)}")
                if self.device is None and not self._explicit_device:
                    self.device = p.device

                new_g = dict(g_base)
                new_g["params"] = [p]
                new_g["initial_weight_decay"] = g_wd
                new_g["weight_decay"] = g_wd
                new_g["lr"] = float(self.min_lr)

                exploded.append(new_g)
                all_params.append(p)

        return exploded, all_params

    def _warmup_damp(self) -> float:
        if not self.noise_warmup:
            return 1.0
        if self.current_step < self.warmup_steps:
            return float(self.current_step) / float(max(1, self.warmup_steps))
        return 1.0

    def _get_base_lr(self) -> float:
        step = int(self.current_step)

        # after training horizon
        if step > self.max_steps:
            return self.min_lr

        # warmup (linear min->max)
        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.max_lr
            frac = step / float(self.warmup_steps)
            return self.min_lr + (self.max_lr - self.min_lr) * frac

        # main schedule
        if self.schedule_type == "cosine":
            # cosine from warmup_steps..max_steps
            denom = max(1, (self.max_steps - self.warmup_steps))
            progress = (step - self.warmup_steps) / float(denom)  # 0..1
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

        # WSD: stable then cosine in last final_ratio
        stable_until = int(self.max_steps * (1.0 - self.wsd_final_ratio))
        stable_until = max(stable_until, self.warmup_steps)

        if step < stable_until:
            return self.max_lr

        denom = max(1, (self.max_steps - stable_until))
        progress = (step - stable_until) / float(denom)  # 0..1
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def _sample_action_raw(self) -> np.ndarray:
        n = self.num_params
        if n == 0:
            return np.zeros((0,), dtype=np.float32)

        # non-DDP or no broadcast
        if (not dist.is_initialized()) or (not self.ddp_broadcast_action):
            if self.action_std == 0.0:
                return np.full((n,), self.action_mean, dtype=np.float32)
            return (self.action_mean + self.action_std * np.random.randn(n)).astype(np.float32)

        # DDP broadcast: rank0 samples
        if self.rank == 0:
            if self.action_std == 0.0:
                a_np = np.full((n,), self.action_mean, dtype=np.float32)
            else:
                a_np = (self.action_mean + self.action_std * np.random.randn(n)).astype(np.float32)
            a_t = torch.tensor(a_np, device=self.device, dtype=torch.float32)
        else:
            a_t = torch.empty((n,), device=self.device, dtype=torch.float32)

        dist.broadcast(a_t, src=0)
        return a_t.detach().cpu().numpy().astype(np.float32, copy=False)

    def _log_stats(self, stats: _RL_LRS_LogStats):
        if self.rank != 0:
            return

        msg = (
            f"[RLLRS] step={stats.step} schedule={stats.schedule_type} base_lr={stats.base_lr:.3e} "
            f"warmup_damp={stats.warmup_damp:.3f} "
            f"action_scale={stats.action_scale:.4f} "
            f"clip_frac={stats.clip_frac:.3f} (hi={stats.clip_hi_frac:.3f}, lo={stats.clip_lo_frac:.3f}) "
            f"a_raw(mean/std)=({stats.a_raw_mean:.3f},{stats.a_raw_std:.3f}) "
            f"lr_scale(p50/p95/p99)=({stats.lr_scale_p50:.3f},{stats.lr_scale_p95:.3f},{stats.lr_scale_p99:.3f}) "
            f"ceil(exp(k*clip))={stats.lr_scale_ceiling:.3f} "
            f"lr(p50/p95/p99)=({stats.lr_p50:.3e},{stats.lr_p95:.3e},{stats.lr_p99:.3e}) "
            f"lr(min/max)=({stats.lr_min:.3e},{stats.lr_max:.3e})"
        )
        if stats.lr_cap_factor is not None:
            msg += f" cap_frac_lr={stats.cap_frac_lr:.3f} cap_factor={stats.lr_cap_factor:.2f}"
        print(msg)

        if self._log_fh is not None:
            self._log_fh.write(json.dumps(stats.__dict__, ensure_ascii=False) + "\n")
            self._log_fh.flush()

    def step(self, closure=None, loss=None):
        if loss is None and closure is not None:
            loss = closure()

        base_lr = float(self._get_base_lr())
        warmup_damp = float(self._warmup_damp())

        # RL-LRS action sampling
        a_raw = self._sample_action_raw()
        a_clip = np.clip(a_raw, -self.action_clip, self.action_clip).astype(np.float32, copy=False)

        k_t = float(self.action_scale) * float(warmup_damp)
        lr_scales = np.exp((k_t * a_clip).astype(np.float32, copy=False)).astype(np.float32, copy=False)

        self.last_action_raw = a_raw
        self.last_action_clipped = a_clip
        self.last_lr_scales = lr_scales

        # apply per-tensor lrs; enforce weight_decay unchanged
        capped_lr = 0
        new_lrs = np.empty((self.num_params,), dtype=np.float64)

        lr_cap = None
        if self.lr_cap_factor is not None:
            lr_cap = float(base_lr * float(self.lr_cap_factor))

        for i, group in enumerate(self.param_groups):
            new_lr = float(base_lr * float(lr_scales[i]))
            if lr_cap is not None and new_lr > lr_cap:
                new_lr = lr_cap
                capped_lr += 1
            group["lr"] = new_lr
            new_lrs[i] = new_lr

            init_wd = float(group.get("initial_weight_decay", group.get("weight_decay", 0.0)) or 0.0)
            group["weight_decay"] = init_wd

        out = self.base_optimizer.step(closure)

        # logging
        if self.rank == 0 and (self.current_step % self.log_interval == 0):
            clip_hi = float(np.mean(a_raw > self.action_clip)) if self.num_params > 0 else 0.0
            clip_lo = float(np.mean(a_raw < -self.action_clip)) if self.num_params > 0 else 0.0
            clip_frac = float(clip_hi + clip_lo)

            def _pct(x: np.ndarray, ps: List[float]) -> List[float]:
                if x.size == 0:
                    return [0.0 for _ in ps]
                return np.percentile(x, ps).tolist()

            lr_p50, lr_p95, lr_p99 = _pct(new_lrs, [50, 95, 99])
            ls_p50, ls_p95, ls_p99 = _pct(lr_scales.astype(np.float64), [50, 95, 99])

            lr_scale_ceiling = float(math.exp(k_t * float(self.action_clip)))

            stats = _RL_LRS_LogStats(
                step=int(self.current_step),
                schedule_type=str(self.schedule_type),
                base_lr=float(base_lr),
                warmup_damp=float(warmup_damp),
                action_scale=float(self.action_scale),
                action_mean=float(self.action_mean),
                action_std=float(self.action_std),
                action_clip=float(self.action_clip),
                a_raw_mean=float(np.mean(a_raw)) if self.num_params > 0 else 0.0,
                a_raw_std=float(np.std(a_raw)) if self.num_params > 0 else 0.0,
                a_clip_mean=float(np.mean(a_clip)) if self.num_params > 0 else 0.0,
                a_clip_std=float(np.std(a_clip)) if self.num_params > 0 else 0.0,
                clip_frac=float(clip_frac),
                clip_hi_frac=float(clip_hi),
                clip_lo_frac=float(clip_lo),
                lr_scale_mean=float(np.mean(lr_scales)) if self.num_params > 0 else 0.0,
                lr_scale_p50=float(ls_p50),
                lr_scale_p95=float(ls_p95),
                lr_scale_p99=float(ls_p99),
                lr_scale_min=float(np.min(lr_scales)) if self.num_params > 0 else 0.0,
                lr_scale_max=float(np.max(lr_scales)) if self.num_params > 0 else 0.0,
                lr_mean=float(np.mean(new_lrs)) if self.num_params > 0 else 0.0,
                lr_p50=float(lr_p50),
                lr_p95=float(lr_p95),
                lr_p99=float(lr_p99),
                lr_min=float(np.min(new_lrs)) if self.num_params > 0 else 0.0,
                lr_max=float(np.max(new_lrs)) if self.num_params > 0 else 0.0,
                lr_cap_factor=self.lr_cap_factor,
                cap_frac_lr=float(capped_lr / max(1, self.num_params)),
                lr_scale_ceiling=float(lr_scale_ceiling),
            )
            self._log_stats(stats)

        self.current_step += 1

        if loss is not None:
            return loss, False
        return out

    def save_agent(self, path):
        # no-op for compatibility
        return

    def __del__(self):
        try:
            if getattr(self, "_log_fh", None) is not None:
                self._log_fh.close()
        except Exception:
            pass