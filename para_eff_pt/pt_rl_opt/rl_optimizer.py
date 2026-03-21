import torch
from torch.optim import Optimizer
import numpy as np
import math
from collections import deque
import os
from .rl_agent import *
import torch.distributed as dist
from tqdm import tqdm


class RL_AdamW_Wrapper(Optimizer):
    def __init__(
        self,
        params,
        base_optimizer_cls=torch.optim.AdamW,
        rl_agent_lr=3e-4,
        base_lr=1e-3,
        max_lr=1e-3,
        min_lr=1e-5,
        warmup_steps=100,
        max_steps=1000,
        agent_load_path=None,
        agent_save_dir='rl_checkpoints',
        run_id='run_001',
        rank=0,
        mode='train',
        device='cuda',
        history_len=5,
        stats_window=20,
        action_scale=1.0,
        round=0,
        log_interval=500,
        **optimizer_kwargs
    ):
        # --- 1. 强制重构 Param Groups ---
        param_list = list(params)
        grouped_params = []
        for p in param_list:
            wd = optimizer_kwargs.get('weight_decay', 1e-2)
            grouped_params.append({'params': [p], 'initial_weight_decay': wd})

        if 'lr' in optimizer_kwargs:
            base_lr = optimizer_kwargs.pop('lr')

        self.base_optimizer = base_optimizer_cls(grouped_params, lr=base_lr, **optimizer_kwargs)

        self.all_params = param_list
        self.num_params = len(self.all_params)

        # --- round / flags 解析 ---
        self.round = int(round)
        self.round_flags = self._resolve_round_flags(self.round)

        self.log_interval = int(log_interval)

        if rank == 0:
            print(f"RL Optimizer: Controlling {self.num_params} tensors individually.")
            print(
                f"[RL ROUND] round={self.round}, round_flags={self.round_flags} "
                f"(1=GlobalLR,2=ShuffleLocal,4=AlignReward,8=Rank0OnlyAgent,16=BroadcastLoss,"
                f"32=DeterministicAct,64=SharedNoise,128=ShuffleActions,256=TanhNormal,512=GlobalOnlyFeats,"
                f"1024=WSD_Scheduler)"
            )

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.mode = mode
        self.device = device
        self.rank = rank
        self.run_id = run_id
        self.agent_save_dir = agent_save_dir

        self.current_step = 0
        self.last_loss = None

        # --- 2. 特征维度定义 ---
        self.global_feat_dim = 4
        self.local_feat_dim = 9
        # flag512: 去掉 local_feats，只保留 global_feats
        self.feature_dim = self.global_feat_dim + (0 if self._flag_enabled(512) else self.local_feat_dim)
        self.action_dim = 2

        self.loss_history = deque(maxlen=stats_window)
        self.drop_history = deque(maxlen=50)
        self.ema_short = None
        self.ema_long = None
        self.grad_norm_avg = 0.0
        self.gss_beta = 0.99

        # 保持默认行为尽量贴近 old
        self.prev_actions = np.zeros((self.num_params, self.action_dim), dtype=np.float64)
        self.prev_grad_norms = np.zeros(self.num_params, dtype=np.float64)

        # reward/action 对齐（flag=4）所需的 pending transition
        self._pending_transition = None  # (state_batch, actions, logprobs)

        # 初始化 Agent
        ddp_sync_agent = not self._flag_enabled(8)  # 只有 rank0-only 时关闭
        squash_actions = self._flag_enabled(256)

        self.agent = PPOAgent(
            state_dim=self.feature_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            device=device,
            ddp_sync=ddp_sync_agent,
            lr=rl_agent_lr,
            squash_actions=squash_actions,
        )
        self.action_scale = action_scale

        if agent_load_path and os.path.exists(agent_load_path):
            try:
                self.agent.load(agent_load_path)
                if self.rank == 0:
                    print(f"SUCCESS: Loaded RL Agent from {agent_load_path}")
            except Exception as e:
                if self.rank == 0:
                    print(f"WARNING: Load failed (possibly dim mismatch): {e}")

        if self.rank == 0 and not os.path.exists(self.agent_save_dir):
            os.makedirs(self.agent_save_dir)

        self.last_debug_stats = {}

    # ---- Optimizer interface passthrough ----
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @property
    def state(self):
        return self.base_optimizer.state

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def save_agent(self, path):
        self.agent.save(path)

    # ---- round helpers ----
    def _resolve_round_flags(self, round_id: int) -> int:
        mapping = {
            0: 0,
            1: 1,     # Global LR action
            2: 2,     # Shuffle local feats
            4: 4,     # Reward/Action 对齐
            5: 8,     # Rank0-only agent learning
            6: 16,    # Broadcast rank0 loss for RL
            7: 512,   # GlobalOnlyFeats
            8: 1024,  # WSD Scheduler
        }
        return mapping.get(round_id, round_id)

    def _flag_enabled(self, flag_bit: int) -> bool:
        return (self.round_flags & flag_bit) != 0

    def _ddp_broadcast_scalar_from_rank0(self, x: float) -> float:
        if not dist.is_initialized():
            return float(x)
        t = torch.tensor([float(x)], device=self.device, dtype=torch.float32)
        dist.broadcast(t, src=0)
        return float(t.item())

    def _ddp_broadcast_perm_from_rank0(self, perm_np: np.ndarray) -> np.ndarray:
        if not dist.is_initialized():
            return perm_np
        if self.rank == 0:
            t = torch.tensor(perm_np, device=self.device, dtype=torch.int64)
        else:
            t = torch.empty((self.num_params,), device=self.device, dtype=torch.int64)
        dist.broadcast(t, src=0)
        return t.cpu().numpy()

    def _get_cosine_lr(self):
        step = self.current_step
        if step < self.warmup_steps:
            return self.min_lr + (self.max_lr - self.min_lr) * (step / self.warmup_steps)
        elif step > self.max_steps:
            return self.min_lr
        else:
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def _get_wsd_lr(self):
        """
        WSD (Warmup - Stable - Decay) scheduler.
        """
        step = int(self.current_step)
        warmup = int(self.warmup_steps)
        max_steps = int(self.max_steps)

        if step > max_steps:
            return float(self.min_lr)

        if max_steps <= 0:
            return float(self.min_lr)

        if warmup > 0 and step < warmup:
            return float(self.min_lr + (self.max_lr - self.min_lr) * (step / warmup))

        if warmup >= max_steps:
            return float(self.min_lr)

        remaining = max_steps - warmup
        stable_ratio = 0.90
        stable_steps = int(round(remaining * stable_ratio))
        decay_steps = remaining - stable_steps

        if decay_steps <= 0:
            return float(self.max_lr)

        decay_start = warmup + stable_steps

        if step < decay_start:
            return float(self.max_lr)

        decay_t = step - decay_start
        progress = min(1.0, max(0.0, decay_t / float(decay_steps)))
        lr = float(self.max_lr - (self.max_lr - self.min_lr) * progress)
        return lr

    # ---- logging helpers (仅抽取打印逻辑，不改变功能) ----
    def _print_rl_stats(
        self,
        loss_val_used,
        reward,
        base_lr,
        new_lrs,
        lr_scales,
        clip_frac_lr,
        clip_frac_wd,
        cap_frac,
        current_scale,
        state_batch,
        actions,
        logprobs,
    ):
        avg_lr = float(np.mean(new_lrs))
        lr_p50, lr_p95, lr_p99 = np.percentile(new_lrs, [50, 95, 99]).tolist()
        lrs_p50, lrs_p95, lrs_p99 = np.percentile(lr_scales, [50, 95, 99]).tolist()

        logstd = self.agent.get_actor_logstd() if hasattr(self.agent, "get_actor_logstd") else None

        for j in range(min(5, self.num_params)):
            print(f"    Param {j}: LR_action={actions[j,0]:.4f}, WD_action={actions[j,1]:.4f}")

        print(f"[RL STATS] step={self.current_step} ")
        if reward is None:
            print(f"loss_used={loss_val_used:.6f} reward=NA ")
        else:
            print(f"loss_used={loss_val_used:.6f} reward={reward:.4f} ")
        print(f"base_lr={base_lr:.3e} avg_lr={avg_lr:.3e} ")
        print(f"lr(p50/p95/p99)=({lr_p50:.3e},{lr_p95:.3e},{lr_p99:.3e}) ")
        print(f"lr_scale(p50/p95/p99)=({lrs_p50:.3f},{lrs_p95:.3f},{lrs_p99:.3f}) ")
        print(f"clip_frac(lr/wd)=({clip_frac_lr:.3f},{clip_frac_wd:.3f}) ")
        print(f"cap_frac={cap_frac:.3f} action_scale={current_scale:.3f}")
        print("state_hash", float(np.mean(state_batch)), float(np.std(state_batch)))
        print("actions0", actions[:5, 0])
        print("logp0", logprobs[:5])
        print("base_lr", base_lr, "scale", current_scale)
        print("avg_lr", float(np.mean([g['lr'] for g in self.param_groups])))

        if logstd is not None:
            print(f"actor_logstd={logstd}")

    def _print_rl_action_debug(self, state_batch, actions, logprobs):
        try:
            with torch.no_grad():
                s_t = torch.tensor(state_batch, device=self.device, dtype=torch.float32)
                x_t = self.agent.policy_old.feature_net(s_t)

                logstd_t = self.agent.policy_old.actor_logstd
                std_t = torch.exp(logstd_t)

                a_t = torch.tensor(actions, device=self.device, dtype=torch.float32)

                if getattr(self.agent, "squash_actions", False):
                    mu_r_t = self.agent.policy_old.actor_head(x_t)
                    std_expand = std_t.expand_as(mu_r_t)

                    a_clip = torch.clamp(a_t, -1.0 + 1e-6, 1.0 - 1e-6)
                    pre_tanh_t = 0.5 * (torch.log1p(a_clip) - torch.log1p(-a_clip))

                    eps_t = (pre_tanh_t - mu_r_t) / (std_expand + 1e-8)
                    mu_action_t = torch.tanh(mu_r_t)

                    mu_np = mu_action_t.detach().cpu().numpy()
                    mu_r_np = mu_r_t.detach().cpu().numpy()
                    std_np = std_t.detach().cpu().numpy().reshape(-1)
                    eps_np = eps_t.detach().cpu().numpy()
                else:
                    mu_action_t = torch.tanh(self.agent.policy_old.actor_head(x_t))
                    std_expand = std_t.expand_as(mu_action_t)
                    eps_t = (a_t - mu_action_t) / (std_expand + 1e-8)

                    mu_np = mu_action_t.detach().cpu().numpy()
                    std_np = std_t.detach().cpu().numpy().reshape(-1)
                    eps_np = eps_t.detach().cpu().numpy()
                    mu_r_np = None

            try:
                a_np = actions.astype(np.float32, copy=False)
                abs_a = np.abs(a_np)

                thr_list = [0.99, 0.999]
                for thr in thr_list:
                    frac = abs_a > thr
                    frac_per_dim = frac.mean(axis=0)
                    print(f"[RL256 DEBUG] frac(|a|>{thr}) per-dim = {frac_per_dim}")

                exact1_per_dim = (abs_a >= 1.0).mean(axis=0)
                print(f"[RL256 DEBUG] frac(|a|==1.0) per-dim = {exact1_per_dim}")

                max_abs_per_dim = abs_a.max(axis=0)
                print(f"[RL256 DEBUG] max(|a|) per-dim = {max_abs_per_dim}")

                lp_recalc = self.agent.logprob_old(state_batch, actions).astype(np.float32, copy=False)
                lp_saved = logprobs.astype(np.float32, copy=False)

                lp_diff = lp_recalc - lp_saved
                print(
                    "[RL256 DEBUG] logprob_old self-consistency: "
                    f"mean_abs={float(np.mean(np.abs(lp_diff))):.6f} "
                    f"max_abs={float(np.max(np.abs(lp_diff))):.6f} "
                    f"mean_diff={float(np.mean(lp_diff)):.6f}"
                )

                if float(np.max(np.abs(lp_diff))) > 1.0:
                    idx = np.argmax(np.abs(lp_diff))
                    print(
                        f"[RL256 DEBUG] worst_idx={int(idx)} "
                        f"a={a_np[idx]} lp_saved={float(lp_saved[idx]):.6f} lp_recalc={float(lp_recalc[idx]):.6f}"
                    )

            except Exception as e:
                print(f"[RL256 DEBUG] failed: {e}")

            mu_mean = mu_np.mean(axis=0)
            mu_std_across_groups = mu_np.std(axis=0)
            eps_mean = eps_np.mean(axis=0)
            eps_std = eps_np.std(axis=0)

            print("[RL ACTION DEBUG] policy_old per-group stats:")
            print(f"  action_dim={self.action_dim} num_groups={self.num_params}")
            if getattr(self.agent, "squash_actions", False):
                print("  mode=256(TanhNormal)  eps is computed in pre_tanh space: eps=(atanh(a)-mu_R)/sigma")
            else:
                print("  mode=baseline(Normal on tanh-mean) eps=(a-mu)/sigma")

            print(f"  sigma(global, per-dim)={std_np}  (shared across all groups)")
            print(f"  mu_mean(per-dim)={mu_mean}")
            print(f"  mu_std_across_groups(per-dim)={mu_std_across_groups}")
            print(f"  eps_mean(per-dim)={eps_mean}")
            print(f"  eps_std(per-dim)={eps_std}")

            def _p(a, ps=(50, 95, 99)):
                return np.percentile(a, ps, axis=0)

            mu_p = _p(mu_np)
            eps_p = _p(eps_np)

            print(f"  mu_percentile(p50/p95/p99) per-dim: {mu_p[0]} / {mu_p[1]} / {mu_p[2]}")
            print(f"  eps_percentile(p50/p95/p99) per-dim: {eps_p[0]} / {eps_p[1]} / {eps_p[2]}")

            N = self.num_params
            if N <= 32:
                k = N
            else:
                k = 16

            print(f"[RL ACTION DEBUG] showing first {k} groups (i: mu, eps):")
            for i in range(k):
                if mu_r_np is not None:
                    print(f"  i={i:5d} mu(tanh_muR)={mu_np[i]}  muR={mu_r_np[i]}  eps={eps_np[i]}")
                else:
                    print(f"  i={i:5d} mu={mu_np[i]}  eps={eps_np[i]}")

        except Exception as e:
            print(f"[RL ACTION DEBUG] failed to compute mu/sigma/eps: {e}")

    def step(self, closure=None, loss=None):
        """
        return (loss, abort_training)
        abort_training=True: Loss Spike 熔断建议外部回滚
        """
        if loss is None and closure is not None:
            loss = closure()

        if loss is None:
            self.base_optimizer.step(closure)
            return None, False

        loss_val_local = float(loss.item())
        abort_training = False

        # flag16：用 rank0 loss 广播，保证 reward/feature 一致
        loss_val_used = loss_val_local
        if self._flag_enabled(16):
            loss_val_used = self._ddp_broadcast_scalar_from_rank0(loss_val_local)

        # --- A. Global Features ---
        progress = self.current_step / self.max_steps
        log_loss = np.log10(loss_val_used + 1e-10)

        self.loss_history.append(loss_val_used)
        loss_arr = np.array(self.loss_history, dtype=np.float64)
        if len(loss_arr) > 1:
            loss_std = np.std(loss_arr) / (loss_val_used + 1e-8)
        else:
            loss_std = 0.0

        if self.ema_short is None:
            self.ema_short = loss_val_used
            self.ema_long = loss_val_used
        else:
            self.ema_short = 0.9 * self.ema_short + 0.1 * loss_val_used
            self.ema_long = 0.99 * self.ema_long + 0.01 * loss_val_used

        ema_trend = (self.ema_short - self.ema_long) / (self.ema_long + 1e-8)
        global_feats = np.array([progress, log_loss, loss_std, ema_trend * 10], dtype=np.float32)

        # --- B. Local Features ---
        local_feats_list = []
        total_norm_sq = 0.0

        for i, p in enumerate(self.all_params):
            if p.grad is None:
                local_feats_list.append(np.zeros(self.local_feat_dim, dtype=np.float32))
                continue

            g_norm = torch.norm(p.grad).item()
            total_norm_sq += g_norm ** 2
            log_gn = np.log10(g_norm + 1e-10)

            state = self.state[p]

            consistency = 0.0
            if 'exp_avg' in state:
                exp_avg = state['exp_avg']
                dot = torch.sum(p.grad * exp_avg).item()
                exp_norm = torch.norm(exp_avg).item()
                if g_norm > 1e-8 and exp_norm > 1e-8:
                    consistency = dot / (g_norm * exp_norm)

            current_lr = self.param_groups[i]['lr']
            log_lr = np.log10(current_lr + 1e-10)

            prev_act = self.prev_actions[i, 0]
            depth = i / self.num_params

            p_norm = torch.norm(p.data).item()
            log_p_norm = np.log10(p_norm + 1e-10)

            trust_ratio = g_norm / (p_norm + 1e-8)
            log_trust_ratio = np.log10(trust_ratio + 1e-10)

            adam_snr = 0.0
            if 'exp_avg' in state and 'exp_avg_sq' in state:
                m = state['exp_avg']
                v = state['exp_avg_sq']
                snr_tensor = m.abs() / (v.sqrt() + 1e-8)
                adam_snr = snr_tensor.mean().item()

            prev_gn = self.prev_grad_norms[i]
            delta_gn = log_gn - prev_gn
            self.prev_grad_norms[i] = log_gn

            local_feats_list.append(np.array([
                log_lr,
                log_gn,
                consistency,
                prev_act,
                depth,
                log_p_norm,
                log_trust_ratio,
                adam_snr,
                delta_gn
            ], dtype=np.float32))

        global_grad_norm = float(total_norm_sq ** 0.5)
        if self._flag_enabled(16):
            global_grad_norm = self._ddp_broadcast_scalar_from_rank0(global_grad_norm)

        if self.current_step == 0:
            self.grad_norm_avg = global_grad_norm
        else:
            self.grad_norm_avg = self.gss_beta * self.grad_norm_avg + (1 - self.gss_beta) * global_grad_norm

        global_feats_batch = np.tile(global_feats, (self.num_params, 1))
        local_feats_batch = np.stack(local_feats_list, axis=0)

        if self._flag_enabled(2):
            perm = np.random.permutation(self.num_params)
            local_feats_batch = local_feats_batch[perm]

        if self._flag_enabled(512):
            state_batch = global_feats_batch
        else:
            state_batch = np.concatenate([global_feats_batch, local_feats_batch], axis=1)

        # --- C. Agent 决策 ---
        rank0_only_agent = self._flag_enabled(8) and dist.is_initialized()

        if rank0_only_agent and self.rank != 0:
            actions = np.zeros((self.num_params, self.action_dim), dtype=np.float32)
            logprobs = np.zeros((self.num_params,), dtype=np.float32)
        else:
            if self._flag_enabled(32):
                pre_tanh, actions, logprobs = self.agent.select_action_deterministic(state_batch)
            elif self._flag_enabled(64):
                pre_tanh, actions, logprobs = self.agent.select_action_shared_noise(state_batch)
            else:
                pre_tanh, actions, logprobs = self.agent.select_action(state_batch)

            actions = actions.astype(np.float32, copy=False)
            logprobs = logprobs.astype(np.float32, copy=False)

            if self._flag_enabled(1):
                global_lr_action = float(actions[:, 0].mean())
                actions[:, 0] = global_lr_action
                logprobs = self.agent.logprob_old(state_batch, actions).astype(np.float32, copy=False)

        # DDP：以 rank0 actions 为准
        if dist.is_initialized():
            if self.rank == 0:
                actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.float32)
                pt_tensor = torch.tensor(pre_tanh, device=self.device, dtype=torch.float32) if pre_tanh is not None else torch.zeros(1)
            else:
                actions_tensor = torch.zeros((self.num_params, self.action_dim), device=self.device, dtype=torch.float32)
                pt_tensor = torch.zeros_like(a_tensor) if self._flag_enabled(256) else torch.zeros(1)
            dist.broadcast(actions_tensor, src=0)
            actions = actions_tensor.cpu().numpy()
            if self._flag_enabled(256):
                dist.broadcast(pt_tensor, src=0)
                pre_tanh = pt_tensor.cpu().numpy()

        # Shuffle-actions
        if self._flag_enabled(128):
            if dist.is_initialized():
                if self.rank == 0:
                    perm_np = np.random.permutation(self.num_params).astype(np.int64)
                else:
                    perm_np = np.empty((self.num_params,), dtype=np.int64)
                perm_np = self._ddp_broadcast_perm_from_rank0(perm_np)
            else:
                perm_np = np.random.permutation(self.num_params).astype(np.int64)

            actions = actions[perm_np]
            if pre_tanh is not None:
                pre_tanh = pre_tanh[perm_np]

            if not (rank0_only_agent and self.rank != 0):
                logprobs = self.agent.logprob_old(state_batch, actions).astype(np.float32, copy=False)

        # 方案A：仅修 round_flags=256 的数值边界问题
        if self._flag_enabled(256):
            _A_EPS = 1e-4
            actions = np.clip(actions, -1.0 + _A_EPS, 1.0 - _A_EPS).astype(np.float32, copy=False)

            if not (rank0_only_agent and self.rank != 0):
                logprobs = self.agent.logprob_old(state_batch, actions).astype(np.float32, copy=False)

        self.prev_actions = actions

        # --- D. 应用动作 ---
        if self._flag_enabled(1024):
            base_lr = self._get_wsd_lr()
        else:
            base_lr = self._get_cosine_lr()

        if self.current_step < self.warmup_steps:
            warmup_dampening = float(self.current_step) / float(self.warmup_steps)
            current_scale = float(self.action_scale) * warmup_dampening
        else:
            current_scale = float(self.action_scale)

        clip_frac_lr = float(np.mean(np.abs(actions[:, 0]) > 1.0))
        clip_frac_wd = float(np.mean(np.abs(actions[:, 1]) > 1.0))

        if self._flag_enabled(256):
            a_exec_lr = actions[:, 0] * current_scale
            a_exec_wd = actions[:, 1] * current_scale
        else:
            a_exec_lr = np.clip(actions[:, 0], -1.0, 1.0) * current_scale
            a_exec_wd = np.clip(actions[:, 1], -1.0, 1.0) * current_scale

        lr_scales = np.exp(a_exec_lr)
        wd_scales = np.exp(a_exec_wd)

        capped_count = 0
        new_lrs = np.empty((self.num_params,), dtype=np.float64)

        for i, group in enumerate(self.param_groups):
            new_lr = float(base_lr * lr_scales[i])
            MAX_LR_CAP = float(base_lr * 7.0)
            if new_lr > MAX_LR_CAP:
                new_lr = MAX_LR_CAP
                capped_count += 1
            group['lr'] = new_lr
            new_lrs[i] = new_lr

            if group.get('initial_weight_decay', 0.0) > 0:
                group['weight_decay'] = group['initial_weight_decay'] * (float(wd_scales[i]) ** 0.35)

        cap_frac = float(capped_count / max(1, self.num_params))

        # --- E. Step (Base Optimizer) ---
        _ = self.base_optimizer.step(closure)

        # --- F. Reward & Update & Spike Detection / Eval Logging ---
        if self.mode == 'train':
            if self.current_step > 0 and self.last_loss is not None:
                loss_ratio = self.last_loss / (loss_val_used + 1e-10)
                loss_ratio = np.clip(loss_ratio, 0.8, 1.2)
                r_perf = np.log(loss_ratio) * 20.0

                r_trend = 0.0
                if self.ema_long is not None:
                    r_trend = (self.ema_long - loss_val_used) / (self.ema_long + 1e-8)
                    r_trend = np.clip(r_trend, -0.2, 0.2) * 2.0

                p_stability = 0.0
                grad_ratio = global_grad_norm / (self.grad_norm_avg + 1e-8)
                if grad_ratio > 1.5:
                    p_stability = (grad_ratio - 1.5) * 5.0

                reward = float(r_perf + r_trend - p_stability)

                if grad_ratio > 3.0:
                    reward -= 20.0

                spike_penalty = 0.0
                if self.ema_long is not None:
                    spike_ratio = loss_val_used / (self.ema_long + 1e-8)
                    if spike_ratio > 1.5:
                        spike_penalty = -100.0
                        abort_training = True
                        if self.rank == 0:
                            print(f"\n[RL WARNING] Loss Spike Detected! Ratio: {spike_ratio:.2f}. Aborting Epoch.")
                reward += spike_penalty

                should_store = True
                if rank0_only_agent and self.rank != 0:
                    should_store = False
                # 对于 round_flags=256，store pre_tanh 以避免 logprob 计算的数值问题；其他情况正常存 actions
                if should_store:
                    action_to_store = pre_tanh if self._flag_enabled(256) else actions
                    if self._flag_enabled(4):
                        if self._pending_transition is not None:
                            # 这里的 pa 现在代表的是上一时刻的 pre_tanh
                            ps, pa, plp = self._pending_transition
                            self.agent.store_transition((ps, pa, plp, reward))
                        self._pending_transition = (state_batch, action_to_store, logprobs)
                    else:
                        self.agent.store_transition((state_batch, action_to_store, logprobs, reward))

                if self.rank == 0 and (self.current_step % self.log_interval == 0):
                    self._print_rl_stats(
                        loss_val_used=loss_val_used,
                        reward=reward,
                        base_lr=base_lr,
                        new_lrs=new_lrs,
                        lr_scales=lr_scales,
                        clip_frac_lr=clip_frac_lr,
                        clip_frac_wd=clip_frac_wd,
                        cap_frac=cap_frac,
                        current_scale=current_scale,
                        state_batch=state_batch,
                        actions=actions,
                        logprobs=logprobs,
                    )
                    self._print_rl_action_debug(
                        state_batch=state_batch,
                        actions=actions,
                        logprobs=logprobs,
                    )

                if abort_training and (not rank0_only_agent or self.rank == 0):
                    if self.rank == 0:
                        print("[RL CRITICAL] Aborting triggered. Forcing immediate agent update to learn from this crash.")
                    self.agent.update()

            if (not abort_training) and self.current_step > 0 and (self.current_step % 50 == 0):
                if (not rank0_only_agent) or (self.rank == 0):
                    self.agent.update()

            if self.current_step > 0 and self.current_step % 500 == 0:
                self._save_agent_checkpoint()

        else:
            if self.rank == 0 and (self.current_step % self.log_interval == 0):
                self._print_rl_stats(
                    loss_val_used=loss_val_used,
                    reward=None,
                    base_lr=base_lr,
                    new_lrs=new_lrs,
                    lr_scales=lr_scales,
                    clip_frac_lr=clip_frac_lr,
                    clip_frac_wd=clip_frac_wd,
                    cap_frac=cap_frac,
                    current_scale=current_scale,
                    state_batch=state_batch,
                    actions=actions,
                    logprobs=logprobs,
                )
                self._print_rl_action_debug(
                    state_batch=state_batch,
                    actions=actions,
                    logprobs=logprobs,
                )

        self.last_loss = float(loss_val_used)
        self.current_step += 1

        if dist.is_initialized():
            abort_tensor = torch.tensor([1.0 if abort_training else 0.0], device=self.device)
            dist.broadcast(abort_tensor, src=0)
            abort_training = (abort_tensor.item() > 0.5)

        return loss, abort_training

    def _save_agent_checkpoint(self):
        if self.rank == 0:
            path = os.path.join(self.agent_save_dir, f"agent_{self.run_id}_latest.pth")
            self.agent.save(path)