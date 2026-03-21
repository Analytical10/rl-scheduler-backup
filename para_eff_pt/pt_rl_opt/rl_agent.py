import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import torch.distributed as dist


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # --- 特征提取层 (纯 MLP) ---
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # --- Actor Head ---
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        # --- Critic Head ---
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Actor LogStd (可学习)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # 初始化权重
        self.apply(self._init_weights)

        # 保持 Actor 初始输出接近 0
        nn.init.constant_(self.actor_head[-1].weight, 0.01)
        nn.init.constant_(self.actor_head[-1].bias, 0.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        # 数值稳定 atanh
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    @staticmethod
    def _stable_log_det_tanh(pre_tanh):
        """
        使用 Softplus 实现数学等价于 log(1 - tanh^2(u)) 的稳定计算
        公式：2.0 * (log(2.0) - u - softplus(-2.0 * u))
        """
        return 2.0 * (np.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh))

    def forward(self, state):
        x = self.feature_net(state)
        return x

    def act(self, state, deterministic: bool = False, shared_noise: bool = False, squash_actions: bool = False):
        """
        默认（squash_actions=False, deterministic=False, shared_noise=False）路径：
        尽量保持与 old 版本逐行等价：tanh(mean) + Normal.sample()

        state: [N, state_dim]
        return:
          action: [N, action_dim]
          action_logprob: [N]
        """
        x = self.feature_net(state)

        if squash_actions:
            # Tanh-Normal：先在 R 上建模，再 tanh squash
            mean = self.actor_head(x)                  # [N, action_dim] in R
            std = torch.exp(self.actor_logstd)         # [1, action_dim]（保持旧式广播）
            dist_ = Normal(mean, std)

            if deterministic:
                pre_tanh = mean
            else:
                if shared_noise:
                    eps = torch.randn((1, mean.shape[-1]), device=mean.device, dtype=mean.dtype)
                    pre_tanh = mean + std * eps
                else:
                    # 这里用 rsample 更常见；但只在 squash 模式下启用，不影响 old baseline
                    pre_tanh = dist_.rsample()

            action = torch.tanh(pre_tanh)
            logprob_u = dist_.log_prob(pre_tanh)
            if logprob_u.dim() > 1:
                logprob_u = logprob_u.sum(dim=-1)

            # log_det = torch.log(1.0 - action.pow(2) + 1e-6)
            # if log_det.dim() > 1:
            #     log_det = log_det.sum(dim=-1)
            log_det = self._stable_log_det_tanh(pre_tanh)
            if log_det.dim() > 1:
                log_det = log_det.sum(dim=-1)
            action_logprob = logprob_u - log_det
            return pre_tanh.detach(), action.detach(), action_logprob.detach()

        # ===== 非 squash（旧 baseline 路径）=====
        action_mean = torch.tanh(self.actor_head(x))   # old: mean in (-1,1)
        action_std = torch.exp(self.actor_logstd)      # old: [1, action_dim]
        dist_ = Normal(action_mean, action_std)

        if deterministic:
            action = action_mean
        else:
            if shared_noise:
                eps = torch.randn((1, action_mean.shape[-1]), device=action_mean.device, dtype=action_mean.dtype)
                action = action_mean + action_std * eps
            else:
                # 关键：恢复 old 的 sample()，避免 RNG 消耗方式变化导致训练轨迹大变
                action = dist_.sample()

        action_logprob = dist_.log_prob(action)
        if action_logprob.dim() > 1:
            action_logprob = action_logprob.sum(dim=-1)

        return None, action.detach(), action_logprob.detach()

    def evaluate(self, state, action_input, squash_actions: bool = False):
        """
        PPO 更新时用：给定 (state, action) 计算 logprob / V(s) / entropy
        """
        x = self.feature_net(state)

        if squash_actions:
            mean = self.actor_head(x)                  # [B, action_dim]
            std = torch.exp(self.actor_logstd)         # [1, action_dim]
            dist_ = Normal(mean, std)

            # a = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
            # pre_tanh = self._atanh(a)
            pre_tanh = action_input  # PPO update 时直接传入 pre_tanh，避免数值不稳定的 atanh 计算
            logprobs = dist_.log_prob(pre_tanh)
            if logprobs.dim() > 1:
                logprobs = logprobs.sum(dim=-1)

            log_det = self._stable_log_det_tanh(pre_tanh)
            if log_det.dim() > 1:
                log_det = log_det.sum(dim=-1)

            action_logprobs = logprobs - log_det

            # entropy 用 base_dist 的 entropy 作为 proxy（常见做法）
            dist_entropy = dist_.entropy().sum(dim=-1)

            state_values = self.critic_head(x)
            return action_logprobs, state_values, dist_entropy

        # ===== 非 squash：保持 old evaluate 行为 =====
        action_mean = torch.tanh(self.actor_head(x))
        action_std = torch.exp(self.actor_logstd)
        dist_ = Normal(action_mean, action_std)

        action_logprobs = dist_.log_prob(action_input)
        if action_logprobs.dim() > 1:
            action_logprobs = action_logprobs.sum(dim=-1)

        dist_entropy = dist_.entropy().sum(dim=-1)
        state_values = self.critic_head(x)

        return action_logprobs, state_values, dist_entropy


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim=1,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=4,
        device='cuda',
        ddp_sync=True,          # 控制 update 时是否做 all_reduce
        entropy_coef=0.05,      # 默认保持原值
        squash_actions=False    # round=256 时启用 tanh-normal
    ):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.action_dim = action_dim
        self.ddp_sync = ddp_sync
        self.entropy_coef = entropy_coef
        self.squash_actions = bool(squash_actions)

        self.policy = ActorCritic(state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCritic(state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = []

    def select_action(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            pre_tanh, action, action_logprob = self.policy_old.act(
                state_t, deterministic=False, shared_noise=False, squash_actions=self.squash_actions
            )
        return (pre_tanh.cpu().numpy() if pre_tanh is not None else None), action.cpu().numpy(), action_logprob.cpu().numpy()

    def select_action_deterministic(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            pre_tanh, action, action_logprob = self.policy_old.act(
                state_t, deterministic=True, shared_noise=False, squash_actions=self.squash_actions
            )
        return (pre_tanh.cpu().numpy() if pre_tanh is not None else None), action.cpu().numpy(), action_logprob.cpu().numpy()

    def select_action_shared_noise(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            pre_tanh, action, action_logprob = self.policy_old.act(
                state_t, deterministic=False, shared_noise=True, squash_actions=self.squash_actions
            )
        return (pre_tanh.cpu().numpy() if pre_tanh is not None else None), action.cpu().numpy(), action_logprob.cpu().numpy()
    #action_input 是 pre_tanh 还是 action 取决于是否开启 squash_actions
    def logprob_old(self, state, action_input):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
            a = torch.tensor(action_input, dtype=torch.float32, device=self.device)
            logprobs, _, _ = self.policy_old.evaluate(s, a, squash_actions=self.squash_actions)
        return logprobs.detach().cpu().numpy()

    def get_actor_logstd(self):
        with torch.no_grad():
            return self.policy_old.actor_logstd.detach().cpu().numpy().reshape(-1)

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):
        if len(self.buffer) == 0:
            return

        s_list, a_list, lp_list, r_list = [], [], [], []

        for (s, a, lp, r) in self.buffer:
            N = s.shape[0]
            s_list.append(s)
            a_list.append(a)
            lp_list.append(lp)
            r_list.append(np.full((N,), r))

        states = torch.tensor(np.concatenate(s_list), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.concatenate(a_list), dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(np.concatenate(lp_list), dtype=torch.float32).to(self.device)
        rewards = np.concatenate(r_list)

        returns = torch.tensor(rewards, dtype=torch.float32).to(self.device).view(-1, 1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                states, actions, squash_actions=self.squash_actions
            )

            ratios = torch.exp(logprobs - old_logprobs.detach())

            #修复形状
            ratios = ratios.view(-1,1)

            advantages = returns - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * nn.MSELoss()(state_values, returns)
                - self.entropy_coef * dist_entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.mean().backward()

            if dist.is_initialized() and self.ddp_sync:
                for param in self.policy.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = []

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())