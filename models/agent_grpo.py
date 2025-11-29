import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.networks import Actor
from models.replay_buffer import Replay_Buffer


class Agent_GRPO:
    """
    GRPO Agent (优化版 - Risk Aware v2)
    核心修复：
    1. calculate_risk: 归一化到 [0, 1]，防止数值爆炸。
    2. Dynamic Beta: 逻辑反转 -> 危险时放松约束 (Beta变小)，允许大幅修正动作。
    """

    def __init__(self, state_size, action_size, bs, lr, dr, decay_step_size, tau, gamma, lam, eps_clip, K_epochs,
                 critic_loss_coef, entropy_coef, device, shared=False, manager=None, is_worker=False,
                 use_dynamic_beta=True):
        self.state_size = state_size
        self.action_size = action_size
        self.bs = bs
        self.lr = lr
        self.dr = dr
        self.decay_step_size = decay_step_size
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.device = device

        # 是否使用动态 Beta
        self.use_dynamic_beta = use_dynamic_beta

        # Beta 参数设置
        self.beta_init = 0.01  # 初始 KL 惩罚系数
        self.beta_min = 0.001  # 最小 Beta (危险时保留微弱约束)
        self.beta = self.beta_init

        # Buffer 大小需能容纳多进程并行产生的数据
        self.memory = Replay_Buffer(int(50000), bs, manager)

        if shared:
            self.actor = Actor(state_size, action_size).share_memory().to(device)
        else:
            self.actor = Actor(state_size, action_size).to(device)

        if not is_worker:
            self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.entropy = 0

    def act(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state)

        if deterministic:
            action_tensor = torch.argmax(action_probs, dim=1)
            log_prob = 0.0
        else:
            dist = torch.distributions.Categorical(action_probs)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor).detach().cpu().numpy().item()

        action_val = action_tensor.cpu().numpy().flatten()
        return int(action_val[0]), log_prob

    def calculate_risk(self, state):
        """
        计算风险值 (Risk) - 线性映射版
        输出: [0.0, 1.0]
        Logic:
            Distance > 30m -> Risk = 0.0 (Safe)
            Distance < 5m  -> Risk = 1.0 (Danger)
        """
        if state.shape[1] < 10: return 0.0

        batch_size = state.shape[0]
        num_feats = 5  # x, y, vx, vy, heading

        # Reshape: [Batch, Vehicles, Feats]
        # 假设 obs 是 fixed order, index 0 是 ego
        reshaped = state.view(batch_size, -1, num_feats)

        ego_x = reshaped[:, 0, 0]
        ego_y = reshaped[:, 0, 1]

        others_x = reshaped[:, 1:, 0]
        others_y = reshaped[:, 1:, 1]

        # 计算归一化距离
        dists_norm = torch.sqrt((others_x - ego_x.unsqueeze(1)) ** 2 + (others_y - ego_y.unsqueeze(1)) ** 2)

        # 还原真实距离 (highway-env 坐标范围 approx 100)
        dists_real = dists_norm * 100.0

        # 过滤 Ghost Vehicles (距离极近通常是 padding 0,0)
        valid_mask = dists_norm > 0.001
        dists_real_masked = dists_real.clone()
        dists_real_masked[~valid_mask] = float('inf')

        # 找到最近的前车距离
        min_dist, _ = torch.min(dists_real_masked, dim=1)

        # --- [关键修改] 线性映射风险 ---
        safe_threshold = 30.0
        danger_threshold = 5.0

        # 线性插值: (30 - dist) / (30 - 5)
        # 当 dist=30, risk=0; 当 dist=5, risk=1
        risk = (safe_threshold - min_dist) / (safe_threshold - danger_threshold)

        # 截断到 [0, 1] 区间
        risk = torch.clamp(risk, 0.0, 1.0)

        # 处理 NaN (如果没有其他车辆)
        risk = torch.nan_to_num(risk, nan=0.0)

        return risk.mean().item()

    def learn(self, current_total_timesteps):
        # 获取数据: advantages 字段存储的是 Worker 算好的 Group Advantage
        states, actions, advantages, next_states, dones, logprobs = self.memory.get_all_and_clear()

        if states is None or len(states) < self.bs: return

        states = states.to(self.device)
        actions = actions.to(self.device)
        advantages = advantages.to(self.device)
        logprobs = logprobs.to(self.device)

        # Clip Advantage 保证数值稳定
        advantages = torch.clamp(advantages, -4.0, 4.0)

        # --- [关键修改] 反转的动态 Beta 更新逻辑 ---
        if self.use_dynamic_beta:
            current_risk = self.calculate_risk(states)

            # 新逻辑：Risk 越高 (危险)，Beta 越小 (放松约束)
            # 目的：允许策略大幅更新以逃离危险状态
            target_beta = self.beta_init * (1.0 - current_risk)

            # 确保 Beta 不会变成 0 (保留微弱的 Trust Region)
            self.beta = max(target_beta, self.beta_min)
        else:
            self.beta = self.beta_init

        # 学习率衰减
        decay = self.dr ** (current_total_timesteps // self.decay_step_size)
        for pg in self.optimizer.param_groups: pg['lr'] = self.lr * decay

        # PPO Update Loop
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        for _ in range(self.K_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.bs):
                end = start + self.bs
                idx = indices[start:end]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log = logprobs[idx]
                mb_adv = advantages[idx]

                action_probs = self.actor(mb_states)
                dist = torch.distributions.Categorical(action_probs)
                mb_new_log = dist.log_prob(mb_actions)
                dist_entropy = dist.entropy().mean()

                ratio = torch.exp(mb_new_log - mb_old_log)

                with torch.no_grad():
                    approx_kl = 0.5 * ((mb_new_log - mb_old_log) ** 2).mean()

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv

                # Loss = PPO_Loss + Beta * KL - Entropy
                loss = -torch.min(surr1, surr2).mean() + \
                       self.beta * approx_kl - \
                       self.entropy_coef * decay * dist_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer.step()

        self.entropy = dist_entropy.item()

    def get_state_dict(self):
        return {'actor': self.actor.state_dict()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd['actor'])