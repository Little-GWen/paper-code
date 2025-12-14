import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.networks import Actor
from models.replay_buffer import Replay_Buffer


class Agent_GRPO:
    def __init__(self, state_size, action_size, bs, lr, decay_max_step, gamma, lam, eps_clip, K_epochs,
                 entropy_coef, device, shared=False, manager=None, is_worker=False, use_dynamic_beta=True,
                 global_baseline=None, baseline_lock=None):
        self.state_size = state_size
        self.action_size = action_size
        self.bs = bs
        self.lr = lr
        self.decay_max_step = decay_max_step
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.device = device
        self.scales = {'x': 1000, 'y': 40, 'vx': 40, 'vy': 40}
        self.entropy = 0

        self.use_dynamic_beta = use_dynamic_beta
        self.beta_init = 0.1
        self.beta_min = 0.1
        self.beta_max = 50000
        self.beta = self.beta_init

        self.memory = Replay_Buffer(int(50000), bs, manager)

        if shared:
            self.actor = Actor(state_size, action_size).share_memory().to(device)
        else:
            self.actor = Actor(state_size, action_size).to(device)

        if not is_worker:
            self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.global_baseline = global_baseline
        self.baseline_lock = baseline_lock
        self.ema_beta = 0.95

    def act(self, state, deterministic=False):
        if isinstance(state, torch.Tensor):
            state_tensor = state.to(self.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            current_vx_norm = state_tensor[0, 2].item()
        else:
            current_vx_norm = state[2]
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        current_speed = current_vx_norm * self.scales['vx']

        with torch.no_grad():
            action_probs = self.actor(state_tensor)

        if current_speed > 40.0:
            action_probs[0, 3] = 0.0
            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum()
            else:
                action_probs[0, 4] = 1.0

        # [保护] act 阶段防止 NaN
        if torch.isnan(action_probs).any():
            action_probs = torch.ones_like(action_probs) / self.action_size

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
        ttc_threshold = 8.0
        batch_size = state.shape[0]
        num_features = 5
        num_vehicles = state.shape[1] // num_features
        reshaped_state = state.view(batch_size, num_vehicles, num_features)

        delta_x = reshaped_state[:, 1:, 0] * self.scales['x']
        delta_y = reshaped_state[:, 1:, 1] * self.scales['y']
        rel_vx = reshaped_state[:, 1:, 2] * self.scales['vx']
        rel_vy = reshaped_state[:, 1:, 3] * self.scales['vy']

        dists = torch.sqrt(delta_x ** 2 + delta_y ** 2) + 1e-6
        dot_product = delta_x * rel_vx + delta_y * rel_vy
        closing_speed = -dot_product / dists

        is_not_padding = dists > 0.1
        is_not_crashed = dists > 0.6
        valid_mask = is_not_padding & is_not_crashed
        is_closing = (closing_speed > 0.05) & valid_mask

        dists = torch.clamp(dists, min=0.5)
        ttc_risk = torch.zeros_like(dists)
        ttc_risk[is_closing] = ttc_threshold * closing_speed[is_closing] / dists[is_closing]

        dist_risk = torch.zeros_like(dists)
        dist_risk[valid_mask] = 20.0 / dists[valid_mask]

        total_risk = torch.max(ttc_risk, dist_risk)
        total_risk = total_risk * valid_mask.float()
        max_total_risk, _ = torch.max(total_risk, dim=1)
        normalized_risk = torch.tanh(max_total_risk * 0.2)

        return normalized_risk.mean().item()

    def calculate_group_advantages(self, group_trajectories):
        group_returns = []
        group_crashed = []
        group_rewards = []

        for traj in group_trajectories:
            G = 0
            crashed = False
            ep_reward = 0
            for t in reversed(range(len(traj))):
                r = traj[t][2]
                term = traj[t][4]
                if term: crashed = True
                G = r + self.gamma * G
                ep_reward += r
            group_returns.append(G)
            group_crashed.append(crashed)
            group_rewards.append(ep_reward)

        group_returns_arr = np.array(group_returns)
        group_crashed_arr = np.array(group_crashed)
        avg_group_reward = np.mean(group_rewards)

        # 1. 计算这一组的基础统计量 (Base Statistics)
        group_mean = group_returns_arr.mean()
        group_std = group_returns_arr.std() + 1e-8

        # 2. 计算标准化的相对优势 (Base Advantage)
        # 这反映了每个样本相对于平均水平的好坏
        # 此时 standard_adv 里既包含了死人也包含了活人
        standard_adv = (group_returns_arr - group_mean) / group_std

        # --- 趋势信号 (保持不变) ---
        trend_signal = 0.0
        if self.global_baseline is not None and self.baseline_lock is not None:
            with self.baseline_lock:
                current_baseline = self.global_baseline.value
                if current_baseline == 0.0:
                    new_baseline = group_mean
                else:
                    new_baseline = self.ema_beta * current_baseline + (1 - self.ema_beta) * group_mean
                self.global_baseline.value = new_baseline
            trend_signal = np.clip((group_mean - new_baseline), -1.0, 1.0)

        # --- [核心修改] 混合局的逻辑优化 ---
        has_survivor = np.any(~group_crashed_arr)
        has_crasher = np.any(group_crashed_arr)
        final_advantages = np.zeros_like(group_returns_arr, dtype=np.float32)

        if has_survivor and has_crasher:
            # === 旧逻辑 (有问题) ===
            # final_advantages[~group_crashed_arr] = 2.0  <-- "大锅饭"，导致细粒度奖励失效
            # final_advantages[group_crashed_arr] = -5.0

            # === 新逻辑 (引入内卷) ===
            # 1. 幸存者：在标准优势的基础上，给予 +1.0 的生存奖励
            # 这样，同样是活下来，分高的人(比如 standard_adv=0.5) 拿 1.5，分低的(standard_adv=-0.2) 拿 0.8
            # 这里的 standard_adv 可能会受到 crashers 低分的干扰而整体偏大，但相对排名是对的

            # 这里的逻辑是：standard_adv 保留了组内排名的相对大小
            # +1.0 保证了他们总体上是优于 crashers 的
            final_advantages[~group_crashed_arr] = standard_adv[~group_crashed_arr] + 1.0

            # 2. 撞车者：不要直接给 -5.0，而是在相对优势上扣分
            # 这样坚持得久的撞车者优势会比秒撞的高，模型能学到“坚持就是胜利”
            # -2.0 保证了他们总体上是劣于 survivors 的
            final_advantages[group_crashed_arr] = standard_adv[group_crashed_arr] - 2.0

        elif has_survivor and not has_crasher:
            # 全员存活：标准 GRPO + 趋势奖励
            final_advantages = standard_adv + 0.2 * trend_signal

        else:
            # 全员撞车：标准 GRPO - 失败惩罚
            # 比谁撞得晚，撞得晚的分数高，standard_adv 就高
            final_advantages = standard_adv - 1.0

        # 截断，防止梯度爆炸
        final_advantages = np.clip(final_advantages, -5.0, 5.0)
        return final_advantages, avg_group_reward

    def learn(self, current_total_timesteps):
        # 1. 取出数据
        states, actions, advantages, next_states, dones, logprobs = self.memory.get_all_and_clear()

        # 保护：如果数据不足 Batch Size，不学习
        if states is None or len(states) < self.bs:
            return

        # 转为 Tensor
        states = states.to(self.device)
        actions = actions.to(self.device)
        advantages = advantages.to(self.device)
        logprobs = logprobs.to(self.device)

        # 2. 计算时间因子 (Progress)
        # 范围：0.0 (训练开始) -> 1.0 (训练结束)
        # current_total_timesteps 是从 train_grpo.py 传进来的全局累积步数
        progress = current_total_timesteps / self.decay_max_step
        progress = min(max(progress, 0.0), 1.0)  # 限制在 [0, 1] 之间

        # 3. 计算时空自适应 Beta (Spatio-Temporal Beta)
        if self.use_dynamic_beta:
            # 获取当前的空间风险 (0.0 ~ 1.0)
            current_risk = self.calculate_risk(states)
            target_beta = self.beta_min + (self.beta_max - self.beta_min) * current_risk * progress
            self.beta = target_beta
        else:
            # 静态实验保持不变 (或者你可以设为某个固定值)
            self.beta = 0.1

            # 4. 学习率衰减 (保持原逻辑)
        decay = 1.0 - progress
        current_lr = self.lr * decay
        for pg in self.optimizer.param_groups:
            pg['lr'] = max(current_lr, 1e-6)

        # 5. PPO 更新循环
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

                # NaN 保护
                if torch.isnan(action_probs).any():
                    continue

                dist = torch.distributions.Categorical(action_probs)
                mb_new_log = dist.log_prob(mb_actions)
                dist_entropy = dist.entropy().mean()

                with torch.no_grad():
                    approx_kl = 0.5 * ((mb_new_log - mb_old_log) ** 2).mean()

                ratio = torch.exp(mb_new_log - mb_old_log)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv

                loss = -torch.min(surr1, surr2).mean() + \
                       self.beta * approx_kl - \
                       self.entropy_coef * dist_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer.step()

        self.entropy = dist_entropy.item()

    def get_state_dict(self):
        return {'actor': self.actor.state_dict()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd['actor'])