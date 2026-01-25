import torch
import torch.optim as optim
import numpy as np
from models.networks import Actor


class Agent_GRPO_Ray:
    def __init__(self, state_size, action_size, bs, lr, decay_max_step, gamma, eps_clip, K_epochs, entropy_coef,
                 device, is_worker=False, adv_mode="tiered", use_dynamic_beta=True):
        self.state_size = state_size
        self.action_size = action_size
        self.bs = bs
        self.lr = lr
        self.decay_max_step = decay_max_step
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.device = device
        # self.scales 已删除，因为不再需要物理坐标反归一化
        self.entropy = 0

        # 网络初始化
        self.actor = Actor(state_size, action_size).to(device)

        if not is_worker:
            self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        else:
            self.optimizer = None

        # GRPO 变体配置
        self.adv_mode = adv_mode
        self.use_dynamic_beta = use_dynamic_beta
        self.beta = 0.1  # 初始 beta

    def act(self, state, deterministic=False):
        # 1. 转为 Tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        # === 🛡️ 鲁棒性保护：输入清洗 ===
        # 防止环境偶尔产生的 NaN 导致网络崩溃
        if torch.isnan(state).any() or torch.isinf(state).any():
            state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action_probs = self.actor(state)

        # === 🛡️ 鲁棒性保护：输出清洗 ===
        if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
            action_probs = torch.ones_like(action_probs) / self.action_size

        if deterministic:
            action_tensor = torch.argmax(action_probs, dim=1)
            log_prob = 0.0
        else:
            dist = torch.distributions.Categorical(action_probs)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor).detach().cpu().numpy().item()

        return int(action_tensor.cpu().numpy().flatten()[0]), log_prob

    def get_weights(self):
        """导出权重到 CPU (Ray 传输用)"""
        return {k: v.cpu() for k, v in self.actor.state_dict().items()}

    def set_weights(self, weights):
        """加载权重"""
        self.actor.load_state_dict(weights)

    # [已删除] calculate_risk 函数

    def calculate_group_advantages(self, group_trajectories, baseline_value):
        group_returns = []
        group_crashed = []

        # 1. 计算每条轨迹的回报 (Return)
        for traj in group_trajectories:
            G = 0
            # 倒序计算
            for t in reversed(range(len(traj))):
                s, a, r, ns, term, lp = traj[t]
                G = r + self.gamma * G

            group_returns.append(G)
            # term 位于 tuple 的第4位 (index 4)
            group_crashed.append(traj[-1][4])

        group_returns_arr = np.array(group_returns)

        # 2. 计算标准优势 (Z-Score Normalization)
        # 这是 GRPO 的核心：(Return - Mean) / Std
        group_mean = group_returns_arr.mean()
        group_std = group_returns_arr.std() + 1e-8
        final_adv = (group_returns_arr - group_mean) / group_std

        # 3. Tiered Advantage 调整 (无 Risk 版本)
        # 逻辑：保留“幸存者”与“撞车者”的分层，保留“Global Trend”奖励
        if self.adv_mode == "tiered":
            # 计算趋势信号：这组表现比历史基准(baseline)好了多少？
            trend = np.clip((group_mean - baseline_value), -1.0, 1.0)

            survivor_mask = ~np.array(group_crashed, dtype=bool)
            crasher_mask = np.array(group_crashed, dtype=bool)

            if np.any(survivor_mask):
                # 幸存者优势 = 原始排名 + 趋势奖励
                # (去掉了之前的 "- surv_risks")
                surv_adv = final_adv[survivor_mask] + (0.2 * trend)

                # 幸存者内部再次归一化 (让幸存者之间也能分出高下)
                if len(surv_adv) > 1:
                    surv_adv = (surv_adv - surv_adv.mean()) / (surv_adv.std() + 1e-8)

                final_adv[survivor_mask] = surv_adv

            if np.any(crasher_mask):
                if np.any(survivor_mask):
                    # 撞车者的得分，必须比最差的幸存者还要低 (分层打击)
                    final_adv[crasher_mask] = final_adv[survivor_mask].min() - 2.0
                else:
                    # 全员撞车，全员扣分
                    final_adv[crasher_mask] -= 1.0

        # 截断，防止梯度爆炸
        final_adv = np.clip(final_adv, -4.0, 4.0)

        # 计算单纯的 Reward 之和 (用于日志显示，不参与更新)
        avg_rew = np.mean([sum(t[2] for t in tr) for tr in group_trajectories])

        return final_adv, avg_rew, group_mean

    def learn(self, memory_list, current_total_timesteps):
        if not memory_list or len(memory_list) < self.bs: return

        # 解包数据 (Ray 传回来的是 list of tuples)
        states, actions, rewards, next_states, dones, log_probs = zip(*memory_list)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        advantages = torch.tensor(np.array(rewards), dtype=torch.float32).to(
            self.device)  # 注意：这里的 rewards 实际上存的是 Advantage
        old_log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32).to(self.device)

        # === 🛡️ 训练数据保护 ===
        if torch.isnan(states).any(): states = torch.nan_to_num(states, nan=0.0)
        if torch.isnan(advantages).any(): advantages = torch.nan_to_num(advantages, nan=0.0)

        # 学习率线性衰减
        decay = 1.0 - min(max(current_total_timesteps / self.decay_max_step, 0.0), 1.0)
        cur_lr = self.lr * decay
        for pg in self.optimizer.param_groups: pg['lr'] = max(cur_lr, 1e-6)

        ds_size = states.size(0)
        indices = np.arange(ds_size)

        for _ in range(self.K_epochs):
            np.random.shuffle(indices)
            for start in range(0, ds_size, self.bs):
                idx = indices[start:start + self.bs]
                mb_s, mb_a, mb_adv, mb_old_lp = states[idx], actions[idx], advantages[idx], old_log_probs[idx]

                action_probs = self.actor(mb_s)
                dist = torch.distributions.Categorical(action_probs)
                mb_new_lp = dist.log_prob(mb_a)
                dist_entropy = dist.entropy().mean()

                ratio = torch.exp(mb_new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv

                # KL 散度近似 (用于动态 Beta)
                with torch.no_grad():
                    approx_kl = 0.5 * ((mb_new_lp - mb_old_lp) ** 2).mean()

                # Loss 计算
                loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist_entropy

                if self.use_dynamic_beta:
                    loss += self.beta * approx_kl

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer.step()

        self.entropy = dist_entropy.item()