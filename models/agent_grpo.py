import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.networks import Actor
from models.replay_buffer import Replay_Buffer


class Agent_GRPO:
    """
    GRPO Agent (修复版：补全 eps_clip 属性)
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
        self.lam = lam  # [修复] 补全 lambda
        self.eps_clip = eps_clip  # [修复] 补全 eps_clip (报错原因)
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.device = device

        # --- [核心开关] 是否启用动态 Beta ---
        self.use_dynamic_beta = use_dynamic_beta

        # 初始 Beta 和最大 Beta 限制
        self.beta_init = 0.01
        self.beta_max = 0.2
        self.beta = self.beta_init

        self.memory = Replay_Buffer(int(20000), bs, manager)

        # GRPO 是 Critic-Free 的，所以只有一个 Actor
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

    # --- [核心函数] 计算当前 Batch 的平均风险 ---
    def calculate_risk(self, state):
        """
        基于车辆间距计算风险值。
        Risk = mean(10.0 / min_distance)
        距离越近，Risk 越大。
        """
        # 状态维度检查 (确保包含周围车辆信息)
        if state.shape[1] < 10: return 0.0

        batch_size = state.shape[0]
        num_feats = 5
        # 假设 state 是 [Batch, N*5] 的扁平向量
        num_vehicles = state.shape[1] // num_feats

        # 重塑为 [Batch, Vehicles, Feats]
        reshaped = state.view(batch_size, num_vehicles, num_feats)

        # Ego 车位置 (通常是第0个)
        ego_x = reshaped[:, 0, 0]
        ego_y = reshaped[:, 0, 1]

        # 周围车位置
        others_x = reshaped[:, 1:, 0]
        others_y = reshaped[:, 1:, 1]

        # 计算欧氏距离
        dists = torch.sqrt((others_x - ego_x.unsqueeze(1)) ** 2 + (others_y - ego_y.unsqueeze(1)) ** 2)

        # 找到最近的一辆车的距离，加一个极小值防止除零，并限制最小距离以防数值爆炸
        min_dist, _ = torch.min(torch.clamp(dists, min=1.0), dim=1)

        # 风险公式：距离越小，风险倒数越大
        # 例如距离 2米 -> Risk 5.0; 距离 20米 -> Risk 0.5
        risk_val = (10.0 / min_dist).mean().item()
        return risk_val

    def learn(self, current_total_timesteps):
        states, actions, rewards, next_states, dones, logprobs = self.memory.get_all_and_clear()
        if states is None or len(states) < self.bs: return

        states = states.to(self.device)
        actions = actions.to(self.device)
        logprobs = logprobs.to(self.device)

        # --- [修复开始] 必须手动计算折扣回报 (Monte-Carlo Returns) ---
        # 之前的代码漏掉了这一步，直接用了 raw rewards，这是不对的。
        returns = []
        G = 0
        # 逆序遍历，计算累积回报
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # --- [修复结束] ---

        # --- GRPO 核心 1: 组归一化 Advantage ---
        adv_mean = returns_tensor.mean()
        adv_std = returns_tensor.std() + 1e-5
        advantages = (returns_tensor - adv_mean) / adv_std
        advantages = torch.clamp(advantages, -4.0, 4.0)

        # --- 动态 Beta 逻辑 ---
        if self.use_dynamic_beta:
            current_risk = self.calculate_risk(states)
            target_beta = self.beta_init * (1 + current_risk)
            self.beta = min(target_beta, self.beta_max)
        else:
            self.beta = self.beta_init

        # ... (后续的 PPO 更新循环保持不变) ...
        decay = self.dr ** (current_total_timesteps // self.decay_step_size)
        for pg in self.optimizer.param_groups: pg['lr'] = self.lr * decay

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
                # 这里的 eps_clip 已经在 __init__ 里修复了
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv

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