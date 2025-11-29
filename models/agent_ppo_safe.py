import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.networks import Actor, Critic
from models.replay_buffer import Replay_Buffer


class Agent_PPO_Safe:
    def __init__(self, state_size, action_size, bs, lr, dr, decay_step_size, gamma, lam, eps_clip, K_epochs,
                 critic_loss_coef, entropy_coef, device, shared=False, manager=None, is_worker=False,
                 use_dynamic_beta=True):
        self.state_size, self.action_size = state_size, action_size
        self.bs, self.lr, self.dr, self.decay_step_size = bs, lr, dr, decay_step_size
        self.gamma, self.lam, self.eps_clip, self.K_epochs = gamma, lam, eps_clip, K_epochs
        self.critic_loss_coef, self.entropy_coef = critic_loss_coef, entropy_coef
        self.device = device
        self.use_dynamic_beta = use_dynamic_beta

        self.beta_init = 0.005
        self.beta = self.beta_init
        self.beta_max = 5.0

        self.memory = Replay_Buffer(int(20000), bs, manager)
        if shared:
            self.actor = Actor(state_size, action_size).share_memory().to(device)
            self.critic = Critic(state_size).share_memory().to(device)
        else:
            self.actor = Actor(state_size, action_size).to(device)
            self.critic = Critic(state_size).to(device)

        if not is_worker:
            self.optimizer = optim.Adam([
                {'params': self.actor.parameters()}, {'params': self.critic.parameters()}
            ], lr=lr)
        self.mse_loss = nn.MSELoss()
        self.entropy = 0

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action_tensor = dist.sample()
        return action_tensor.cpu().numpy().item(), dist.log_prob(action_tensor).detach().cpu().numpy().item()

    def calculate_risk(self, state):
        # --- 维度重塑 ---
        # state 输入形状: (Batch_Size, 75) -> 扁平的向量
        # 我们需要把 state 重塑成: (Batch_Size, 15, 5) -> (Batch, 车辆数, 特征数)，方便后续使用 [:, 0, 0] 这种索引
        batch_size = state.shape[0]
        num_features = 5  # x, y, vx, vy, heading
        num_vehicles = state.shape[1] // num_features  # 车辆数 (75 / 5 = 15)
        reshaped_state = state.view(batch_size, num_vehicles, num_features)  # 不改变原数据

        # 第 0 辆车是自车 (Ego)，使用相对坐标时，自车坐标 = (0, 0)
        # ego_x, ego_y = reshaped_state[:, 0, 0], reshaped_state[:, 0, 1]

        # 第 1 到 N 辆车是周围车辆
        # 周围车辆相对自车的 delta_x, delta_y
        delta_x = reshaped_state[:, 1:, 0]
        delta_y = reshaped_state[:, 1:, 1]

        # 计算欧氏距离
        dists = torch.sqrt(delta_x ** 2 + delta_y ** 2)

        # 找到最近的一辆车的距离
        min_dist, _ = torch.min(dists, dim=1)

        # 使用 1e-6 作为安全平滑项，避免除零
        # 距离越近，Risk 越大 (归一化: 30米距离时 Risk=1.0)
        # 钳制 Risk 的最大值（50.0），防止 min_dist 极小时 beta 爆炸
        safe_risk = torch.clamp(30.0 / (min_dist + 1e-6), max=50.0)

        # 抗噪处理: 取最危险的 10% 数据的平均值
        k = int(safe_risk.size(0) * 0.1)
        top_k_risk, _ = torch.topk(safe_risk, k)
        risk = top_k_risk.mean().item()
        # 激进处理: 直接取最危险的数据
        #risk = safe_risk.max().item()

        return risk

    def learn(self, current_total_timesteps):
        states, actions, rewards, next_states, dones, logprobs = self.memory.get_all_and_clear()
        if states is None or len(states) < self.bs:
            return
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        logprobs = logprobs.to(self.device)

        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            advantages = torch.zeros_like(rewards).to(self.device)
            delta = rewards + self.gamma * next_values * (1 - dones) - values
            advantage = 0
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    advantage = 0
                advantage = delta[t] + self.gamma * self.lam * advantage
                advantages[t] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.use_dynamic_beta:
            # 根据风险值动态放大 beta 值，能够促使策略更稳定、更保守
            current_risk = self.calculate_risk(states)
            self.beta = min(self.beta_init * (1 + current_risk), self.beta_max)
        else:
            self.beta = self.beta_init

        # 学习率衰减，有助于后期训练的稳定
        decay = self.dr ** (current_total_timesteps // self.decay_step_size)
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.lr * decay

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        for _ in range(self.K_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.bs):
                end = start + self.bs
                idx = indices[start:end]
                mb_states, mb_actions = states[idx], actions[idx]
                mb_old_log, mb_adv, mb_ret = logprobs[idx], advantages[idx], returns[idx]

                # 计算新策略的概率
                action_probs = self.actor(mb_states)
                dist = torch.distributions.Categorical(action_probs)
                mb_new_log = dist.log_prob(mb_actions)
                dist_entropy = dist.entropy().mean()
                mb_val = self.critic(mb_states).squeeze(-1)

                with torch.no_grad():
                    approx_kl = 0.5 * ((mb_new_log - mb_old_log) ** 2).mean()

                ratio = torch.exp(mb_new_log - mb_old_log)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv

                loss = -torch.min(surr1, surr2).mean() + \
                        self.critic_loss_coef * self.mse_loss(mb_val, mb_ret) + \
                        self.beta * approx_kl - \
                        self.entropy_coef * decay * dist_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

        self.entropy = dist_entropy.item() * decay # 衰减后的策略熵
        print(f" | approx_kl: {approx_kl} | ")

    def get_state_dict(self):
        return {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd['actor'])
        self.critic.load_state_dict(sd['critic'])