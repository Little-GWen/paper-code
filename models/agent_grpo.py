import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.networks import Actor
from models.replay_buffer import Replay_Buffer


class Agent_GRPO:
    def __init__(self, state_size, action_size, bs, lr, dr, decay_step_size, tau, gamma, lam, eps_clip, K_epochs,
                 critic_loss_coef, entropy_coef, device, shared=False, manager=None, is_worker=False,
                 use_dynamic_beta=True):
        self.state_size, self.action_size = state_size, action_size
        self.bs, self.lr, self.dr, self.decay_step_size = bs, lr, dr, decay_step_size
        self.gamma, self.K_epochs, self.entropy_coef = gamma, K_epochs, entropy_coef
        self.device = device
        self.use_dynamic_beta = use_dynamic_beta

        # --- [优化] 降低初始Beta，防止早期锁死 ---
        self.beta_init = 0.01
        self.beta = self.beta_init

        self.memory = Replay_Buffer(int(20000), bs, manager)
        if shared:
            self.actor = Actor(state_size, action_size).share_memory().to(device)
        else:
            self.actor = Actor(state_size, action_size).to(device)

        if not is_worker: self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.entropy = 0

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action_tensor = dist.sample()
        return action_tensor.cpu().numpy().item(), dist.log_prob(action_tensor).detach().cpu().numpy().item()

    def calculate_risk(self, state):
        # 修复维度问题: (Batch, 75) -> (Batch, 15, 5)
        if state.shape[1] < 10: return 0.0
        batch_size = state.shape[0]
        num_feats = 5
        num_vehicles = state.shape[1] // num_feats
        reshaped = state.view(batch_size, num_vehicles, num_feats)

        ego_x = reshaped[:, 0, 0]
        ego_y = reshaped[:, 0, 1]
        others_x = reshaped[:, 1:, 0]
        others_y = reshaped[:, 1:, 1]

        dists = torch.sqrt((others_x - ego_x.unsqueeze(1)) ** 2 + (others_y - ego_y.unsqueeze(1)) ** 2)
        min_dist, _ = torch.min(torch.clamp(dists, min=0.5), dim=1)
        risk = torch.clamp(10.0 / min_dist, 0.0, 5.0)
        return risk.mean().item()

    def learn(self, current_total_timesteps):
        states, actions, rewards, next_states, dones, logprobs = self.memory.get_all_and_clear()
        if states is None or len(states) < self.bs: return

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        logprobs = logprobs.to(self.device)

        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done: G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Group Norm (防止数值爆炸)
        adv_mean, adv_std = returns.mean(), returns.std() + 1e-5
        advantages = torch.clamp((returns - adv_mean) / adv_std, -3.0, 3.0)

        if self.use_dynamic_beta:
            current_risk = self.calculate_risk(states)
            self.beta = min(self.beta_init * (1 + current_risk), 0.5)  # 限制上限
        else:
            self.beta = self.beta_init

        decay = self.dr ** (current_total_timesteps // self.decay_step_size)
        for pg in self.optimizer.param_groups: pg['lr'] = self.lr * decay

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        for _ in range(self.K_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.bs):
                end = start + self.bs
                idx = indices[start:end]
                mb_states, mb_actions = states[idx], actions[idx]
                mb_old_log, mb_adv = logprobs[idx], advantages[idx]

                action_probs = self.actor(mb_states)
                dist = torch.distributions.Categorical(action_probs)
                mb_new_log = dist.log_prob(mb_actions)
                dist_entropy = dist.entropy().mean()

                ratio = torch.exp(mb_new_log - mb_old_log)
                with torch.no_grad(): approx_kl = 0.5 * ((mb_new_log - mb_old_log) ** 2).mean()

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * mb_adv
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