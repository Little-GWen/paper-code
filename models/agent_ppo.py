import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.networks import Actor, Critic
from models.replay_buffer import Replay_Buffer


class Agent_PPO:
    def __init__(self, state_size, action_size, bs, lr, dr, decay_step_size, tau, gamma, lam, eps_clip, K_epochs,
                 critic_loss_coef, entropy_coef, device, shared=False, manager=None, is_worker=False):
        self.state_size, self.action_size = state_size, action_size
        self.bs, self.lr, self.dr, self.decay_step_size = bs, lr, dr, decay_step_size
        self.gamma, self.lam, self.eps_clip, self.K_epochs = gamma, lam, eps_clip, K_epochs
        self.critic_loss_coef, self.entropy_coef, self.device = critic_loss_coef, entropy_coef, device

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

        # --- [终极修复] 强制转为 Python int ---
        action_val = action_tensor.cpu().numpy()
        action_val = action_val.flatten()
        return int(action_val[0]), log_prob

    def learn(self, current_total_timesteps):
        states, actions, rewards, next_states, dones, logprobs = self.memory.get_all_and_clear()
        if states is None or len(states) < self.bs: return

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        logprobs = logprobs.to(self.device)

        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            advantages = torch.zeros_like(rewards).to(self.device)
            delta = rewards + self.gamma * next_values * (1 - dones) - values
            advantage = 0
            for t in reversed(range(len(rewards))):
                if dones[t]: advantage = 0
                advantage = delta[t] + self.gamma * self.lam * advantage
                advantages[t] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
                mb_old_log, mb_adv, mb_ret = logprobs[idx], advantages[idx], returns[idx]

                action_probs = self.actor(mb_states)
                dist = torch.distributions.Categorical(action_probs)
                mb_new_log = dist.log_prob(mb_actions)
                dist_entropy = dist.entropy().mean()
                mb_val = self.critic(mb_states).squeeze(-1)

                ratio = torch.exp(mb_new_log - mb_old_log)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv
                loss = -torch.min(surr1, surr2).mean() + \
                       self.critic_loss_coef * self.mse_loss(mb_val, mb_ret) - \
                       self.entropy_coef * decay * dist_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()
        self.entropy = dist_entropy.item()

    def get_state_dict(self):
        return {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd['actor'])
        self.critic.load_state_dict(sd['critic'])