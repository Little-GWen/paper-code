import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from models.networks import QNetwork
from models.replay_buffer import Replay_Buffer


# --- 注意：这里千万不要有 "from models.agent_dqn import Agent_DQN" ---

class Agent_DQN:
    def __init__(self, state_size, action_size, bs, lr, gamma, epsilon_start, epsilon_end, epsilon_decay, device,
                 manager=None):
        self.state_size = state_size
        self.action_size = action_size
        self.bs = bs
        self.gamma = gamma
        self.device = device

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        # DQN 这里的 replay buffer 大小设为 50000
        self.memory = Replay_Buffer(int(50000), bs, manager)
        self.update_counter = 0

    # --- act 函数 (兼容 deterministic 参数) ---
    def act(self, state, deterministic=False):
        """
        deterministic=True: 测试模式，完全贪婪 (epsilon=0)
        deterministic=False: 训练模式，使用 epsilon-greedy
        """
        eps = 0.0 if deterministic else self.epsilon

        if random.random() > eps:
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state)
            action = torch.argmax(q_values).item()
        else:
            action = random.randrange(self.action_size)

        # 返回 (action, log_prob=0) 以兼容接口
        return action, 0.0

    def learn(self, current_step=None):
        states, actions, rewards, next_states, dones, _ = self.memory.sample(self.bs)
        if states is None: return

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards.unsqueeze(1) + (self.gamma * q_next * (1 - dones.unsqueeze(1)))

        q_expected = self.q_net(states).gather(1, actions.unsqueeze(1))
        loss = F.mse_loss(q_expected, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def get_state_dict(self):
        return self.q_net.state_dict()

    def load_state_dict(self, state_dict):
        self.q_net.load_state_dict(state_dict)