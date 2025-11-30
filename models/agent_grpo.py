import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.networks import Actor
from models.replay_buffer import Replay_Buffer


class Agent_GRPO:
    """
    GRPO Agent (修复版)
    核心变动：
    1. learn() 函数不再计算 Monte-Carlo Return，也不再进行全局归一化。
       它假设 Buffer 中取出的 'rewards' 字段已经是 Worker 计算好的 'Group Relative Advantage'。
    2. calculate_risk() 修复了索引逻辑，配合 Observation Fixed 模式使用。
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

        self.beta_init = 0.01
        self.beta_min = 0.0001
        self.beta_max = 0.05
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
        计算风险值 (Risk)
        引入相对速度 (Time-to-Collision)
        """
        # 设定 TTC 阈值
        ttc_threshold = 5.0

        # --- 维度重塑 ---
        # state 输入形状: (Batch_Size, 75) -> 扁平的向量
        # 我们需要把 state 重塑成: (Batch_Size, 15, 5) -> (Batch, 车辆数, 特征数)，方便后续使用 [:, 0, 0] 这种索引
        batch_size = state.shape[0]
        num_features = 5  # x, y, vx, vy, heading
        num_vehicles = state.shape[1] // num_features  # 车辆数 (75 / 5 = 15)
        reshaped_state = state.view(batch_size, num_vehicles, num_features)  # 不改变原数据

        # 获取与周围车辆相对位置与相对速度（接近速度）
        # 注意：需要反归一化（根据 custom_env 中的环境配置）
        delta_x = reshaped_state[:, 1:, 0] * 100.0
        delta_y = reshaped_state[:, 1:, 1] * 100.0
        rel_vx = reshaped_state[:, 1:, 2] * 20.0
        rel_vy = reshaped_state[:, 1:, 3] * 20.0

        # 计算欧式距离，限制撞车时最小值为 1e-6
        dists = torch.sqrt(delta_x ** 2 + delta_y ** 2) + 1e-6

        # 计算接近速度 - -- 使用向量点积投影: (pos · vel)
        # 结果 > 0: 正在远离 (位置向量和速度向量方向一致)
        # 结果 < 0: 正在靠近 (位置向量和速度向量方向相反)
        dot_product = delta_x * rel_vx + delta_y * rel_vy

        # 将速度投影到距离方向上, 只关心连线方向上的速度 > 0 为远离， < 0 为靠近
        # 取反，变成正数表示靠近速度
        closing_speed = -dot_product / dists

        # 构建掩码
        # 判断真车：距离 > 0.1米 认为是真车，0.0 是填充位
        is_not_padding = dists > 0.1
        # 判断碰撞，Reward 会给惩罚，Risk 此时应该设为 0，避免干扰 Beta 的学习
        is_not_crashed = dists > 0.6
        # 综合有效掩码: 既不是填充，也不是已经撞上的烂摊子
        valid_mask = is_not_padding & is_not_crashed
        # 只计算 "正在靠近，速度显著" 且 "有效目标" 的情况
        is_closing = (closing_speed > 0.05) & valid_mask

        # TTC = dist / speed
        # Risk = threshold / TTC = threshold * speed / dist
        # 设定安全分母 (防止除以极小数)
        dists = torch.clamp(dists, min=0.5)
        ttc_risk = torch.zeros_like(dists)
        ttc_risk[is_closing] = ttc_threshold * closing_speed[is_closing] / dists[is_closing]

        # 保底计算距离风险: 即使相对静止，贴太近也是高风险
        dist_risk = torch.zeros_like(dists)
        dist_risk[valid_mask] = 20.0 / dists[valid_mask]

        # 计算综合风险
        total_risk = torch.max(ttc_risk, dist_risk)
        total_risk = total_risk * valid_mask.float()       # 确认强制把无效位置归零

        # 找出每个样本中，周围最危险的那辆车
        max_total_risk, _ = torch.max(total_risk, dim=1)

        # [核心] 数值归一化映射 ---
        # 优化线性截断，用 tanh 函数做软饱和，尽量避免使用 clamp 做硬截断
        # 逻辑:
        #   Risk=0 -> 0.0
        #   Risk=5 (距离4米) -> 0.5 (中等危险)
        #   Risk=20 (贴脸) -> 接近 1.0 (极度危险)

        # 固定物理缩放系数（0.2），不再调整
        #   距离 10m (Raw=2) -> tanh(0.4) = 0.38 (低风险)
        #   距离 5m  (Raw=4) -> tanh(0.8) = 0.66 (中高风险)
        #   距离 2m  (Raw=10)-> tanh(2.0) = 0.96 (极高风险)
        normalized_risk = torch.tanh(max_total_risk * 0.2)

        return normalized_risk.mean().item()


    def calculate_group_advantages(self, group_trajectories):
        """
        纯数学计算函数。
        Args:
            group_trajectories: List[List[Tuple]], 原始轨迹数据 (Worker 采集的)
        Returns:
            normalized_advantages: np.array, 形状为 (Group_Size,)
                对应每条轨迹的 Advantage 值。
            group_mean: float
                组平均回报 (用于日志)。
        """
        group_returns = []
        group_crashed = []  # 新增：记录每条轨迹是否最终撞车

        # 1. 计算每条轨迹的 Monte-Carlo Return
        for traj in group_trajectories:
            G = 0
            crashed = False
            # tuple 结构是 (s, a, r, ns, d, lp)，索引 2 是 reward
            for t in reversed(range(len(traj))):
                r = traj[t][2]
                # 判定撞车：根据 reward 阈值或者 done 状态
                # 假设你的环境 collision_reward 是 -20
                if r <= -10.0:
                    crashed = True
                G = r + self.gamma * G
            group_returns.append(G)
            group_crashed.append(crashed)

        # 2. 归一化计算
        group_returns_arr = np.array(group_returns)
        group_mean = group_returns_arr.mean()
        group_std = group_returns_arr.std() + 1e-8

        # 计算标准化优势 (GRPO 核心公式)
        # 结果是一个列表，例如: [0.5, -1.2, 0.8, ...]，对应第 0, 1, 2... 条轨迹
        normalized_advantages = (group_returns_arr - group_mean) / group_std

        # 3. [关键优化] 绝对惩罚掩码
        # 如果这条轨迹撞车了，无论它相对于平均值如何，强制它的优势为负
        for i in range(len(normalized_advantages)):
            if group_crashed[i]:
                # 使用 np.minimum 做截断，而不是强制赋值
                # 这样：-1.5 还是 -1.5 (保留更差的信号)
                #      +2.0 变成 -0.5 (防止被奖励，但比 -1.5 好)
                normalized_advantages[i] = np.minimum(normalized_advantages[i], -0.5)

        return normalized_advantages, group_mean

    def learn(self, current_total_timesteps):
        # [核心修正]
        # get_all_and_clear 返回的第三个参数原本是 rewards，现在存储的是 advantages
        states, actions, advantages, next_states, dones, logprobs = self.memory.get_all_and_clear()

        if states is None or len(states) < self.bs:
            return

        states = states.to(self.device)
        actions = actions.to(self.device)
        advantages = advantages.to(self.device)  # 这是 Worker 算好的相对优势
        logprobs = logprobs.to(self.device)

        # 放大优势
        # Clip Advantage 保证数值稳定
        advantages = advantages * 5.0
        advantages = torch.clamp(advantages, -4.0, 4.0)

        # 动态 Beta 更新逻辑
        if self.use_dynamic_beta:
            current_risk = self.calculate_risk(states)
            #print(f" | current_risk: {current_risk} | ")

            # 使用反比例公式，保证 Beta 永远 > 0
            # 敏感度因子 risk_sensitivity 建议值:
            #   1.0  -> 危险时(Risk=1), Beta 降为原来的 1/2 (保守，防止做出过激行为)
            #   5.0  -> 危险时(Risk=1), Beta 降为原来的 1/6 (适中)
            #   10.0 -> 危险时(Risk=1), Beta 降为原来的 1/11 (激进，允许剧烈改变)
            risk_sensitivity = 1.0
            #target_beta = self.beta_init / (1.0 + risk_sensitivity * current_risk)
            #self.beta = max(target_beta, self.beta_min)

            # 使用乘法公式（更加稳健，用于解决超速震荡）
            target_beta = self.beta_init / (1.0 + risk_sensitivity * current_risk)
            self.beta = min(target_beta, self.beta_max)
        else:
            self.beta = self.beta_init  # 消融实验会一直跑这里

        # 学习率衰减
        decay = self.dr ** (current_total_timesteps // self.decay_step_size)
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.lr * decay

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
                mb_adv = advantages[idx]  # 直接使用

                action_probs = self.actor(mb_states)
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
                       self.entropy_coef * decay * dist_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer.step()

        self.entropy = dist_entropy.item()          # 策略熵
        #print(f" | approx_kl: {approx_kl} | ")      # 最后一次的 approx_kl

    def get_state_dict(self):
        return {'actor': self.actor.state_dict()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd['actor'])