import torch
import numpy as np
import random
import os

# 设备配置
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

# 环境名
RAM_ENV_NAME = 'my-highway-v0'

# 训练参数
# [优化] 增大 Batch Size 以稳定梯度估计
NUM_PROCESSES = 15
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4

# 衰减参数
DECAY_RATE = 0.999
DECAY_STEP_SIZE = 2048

# 算法通用参数
GAMMA = 0.99       # GRPO中，0.99 是底线，不要降低它
TAU = 0.001

# PPO/GRPO 特有参数
LAMDA = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 10
CRITIC_LOSS_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
GRPO_ENTROPY_COEF = 0.001   # GRPO 必须比 PPO 更低，否则它不敢确定策略
GROUP_SIZE = 32

# 训练时长
RAM_NUM_EPISODE = 50000
MAX_T = 300

# 可视化参数
VISUAL_NUM_EPISODE = 3000

# --- 全局随机种子设置函数 ---
def set_seed(seed):
    if seed is None: return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"✅ Global Seed set to: {seed}")