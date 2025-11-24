import torch
import numpy as np
import random
import os

# 设备配置
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

# 环境名
RAM_ENV_NAME = 'highway-v0'

# 训练参数
NUM_PROCESSES = 30        # 多进程数量
BATCH_SIZE = 512
LEARNING_RATE = 0.0003

# 衰减参数
DECAY_RATE = 0.999       # 学习率衰减率
DECAY_STEP_SIZE = 50000  # 衰减步长 (Steps)

# 算法通用参数
GAMMA = 0.99
TAU = 0.001

# PPO/GRPO 特有参数
LAMDA = 0.95
EPS_CLIP = 0.3
K_EPOCHS = 6
CRITIC_LOSS_COEF = 0.5
ENTROPY_COEF = 0.05

# 训练时长
RAM_NUM_EPISODE = 8000
MAX_T = 1000             # 单回合最大步数 (需 >= custom_env.py 中的 duration)

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