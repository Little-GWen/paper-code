
import torch
import numpy as np
import random
import os

# 设备配置
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

# 环境名
RAM_ENV_NAME = 'my-merge-v0'

# --- 训练时长控制 (Unified Training Steps) ---
# 统一设定总步数，替代 Episode 控制
# 3,000,000 步 ≈ 50,000 Episodes * 60 Steps
TOTAL_TRAIN_STEPS = 5000000

# 原有参数 (保留，但不再作为主循环控制)
RAM_NUM_EPISODE = 50000
MAX_T = 50

# 训练参数
NUM_PROCESSES = 30

# --- PPO/DQN 参数 ---
BATCH_SIZE = 8192
LEARNING_RATE = 1e-4
GAMMA = 0.985
# 衰减参数：对齐总步数
DECAY_MAX_STEP = TOTAL_TRAIN_STEPS
LAMDA = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 10
CRITIC_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01

# --- 全局随机种子设置函数 ---
def set_seed(seed):
    if seed is None: return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"✅ Global Seed set to: {seed}")