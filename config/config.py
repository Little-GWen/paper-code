import torch
import numpy as np
import random
import os

# 设备配置
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

# 环境名
RAM_ENV_NAME = 'my-merge-v0'

# 训练时长
RAM_NUM_EPISODE = 50000
MAX_T = 60

# 训练参数
NUM_PROCESSES = 30


# --- 废弃参数 ---
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
GAMMA = 0.985
DECAY_MAX_STEP = RAM_NUM_EPISODE * 40   # 衰减参数（线性衰减）：训练回合数 * 每回合最大时间步
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