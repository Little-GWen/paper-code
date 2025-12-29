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
RAM_NUM_EPISODE = 80000
MAX_T = 60

# 训练参数
NUM_PROCESSES =30
BATCH_SIZE = 8192       # 采集 8192 步训练一次
MINI_BATCH_SIZE = 512   # 与 8192 采集量配合
LEARNING_RATE = 3e-4
GAMMA = 0.99            # GRPO中，0.99 是底线，不要降低它

# 衰减参数（线性衰减）
DECAY_MAX_STEP = RAM_NUM_EPISODE * MAX_T

# GRPO 特有参数
EPS_CLIP = 0.1          # 0.2 -> 0.1 配合强力的 Tiered Advantage 信号，防止策略更新过猛导致震荡
K_EPOCHS = 10
ENTROPY_COEF = 0.001    # 必须比 PPO 更低，否则它不敢确定策略
GROUP_SIZE = 32

# --- 全局随机种子设置函数 ---
def set_seed(seed):
    if seed is None: return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"✅ Global Seed set to: {seed}")