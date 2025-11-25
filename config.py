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
# [优化] 增大 Batch Size 以稳定梯度估计
NUM_PROCESSES = 30
BATCH_SIZE = 1024         # 原 512，建议 1024 或 2048
LEARNING_RATE = 0.0005    # 原 0.0003，配合新的 Reward Scale 稍微调大

# 衰减参数
DECAY_RATE = 0.999
DECAY_STEP_SIZE = 50000

# 算法通用参数
GAMMA = 0.99
TAU = 0.001

# PPO/GRPO 特有参数
LAMDA = 0.95
EPS_CLIP = 0.2           # 原 0.3，稍微保守一点
K_EPOCHS = 10            # 原 6，数据利用率稍微提高
CRITIC_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01      # 原 0.05，降低熵正则化，避免过于随机

# 训练时长
RAM_NUM_EPISODE = 50000  # 稍微增加总轮次
MAX_T = 1000

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