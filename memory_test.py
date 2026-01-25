import matplotlib

# ⚠️ 必须放在 pyplot 导入之前！强制使用非交互式后端，防止服务器报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import seaborn as sns
import os
import gc
import sys

# ================= 配置区域 (服务器版) =================
OUTPUT_DIR = 'analysis/memory_plots'
CSV_NAME = 'memory_benchmark_data.csv'

# 1. Batch Size 扩展性测试 (固定模型大小 Hidden=256)
# 范围：从 2^8 (256) 到 2^24 (约1677万)
# 目的：测出 PPO 在哪里崩，GRPO 能撑多远
BATCH_SIZES = [2 ** i for i in range(8, 25)]
FIXED_HIDDEN_FOR_BS = 256

# 2. 模型参数扩展性测试 (固定 Batch Size=2048)
HIDDEN_DIMS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
FIXED_BS_FOR_MODEL = 2048

# 绘图风格
plt.rcParams.update({
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

PALETTE = {'GRPO': '#2ca02c', 'PPO': '#ff7f0e', 'DQN': '#1f77b4'}
MARKERS = {'GRPO': '^', 'PPO': 'o', 'DQN': 's'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= 模拟网络结构 =================
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ================= 核心测试函数 =================
def run_single_test(algo, batch_size, hidden_dim):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    try:
        input_dim = 60
        action_dim = 5
        x = torch.randn(batch_size, input_dim).to(DEVICE)

        if algo == 'PPO':
            actor = SimpleNet(input_dim, hidden_dim, action_dim).to(DEVICE)
            critic = SimpleNet(input_dim, hidden_dim, 1).to(DEVICE)
            opt_actor = optim.Adam(actor.parameters(), lr=1e-4)
            opt_critic = optim.Adam(critic.parameters(), lr=1e-4)

            loss = actor(x).sum() + critic(x).sum()
            loss.backward()
            opt_actor.step()
            opt_critic.step()

        elif algo == 'DQN':
            q_net = SimpleNet(input_dim, hidden_dim, action_dim).to(DEVICE)
            target_net = SimpleNet(input_dim, hidden_dim, action_dim).to(DEVICE)
            opt = optim.Adam(q_net.parameters(), lr=1e-4)

            loss = q_net(x).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                _ = target_net(x)

        elif algo == 'GRPO':
            actor = SimpleNet(input_dim, hidden_dim, action_dim).to(DEVICE)
            opt = optim.Adam(actor.parameters(), lr=1e-4)

            loss = actor(x).sum()
            loss.backward()
            opt.step()

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        else:
            peak_mem = 0
        return peak_mem

    except RuntimeError as e:
        if 'out of memory' in str(e):
            return None
        else:
            raise e


# ================= 主流程 =================
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🚀 Server Benchmark Started on: {gpu_name}")

    results = []

    # --- 1. Batch Size Test ---
    print("\n[1/2] Batch Size Scalability (Log Scale)...")
    for algo in ['PPO', 'DQN', 'GRPO']:
        print(f"  > {algo}: ", end="")
        for bs in BATCH_SIZES:
            mem = run_single_test(algo, bs, FIXED_HIDDEN_FOR_BS)
            if mem is not None:
                results.append({'Type': 'Batch', 'Algo': algo, 'X': bs, 'Memory': mem})
                # print(".", end="", flush=True)
            else:
                print(f"OOM@{bs}", end="")
                break
        print("")

    # --- 2. Model Size Test ---
    print("\n[2/2] Model Parameter Scalability...")
    for algo in ['PPO', 'DQN', 'GRPO']:
        print(f"  > {algo}: ", end="")
        for hd in HIDDEN_DIMS:
            mem = run_single_test(algo, FIXED_BS_FOR_MODEL, hd)
            if mem is not None:
                results.append({'Type': 'Model', 'Algo': algo, 'X': hd, 'Memory': mem})
                # print(".", end="", flush=True)
            else:
                print(f"OOM@{hd}", end="")
                break
        print("")

    # 保存数据
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, CSV_NAME), index=False)

    # --- 绘图 ---
    print("\n🎨 Generating Plots...")

    # 图1
    plt.figure(figsize=(10, 6))
    df_batch = df[df['Type'] == 'Batch']
    sns.lineplot(data=df_batch, x='X', y='Memory', hue='Algo', style='Algo',
                 palette=PALETTE, markers=MARKERS, dashes=False, linewidth=2.5)
    plt.xscale('log', base=2)
    plt.xlabel('Batch Size (Log Scale)', fontsize=14)
    plt.ylabel('Peak GPU Memory (MB)', fontsize=14)
    plt.title(f'Batch Size Scalability ({gpu_name})', fontsize=16, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'server_batch_scaling.png'), dpi=300)

    # 图2
    plt.figure(figsize=(10, 6))
    df_model = df[df['Type'] == 'Model']
    sns.lineplot(data=df_model, x='X', y='Memory', hue='Algo', style='Algo',
                 palette=PALETTE, markers=MARKERS, dashes=False, linewidth=2.5)
    plt.xlabel('Hidden Dimension (Model Size)', fontsize=14)
    plt.ylabel('Peak GPU Memory (MB)', fontsize=14)
    plt.title(f'Model Size Scalability ({gpu_name})', fontsize=16, fontweight='bold')
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'server_model_scaling.png'), dpi=300)

    print(f"✅ All Done! Check {OUTPUT_DIR}")


if __name__ == "__main__":
    main()