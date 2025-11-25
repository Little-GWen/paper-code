import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

RESULTS_DIR = '../results'


def get_data(algo_folder):
    path = os.path.join(RESULTS_DIR, algo_folder)
    if not os.path.exists(path):
        print(f"⚠️  文件夹未找到 (跳过): {path}")
        return None

    all_data = []
    min_len = float('inf')

    # 遍历所有种子文件夹 seed_0, seed_1 ...
    seed_folders = [d for d in os.listdir(path) if d.startswith('seed_')]

    if not seed_folders:
        print(f"⚠️  {algo_folder} 下没有找到 seed_X 文件夹")
        return None

    for seed_folder in seed_folders:
        file_path = os.path.join(path, seed_folder, 'rewards.npy')
        if os.path.exists(file_path):
            try:
                rewards = np.load(file_path)
                # 数据平滑 (Smoothing)
                rewards = pd.Series(rewards).rolling(window=50, min_periods=1).mean().values
                all_data.append(rewards)
                if len(rewards) < min_len: min_len = len(rewards)
            except:
                print(f"❌ 读取错误: {file_path}")

    if not all_data: return None

    # 截断到最小长度，保证矩阵整齐
    data_matrix = np.array([d[:min_len] for d in all_data])

    # 转换为 DataFrame
    df = pd.DataFrame(data_matrix).melt()
    df.columns = ['Episode', 'Reward']
    return df


def plot_all():
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # 定义要画的算法 (Label名称 : 文件夹名称)
    exps = {
        'GRPO (Ours)': 'grpo_main',  # 红色 (默认第一色)
        'PPO (Baseline)': 'ppo',  # 蓝色
        #'GRPO (No Dynamic Beta)': 'grpo_no_dynamic',  # 绿色/橙色 (消融)
        #'DQN (Baseline)': 'dqn'  # 紫色/其他
    }

    # 自动分配颜色，或者你可以手动指定 palette
    # palette = {'GRPO (Ours)': 'red', ...}

    plotted_count = 0
    for label, folder in exps.items():
        df = get_data(folder)
        if df is not None:
            sns.lineplot(data=df, x='Episode', y='Reward', label=label, linewidth=2.5)
            plotted_count += 1

    if plotted_count == 0:
        print("❌ 没有找到任何数据，请先运行 run_experiments.py")
        return

    plt.title("Training Performance Comparison (4-Lane High Density)", fontsize=15, fontweight='bold')
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.legend(fontsize=11, loc='lower right')
    plt.tight_layout()

    save_path = "learning_curve_full.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 图表已保存至 analysis/{save_path}")
    print("打开它看看你的论文核心结果吧！")
    plt.show()


if __name__ == "__main__":
    plot_all()