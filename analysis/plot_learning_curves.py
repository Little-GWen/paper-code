import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

RESULTS_DIR = '../results'
SAVE_DIR = '../analysis'

# 定义实验配置
experiments_config = {
    # GRPO 的GROUP_SIZE 是 32
    'GRPO (Ours)': {'folder': 'grpo_main', 'step_scale': 32, 'color': '#1f77b4'},
    'PPO (Baseline)': {'folder': 'ppo', 'step_scale': 1, 'color': '#ff7f0e'},
    'DQN (Baseline)': {'folder': 'dqn', 'step_scale': 1, 'color': '#d62728'},
    'GRPO (Static-Beta)': {'folder': 'ppo_static_beta', 'step_scale': 32, 'color': '#2ca03d'},
    #'GRPO (Standard)': {'folder': 'grpo_standard', 'step_scale': 32, 'color': '#2ca02c'},
}


def load_data_robust(label, config, metric_type):
    folder = config['folder']
    step_scale = config['step_scale']
    path = os.path.join(RESULTS_DIR, 'train', folder)

    # 检查文件夹是否存在，不存在则跳过而不报错
    if not os.path.exists(path):
        return None

    all_dfs = []
    # 读取所有 seed_ 开头的文件夹
    seed_folders = [d for d in os.listdir(path) if d.startswith('seed_')]

    for seed_folder in seed_folders:
        try:
            reward_path = os.path.join(path, seed_folder, 'rewards.npy')
            collision_path = os.path.join(path, seed_folder, 'collision.npy')
            speed_path = os.path.join(path, seed_folder, 'speed.npy')

            # 处理速度数据
            if metric_type == 'speed':
                if os.path.exists(speed_path):
                    data = np.load(speed_path)

                    # 反归一化，还原真实速度
                    data *= 40

                    # 平滑处理
                    data_smooth = pd.Series(data).rolling(window=50, min_periods=1).mean().values
                    steps = np.arange(len(data_smooth)) * step_scale
                    all_dfs.append(pd.DataFrame({'Episode': steps, 'Value': data_smooth, 'Algorithm': label}))
                continue

            # 处理奖励与碰撞
            if not os.path.exists(reward_path): continue
            rewards = np.load(reward_path)
            collisions = None
            if os.path.exists(collision_path):
                try:
                    loaded = np.load(collision_path)
                    if len(loaded) > 0: collisions = loaded
                except:
                    pass
            # 如果没有 collision 文件，尝试从 reward 推断 (后备方案)
            if collisions is None:
                collisions = np.where(rewards < -50.0, 1.0, 0.0)  # 修正判定阈值
            # 对齐数据长度
            min_len = min(len(rewards), len(collisions))
            rewards = rewards[:min_len]
            collisions = collisions[:min_len]

            win_size = 100

            # 根据指标类型计算数据
            if metric_type == 'safety_rate':
                final_data = 1.0 - collisions  # 0=Crash -> 1=Safe
                win_size = 150

            elif metric_type == 'weighted_reward':
                is_safe = 1.0 - collisions
                final_data = rewards * is_safe

            elif metric_type == 'reward':
                final_data = rewards

            # 平滑处理
            data_smooth = pd.Series(final_data).rolling(window=win_size, min_periods=1).mean().values
            steps = np.arange(len(data_smooth)) * step_scale

            all_dfs.append(pd.DataFrame({
                'Episode': steps,
                'Value': data_smooth,
                'Algorithm': label
            }))

        except Exception as e:
            print(f"⚠️ 跳过 {seed_folder}: {e}")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None


def plot_paper_graph(metric_type, title, ylabel, filename, y_limit=None):
    plt.figure(figsize=(8, 5))
    # 设置 Seaborn 风格
    sns.set_theme(style="whitegrid", font_scale=1.1)

    all_data = []
    # 遍历配置加载数据
    for label, config in experiments_config.items():
        df = load_data_robust(label, config, metric_type)
        if df is not None:
            all_data.append(df)

    if not all_data:
        print(f"❌ 无法绘制 {metric_type}")
        return

    final_df = pd.concat(all_data, ignore_index=True)

    # 动态生成颜色调色板，只包含存在的算法
    unique_algos = final_df['Algorithm'].unique()
    palette = {k: v['color'] for k, v in experiments_config.items() if k in unique_algos}

    # 绘图
    sns.lineplot(
        data=final_df, x='Episode', y='Value', hue='Algorithm',
        palette=palette, linewidth=2.5, alpha=0.9
    )

    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Training Episodes (Equivalent)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if y_limit:
        plt.ylim(y_limit)

    # 优化图例
    plt.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")


if __name__ == "__main__":
    print("🎨 Generating Plots...")
    # 1. 安全率对比
    plot_paper_graph('safety_rate', 'Safety Performance (Survival Rate)', 'Safety Rate', 'train_safety.png', (0.0, 1.05))
    # 2. 有效奖励对比 (撞车=0分)
    plot_paper_graph('weighted_reward', 'Effective Performance', 'Reward (Zero on Crash)', 'train_effective.png')
    # 3. 速度对比
    plot_paper_graph('speed', 'Driving Efficiency', 'Speed (km/h)', 'train_speed.png')
    # 4. 原始奖励对比
    plot_paper_graph('reward', 'Raw Training Reward', 'Average Reward', 'train_raw_reward.png')