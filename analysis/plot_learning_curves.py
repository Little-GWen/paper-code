
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# ================= 1. 路径与核心配置 =================
RESULTS_DIR = '../results'
SAVE_DIR = '../analysis'
SAMPLE_RATE = 100  # 采样率：每100个点取1个，数据点多了，采样可以稀疏一点

# ⚠️ 关键设置：这必须和你训练时的 TOTAL_TRAIN_STEPS 一致
TOTAL_TRAIN_STEPS = 3000000

experiments_config = {
    'GRPO (Ours)': {
        'folder': 'grpo_main',
        'color': '#1f77b4',
        'reward_fn': 'rew.npy'  # GRPO 保存的是 rew.npy
    },
    'PPO (Baseline)': {
        'folder': 'ppo',
        'color': '#ff7f0e',
        'reward_fn': 'rewards.npy'
    },
    'DQN (Baseline)': {
        'folder': 'dqn',
        'color': '#d62728',
        'reward_fn': 'rewards.npy'
    },
    # 如果有 Standard 对比
    'GRPO (Standard)': {
        'folder': 'grpo_standard',
        'color': '#2ca02c',
        'reward_fn': 'rew.npy'
    },
}

# 论文级绘图风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3
})


# ================= 2. 数据处理引擎 (自动对齐 X 轴) =================

def load_algorithm_data(label, config, metric_type):
    folder = config['folder']
    rew_file = config['reward_fn']

    path = os.path.join(RESULTS_DIR, 'train', folder)
    if not os.path.exists(path):
        print(f"❌ 路径不存在: {path}")
        return None

    all_dfs = []
    seed_folders = [d for d in os.listdir(path) if d.startswith('seed_')]

    print(f"读取 {label}...", end=" ")

    for seed in seed_folders:
        sp = os.path.join(path, seed)
        try:
            # --- 1. 读取数据 ---
            if metric_type == 'reward':
                target_path = os.path.join(sp, rew_file)
                if not os.path.exists(target_path): continue
                data = np.load(target_path)

            elif metric_type == 'safety':
                col_path = os.path.join(sp, 'collision.npy')  # 优先找 collision
                if os.path.exists(col_path):
                    # collision: 1=撞, 0=安全 -> safety: 0=撞, 1=安全
                    # 也可以做滑动平均安全率
                    raw_col = np.load(col_path)
                    data = 1.0 - raw_col
                else:
                    # 找不到 collision 文件则跳过
                    continue

            elif metric_type == 'speed':
                speed_path = os.path.join(sp, 'speed.npy')
                if not os.path.exists(speed_path): continue
                # 假设环境归一化了，这里简单还原 (视你的环境而定)
                # 如果你的 speed.npy 已经是真实速度，就不需要 * 40
                data = np.load(speed_path)  # * 40

            else:
                return None

            # --- 2. 数据平滑 ---
            # 窗口大小根据数据量动态调整，保证曲线平滑
            # PPO 数据点多，窗口大一点；GRPO 数据点少，窗口小一点
            win_size = max(int(len(data) * 0.01), 10)
            v_smooth = pd.Series(data).rolling(window=win_size, min_periods=1).mean().values

            # --- 3. 下采样 (防止画图太卡) ---
            # 我们不需要画几十万个点，只需要画几千个点
            target_plot_points = 1000
            if len(v_smooth) > target_plot_points:
                indices = np.linspace(0, len(v_smooth) - 1, target_plot_points).astype(int)
                v_down = v_smooth[indices]
            else:
                v_down = v_smooth

            # --- 4. 关键修正：X 轴强行拉伸对齐 ---
            # 无论原来有多少个点，我们都认为最后一个点对应 TOTAL_TRAIN_STEPS
            steps = np.linspace(0, TOTAL_TRAIN_STEPS, len(v_down))

            all_dfs.append(pd.DataFrame({'Steps': steps, 'Value': v_down, 'Algorithm': label}))

        except Exception as e:
            print(f"Error {seed}: {e}")
            continue

    if not all_dfs:
        print("无数据。")
        return None

    df_final = pd.concat(all_dfs, ignore_index=True)
    print(f"成功! (Samples: {len(df_final)})")
    return df_final


# ================= 3. 绘图与保存 =================

def run_plot(metric_type, title, ylabel, filename):
    plt.figure()
    sns.set_style("whitegrid", {'grid.linestyle': '--'})

    # 调色板
    palette = {k: v['color'] for k, v in experiments_config.items()}

    plot_data = []
    for label, config in experiments_config.items():
        df = load_algorithm_data(label, config, metric_type)
        if df is not None:
            plot_data.append(df)

    if not plot_data:
        print(f"⚠️ {metric_type} 没有数据可画")
        return

    final_df = pd.concat(plot_data, ignore_index=True)

    # 绘制带置信区间的曲线
    sns.lineplot(
        data=final_df, x='Steps', y='Value', hue='Algorithm',
        palette=palette, linewidth=2.0, errorbar='sd', alpha=0.8
    )

    plt.title(title, fontweight='bold', fontsize=14, pad=15)
    plt.xlabel("Total Environment Steps", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # 科学计数法显示 X 轴
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # 设置 X 轴范围
    plt.xlim(0, TOTAL_TRAIN_STEPS)

    plt.legend(loc='lower right', frameon=True, framealpha=0.9)
    plt.tight_layout()

    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"🎉 图片已保存: {save_path}\n")
    plt.close()


if __name__ == "__main__":
    print(f"🚀 开始绘图 (Target Steps: {TOTAL_TRAIN_STEPS})...\n")

    # 1. 奖励曲线
    run_plot('reward', 'Training Reward Convergence', 'Average Reward', 'train_reward.png')

    # 2. 安全率曲线 (Survival Rate)
    # 这里的 safety 是 0~1 的值
    run_plot('safety', 'Safety Rate during Training', 'Safety Rate (0-1)', 'train_safety.png')

    # 3. 速度曲线
    run_plot('speed', 'Average Speed during Training', 'Speed', 'train_speed.png')