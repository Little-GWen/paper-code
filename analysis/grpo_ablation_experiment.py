import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import re

# ================= 1. 配置参数 =================
RESULTS_DIR = '../results/train'  # 数据根目录
SAVE_DIR = '../analysis/ablation'  # 图片保存目录
TOTAL_TRAIN_STEPS = 5000000  # 你的统一训练步数
SAMPLE_RATE = 100  # 下采样率，防止点太多画图卡顿

# 绘图风格设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'figure.figsize': (8, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12
})

# 定义颜色 (不同 Group Size 使用渐变色或对比色)
# 如果你有更多组，可以继续添加
PALETTE = {
    16: '#8c564b',  # 棕色
    32: '#1f77b4',  # 蓝色 (基准)
    64: '#ff7f0e',  # 橙色
    96: '#9467bd',  # 紫色
    128: '#2ca02c',  # 绿色
    256: '#d62728'  # 红色
}


# ================= 2. 数据加载逻辑 =================

def load_ablation_data(metric_type):
    """
    扫描文件夹，加载不同 Group Size 的数据
    metric_type: 'reward', 'safety', 'speed'
    """
    if not os.path.exists(RESULTS_DIR):
        print(f"❌ 路径不存在: {RESULTS_DIR}")
        return None

    all_dfs = []

    # 遍历 results/train 下的所有文件夹
    for folder_name in os.listdir(RESULTS_DIR):
        # 1. 正则匹配：查找类似 grpo_16, grpo_32, G32, GroupSize64 等命名
        # 只要文件夹名字里包含数字，且包含 grpo，我们就认为它是消融实验对象
        if not ('grpo' in folder_name.lower()):
            continue

        # 提取数字部分作为 Group Size
        match = re.search(r'(\d+)', folder_name)
        if not match:
            continue

        group_size = int(match.group(1))
        algo_label = f"G={group_size}"  # 图例显示的名称

        folder_path = os.path.join(RESULTS_DIR, folder_name)
        seed_folders = [d for d in os.listdir(folder_path) if d.startswith('seed_')]

        print(f"处理 Group Size: {group_size} ({len(seed_folders)} seeds)...")

        for seed in seed_folders:
            seed_path = os.path.join(folder_path, seed)

            try:
                # --- 根据类型读取不同文件 ---
                # 兼容 GRPO (rewards.npy) 和 PPO (rewards.npy) 的命名习惯
                if metric_type == 'reward':
                    if os.path.exists(os.path.join(seed_path, 'rewards.npy')):
                        data = np.load(os.path.join(seed_path, 'rewards.npy'))
                    elif os.path.exists(os.path.join(seed_path, 'rewards.npy')):
                        data = np.load(os.path.join(seed_path, 'rewards.npy'))
                    else:
                        continue

                elif metric_type == 'safety':
                    # 读取 collision (0/1)，转为 safety (1/0)
                    if os.path.exists(os.path.join(seed_path, 'collision.npy')):
                        col_data = np.load(os.path.join(seed_path, 'collision.npy'))
                    elif os.path.exists(os.path.join(seed_path, 'collision.npy')):
                        col_data = np.load(os.path.join(seed_path, 'collision.npy'))
                    else:
                        continue
                    # 平滑一下再算安全率，视觉效果更好
                    data = 1.0 - col_data

                elif metric_type == 'speed':
                    if os.path.exists(os.path.join(seed_path, 'speed.npy')):
                        data = np.load(os.path.join(seed_path, 'speed.npy'))
                    elif os.path.exists(os.path.join(seed_path, 'speed.npy')):
                        data = np.load(os.path.join(seed_path, 'speed.npy'))
                    else:
                        continue

                else:
                    continue

                # --- 数据对齐与平滑 ---
                # 1. 简单平滑 (Rolling Mean)
                win_size = max(int(len(data) * 0.02), 5)  # 动态窗口
                data_smooth = pd.Series(data).rolling(window=win_size, min_periods=1).mean().values

                # 2. 下采样 (Downsampling)
                # 无论原始数据多少个点，都插值到 1000 个点，方便画图
                target_points = 1000
                indices = np.linspace(0, len(data_smooth) - 1, target_points).astype(int)
                data_down = data_smooth[indices]

                # 3. 生成 X 轴 (Environment Steps)
                steps = np.linspace(0, TOTAL_TRAIN_STEPS, target_points)

                all_dfs.append(pd.DataFrame({
                    'Steps': steps,
                    'Value': data_down,
                    'Group Size': group_size,  # 用于排序和颜色映射
                    'Label': algo_label
                }))

            except Exception as e:
                print(f"  ⚠️ 读取错误 {seed_path}: {e}")
                continue

    if not all_dfs:
        return None

    return pd.concat(all_dfs, ignore_index=True)


# ================= 3. 绘图函数 =================

def plot_metric(metric_type, title, ylabel, filename):
    df = load_ablation_data(metric_type)
    if df is None:
        print(f"⚠️ {metric_type} 没有找到有效数据，跳过。")
        return

    plt.figure()
    sns.set_style("whitegrid", {'grid.linestyle': '--'})

    # 获取存在的 Group Size 并排序
    unique_groups = sorted(df['Group Size'].unique())

    # 动态生成颜色板 (如果在 PALETTE 里有定义就用定义的，没有就用 Seaborn 默认)
    custom_palette = {}
    for g in unique_groups:
        custom_palette[f"G={g}"] = PALETTE.get(g, None)  # None 会让 sns 自动分配

    # 绘图
    sns.lineplot(
        data=df, x='Steps', y='Value', hue='Label',
        # 按照 Group Size 数字大小排序图例
        hue_order=[f"G={g}" for g in unique_groups],
        palette=custom_palette,
        linewidth=2.5, errorbar='sd', alpha=0.85
    )

    plt.title(title, fontweight='bold', fontsize=14, pad=15)
    plt.xlabel("Total Environment Steps", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # X轴科学计数法
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlim(0, TOTAL_TRAIN_STEPS)

    # 图例优化
    plt.legend(title="Group Size", loc='best', frameon=True, framealpha=0.9, fancybox=True)
    plt.tight_layout()

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 图片保存成功: {save_path}")
    plt.close()


if __name__ == "__main__":
    print(f"🚀 开始绘制消融实验图表 (Group Size Ablation)...")
    print(f"📂 扫描目录: {RESULTS_DIR}")
    print("-" * 60)

    # 1. 奖励曲线
    plot_metric('reward',
                'Effect of Group Size on Average Reward',
                'Average Reward',
                'ablation_reward.png')

    # 2. 安全率曲线
    plot_metric('safety',
                'Effect of Group Size on Safety Rate',
                'Safety Rate (0-1)',
                'ablation_safety.png')

    # 3. 速度曲线
    plot_metric('speed',
                'Effect of Group Size on Driving Efficiency',
                'Average Speed',
                'ablation_speed.png')

    print("-" * 60)
    print("✅ 所有绘图任务完成！请查看 results/analysis/ablation 文件夹。")