import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ================= 配置 =================
# 测试结果的根目录
TEST_RESULT_DIR = '../results/test'
# 图片/表格保存目录
OUTPUT_DIR = '../analysis/report'

# 想要统计的指标文件名
METRICS_FILES = {
    'Reward': 'rewards.npy',
    'Speed': 'speed.npy',
    'Collision': 'collision.npy'
}

# 绘图风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'figure.figsize': (12, 6),
    'font.size': 12
})


def get_algo_display_name(folder_name):
    """美化算法名称用于显示"""
    name = folder_name.replace('_', ' ').upper()
    # 特殊处理 GRPO 的 Group Size 显示
    if 'GRPO' in name:
        # 尝试提取数字
        match = re.search(r'(\d+)', name)
        if match:
            return f"GRPO (G={match.group(1)})"
        if 'MAIN' in name:
            return "GRPO (Main)"
        if 'STANDARD' in name:
            return "GRPO (Standard)"
    return name


def load_and_aggregate():
    """核心数据加载逻辑"""
    if not os.path.exists(TEST_RESULT_DIR):
        print(f"❌ 目录不存在: {TEST_RESULT_DIR}")
        return []

    summary_data = []
    raw_data_list = []

    # 1. 遍历算法文件夹
    algo_folders = sorted(os.listdir(TEST_RESULT_DIR))

    for algo_folder in algo_folders:
        algo_path = os.path.join(TEST_RESULT_DIR, algo_folder)
        if not os.path.isdir(algo_path): continue

        display_name = get_algo_display_name(algo_folder)

        # 临时存储该算法所有种子的均值
        seed_means = {'Reward': [], 'Speed': [], 'Success Rate': []}

        # 2. 遍历种子文件夹
        seeds = [s for s in os.listdir(algo_path) if s.startswith('seed_')]

        for seed in seeds:
            seed_path = os.path.join(algo_path, seed)

            try:
                # 读取 Reward
                rew = np.load(os.path.join(seed_path, METRICS_FILES['Reward']))
                # 读取 Speed
                spd = np.load(os.path.join(seed_path, METRICS_FILES['Speed']))
                # 读取 Collision
                col = np.load(os.path.join(seed_path, METRICS_FILES['Collision']))

                # --- 计算单一种子的统计值 ---
                # 成功率 = (1 - 碰撞率) * 100
                success_rate = (1.0 - np.mean(col)) * 100
                avg_rew = np.mean(rew)
                avg_spd = np.mean(spd)

                # 存入列表用于计算算法总均值
                seed_means['Reward'].append(avg_rew)
                seed_means['Speed'].append(avg_spd)
                seed_means['Success Rate'].append(success_rate)

                # 存入原始数据用于画图 (Seaborn 会自动算误差)
                # 假设每个种子测了 N 轮，我们把 N 轮的均值作为一个数据点
                # 或者如果你想画箱线图，可以存入每一轮的数据
                raw_data_list.append({
                    'Algorithm': display_name,
                    'Reward': avg_rew,
                    'Speed': avg_spd,
                    'Success Rate': success_rate,
                    'Seed': seed
                })

            except Exception as e:
                print(f"⚠️  跳过 {algo_folder}/{seed}: 文件缺失或损坏 ({e})")
                continue

        # --- 计算该算法所有种子的聚合统计 (Mean ± Std) ---
        if seed_means['Reward']:
            n = len(seed_means['Reward'])
            entry = {'Algorithm': display_name, 'Seeds': n}

            for metric in ['Reward', 'Speed', 'Success Rate']:
                m = np.mean(seed_means[metric])
                s = np.std(seed_means[metric])
                entry[f'{metric} Mean'] = m
                entry[f'{metric} Std'] = s
                # 格式化字符串用于表格展示
                entry[metric] = f"{m:.2f} ± {s:.2f}"

            summary_data.append(entry)

    return pd.DataFrame(summary_data), pd.DataFrame(raw_data_list)


def plot_bar_charts(raw_df):
    """绘制带误差线的柱状图"""
    if raw_df.empty: return

    metrics = ['Reward', 'Success Rate', 'Speed']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 定义颜色调色板
    palette = sns.color_palette("muted")

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # 使用 Barplot，Seaborn 会自动计算置信区间或标准差
        sns.barplot(
            data=raw_df,
            x='Algorithm',
            y=metric,
            ax=ax,
            palette=palette,
            capsize=.1,
            errorbar='sd'  # 显示标准差
        )

        ax.set_title(f"Average {metric}", fontweight='bold', fontsize=14)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # 特殊处理：成功率上限设为 100+
        if metric == 'Success Rate':
            ax.set_ylim(0, 105)
            ax.set_ylabel("Success Rate (%)")
        elif metric == 'Speed':
            ax.set_ylabel("Speed (m/s)")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'benchmark_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"📊 图表已保存: {save_path}")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("🔄 正在扫描并计算测试结果...")

    # 1. 获取数据
    summary_df, raw_df = load_and_aggregate()

    if summary_df.empty:
        print("❌ 未找到有效数据，请确保已经运行了 run_batch_test.py")
        return

    # 2. 整理表格 (按名称排序，让 GRPO 排在一起)
    # 简单的排序逻辑：先把名称排序
    summary_df.sort_values('Algorithm', inplace=True)

    # 提取用于显示的列
    display_table = summary_df[['Algorithm', 'Reward', 'Success Rate', 'Speed', 'Seeds']]

    # 3. 输出 Markdown 表格到控制台
    print("\n" + "=" * 80)
    print("🏆 FINAL TEST RESULTS SUMMARY (Mean ± Std over Seeds)")
    print("=" * 80)
    print(display_table.to_markdown(index=False))
    print("=" * 80)

    # 4. 保存 CSV
    csv_path = os.path.join(OUTPUT_DIR, 'test_statistics.csv')
    display_table.to_csv(csv_path, index=False)
    print(f"💾 表格已保存: {csv_path}")

    # 5. 绘图
    plot_bar_charts(raw_df)


if __name__ == "__main__":
    main()