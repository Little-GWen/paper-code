import os
import subprocess
import time

# --- 实验配置 ---
# 为了发论文，建议跑 3 个种子以画出阴影图。
# 如果只是想快速测试代码是否报错，可以临时改为 [0]
SEEDS = [1,2,]

ALGORITHMS = [
    # 1. GRPO 主模型 (Ours)
    #("experiments/train_grpo.py", "GRPO (Ours)"),

    # 2. PPO 基线 (Baseline)
    ("experiments/train_ppo.py", "PPO (Baseline)"),

    # 3. DQN 基线 (Baseline)
    ("experiments/train_dqn.py", "DQN (Baseline)"),

    # 4. GRPO 消融实验 (Static_Beta)
    ("experiments/train_grpo_static_beta.py", "GRPO (Static-Beta)")
]


def run():
    print(f"🚀 开始全量实验! 总共 {len(ALGORITHMS)} 个算法, 每个跑 {len(SEEDS)} 个种子。")
    print("⚠️ 警告：全量运行时间较长，请确保电源连接，不要让电脑休眠。\n")

    start_all = time.time()

    for script_path, algo_name in ALGORITHMS:
        for seed in SEEDS:
            print(f"{'=' * 60}")
            print(f"▶️  正在运行: {algo_name} | Seed: {seed}")
            print(f"   Command: python {script_path} --seed {seed}")
            print(f"{'=' * 60}\n")

            algo_start = time.time()

            try:
                # check=True: 遇到报错立即停止，方便你修 Bug
                subprocess.run(["python", script_path, "--seed", str(seed)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n❌ 实验失败: {algo_name} (Seed {seed})")
                print(f"错误信息: {e}")
                # 这里选择 continue，即使 DQN 挂了，也不影响其他算法继续跑
                continue
            except KeyboardInterrupt:
                print("\n🛑 用户手动停止实验。")
                return

            duration = (time.time() - algo_start) / 60
            print(f"\n✅ {algo_name} (Seed {seed}) 完成! 耗时: {duration:.2f} 分钟\n")

    total_duration = (time.time() - start_all) / 3600
    print(f"🎉 所有实验已完成！总耗时: {total_duration:.2f} 小时。")
    print("请运行 analysis/plot_learning_curves.py 查看最终对比图。")


if __name__ == "__main__":
    run()