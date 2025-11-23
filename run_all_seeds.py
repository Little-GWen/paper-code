import os
import subprocess
import time

SEEDS = [0, 1, 2]
ALGORITHMS = [
    # 建议先跑这俩
    ("experiments/train_grpo.py", "GRPO (Ours)"),
    ("experiments/train_ppo.py", "PPO (Baseline)"),
    # ("experiments/train_dqn.py", "DQN (Baseline)"),
    # ("experiments/train_grpo_ablation.py", "GRPO (Ablation)")
]

def run():
    print(f"🚀 开始批量实验!")
    for script_path, algo_name in ALGORITHMS:
        for seed in SEEDS:
            print(f"\n{'='*50}\n▶️  运行: {algo_name} | Seed: {seed}\n{'='*50}")
            try:
                # check=True 会在报错时停止，方便你发现问题
                subprocess.run(["python", script_path, "--seed", str(seed)], check=True)
            except subprocess.CalledProcessError:
                print(f"❌ 失败: {algo_name} (Seed {seed})")
                continue
            except KeyboardInterrupt:
                return

if __name__ == "__main__":
    run()