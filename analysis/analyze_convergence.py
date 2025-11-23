import numpy as np
import os

DATA_DIR = '../results'


def analyze(algo_folder, name):
    path = os.path.join(DATA_DIR, algo_folder)
    if not os.path.exists(path): return
    scores = []
    for seed in os.listdir(path):
        if seed.startswith('seed_'):
            f = os.path.join(path, seed, 'rewards.npy')
            if os.path.exists(f):
                data = np.load(f)
                scores.append(np.mean(data[-500:]))

    if scores:
        print(f"--- {name} ---")
        print(f"Final Reward (Mean ± Std): {np.mean(scores):.2f} ± {np.std(scores):.2f}")


if __name__ == "__main__":
    analyze('ppo', 'PPO')
    analyze('grpo_main', 'GRPO')
    analyze('grpo_no_dynamic', 'GRPO (Ablation)')