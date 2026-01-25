import sys
import os
import numpy as np
import torch
import gymnasium as gym
import ray

# --- 1. 路径与环境初始化 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入你的配置
from config.grpo_config import (
    RAM_ENV_NAME, MAX_T, DEVICE, BATCH_SIZE, LEARNING_RATE,
    DECAY_MAX_STEP, GAMMA, EPS_CLIP, K_EPOCHS, ENTROPY_COEF
)
from models.agent_grpo_ray import Agent_GRPO_Ray
import envs.custom_merge_env

# 实验配置
ALGORITHMS = ['grpo_256']
SEEDS = [0, 1, 2]
TOTAL_TEST_EPISODES = 10000
NUM_CPUS_PER_TASK = 6  # 每个模型开启多少个并行进程

TRAIN_DIR_BASE = 'results/train'
TEST_DIR_BASE = 'results/test'


# --- 2. 分布式评估 Worker ---
@ray.remote
class EvalWorker:
    def __init__(self, algo, weight_file, env_name):
        # 每个进程独立导入环境
        import envs.custom_merge_env
        self.env = gym.make(env_name)
        self.algo = algo

        state_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = self.env.action_space.n

        # --- 精确复用你原始代码的加载逻辑 ---
        try:
            if 'grpo' in algo:
                self.agent = Agent_GRPO_Ray(
                    state_size=state_dim, action_size=act_dim,
                    bs=BATCH_SIZE, lr=LEARNING_RATE, decay_max_step=DECAY_MAX_STEP,
                    gamma=GAMMA, eps_clip=EPS_CLIP, K_epochs=K_EPOCHS,
                    entropy_coef=ENTROPY_COEF, device='cpu', is_worker=False
                )
                checkpoint = torch.load(weight_file, map_location='cpu')
                sd = checkpoint['actor'] if (isinstance(checkpoint, dict) and 'actor' in checkpoint) else checkpoint
                self.agent.actor.load_state_dict(sd)
                self.agent.actor.eval()

            elif algo == 'ppo':
                from stable_baselines3 import PPO
                policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[512, 512], vf=[512, 512]))
                model = PPO("MlpPolicy", self.env, policy_kwargs=policy_kwargs, device='cpu')
                model.policy.load_state_dict(torch.load(weight_file, map_location='cpu'))
                self.agent = model

            elif algo == 'dqn':
                from stable_baselines3 import DQN
                policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512, 512])
                model = DQN("MlpPolicy", self.env, policy_kwargs=policy_kwargs, device='cpu')
                model.policy.load_state_dict(torch.load(weight_file, map_location='cpu'))
                self.agent = model
        except Exception as e:
            print(f"Worker Error loading {algo}: {e}")

    def work(self, episodes):
        rews, spds, cols = [], [], []
        for _ in range(episodes):
            state, _ = self.env.reset()
            if isinstance(state, tuple): state = state[0]
            state = np.nan_to_num(state, nan=0.0)

            ep_rew, ep_spd_list, crashed = 0, [], False

            for _ in range(MAX_T):
                # 动作选择逻辑
                if 'grpo' in self.algo:
                    with torch.no_grad():
                        action, _ = self.agent.act(state.flatten(), deterministic=True)
                else:
                    # SB3 预测
                    action, _ = self.agent.predict(state, deterministic=True)
                    action = action.item()

                state, reward, term, trunc, info = self.env.step(action)
                state = np.nan_to_num(state, nan=0.0)
                ep_rew += reward

                # 速度记录
                spd = info.get('speed', 0)
                if hasattr(self.env.unwrapped, 'vehicle'):
                    spd = self.env.unwrapped.vehicle.speed
                ep_spd_list.append(spd)

                if term or trunc:
                    if info.get('crashed') or (
                            hasattr(self.env.unwrapped, 'vehicle') and self.env.unwrapped.vehicle.crashed):
                        crashed = True
                    break

            rews.append(ep_rew)
            spds.append(np.mean(ep_spd_list) if ep_spd_list else 0)
            cols.append(1.0 if crashed else 0.0)
        return rews, spds, cols


# --- 3. 主调度程序 ---
def main():
    ray.init(ignore_reinit_error=True)
    print(f"🚀 分布式批量评估启动 | 目标环境: {RAM_ENV_NAME}")

    for algo in ALGORITHMS:
        for seed in SEEDS:
            source_path = os.path.join(TRAIN_DIR_BASE, algo, f"seed_{seed}")
            weight_file = os.path.join(source_path, 'weights.pth')
            target_path = os.path.join(TEST_DIR_BASE, algo, f"seed_{seed}")

            if not os.path.exists(weight_file):
                print(f"⏩ 缺失权重，跳过: {algo} seed {seed}")
                continue

            if os.path.exists(os.path.join(target_path, 'rewards.npy')):
                print(f"✅ 已存在结果，跳过: {algo} seed {seed}")
                continue

            print(f"▶️  并行测试中: [{algo}] Seed {seed}...")
            os.makedirs(target_path, exist_ok=True)

            # 分配任务到多个并行 Worker
            eps_per_worker = TOTAL_TEST_EPISODES // NUM_CPUS_PER_TASK
            workers = [EvalWorker.remote(algo, weight_file, RAM_ENV_NAME) for _ in range(NUM_CPUS_PER_TASK)]

            # 并行执行
            results = ray.get([w.work.remote(eps_per_worker) for w in workers])

            # 汇总结果
            all_r, all_s, all_c = [], [], []
            for r, s, c in results:
                all_r.extend(r);
                all_s.extend(s);
                all_c.extend(c)

            # 保存
            np.save(os.path.join(target_path, 'rewards.npy'), np.array(all_r))
            np.save(os.path.join(target_path, 'speed.npy'), np.array(all_s))
            np.save(os.path.join(target_path, 'collision.npy'), np.array(all_c))

            print(f"    📊 Avg Reward: {np.mean(all_r):.2f} | Crash Rate: {np.mean(all_c) * 100:.1f}%")

    ray.shutdown()
    print("🎉 所有并行评估已完成！")


if __name__ == "__main__":
    main()