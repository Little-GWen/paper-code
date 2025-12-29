import sys
import os
import argparse

import gymnasium
import numpy as np
import torch
import gymnasium as gym
import ray

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from config.config import *
from envs.custom_merge_env import MergeEnv

# 主进程注册环境
gym.register(id='my-merge-v0', entry_point='envs.custom_merge_env:MergeEnv')


# ================= 自定义 Ray Worker (同 PPO) =================
@ray.remote
class RayWorker:
    def __init__(self, env_id, seed):
        import gymnasium as gym
        from envs.custom_merge_env import MergeEnv
        try:
            gym.make(env_id)
        except gymnasium.error.NameNotFound:
            gym.register(id=env_id, entry_point='envs.custom_merge_env:MergeEnv')

        self.env = gym.make(env_id)
        self.env.reset(seed=seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            obs, _ = self.env.reset()

        # 🛡️ NaN 清洗
        if np.isnan(obs).any() or np.isinf(obs).any():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs, reward, terminated or truncated, info

    def reset(self):
        obs, _ = self.env.reset()
        # 🛡️ NaN 清洗
        if np.isnan(obs).any() or np.isinf(obs).any():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def close(self):
        self.env.close()


# ================= 自定义 Ray VecEnv (同 PPO) =================
class RayVecEnv(VecEnv):
    def __init__(self, env_id, num_envs, seed_start=0):
        ray.init(ignore_reinit_error=True)
        self.workers = [RayWorker.remote(env_id, seed_start + i) for i in range(num_envs)]

        dummy = gym.make(env_id)

        self.num_envs = num_envs
        self.observation_space = dummy.observation_space
        self.action_space = dummy.action_space
        self.render_mode = getattr(dummy, "render_mode", None)
        self.metadata = getattr(dummy, "metadata", {"render_modes": []})

        dummy.close()

    def step_async(self, actions):
        self.futures = [w.step.remote(a) for w, a in zip(self.workers, actions)]

    def step_wait(self):
        results = ray.get(self.futures)
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), list(infos)

    def reset(self):
        return np.stack(ray.get([w.reset.remote() for w in self.workers]))

    def close(self):
        ray.get([w.close.remote() for w in self.workers])
        ray.shutdown()

    def get_attr(self, attr_name, indices=None): return []

    def set_attr(self, attr_name, value, indices=None): pass

    def env_method(self, method_name, *args, **kwargs): return []

    def env_is_wrapped(self, wrapper, indices=None): return [False] * self.num_envs

    def seed(self, seed=None): return


# ================= 回调函数 =================
class MatchCallback(BaseCallback):
    def __init__(self, save_dir):
        super().__init__(verbose=0)
        self.save_dir = save_dir
        self.log_rew = []
        self.log_spd = []
        self.log_col = []

        self.ep_rew = np.zeros(NUM_PROCESSES)
        self.ep_spd = np.zeros(NUM_PROCESSES)
        self.ep_len = np.zeros(NUM_PROCESSES)

    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        infos = self.locals['infos']

        self.ep_rew += rewards
        self.ep_len += 1

        for i in range(len(infos)):
            self.ep_spd[i] += infos[i].get('speed', 0)

            if dones[i]:
                self.log_rew.append(self.ep_rew[i])
                self.log_spd.append(self.ep_spd[i] / max(1, self.ep_len[i]))
                self.log_col.append(1.0 if infos[i].get('crashed', False) else 0.0)

                self.ep_rew[i] = 0
                self.ep_spd[i] = 0
                self.ep_len[i] = 0

        # DQN 可以在 step 中定期保存
        if self.num_timesteps % 5000 == 0:
            self._save_logs()
        return True

    def _save_logs(self):
        if len(self.log_rew) > 0:
            print(f"   [DQN] Steps: {self.num_timesteps} | Mean Rew: {np.mean(self.log_rew[-100:]):.2f}")
            np.save(os.path.join(self.save_dir, 'rewards.npy'), self.log_rew)
            np.save(os.path.join(self.save_dir, 'speed.npy'), self.log_spd)
            np.save(os.path.join(self.save_dir, 'collision.npy'), self.log_col)
            torch.save(self.model.policy.state_dict(), os.path.join(self.save_dir, 'weights.pth'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=NUM_PROCESSES)
    args = parser.parse_args()

    save_dir = f'results/train/dqn/seed_{args.seed}'
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # 初始化并行环境
    env = RayVecEnv(RAM_ENV_NAME, args.n_workers, seed_start=args.seed)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512, 512])

    # 初始化 DQN
    # DQN 比较吃内存，buffer_size 设为 50000 比较安全
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        device=DEVICE,
        policy_kwargs=policy_kwargs,
        buffer_size=50000,
        learning_starts=5000,
        batch_size=256
    )

    print(f"🚀 DQN (Ray) Training Started | Workers: {args.n_workers}")
    try:
        model.learn(total_timesteps=RAM_NUM_EPISODE * MAX_T, callback=MatchCallback(save_dir))
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()
        torch.save(model.policy.state_dict(), os.path.join(save_dir, 'weights.pth'))


if __name__ == '__main__':
    main()