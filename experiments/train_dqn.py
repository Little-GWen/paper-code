import sys, os, argparse

# --- 1. 路径修正 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import Manager
import gymnasium as gym
import torch
import numpy as np

from models.agent_dqn import Agent_DQN
from config.config import *

# --- 2. [关键修复] 必须导入自定义环境以触发注册 ---
import custom_merge_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main_optimizer(env_id, num_episodes, agent, save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    weights_path = os.path.join(save_dir, 'weights.pth')

    # 创建环境
    env = gym.make(env_id)

    # 日志列表
    rewards_log = []
    speed_log = []
    collision_log = []

    print(f"🚀 DQN Training Started! Episodes: {num_episodes}")
    print(f"   Device: {agent.device}")
    print(f"   Batch Size: {agent.bs}")

    for i in range(num_episodes):
        state, _ = env.reset()  # Gym 新版 reset 返回 (obs, info)
        state = state.flatten()

        ep_reward, steps, done = 0, 0, False
        ep_speed = 0
        is_crashed = 0

        while not done and steps < MAX_T:
            action, _ = agent.act(state)  # DQN act 返回 (action, 0.0)

            res = env.step(action)
            # 兼容 4-tuple 或 5-tuple 返回
            if len(res) == 5:
                next_state, reward, term, trunc, info = res
                done = term or trunc
            else:
                next_state, reward, done, info = res

            if len(state) >= 3: ep_speed += state[2]  # 记录速度

            # 记录碰撞状态
            if done:
                is_crashed = info.get('crashed', False) or getattr(env.unwrapped.vehicle, 'crashed', False)

            next_state = next_state.flatten()

            # 存入经验回放 (log_prob 占位符填 0)
            agent.memory.remember((state, action, reward, next_state, done, 0))

            # 学习步骤
            if steps % 10 == 0:
                agent.learn()

            state = next_state
            ep_reward += reward
            steps += 1

        # 记录日志
        rewards_log.append(ep_reward)
        speed_log.append(ep_speed / max(1, steps))
        collision_log.append(1 if is_crashed else 0)

        # 打印进度
        if i % 10 == 0:
            print(
                f"\rEp {i}/{num_episodes} | Rew: {ep_reward:.1f} | Eps: {agent.epsilon:.2f} | Mem: {len(agent.memory)}",
                end='')

        # 保存模型和数据
        if i % 50 == 0:
            torch.save(agent.q_net.state_dict(), weights_path)
            np.save(os.path.join(save_dir, 'rewards.npy'), rewards_log)
            np.save(os.path.join(save_dir, 'speed.npy'), speed_log)
            np.save(os.path.join(save_dir, 'collision.npy'), collision_log)

    # 训练结束保存
    torch.save(agent.q_net.state_dict(), weights_path)
    np.save(os.path.join(save_dir, 'rewards.npy'), rewards_log)
    np.save(os.path.join(save_dir, 'speed.npy'), speed_log)
    np.save(os.path.join(save_dir, 'collision.npy'), collision_log)
    print(f"\nDQN Finished! Saved to {save_dir}")


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    save_dir = f'results/dqn/seed_{args.seed}'

    # 使用 Manager 只是为了兼容 ReplayBuffer 接口，DQN 本身是单进程的
    with Manager() as manager:
        # 创建 Dummy 环境获取维度
        dummy = gym.make(RAM_ENV_NAME)
        state_dim = int(np.prod(dummy.observation_space.shape))
        act_dim = dummy.action_space.n
        dummy.close()

        # --- 3. [关键调整] 超参数优化 ---
        # 原来是 64，建议改成 256 或 512，否则高密度下学不动
        BATCH_SIZE_DQN = 256

        # 增加总回合数，原来 //4 可能太少了，跑 10000 轮看看
        TOTAL_EPISODES = 80000

        agent = Agent_DQN(
            state_dim,
            act_dim,
            bs=BATCH_SIZE_DQN,
            lr=5e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9995,  # 衰减慢一点，多探索一会儿
            device=DEVICE,
            manager=manager
        )

        main_optimizer(RAM_ENV_NAME, TOTAL_EPISODES, agent, save_dir)