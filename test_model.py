import sys
import os

# --- 1. 路径锚定：获取当前脚本所在的绝对路径，确保根目录正确 ---
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
sys.path.append(project_root)

import gymnasium as gym
import numpy as np
import torch
import time
import argparse

# 尝试导入环境
try:
    import custom_merge_env
except ImportError:
    try:
        import envs.custom_merge_env as custom_merge_env
    except:
        print("❌ Error: 找不到 custom_merge_env.py，请确保它在根目录或 envs 目录下")
        exit()

from config.config import *
# 导入模型
from models.agent_ppo import Agent_PPO
from models.agent_grpo import Agent_GRPO
from models.agent_dqn import Agent_DQN


def parse_args():
    parser = argparse.ArgumentParser(description="模型测试脚本")
    # [关键] 确保 choices 包含 grpo_static_beta
    parser.add_argument('--algo', type=str, default='dqn',
                        choices=['ppo', 'grpo', 'grpo_static_beta', 'dqn'],
                        help='选择算法: ppo, grpo, grpo_static_beta, dqn')
    parser.add_argument('--seed', type=int, default=0, help='加载哪个种子的权重')
    parser.add_argument('--episodes', type=int, default=5, help='测试几轮')
    parser.add_argument('--render', action='store_true', default=True, help='是否渲染画面')
    parser.add_argument('--no-render', action='store_false', dest='render', help='关闭渲染')
    return parser.parse_args()


def get_model_path(algo, seed):
    # [修改] 文件夹映射字典
    # 键是命令行输入的 --algo 参数
    # 值是 results 文件夹下实际的文件夹名字
    folder_map = {
        'ppo': 'ppo',
        'grpo': 'grpo_main',  # GRPO (Ours) 的文件夹名
        'grpo_static_beta': 'grpo_static_beta',  # [关键] Static Beta 的文件夹名
        'dqn': 'dqn'
    }

    folder_name = folder_map.get(algo, algo)

    # 拼接绝对路径
    # results_dir = project_root/results/folder_name/seed_X/weights.pth
    return os.path.join(project_root, 'results', folder_name, f'seed_{seed}', 'weights.pth')


def test(env, agent, episodes, max_t, render):
    print(f"\n🚗 开始测试 | 算法: {agent.__class__.__name__} | 轮数: {episodes}")

    for i in range(episodes):
        # 锁定测试种子，方便复现
        test_seed = 100 + i
        state, _ = env.reset(seed=test_seed)
        state = state.flatten()

        ep_reward = 0
        ep_speed = []
        done = False
        t = 0

        print(f"\n--- Episode {i + 1} (Seed {test_seed}) ---")

        while not done and t < max_t:
            t += 1
            # 测试时开启确定性模式 (Deterministic)
            action, _ = agent.act(state, deterministic=True)

            step_res = env.step(action)
            if len(step_res) == 5:
                next_state, reward, term, trunc, info = step_res
                done = term or trunc
            else:
                next_state, reward, done, info = step_res

            if hasattr(env.unwrapped, 'vehicle'):
                ep_speed.append(env.unwrapped.vehicle.speed)

            state = next_state.flatten()
            ep_reward += reward

            if render:
                time.sleep(0.02)  # 稍微慢点

            if done:
                is_crashed = info.get('crashed', False) or getattr(env.unwrapped.vehicle, 'crashed', False)
                reason = "💥 撞车" if is_crashed else "✅ 完成/超时"
                print(f"   -> 结束步骤: {t} | 原因: {reason}")

        avg_spd = np.mean(ep_speed) if ep_speed else 0
        print(f"   Reward: {ep_reward:.2f} | Avg Speed: {avg_spd:.2f}")


def main():
    args = parse_args()

    # 1. 查找模型路径
    model_path = get_model_path(args.algo, args.seed)

    # [诊断] 打印绝对路径，让你看清楚它到底在找哪里
    print(f"🔍 正在寻找模型文件: {model_path}")

    if not os.path.exists(model_path):
        print(f"\n❌ 错误: 找不到模型文件！")
        print(f"   请检查 results 文件夹下是否有 '{args.algo}' 或者是映射后的文件夹。")
        print(f"   尝试去 train_grpo_static_beta.py 里看看 save_dir 是怎么写的。")
        return

    # 2. 创建环境
    render_mode = 'human' if args.render else None
    try:
        env = gym.make(RAM_ENV_NAME, render_mode=render_mode)
        # 强制同步配置
        env.unwrapped.configure({
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 40,
            "vehicles_count": 20,
            "collision_reward": -500
        })
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return

    # 3. 初始化 Agent
    state_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n

    # 这里不需要具体的 lr 等参数，只需要网络结构匹配即可
    if args.algo == 'ppo':
        agent = Agent_PPO(state_dim, act_dim, 0, 0, 0, 0, 0, 0, 0, 0, 0, DEVICE)

    elif args.algo in ['grpo', 'grpo_static_beta']:
        # 无论是 Dynamic 还是 Static，模型结构是一样的 (Agent_GRPO)
        agent = Agent_GRPO(state_dim, act_dim, 0, 0, 0, 0, 0, 0, 0, 0, DEVICE)

    elif args.algo == 'dqn':
        agent = Agent_DQN(state_dim, act_dim, 0, 0, 0, 0.0, 0.0, 0.0, DEVICE)

    else:
        print("未知的算法类型")
        return

    # 4. 加载权重
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)

        if args.algo == 'dqn':
            agent.q_net.load_state_dict(checkpoint)
        else:
            # PPO/GRPO 通常保存的是 actor state_dict，或者是包含 'actor' key 的字典
            if isinstance(checkpoint, dict) and 'actor' in checkpoint:
                agent.actor.load_state_dict(checkpoint['actor'])
            else:
                agent.actor.load_state_dict(checkpoint)

        print(f"✅ 成功加载权重: {model_path}")

    except Exception as e:
        print(f"❌ 加载权重失败: {e}")
        return

    # 5. 开始测试
    try:
        test(env, agent, args.episodes, 1000, args.render)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()