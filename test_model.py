import sys
import os

# --- 1. 路径修正：确保能导入根目录的模块 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# ----------------------------------------

import numpy as np
import gymnasium as gym
import torch
import time
import argparse
from config import *
import custom_env

# 动态导入所有 Agent
from models.agent_ppo import Agent_PPO
from models.agent_grpo import Agent_GRPO
from models.agent_dqn import Agent_DQN


def parse_args():
    parser = argparse.ArgumentParser(description="模型测试脚本")
    parser.add_argument('--algo', type=str, default='grpo', choices=['ppo', 'grpo', 'dqn'],
                        help='选择算法: ppo, grpo, dqn')
    parser.add_argument('--seed', type=int, default=0, help='加载哪个种子的权重')
    parser.add_argument('--episodes', type=int, default=5, help='测试几轮')
    parser.add_argument('--render', action='store_true', default=True, help='是否渲染画面 (默认开启)')
    parser.add_argument('--no-render', action='store_false', dest='render', help='关闭画面渲染')
    return parser.parse_args()


def get_model_path(algo, seed):
    folder_map = {
        'ppo': 'ppo',
        'grpo': 'grpo_main',
        'dqn': 'dqn'
    }
    folder = folder_map.get(algo, algo)
    return os.path.join('results', folder, f'seed_{seed}', 'weights.pth')


def test(env, agent, episodes, max_t, render):
    print(f"\n🚗 开始测试: {episodes} 轮 | 渲染: {render}")

    for i in range(episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        state = state.flatten()

        ep_reward = 0
        ep_speed = []
        done = False
        t = 0

        print(f"\n--- Episode {i + 1} Start ---")

        while not done and t < max_t:
            t += 1

            state_tensor = torch.FloatTensor(state).to(agent.device).unsqueeze(0)

            # 开启确定性模式
            action, _ = agent.act(state_tensor, deterministic=True)

            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            if hasattr(env.unwrapped, 'vehicle'):
                ep_speed.append(env.unwrapped.vehicle.speed)

            if done:
                is_crashed = info.get('crashed', False) or getattr(env.unwrapped.vehicle, 'crashed', False)
                end_reason = "💥 CRASHED" if is_crashed else "⏱️ TIME LIMIT"
                print(f"   -> Terminated at Step {t} | Reason: {end_reason}")

            next_state = next_state.flatten()
            state = next_state.copy()

            # --- [关键修复] 渲染逻辑增强 ---
            if render:
                try:
                    env.render()  # 新版尝试
                except:
                    try:
                        env.render(mode='human')  # 旧版尝试
                    except Exception as e:
                        pass  # 忽略渲染错误，保证程序不崩
                time.sleep(0.03)

            ep_reward += reward

        avg_speed = np.mean(ep_speed) if ep_speed else 0
        print(f"   Episode {i + 1} Reward: {ep_reward:.2f} | Avg Speed: {avg_speed:.2f} m/s")


def main():
    args = parse_args()

    # --- [关键修复] 强制设置渲染模式 ---
    # 我们尝试多种方式来“激活” highway-env 的渲染窗口
    env = None
    try:
        if args.render:
            # 方式 1: 标准新版 Gym
            env = gym.make(RAM_ENV_NAME, render_mode='human')
        else:
            env = gym.make(RAM_ENV_NAME)
    except Exception as e:
        print(f"⚠️ Standard make failed ({e}), trying fallback...")
        # 方式 2: 旧版兼容
        env = gym.make(RAM_ENV_NAME)

    # 方式 3: 暴力补丁 (针对某些 highway-env 版本)
    if args.render:
        try:
            env.unwrapped.render_mode = 'human'
            # 有些版本需要设置 config 里的 mode
            if hasattr(env.unwrapped, 'config'):
                env.unwrapped.config['render_mode'] = 'human'
        except:
            pass

    # 自动获取维度
    state_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n

    print(f"🔍 Environment: {RAM_ENV_NAME}")
    print(f"🔍 State Dim: {state_dim}, Action Dim: {act_dim}")
    print(f"🔍 Algorithm: {args.algo.upper()}")

    if args.algo == 'ppo':
        agent = Agent_PPO(state_dim, act_dim, BATCH_SIZE, LEARNING_RATE, DECAY_RATE, DECAY_STEP_SIZE, TAU, GAMMA, LAMDA,
                          EPS_CLIP, K_EPOCHS, CRITIC_LOSS_COEF, PPO_ENTROPY_COEF, DEVICE)
    elif args.algo == 'grpo':
        agent = Agent_GRPO(state_dim, act_dim, BATCH_SIZE, LEARNING_RATE, DECAY_RATE, DECAY_STEP_SIZE, TAU, GAMMA,
                           LAMDA, EPS_CLIP, K_EPOCHS, CRITIC_LOSS_COEF, GRPO_ENTROPY_COEF, DEVICE)
    elif args.algo == 'dqn':
        agent = Agent_DQN(state_dim, act_dim, BATCH_SIZE, LEARNING_RATE, GAMMA, 0.0, 0.0, 0.0, DEVICE)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # 加载权重
    model_path = get_model_path(args.algo, args.seed)
    print(f"📂 Loading weights from: {model_path}")

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if args.algo == 'dqn':
                agent.q_net.load_state_dict(checkpoint)
            else:
                if isinstance(checkpoint, dict) and 'actor' in checkpoint:
                    agent.actor.load_state_dict(checkpoint['actor'])
                else:
                    agent.actor.load_state_dict(checkpoint)
            print("✅ Weights loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            print("⚠️ Running with RANDOM weights (Expect bad performance)")
    else:
        print(f"❌ File not found: {model_path}")
        print("⚠️ Running with RANDOM weights")

    try:
        test(env, agent, args.episodes, MAX_T, args.render)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user.")
    finally:
        env.close()
        print("Done.")


if __name__ == '__main__':
    main()