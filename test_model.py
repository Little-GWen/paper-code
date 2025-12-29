import sys
import os

# --- 路径修正：确保能导入根目录的模块 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import gymnasium as gym
import time
import argparse
# from config.config import *
from config.grpo_config import *

from envs.custom_highway_env import HighwayEnv
from envs.custom_merge_env import MergeEnv

from models.agent_grpo_ray import Agent_GRPO_Ray
from stable_baselines3 import PPO, DQN

folder_map = {
    'ppo': 'ppo',
    'grpo': 'grpo_main',
    'grpo_standard': 'grpo_standard',
    'dqn': 'dqn'
}

def parse_args():
    parser = argparse.ArgumentParser(description="模型测试脚本")
    parser.add_argument('--algo', type=str,
                        default='grpo',
                        choices=['ppo', 'grpo', 'grpo_standard', 'dqn'],
                        help='选择算法: ppo, grpo, dqn')
    parser.add_argument('--seed', type=int, default=0, help='加载哪个种子的权重')
    parser.add_argument('--episodes', type=int, default=100, help='测试几轮')
    parser.add_argument('--render', action='store_true', default=True, help='是否渲染画面 (默认开启)')
    parser.add_argument('--no-render', action='store_false', dest='render', help='关闭画面渲染')
    return parser.parse_args()


def get_model_path(algo, seed):
    folder = folder_map.get(algo, algo)
    return os.path.join('results', 'train', folder, f'seed_{seed}', 'weights.pth')

def get_save_path(algo, seed):
    folder = folder_map.get(algo, algo)
    return os.path.join('results', 'test', folder, f'seed_{seed}')


# SB3 包装器
class SB3_Wrapper:
    """
    这个类把 SB3 的 model.predict 包装成和你自定义 Agent 一样的 .act 接口
    这样下面的 test 循环就不用改代码了
    """
    def __init__(self, model):
        self.model = model

    def act(self, state, deterministic=True):
        # SB3 的 predict 返回 (action, _states)
        # 这里的 state 已经是 numpy 数组了，可以直接传
        action, _ = self.model.predict(state, deterministic=deterministic)

        # 如果是离散动作，action 是一个标量，直接返回即可
        # 这里的 None 是为了模仿你自定义 Agent 返回的 (action, log_prob) 格式
        return action.item(), None

def test(env, agent, episodes, max_t, render, save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    print(f"\n🚗 开始测试: {episodes} 轮 | 渲染: {render}")
    print(f"📂 结果将保存至: {save_dir}")

    # 初始化日志容器
    log_rewards = []
    log_avg_speeds = []
    log_collisions = []  # 0: 安全, 1: 碰撞

    for i in range(episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        # state = state.flatten()

        ep_reward = 0
        ep_speed = []
        done = False
        t = 0
        is_crashed = False

        print(f"\n--- Episode {i + 1} Start ---")

        while not done and t < max_t:
            t += 1

            # --- 核心适配逻辑 ---
            # 如果是 GRPO，手动在这里 flatten 一下传进去
            # 如果是 PPO/DQN (SB3_Wrapper)，直接传 (10, 6)
            if isinstance(agent, Agent_GRPO_Ray):
                state = state.flatten()

            # 开启确定性模式
            action, _ = agent.act(state, deterministic=True)

            step_result = env.step(action)
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
            ep_reward += reward

            # 记录速度（未归一化）
            if hasattr(env.unwrapped, 'vehicle'):
                ep_speed.append(env.unwrapped.vehicle.speed)
            elif 'speed' in info:
                ep_speed.append(info['speed'])

            if done:
                is_crashed = info.get('crashed', False)
                # 双重保险检查
                if hasattr(env.unwrapped, 'vehicle') and hasattr(env.unwrapped.vehicle, 'crashed'):
                    if env.unwrapped.vehicle.crashed: is_crashed = True

                end_reason = "💥 CRASHED" if is_crashed else "⏱️ TIME LIMIT"
                print(f"   -> Terminated at Step {t} | Reason: {end_reason}")

            state = next_state.copy()

            # --- 渲染逻辑增强 ---
            if render:
                try:
                    env.render()  # 新版尝试
                except:
                    try:
                        env.render(mode='human')  # 旧版尝试
                    except Exception as e:
                        pass  # 忽略渲染错误，保证程序不崩
                time.sleep(0.03)

        avg_speed = np.mean(ep_speed) if ep_speed else 0
        max_speed = np.max(ep_speed) if ep_speed else 0

        log_rewards.append(ep_reward)
        log_avg_speeds.append(avg_speed)
        log_collisions.append(1.0 if is_crashed else 0.0)

        print(f"   Episode {i + 1} Reward: {ep_reward:.2f} | Avg Speed: {avg_speed:.2f} m/s | Max Speed: {max_speed:.2f} m/s")

    # 保存原始数据
    np.save(os.path.join(save_dir, 'rewards.npy'), log_rewards)
    np.save(os.path.join(save_dir, 'speed.npy'), log_avg_speeds)
    np.save(os.path.join(save_dir, 'collision.npy'), log_collisions)
    print(f"✅ 所有测试数据已保存至: {save_dir}")

def main():
    args = parse_args()

    # 环境初始化
    render_mode = 'human' if args.render else None
    try:
        env = gym.make(RAM_ENV_NAME, render_mode=render_mode)
    except:
        env = gym.make(RAM_ENV_NAME)
        if args.render and hasattr(env.unwrapped, 'config'):
            env.unwrapped.config['render_mode'] = 'human'

    # 自动获取维度
    state_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n

    print(f"🔍 Environment: {RAM_ENV_NAME}")
    print(f"🔍 State Dim: {state_dim}, Action Dim: {act_dim}")
    print(f"🔍 Algorithm: {args.algo.upper()}")

    model_path = get_model_path(args.algo, args.seed)           # 加载权重
    save_dir   = get_save_path(args.algo, args.seed)            # 保存路径

    # 加载模型
    if args.algo in ['grpo', 'grpo_standard']:
        agent = Agent_GRPO_Ray(state_dim, act_dim, BATCH_SIZE, LEARNING_RATE, DECAY_MAX_STEP, GAMMA,
                           EPS_CLIP, K_EPOCHS, ENTROPY_COEF, DEVICE)
        if os.path.exists(model_path):
            print(f"📂 Loading GRPO weights from: {model_path}")
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'actor' in checkpoint:
                agent.actor.load_state_dict(checkpoint['actor'])
            else:
                agent.actor.load_state_dict(checkpoint)
        else:
            print(f"❌ File not found: {model_path}")

    elif args.algo == 'ppo':
        # 必须和训练时的 policy_kwargs 一模一样
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[512, 512], vf=[512, 512]))
        sb3_model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, device=DEVICE)
        if os.path.exists(model_path):
            print(f"📂 Loading PPO weights from: {model_path}")
            # 注意：加载到 model.policy 中
            sb3_model.policy.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print("✅ Weights loaded successfully!")
        else:
            print(f"❌ File not found: {model_path}, 使用随机权重测试!")

        # 包装成通用接口
        agent = SB3_Wrapper(sb3_model)

    elif args.algo == 'dqn':
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512, 512])
        sb3_model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, device=DEVICE)
        if os.path.exists(model_path):
            print(f"📂 Loading DQN weights from: {model_path}")
            sb3_model.policy.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print("✅ Weights loaded successfully!")
        else:
            print(f"❌ File not found: {model_path}, 使用随机权重测试!")
        agent = SB3_Wrapper(sb3_model)

    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # 开始测试
    try:
        test(env, agent, args.episodes, MAX_T, args.render, save_dir)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user.")
    finally:
        env.close()
        print("Done.")


if __name__ == '__main__':
    main()