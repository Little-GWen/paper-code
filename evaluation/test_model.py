import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import gym, torch, time, argparse
from config import *
import custom_env
from models.agent_ppo import Agent_PPO
from models.agent_grpo import Agent_GRPO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='grpo', choices=['ppo', 'grpo'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--render', action='store_true', default=True)
    return parser.parse_args()


def test(env, agent, episodes, max_t, render):
    for i in range(episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        state = state.flatten()
        ep_reward, done, t = 0, False, 0
        print(f"\n--- Ep {i + 1} ---")
        while not done and t < max_t:
            t += 1
            state_tensor = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
            action, _ = agent.act(state_tensor)
            res = env.step(action)
            if len(res) == 5:
                next_state, reward, term, trunc, info = res
            else:
                next_state, reward, done, info = res
            done = term or trunc if len(res) == 5 else done
            if done:
                crash = info.get('crashed', False) or (
                            hasattr(env.unwrapped, 'vehicle') and env.unwrapped.vehicle.crashed)
                print(f"End Step {t}. Crashed: {crash}")
            state = next_state.flatten()
            if render:
                env.render()
                time.sleep(0.05)
            ep_reward += reward
        print(f"Reward: {ep_reward:.1f}")


if __name__ == '__main__':
    args = parse_args()
    try:
        env = gym.make(RAM_ENV_NAME, render_mode='human' if args.render else None)
    except:
        env = gym.make(RAM_ENV_NAME)

    # --- [关键修复] ---
    state_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n
    # ----------------

    folder_map = {'ppo': 'ppo', 'grpo': 'grpo_main'}
    path = f'results/{folder_map[args.algo]}/seed_{args.seed}/weights.pth'

    print(f"Initializing Agent with State Dim: {state_dim}...")
    if args.algo == 'ppo':
        agent = Agent_PPO(state_dim, act_dim, BATCH_SIZE, LEARNING_RATE, DECAY_RATE, DECAY_STEP_SIZE, TAU, GAMMA, LAMDA,
                          EPS_CLIP, K_EPOCHS, CRITIC_LOSS_COEF, ENTROPY_COEF, DEVICE)
    else:
        agent = Agent_GRPO(state_dim, act_dim, BATCH_SIZE, LEARNING_RATE, DECAY_RATE, DECAY_STEP_SIZE, TAU, GAMMA,
                           LAMDA, EPS_CLIP, K_EPOCHS, CRITIC_LOSS_COEF, ENTROPY_COEF, DEVICE)

    try:
        ckpt = torch.load(path, map_location=DEVICE)
        agent.actor.load_state_dict(ckpt['actor'] if 'actor' in ckpt else ckpt)
        print(f"✅ Loaded {path}")
    except:
        print(f"❌ Failed to load {path}, using Random Weights.")
    test(env, agent, args.episodes, MAX_T, args.render)