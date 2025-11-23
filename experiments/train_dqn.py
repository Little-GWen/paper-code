import sys, os, argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Manager
import gym, torch
from models.agent_dqn import Agent_DQN
from config import *
import custom_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main_optimizer(env_id, num_episodes, agent, save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    weights_path = os.path.join(save_dir, 'weights.pth')
    env = gym.make(env_id)
    rewards_log = []

    for i in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        state = state.flatten()
        ep_reward, steps, done = 0, 0, False
        while not done and steps < MAX_T:
            action, _ = agent.act(state)
            res = env.step(action)
            if len(res) == 5:
                next_state, reward, term, trunc, _ = res
            else:
                next_state, reward, done, _ = res
            done = term or trunc if len(res) == 5 else done
            next_state = next_state.flatten()

            agent.memory.remember((state, action, reward, next_state, done, 0))
            agent.learn()
            state = next_state
            ep_reward += reward
            steps += 1
        rewards_log.append(ep_reward)
        if i % 10 == 0: print(f"\rEp {i} Reward {ep_reward:.1f} Eps {agent.epsilon:.2f}", end='')
        if i % 50 == 0:
            torch.save(agent.q_net.state_dict(), weights_path)
            np.save(os.path.join(save_dir, 'rewards.npy'), rewards_log)


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    save_dir = f'results/dqn/seed_{args.seed}'

    with Manager() as manager:
        dummy = gym.make(RAM_ENV_NAME)
        # --- [关键修复] ---
        state_dim = int(np.prod(dummy.observation_space.shape))
        act_dim = dummy.action_space.n
        dummy.close()
        # ----------------

        agent = Agent_DQN(state_dim, act_dim, 64, 5e-4, 0.99, 1.0, 0.05, 0.995, DEVICE, manager)
        main_optimizer(RAM_ENV_NAME, 5000, agent, save_dir)