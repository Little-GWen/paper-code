import sys, os, argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Manager
import gymnasium as gym
from models.agent_dqn import Agent_DQN
from config import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main_optimizer(env_id, num_episodes, agent, save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    weights_path = os.path.join(save_dir, 'weights.pth')
    env = gym.make(env_id)

    # 日志列表
    rewards_log = []
    speed_log = []
    collision_log = []

    print(f"🚀 DQN Training Started! Episodes: {num_episodes}")

    for i in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        state = state.flatten()
        ep_reward, steps, done = 0, 0, False
        ep_speed = 0
        is_crashed = 0

        while not done and steps < MAX_T:
            action, _ = agent.act(state)
            res = env.step(action)
            if len(res) == 5:
                next_state, reward, term, trunc, info = res
            else:
                next_state, reward, done, info = res
            done = term or trunc if len(res) == 5 else done

            if len(state) >= 3: ep_speed += state[2]
            if info.get('crashed', False): is_crashed = 1

            next_state = next_state.flatten()
            agent.memory.remember((state, action, reward, next_state, done, 0))
            if steps % 10 == 0: agent.learn()

            state = next_state
            ep_reward += reward
            steps += 1

        rewards_log.append(ep_reward)
        speed_log.append(ep_speed / max(1, steps))
        collision_log.append(is_crashed)

        if i % 10 == 0: print(f"\rEp {i}/{num_episodes} Reward {ep_reward:.1f} Eps {agent.epsilon:.2f}", end='')
        if i % 50 == 0:
            torch.save(agent.q_net.state_dict(), weights_path)
            np.save(os.path.join(save_dir, 'rewards.npy'), rewards_log)
            np.save(os.path.join(save_dir, 'speed.npy'), speed_log)
            np.save(os.path.join(save_dir, 'collision.npy'), collision_log)

    torch.save(agent.q_net.state_dict(), weights_path)
    np.save(os.path.join(save_dir, 'rewards.npy'), rewards_log)
    np.save(os.path.join(save_dir, 'speed.npy'), speed_log)
    np.save(os.path.join(save_dir, 'collision.npy'), collision_log)
    print(f"\nDQN Finished! Saved to {save_dir}")


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    save_dir = f'results/dqn/seed_{args.seed}'
    with Manager() as manager:
        dummy = gym.make(RAM_ENV_NAME)
        state_dim = int(np.prod(dummy.observation_space.shape))
        act_dim = dummy.action_space.n
        dummy.close()
        agent = Agent_DQN(state_dim, act_dim, 64, 5e-4, 0.99, 1.0, 0.05, 0.995, DEVICE, manager)
        main_optimizer(RAM_ENV_NAME, RAM_NUM_EPISODE // 4, agent, save_dir)