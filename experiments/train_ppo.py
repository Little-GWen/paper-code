import sys, os, argparse
import time
import numpy as np
import gymnasium as gym
import torch
import torch.multiprocessing as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Value
from models.agent_ppo import Agent_PPO
from config.config import *
import custom_merge_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main_optimizer(env_id, num_processes, total_episodes, max_t, agent, manager, save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    weights_path = os.path.join(save_dir, 'weights.pth')
    param_queue = manager.Queue(maxsize=num_processes)
    stop_event = manager.Event()
    global_episode = Value('i', 0)
    global_step = Value('i', 0)
    episode_lock = manager.Lock()
    save_lock = manager.Lock()

    rewards_log = manager.list()
    speed_log = manager.list()
    collision_log = manager.list()

    processes = []
    for pid in range(num_processes):
        p = mp.Process(target=sample_worker, args=(
            env_id, agent.memory, global_episode, global_step, total_episodes, episode_lock, pid, param_queue,
            stop_event,
            max_t, save_lock, rewards_log, speed_log, collision_log))
        p.start()
        processes.append(p)

    initial_state_dict = agent.get_state_dict()
    for _ in range(num_processes): param_queue.put(initial_state_dict)

    last_log_time, update_count = time.time(), 0
    try:
        while global_episode.value < total_episodes:
            if len(agent.memory) >= 2048:
                with save_lock:
                    agent.learn(global_step.value)
                    update_count += 1
                    if time.time() - last_log_time > 5:
                        print(f"[PPO] Upd {update_count} | Step {global_step.value} | Ent {agent.entropy:.3f}")
                        last_log_time = time.time()

                    if update_count % 5 == 0:
                        torch.save(agent.actor.state_dict(), weights_path)
                        np.save(os.path.join(save_dir, 'rewards.npy'), list(rewards_log))
                        np.save(os.path.join(save_dir, 'speed.npy'), list(speed_log))
                        np.save(os.path.join(save_dir, 'collision.npy'), list(collision_log))

                new_state_dict = agent.get_state_dict()
                while not param_queue.empty(): param_queue.get()
                for _ in range(num_processes): param_queue.put(new_state_dict)
            time.sleep(0.1)
    finally:
        stop_event.set()
        [p.join() for p in processes]

    torch.save(agent.actor.state_dict(), weights_path)


def sample_worker(env_id, shared_memory, global_episode, global_step, total_episodes, episode_lock, pid, param_queue,
                  stop_event, max_t, save_lock, rewards_log, speed_log, collision_log):
    env = gym.make(env_id)
    state_dim = int(np.prod(env.observation_space.shape))
    local_agent = Agent_PPO(state_dim, env.action_space.n, BATCH_SIZE, LEARNING_RATE, DECAY_MAX_STEP, GAMMA, LAMDA,
                            EPS_CLIP, K_EPOCHS, CRITIC_LOSS_COEF, ENTROPY_COEF, DEVICE, shared=False, manager=None,
                            is_worker=True)
    try:
        local_agent.load_state_dict(param_queue.get(timeout=5))
    except:
        pass

    while not stop_event.is_set():
        with episode_lock:
            if global_episode.value >= total_episodes: break
            global_episode.value += 1
            ep = global_episode.value

        state, _ = env.reset()  # Gym reset 返回 tuple
        state = state.flatten()

        ep_reward, ep_speed, steps = 0, 0, 0
        done = False
        is_crashed = 0

        while not done and steps < max_t:
            with global_step.get_lock():
                global_step.value += 1
            steps += 1
            if len(state) >= 3: ep_speed += state[2]

            # [修复] 直接传 Numpy，不要在这里转 Tensor
            action, log_prob = local_agent.act(state)

            res = env.step(action)
            next_state, reward, term, trunc, info = res
            done = term or trunc

            is_crashed = 1 if term else 0

            next_state = next_state.flatten()
            shared_memory.remember((state, action, reward, next_state, done, log_prob))
            state = next_state.copy()
            ep_reward += reward
            if not param_queue.empty():
                try:
                    local_agent.load_state_dict(param_queue.get_nowait())
                except:
                    pass
        with save_lock:
            rewards_log.append(ep_reward)
            speed_log.append(ep_speed / max(1, steps))
            collision_log.append(is_crashed)

        if pid == 0: print(f'\rEp {ep} Rew {ep_reward:.1f}', end='')
    env.close()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    set_seed(args.seed)
    save_dir = f'results/ppo/seed_{args.seed}'
    with mp.Manager() as manager:
        dummy = gym.make(RAM_ENV_NAME)
        state_dim = int(np.prod(dummy.observation_space.shape))
        act_dim = dummy.action_space.n
        dummy.close()
        agent = Agent_PPO(state_dim, act_dim, BATCH_SIZE, LEARNING_RATE, DECAY_MAX_STEP, GAMMA, LAMDA, EPS_CLIP,
                          K_EPOCHS, CRITIC_LOSS_COEF, ENTROPY_COEF, DEVICE, shared=False, manager=manager,
                          is_worker=False)
        main_optimizer(RAM_ENV_NAME, NUM_PROCESSES, RAM_NUM_EPISODE, MAX_T, agent, manager, save_dir)