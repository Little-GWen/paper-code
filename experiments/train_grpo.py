import sys, os, argparse
import time
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Value
import gymnasium as gym
import torch
import torch.multiprocessing as mp
from models.agent_grpo import Agent_GRPO
from config.grpo_config import *
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

    global_baseline = manager.Value('d', 0.0)
    baseline_lock = manager.Lock()

    rewards_log = manager.list()
    speed_log = manager.list()
    collision_log = manager.list()

    processes = []
    for pid in range(num_processes):
        p = mp.Process(target=sample_worker, args=(
            env_id, agent.memory, global_episode, global_step, total_episodes, episode_lock, pid, param_queue,
            stop_event, max_t, save_lock, rewards_log, speed_log, collision_log, global_baseline, baseline_lock))
        p.start()
        processes.append(p)

    initial_state_dict = agent.get_state_dict()
    for _ in range(num_processes): param_queue.put(initial_state_dict)

    last_log_time, update_count = time.time(), 0
    try:
        while global_episode.value < total_episodes:
            if len(agent.memory) >= BATCH_SIZE * 4:
                with save_lock:
                    agent.learn(global_step.value)
                    update_count += 1
                    if time.time() - last_log_time > 5:
                        print(f"\n[GRPO-Main] Upd {update_count} | Step {global_step.value} | Ent {agent.entropy:.3f} | Beta {agent.beta:.6f} | Base {global_baseline.value:.2f}")
                        last_log_time = time.time()
                    if update_count % 5 == 0:
                        torch.save(agent.actor.state_dict(), weights_path)
                        np.save(os.path.join(save_dir, 'rewards.npy'), list(rewards_log))
                        np.save(os.path.join(save_dir, 'speed.npy'), list(speed_log))
                        np.save(os.path.join(save_dir, 'collision.npy'), list(collision_log))

                new_state_dict = agent.get_state_dict()
                while not param_queue.empty():
                    try:
                        param_queue.get_nowait()
                    except:
                        pass
                for _ in range(num_processes): param_queue.put(new_state_dict)
            time.sleep(0.1)
    finally:
        stop_event.set()
        [p.join() for p in processes]

    torch.save(agent.actor.state_dict(), weights_path)
    np.save(os.path.join(save_dir, 'rewards.npy'), list(rewards_log))
    np.save(os.path.join(save_dir, 'speed.npy'), list(speed_log))
    np.save(os.path.join(save_dir, 'collision.npy'), list(collision_log))


def sample_worker(env_id, shared_memory, global_episode, global_step, total_episodes, episode_lock, pid, param_queue,
                  stop_event, max_t, save_lock, rewards_log, speed_log, collision_log, global_baseline, baseline_lock):
    env = gym.make(env_id)
    state_dim = int(np.prod(env.observation_space.shape))

    local_agent = Agent_GRPO(state_dim, env.action_space.n, BATCH_SIZE, LEARNING_RATE, DECAY_MAX_STEP, GAMMA, LAMDA,
                             EPS_CLIP, K_EPOCHS, ENTROPY_COEF, DEVICE, shared=False, manager=None, is_worker=True,
                             use_dynamic_beta=True, global_baseline=global_baseline, baseline_lock=baseline_lock)
    try:
        local_agent.load_state_dict(param_queue.get(timeout=5))
    except:
        pass

    while not stop_event.is_set():
        group_seed = random.randint(0, 1000000)

        group_trajectories = []
        group_ep_speeds = []
        group_is_crashed = []

        for _ in range(GROUP_SIZE):
            state, _ = env.reset(seed=group_seed)
            state = state.flatten()

            ep_reward, ep_speed = 0, 0
            steps = 0
            is_crashed = 0
            done = False

            traj_buffer = []

            while not done and steps < max_t:
                # [修复] 直接传 Numpy 数组，不要在这里手动转 Tensor
                action, log_prob = local_agent.act(state)

                res = env.step(action)
                next_state, reward, term, trunc, info = res
                next_state = next_state.flatten()

                traj_buffer.append((state, action, reward, next_state, term, log_prob))

                ep_reward += reward
                is_crashed = 1 if term else 0
                ep_speed += state[2]

                state = next_state.copy()
                steps += 1
                done = term or trunc

            group_trajectories.append(traj_buffer)
            group_ep_speeds.append(ep_speed / max(1, steps))
            group_is_crashed.append(is_crashed)

            with global_step.get_lock():
                global_step.value += steps

        group_advantages, avg_group_reward = local_agent.calculate_group_advantages(group_trajectories)

        for raw_trajectory, advantage_val in zip(group_trajectories, group_advantages):
            for step_data in raw_trajectory:
                s, a, raw_reward, ns, d, lp = step_data
                data_to_store = (s, a, advantage_val, ns, d, lp)
                shared_memory.remember(data_to_store)

        with episode_lock:
            if global_episode.value < total_episodes:
                global_episode.value += GROUP_SIZE
                ep = global_episode.value

        with save_lock:
            rewards_log.append(avg_group_reward)
            speed_log.append(np.mean(group_ep_speeds))
            collision_log.append(np.mean(group_is_crashed))

        if pid == 0:
            print(f'\rEp {ep} GroupRew {avg_group_reward:.1f}', end='')

        if not param_queue.empty():
            try:
                local_agent.load_state_dict(param_queue.get_nowait())
            except:
                pass

    env.close()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    set_seed(args.seed)
    save_dir = f'results/grpo_main/seed_{args.seed}'

    with mp.Manager() as manager:
        dummy = gym.make(RAM_ENV_NAME)
        state_dim = int(np.prod(dummy.observation_space.shape))
        act_dim = dummy.action_space.n
        dummy.close()

        agent = Agent_GRPO(state_dim, act_dim, BATCH_SIZE, LEARNING_RATE, DECAY_MAX_STEP, GAMMA, LAMDA, EPS_CLIP,
                           K_EPOCHS, ENTROPY_COEF, DEVICE, shared=False, manager=manager, is_worker=False,
                           use_dynamic_beta=True)

        main_optimizer(RAM_ENV_NAME, NUM_PROCESSES, RAM_NUM_EPISODE, MAX_T, agent, manager, save_dir)