import sys, os, argparse
import time
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Process, Manager, Value, Lock
import gym, torch
import torch.multiprocessing as mp
from models.agent_grpo import Agent_GRPO
from config import *
import custom_env

# --- 消融实验配置 ---
# 必须与主实验保持一致，除了 use_dynamic_beta
GROUP_SIZE = 4


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
        # 注意：这里传入 use_dynamic_beta=False
        p = mp.Process(target=sample_worker, args=(
            env_id, agent.memory, global_episode, global_step, total_episodes, episode_lock, pid, param_queue,
            stop_event,
            max_t, save_lock, rewards_log, speed_log, collision_log, False))
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
                        # 打印 Beta 确认它是否固定 (应该始终接近 0.01)
                        print(
                            f"[Ablation-NoDynamic] Upd {update_count} | Step {global_step.value} | Beta {agent.beta:.3f}")
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
                  stop_event, max_t, save_lock, rewards_log, speed_log, collision_log, use_dynamic_beta):
    env = gym.make(env_id)

    state_dim = int(np.prod(env.observation_space.shape))
    # 关键：初始化时传入 use_dynamic_beta=False
    local_agent = Agent_GRPO(state_dim, env.action_space.n, BATCH_SIZE, LEARNING_RATE, DECAY_RATE, DECAY_STEP_SIZE, TAU,
                             GAMMA, LAMDA, EPS_CLIP, K_EPOCHS, CRITIC_LOSS_COEF, ENTROPY_COEF, DEVICE, shared=False,
                             manager=None, is_worker=True, use_dynamic_beta=use_dynamic_beta)

    try:
        local_agent.load_state_dict(param_queue.get(timeout=5))
    except:
        pass

    while not stop_event.is_set():
        # 1. 同样使用 Group Sampling 逻辑，保证数据分布与主实验一致
        group_seed = random.randint(0, 1000000)

        group_trajectories = []
        group_returns = []

        for _ in range(GROUP_SIZE):
            state, _ = env.reset(seed=group_seed)

            state = state.flatten()
            ep_reward, ep_speed, steps = 0, 0, 0
            done = False
            is_crashed = 0

            trajectory = []

            while not done and steps < max_t:
                state_tensor = torch.FloatTensor(state).to(local_agent.device)
                action, log_prob = local_agent.act(state_tensor)

                res = env.step(action)
                if len(res) == 5:
                    next_state, reward, term, trunc, info = res
                    done = term or trunc
                else:
                    next_state, reward, done, info = res

                if info.get('crashed', False): is_crashed = 1
                if len(state) >= 3: ep_speed += state[2]

                next_state = next_state.flatten()
                trajectory.append((state, action, reward, next_state, done, log_prob))
                state = next_state.copy()
                ep_reward += reward
                steps += 1

            group_trajectories.append(trajectory)
            group_returns.append(ep_reward)

            with global_step.get_lock():
                global_step.value += steps

        for traj in group_trajectories:
            for step_data in traj:
                shared_memory.remember(step_data)

        with episode_lock:
            if global_episode.value < total_episodes:
                global_episode.value += 1
                ep = global_episode.value

        with save_lock:
            avg_group_reward = np.mean(group_returns)
            rewards_log.append(avg_group_reward)
            speed_log.append(ep_speed / max(1, steps))
            collision_log.append(is_crashed)

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
    # 结果保存在 grpo_no_dynamic 文件夹
    save_dir = f'results/grpo_no_dynamic/seed_{args.seed}'

    with mp.Manager() as manager:
        dummy = gym.make(RAM_ENV_NAME)
        state_dim = int(np.prod(dummy.observation_space.shape))
        act_dim = dummy.action_space.n
        dummy.close()

        # 主 Agent 也必须关闭 dynamic beta
        agent = Agent_GRPO(state_dim, act_dim, BATCH_SIZE, LEARNING_RATE, DECAY_RATE, DECAY_STEP_SIZE, TAU, GAMMA,
                           LAMDA, EPS_CLIP, K_EPOCHS, CRITIC_LOSS_COEF, ENTROPY_COEF, DEVICE, shared=False,
                           manager=manager, is_worker=False, use_dynamic_beta=False)

        main_optimizer(RAM_ENV_NAME, NUM_PROCESSES, RAM_NUM_EPISODE, MAX_T, agent, manager, save_dir)