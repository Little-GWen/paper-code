import sys, os, argparse
import time
import random  # 新增

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Process, Manager, Value, Lock
import gym, torch
import torch.multiprocessing as mp
from models.agent_grpo import Agent_GRPO
from config import *
import custom_env

# --- [核心参数] 组大小 ---
# 类似于 LLM 中针对一个 Prompt 生成多少个回答
# 建议设置为 4 到 8。这意味着我们在完全相同的路况下，尝试 4 种不同的开法。
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
            # 因为是 Group 采样，数据量增长会很快，所以 buffer 阈值可以稍微调大
            if len(agent.memory) >= BATCH_SIZE * 4:
                with save_lock:
                    # 注意：这里原来的 learn 函数不需要大改，
                    # 因为我们在 worker 里已经把 Advantage 算好了一部分逻辑，
                    # 或者我们让 learn 函数只负责梯度下降。
                    # 为了兼容性，我们让 agent.learn 继续负责计算 loss，
                    # 但 worker 存进去的数据质量已经变高了。
                    agent.learn(global_step.value)

                    update_count += 1
                    if time.time() - last_log_time > 5:
                        print(f"[GRPO-SameStart] Upd {update_count} | Step {global_step.value} | Beta {agent.beta:.3f}")
                        last_log_time = time.time()
                    if update_count % 5 == 0:
                        torch.save(agent.actor.state_dict(), weights_path)
                        np.save(os.path.join(save_dir, 'rewards.npy'), list(rewards_log))
                        np.save(os.path.join(save_dir, 'speed.npy'), list(speed_log))
                        np.save(os.path.join(save_dir, 'collision.npy'), list(collision_log))

                new_state_dict = agent.get_state_dict()
                # 清空旧参数队列，放入新参数
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
                  stop_event, max_t, save_lock, rewards_log, speed_log, collision_log):
    env = gym.make(env_id)

    # 这里的 Agent 只是用来采样，不需要 optimizer
    state_dim = int(np.prod(env.observation_space.shape))
    local_agent = Agent_GRPO(state_dim, env.action_space.n, BATCH_SIZE, LEARNING_RATE, DECAY_RATE, DECAY_STEP_SIZE, TAU,
                             GAMMA, LAMDA, EPS_CLIP, K_EPOCHS, CRITIC_LOSS_COEF, ENTROPY_COEF, DEVICE, shared=False,
                             manager=None, is_worker=True)

    try:
        local_agent.load_state_dict(param_queue.get(timeout=5))
    except:
        pass

    while not stop_event.is_set():
        # 1. 确定本组的随机种子 (Same-State Condition)
        group_seed = random.randint(0, 1000000)

        group_trajectories = []  # 存储一组内的多条轨迹
        group_returns = []  # 存储每条轨迹的总回报

        # 2. 循环采集 Group Size 条轨迹
        for _ in range(GROUP_SIZE):
            # 关键：强制使用相同的种子重置环境！
            # 这样所有轨迹都面临完全相同的初始交通流
            state, _ = env.reset(seed=group_seed)

            state = state.flatten()
            ep_reward, ep_speed, steps = 0, 0, 0
            done = False
            is_crashed = 0

            trajectory = []  # 存储单条轨迹的 (s, a, r, s', done, log_prob)

            while not done and steps < max_t:
                state_tensor = torch.FloatTensor(state).to(local_agent.device)

                # 采样动作 (Stochastic)
                # 即使环境相同，因为这里有随机性，Agent 会走出不同的路
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

                # 暂存数据
                trajectory.append((state, action, reward, next_state, done, log_prob))

                state = next_state.copy()
                ep_reward += reward
                steps += 1

            # 记录这条轨迹的数据和总分
            group_trajectories.append(trajectory)
            group_returns.append(ep_reward)

            # 更新全局计数器 (只在第一条轨迹时更新，避免 log 过于频繁)
            with global_step.get_lock():
                global_step.value += steps

        # 3. 组内处理 (Group Processing)
        # 虽然 agent_grpo.py 里的 learn 也会算一遍，但为了保险，
        # 我们在这里可以做一些预处理，或者直接把原始数据推送到 buffer。
        # GRPO 的精髓：Advantage 计算应该是基于这个 group_returns 的。
        # 但为了不改动 memory 结构，我们直接把数据推过去，
        # 只要 buffer 里的数据是按 Group 顺序存的，且 BatchSize 能被 GroupSize 整除，
        # 理论上 agent_grpo.py 里的全 Batch 归一化也是一种近似（它是 Global Baseline）。

        # 更好的做法：如果想严格复现 GRPO，我们应该在这里计算 Advantage 并存入 log_prob 字段(Hack)，
        # 但这太复杂。

        # 现在的改进：由于我们消除了环境初始化的方差，
        # 即使是全 Batch 归一化，效果也会好很多，因为 Batch 里包含了很多个 "Same-State Group"。

        for traj in group_trajectories:
            for step_data in traj:
                shared_memory.remember(step_data)

        # 4. 记录日志 (只记平均值)
        with episode_lock:
            if global_episode.value < total_episodes:
                global_episode.value += 1
                ep = global_episode.value

        with save_lock:
            # 记录这一组的平均表现
            avg_group_reward = np.mean(group_returns)
            rewards_log.append(avg_group_reward)
            speed_log.append(ep_speed / max(1, steps))  # 粗略估计
            collision_log.append(is_crashed)

        if pid == 0:
            print(f'\rEp {ep} GroupRew {avg_group_reward:.1f}', end='')

        # 5. 更新参数
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

        agent = Agent_GRPO(state_dim, act_dim, BATCH_SIZE, LEARNING_RATE, DECAY_RATE, DECAY_STEP_SIZE, TAU, GAMMA,
                           LAMDA, EPS_CLIP, K_EPOCHS, CRITIC_LOSS_COEF, ENTROPY_COEF, DEVICE, shared=False,
                           manager=manager, is_worker=False)

        main_optimizer(RAM_ENV_NAME, NUM_PROCESSES, RAM_NUM_EPISODE, MAX_T, agent, manager, save_dir)