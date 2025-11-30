import sys, os, argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Value
import gymnasium as gym
import torch.multiprocessing as mp
from models.agent_grpo import Agent_GRPO
import custom_env
from config.grpo_config import *


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
        # [主实验] 默认 use_dynamic_beta=True (在 Agent 初始化中设置)
        p = mp.Process(target=sample_worker, args=(
            env_id, agent.memory, global_episode, global_step, total_episodes, episode_lock, pid, param_queue,
            stop_event, max_t, save_lock, rewards_log, speed_log, collision_log))
        p.start()
        processes.append(p)

    initial_state_dict = agent.get_state_dict()
    for _ in range(num_processes): param_queue.put(initial_state_dict)

    last_log_time, update_count = time.time(), 0
    try:
        while global_episode.value < total_episodes:
            # 等待足够的数据 (Batch Size * Group Size 确保可以进行几次完整的组更新)
            if len(agent.memory) >= BATCH_SIZE * 4:
                with save_lock:
                    agent.learn(global_step.value)
                    update_count += 1
                    if time.time() - last_log_time > 5:
                        print(f"\n[GRPO-Main] Upd {update_count} | Step {global_step.value} | Ent {agent.entropy:.3f} | Beta {agent.beta:.6f}")
                        last_log_time = time.time()
                    if update_count % 5 == 0:
                        torch.save(agent.actor.state_dict(), weights_path)
                        np.save(os.path.join(save_dir, 'rewards.npy'), list(rewards_log))
                        np.save(os.path.join(save_dir, 'speed.npy'), list(speed_log))
                        np.save(os.path.join(save_dir, 'collision.npy'), list(collision_log))

                new_state_dict = agent.get_state_dict()
                # 清空旧参数，广播新参数
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
    state_dim = int(np.prod(env.observation_space.shape))

    # [主实验] use_dynamic_beta=True
    local_agent = Agent_GRPO(state_dim, env.action_space.n, BATCH_SIZE, LEARNING_RATE, DECAY_RATE, DECAY_STEP_SIZE, TAU,
                             GAMMA, LAMDA, EPS_CLIP, K_EPOCHS, CRITIC_LOSS_COEF, GRPO_ENTROPY_COEF, DEVICE, shared=False,
                             manager=None, is_worker=True, use_dynamic_beta=True)
    try:
        local_agent.load_state_dict(param_queue.get(timeout=5))
    except:
        pass

    while not stop_event.is_set():
        # 1. 采集数据
        # 组采样初始化 (Same-State Logic)
        group_seed = random.randint(0, 1000000)

        group_trajectories = []     # 存储轨迹数据
        group_ep_speeds = []        # 用于日志
        group_is_crashed = []       # 用于日志

        for _ in range(GROUP_SIZE):
            # 强制使用相同的种子重置环境
            state, _ = env.reset(seed=group_seed)
            state = state.flatten()

            ep_reward, ep_speed = 0, 0
            steps = 0
            is_crashed = 0
            done = False

            traj_buffer = []    # 单条轨迹的 Buffer

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
                traj_buffer.append((state, action, reward, next_state, done, log_prob))

                state = next_state.copy()
                ep_reward += reward
                steps += 1

            # 采集完一条，加入组列表
            group_trajectories.append(traj_buffer)

            # 记录辅助信息
            group_ep_speeds.append(ep_speed)
            group_is_crashed.append(is_crashed)

            # 更新全局步数计数器
            with global_step.get_lock():
                global_step.value += steps

        # 2. 计算优势
        group_advantages, avg_group_reward = local_agent.calculate_group_advantages(group_trajectories)

        # 3. 组装与存储
        # 我们同时遍历 "原始轨迹列表" 和 "计算好的优势列表"
        # zip((traj1, traj2...), (adv1, adv2...)) -> [(traj1, adv1), (traj2, adv2)...]
        for raw_trajectory, advantage_val in zip(group_trajectories, group_advantages):
            # advantage_val 是这一整条轨迹共享的优势值 (标量)
            # raw_trajectory 是这一条轨迹里的步骤列表: [(s,a,r...), (s,a,r...)]
            for step_data in raw_trajectory:
                # 1. 解包原始数据
                s, a, raw_reward, ns, d, lp = step_data

                # 2. 构造存入数据
                # 重点：我们用 advantage_val 替换掉了 raw_reward
                # GRPO 的核心思想：整个轨迹的好坏决定了其中每一步的好坏
                data_to_store = (s, a, advantage_val, ns, d, lp)

                # 3. 存入共享内存
                shared_memory.remember(data_to_store)

        # 4. 日志与同步
        with episode_lock:
            if global_episode.value < total_episodes:
                global_episode.value += GROUP_SIZE
                ep = global_episode.value

        with save_lock:
            # 记录平均值
            rewards_log.append(avg_group_reward)
            speed_log.append(np.mean(group_ep_speeds) / max(1, steps)) # 粗略估算
            collision_log.append(np.mean(group_is_crashed))

        if pid == 0:
            print(f'\rEp {ep} GroupRew {avg_group_reward:.1f}', end='')

        # 检查是否有新参数，更新 Worker 的网络
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
                           LAMDA, EPS_CLIP, K_EPOCHS, CRITIC_LOSS_COEF, GRPO_ENTROPY_COEF, DEVICE, shared=False,
                           manager=manager, is_worker=False, use_dynamic_beta=True)

        main_optimizer(RAM_ENV_NAME, NUM_PROCESSES, RAM_NUM_EPISODE, MAX_T, agent, manager, save_dir)