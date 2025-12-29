import sys, os, time, argparse, numpy as np, torch, gymnasium as gym, ray

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.grpo_config import *
from models.agent_grpo_ray import Agent_GRPO_Ray
from envs.custom_merge_env import MergeEnv

gym.register(id='my-merge-v0', entry_point='envs.custom_merge_env:MergeEnv')


# 复用 Worker 类结构，但初始化参数不同
@ray.remote
class GRPOWorker:
    def __init__(self, env_id, seed, worker_id, adv_mode="standard"):
        import gymnasium as gym
        from envs.custom_merge_env import MergeEnv
        try:
            gym.make(env_id)
        except gymnasium.error.NameNotFound:

            gym.register(id=env_id, entry_point='envs.custom_merge_env:MergeEnv')
        self.seed = seed + worker_id * 1000
        np.random.seed(self.seed);
        torch.manual_seed(self.seed)
        self.env = gym.make(env_id)
        self.agent = Agent_GRPO_Ray(
            int(np.prod(self.env.observation_space.shape)), self.env.action_space.n,
            BATCH_SIZE, LEARNING_RATE, DECAY_MAX_STEP, GAMMA, EPS_CLIP, K_EPOCHS,
            ENTROPY_COEF, torch.device('cpu'), is_worker=True, adv_mode=adv_mode
        )

    def sample(self, weights, baseline, max_t):
        self.agent.set_weights(weights)
        group_trajs, metrics = [], {'rew': [], 'spd': [], 'col': []}
        total_steps = 0
        group_seed = np.random.randint(0, 1e6)

        for _ in range(GROUP_SIZE):
            s, _ = self.env.reset(seed=group_seed);
            s = s.flatten()
            traj, ep_r, ep_s = [], 0, []
            for _ in range(max_t):
                with torch.no_grad():
                    a, lp = self.agent.act(torch.FloatTensor(s))
                ns, r, term, trunc, info = self.env.step(a);
                ns = ns.flatten()
                traj.append((s, a, r, ns, term, lp))
                ep_r += r;
                ep_s.append(info.get('speed', 0));
                s = ns
                if term or trunc: break
            group_trajs.append(traj);
            total_steps += len(traj)
            metrics['rew'].append(ep_r);
            metrics['spd'].append(np.mean(ep_s) if ep_s else 0)
            metrics['col'].append(1 if info.get('crashed') else 0)

        advs, avg_rew, grp_mean = self.agent.calculate_group_advantages(group_trajs, baseline)
        batch = []
        for tr, adv in zip(group_trajs, advs):
            for step in tr: batch.append((step[0], step[1], adv, step[3], step[4], step[5]))
        return batch, metrics, grp_mean, total_steps


def train():
    parser = argparse.ArgumentParser();
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    ray.init()
    save_dir = f'results/train/grpo_standard/seed_{args.seed}'
    os.makedirs(save_dir, exist_ok=True)

    dummy = gym.make(RAM_ENV_NAME)
    # 注意这里 adv_mode="standard"
    learner = Agent_GRPO_Ray(
        int(np.prod(dummy.observation_space.shape)), dummy.action_space.n,
        BATCH_SIZE, LEARNING_RATE, DECAY_MAX_STEP, GAMMA, EPS_CLIP, K_EPOCHS,
        ENTROPY_COEF, DEVICE, is_worker=False, adv_mode="standard", use_dynamic_beta=True
    )

    # 启动 Worker 时也传入 standard
    workers = [GRPOWorker.remote(RAM_ENV_NAME, args.seed, i, "standard") for i in range(NUM_PROCESSES)]

    global_base, ema, step, ep, buffer = 0.0, 0.95, 0, 0, []
    logs = {'rew': [], 'spd': [], 'col': []}

    print(f"🚀 GRPO (Standard) Ray Training Started")

    try:
        while ep < RAM_NUM_EPISODE:
            w_ref = ray.put(learner.get_weights())
            results = ray.get([w.sample.remote(w_ref, global_base, MAX_T) for w in workers])

            grp_means = []
            for batch, m, gm, s in results:
                buffer.extend(batch)
                logs['rew'].extend(m['rew']);
                logs['spd'].extend(m['spd']);
                logs['col'].extend(m['col'])
                grp_means.append(gm);
                ep += GROUP_SIZE;
                step += s

            global_base = np.mean(grp_means) if global_base == 0 else ema * global_base + (1 - ema) * np.mean(grp_means)

            if len(buffer) >= BATCH_SIZE:
                learner.learn(buffer, step)
                buffer = []
                print(f"Ep {ep} | Step {step} | Rew {np.mean(logs['rew'][-100:]):.1f}")
                if (ep // GROUP_SIZE) % 50 == 0:
                    torch.save(learner.actor.state_dict(), f"{save_dir}/weights.pth")
                    for k, v in logs.items(): np.save(f"{save_dir}/{k}.npy", v)
    finally:
        ray.shutdown()


if __name__ == '__main__': train()