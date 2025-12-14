更新日志 (Changelog)

本项目旨在将组相对策略优化 (GRPO) 算法应用于自动驾驶场景，并引入风险感知的动态约束机制。以下是针对算法逻辑、环境配置及代码稳定性的主要更新记录。

[2025-11-24] - 核心算法重构与环境修复

🚀 核心算法更新 (Core Algorithm Updates)

引入同状态组采样 (Same-State Group Sampling):

为了解决 GRPO 在高随机性交通流中估值方差过大的问题，重写了 train_grpo.py 和 train_grpo_ablation.py 中的采样逻辑。

机制: 每次采集一组 (GROUP_SIZE=4) 轨迹时，强制使用相同的随机种子 (env.reset(seed=group_seed)) 重置环境。

目的: 确保组内轨迹面临完全相同的初始路况，从而使优势函数 (Advantage) 的计算基于策略本身的差异，而非环境的随机性。

完善动态风险机制 (Dynamic Risk Mechanism):

在 models/agent_grpo.py 中补全了 calculate_risk 函数。

实现了基于周围车辆距离的动态 Beta 调整逻辑：风险越高（距离越近），KL 散度惩罚系数 (self.beta) 越大，强制策略保守更新。

🛠 环境与奖励函数优化 (Environment & Rewards)

奖励函数重塑 (Reward Shaping):

碰撞惩罚: 从激进的 -300.0 调整为 -50.0，降低稀疏负奖励对训练的打击。

距离惩罚: 将原本不稳定的倒数惩罚 (1/d) 改为线性惩罚 (-1.0 * (1 - d/25.0))，防止车辆过近时产生数值爆炸。

正向引导: 增加了归一化的速度奖励和存活奖励，确保在无碰撞情况下 Agent 能获得正向收益，解决训练曲线长期为负的问题。

Gym 兼容性修复:

在 custom_env.py 的注册代码中移除了 max_episode_steps=1000 参数。

解决问题: 修复了 test_model.py 中出现的 ValueError: too many values to unpack 错误，防止 Gym 自动添加不兼容的旧版 TimeLimit 包装器。

🐛 Bug 修复 (Bug Fixes)

Agent 属性缺失修复:

在 models/agent_grpo.py 的 __init__ 方法中补充了 self.eps_clip = eps_clip。

解决问题: 修复了训练时报错 AttributeError: 'Agent_GRPO' object has no attribute 'eps_clip'。

消融实验对齐:

更新了 experiments/train_grpo_ablation.py，使其与主实验采用完全相同的“同状态组采样”策略，仅关闭动态 Beta 功能 (use_dynamic_beta=False)，确保实验对比的公平性和有效性。

[实验指南] 接下来的步骤

清理旧数据: 建议删除 results/ 文件夹下的旧日志，避免不同奖励尺度的数据混淆。

运行主实验: python experiments/train_grpo.py --seed 1

运行消融实验: python experiments/train_grpo_ablation.py --seed 1

运行基线 (PPO): python experiments/train_ppo.py --seed 1

绘图: 使用 analysis/plot_learning_curves.py 查看对比结果（预期 GRPO 曲线将稳步上升且优于消融版本）。