# Dynamic Constraint-Based GRPO for Autonomous Driving

This repository contains the official implementation of **"Dynamic Constraint-Based GRPO with Tiered Rewards for Safe Merging Scenarios"**.

We propose a **Critic-Free** reinforcement learning framework adapted from DeepSeek's GRPO. It is designed to solve "Reward Hacking" and safety issues in high-density highway merging tasks by introducing tiered advantages and physical risk constraints.

## 🚀 Key Innovations

### 1\. Tiered Advantage Estimation (Survival \> Efficiency)

Instead of standard normalization, we calculate Group Relative Advantage based on a hierarchy of objectives:

  * **Tier 1 (Survival):** Strict separation between **Crashed (-5.0)** and **Survived (+2.0)** trajectories.
  * **Tier 2 (Efficiency):** Standard normalization is applied only among surviving trajectories.
  * **Tier 3 (Evolution):** An **EMA (Exponential Moving Average) Baseline** is introduced as a trend signal to reward groups that outperform historical averages.

### 2\. Risk-Aware Dynamic $\beta$

We dynamically adjust the PPO KL-divergence penalty ($\beta$) based on real-time physical risk:

  * **Risk Calculation:** Derived from **Time-to-Collision (TTC)** and relative velocity using a `tanh` normalized scale.
  * **Adaptive Constraint:**
      * **High Risk:** $\beta \to \beta_{max}$ (Forces conservative policy updates).
      * **Low Risk:** $\beta \to \beta_{min}$ (Encourages exploration).

### 3\. Hard-Constraint Action Masking

To ensure fundamental compliance, the `act` function implements rule-based masking to strictly forbid illegal actions (e.g., accelerating beyond 32m/s) before the probability distribution layer.

## 🛣️ Environment: Custom MergeEnv

We customized the `highway-env` to address the "ramp camping" reward hacking issue:

  * **Action Shaping:** Positive reinforcement for `LANE_LEFT` attempts on the ramp; strict penalties for `IDLE` or `LANE_RIGHT` in critical zones.
  * **Global Risk Penalty:** Risk penalties are calculated globally, not just after merging, forcing the agent to yield while still on the ramp.

## 📂 Structure

  * `models/agent_grpo.py`: Core GRPO logic with Tiered Advantage & Dynamic Beta.
  * `envs/custom_merge_env.py`: Custom environment with Urgency & Action Shaping.
  * `run_experiments.py`: Multi-process training loop (Master-Worker architecture).
  * `config/grpo_config.py`: Hyperparameters.

## ⚡ Quick Start

### Installation

```bash
pip install highway-env torch numpy pandas seaborn gymnasium
```

### Training

Train with our proposed method (Tiered Advantage):

```bash
python run_experiments.py --mode tiered --seed 0
```

Train with Standard GRPO (Ablation):

```bash
python run_experiments.py --mode standard --seed 0
```

### Evaluation

Visualize the trained agent:

```bash
python evaluate.py --weights results/grpo_main/seed_0/weights.pth --render
```

## 📝 Citation

If you find this code useful, please cite our work:

```bibtex
@article{dynamic_grpo_2025,
  title={Dynamic Constraint-Based GRPO with Tiered Rewards for Safe Autonomous Driving},
  author={Your Name et al.},
  journal={arXiv preprint},
  year={2025}
}
```