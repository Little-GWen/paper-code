Dynamic Constraint-Based GRPO for Autonomous Driving

This repository contains the official implementation of the paper: "Dynamic Constraint-Based GRPO for Safe and Efficient Lane-Keeping and Changing in Four-Lane Highway Scenarios".

🚗 Introduction

We propose a novel Critic-Free reinforcement learning framework adapted from DeepSeek's GRPO, specifically designed for high-density autonomous driving scenarios.

Key Features:

Critic-Free Architecture: Eliminates value estimation bias.

Dynamic $\beta$ Constraint: Adaptively balances safety and exploration based on real-time risk assessment.

Group Relative Advantage: Utilizes group-based statistics for stable policy updates.

📂 Structure

models/: Core algorithms (GRPO, PPO, DQN).

experiments/: Training scripts for different baselines.

custom_env.py: The 4-lane high-density highway environment.

🚀 Quick Start

1. Installation

pip install highway-env torch numpy pandas seaborn gym


2. Run Training

To reproduce the main results (GRPO vs PPO):

python run_experiments.py


3. Evaluation

Visualize the trained agent:

python evaluation/test_model.py --algo grpo --render


📊 Results

(Run analysis/plot_learning_curves.py to generate your comparison plot)

📝 Citation

If you find this code useful, please cite our paper.