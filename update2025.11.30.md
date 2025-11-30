📅 项目更新日志：GRPO 算法与 Highway 环境深度优化版

  版本摘要：本次更新集中解决了 GRPO 算法在自动驾驶场景中“收敛慢”、“不敢超车”以及“对幽灵车辆误判”的核心问题。通过重构风险计算逻辑（TTC Fusion）、优化动态 Beta 约束机制以及调整环境奖励函数，实现了在保证安全的前提下显著提升行驶速度和训练稳定性。
  
🛠️ 1. 环境层优化 (Environment - custom_env.py)

  💰 奖励函数重构 (Reward Shaping)
  
    降低速度奖励：从 2.0 降至 1.0。避免智能体通过“超速行驶”来获取高分，让智能体更容易活到最后。
    
    消除低速死区：速度奖励范围从 [20, 30] 放宽至 [10, 30]。让智能体在低速起步阶段也能获得正反馈，避免因起步无奖励而陷入停滞。
    
    减小环境复杂程度：将 duration 减小到150，将 vehicles_count 减小到10。让智能体更容易活到最后, 建立正向反馈循环
    
  🐛 兼容性与注册修复
  
    Gymnasium 迁移：彻底修复 gym 与 gymnasium 的混用问题，元数据键名更新为 render_modes 和 render_fps，消除所有黄色警告。
    
    注册表冲突修复：环境 ID 更名为 my-highway-v0 并增加 if not in registry 检查，彻底解决多进程 (multiprocessing) 下的重复注册报错。
    
🧠 2. 算法层升级 (Agent - models/agent_grpo.py)

  🛡️ 风险感知系统重构 (Physics-Aware Risk)
  
    TTC + 距离融合：摒弃单纯的距离判断，引入 TTC (Time-to-Collision) 逻辑。允许在相对静止时近距离跟车（低风险），但在高速对冲时提前预警（高风险）。
    
    幽灵车辆过滤 (Ghost Masking)：修复了填充数据（坐标 0,0）导致 Risk 爆表的问题。增加了 dist > 0.1 和 dist > 0.6 的双重掩码，过滤掉 Padding 数据和已碰撞数据。
    
    Tanh 归一化映射：使用 tanh(raw_risk * 0.2) 将物理风险平滑映射到 [0, 1] 区间，避免梯度突变。
    
  📉 动态 Beta 机制反转 (Inverse Beta Decay)
  
    逻辑反转：从“越危险越保守”改为“越危险越激进”。新公式：$\beta = \beta_{init} / (1 + Risk)$。当检测到高风险时，大幅降低 KL 散度约束，允许策略网络剧烈更新（如急刹、急转）以进行紧急避险。
    
  📉 优势函数修正
  
    代码优化：将优势函数计算移动到 models/agent_grpo.py 中，定义为 calculate_group_advantages，将 GROUP_SIZE 移动到 config 中，增强代码的可读性与可维护性。
    
    绝对失败惩罚优化：计算绝对惩罚掩码时，使用 np.minimum 做截断，而不是强制赋值。
    放大优势值：将优势放大为原来的 5.0 倍，让对奖励更敏感。
    
⚙️ 3. 超参数调优 (Config - config.py)

  🚀 训练加速与收敛
  
    区分熵系数：将 GRPO 与 PPO 熵系数分开为 GRPO_ENTROPY_COEF = 0.001 与 PPO_ENTROPY_COEF = 0.01。解决了 GRPO 因缺乏 Critic 指引而导致的“犹豫不决”（高熵值停滞）问题。
    
    学习率策略调整：DECAY_RATE 提升至 0.9999，减缓衰减速度，确保训练后期仍有足够的梯度更新动力。
