# baseline/config.py
import numpy as np

# 纯RL训练配置（
DIRECT_RL_CONFIG = {
    # 网络结构
    "policy_kwargs": dict(net_arch=[256, 256]),

    # 训练参数
    "learning_rate": 3e-4,
    "buffer_size": 10000,
    "learning_starts": 200,
    "batch_size": 128,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 2,

    # 环境参数
    "max_episode_steps": 100,
    "control_steps_per_env_step": 25,

    # 控制维度
    "control_dim": 3,  # 3维速度
    "state_dim": 14,
    "uncertainty_dim": 14,

    # 奖励权重
    "reward_weights": {
        "goal": 10.0,
        "time": -1.0,
        "safety": -10.0,
        "success": 100.0,
        "uncertainty": -5.0
    },

    # 动作边界
    "action_low": -0.3,
    "action_high": 0.3,
}