import numpy as np

RL_CONFIG= {
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
    # 动作维度：15维（1维感知决策 + 14维子目标）
    "action_dim": 15,
    "action_scale": np.array([1.0] + [
        0.1, 0.1, 0.1,  # scale_a1,a2,a3
        0.2, 0.2,  # epsilon1,epsilon2
        0.05, 0.05, 0.05,  # translation_x,y,z
        0.1, 0.1, 0.1,  # euler_rx,ry,rz
        0.2, 0.2, 0.2 # 几何特征
    ]),

    # 奖励权重
    "reward_weights": {
        "goal": 10.0,
        "time": -1.0,
        "safety": -10.0,
        "success": 100.0,
        "uncertainty": -5.0
    },

    # 控制参数
    "control_dim": 3,
    "state_dim": 14,
    "uncertainty_dim": 14
}