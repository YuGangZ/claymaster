# baseline/train_direct.py
import os
import sys
import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import json
from datetime import datetime

# 添加项目根目录到path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from physical_engine.soft_sim_env import SoftSimEnv
from baseline.motion_controller_direct import DirectMotionController
from baseline.rl_env_direct import DeformationRLEnvDirect
from baseline.config import DIRECT_RL_CONFIG


class DirectRLTrainingCallback(BaseCallback):
    """纯RL训练回调，记录奖励和成功率"""
    def __init__(self, log_dir="./direct_rl_logs", save_interval=100, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_interval = save_interval
        os.makedirs(log_dir, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.total_episodes = 0
        self.best_reward = -float('inf')

    def _on_training_start(self):
        print("\n" + "="*60)
        print("纯RL训练开始 ")
        print(f"日志目录: {self.log_dir}")
        print("="*60)

    def _on_step(self):
        # 检测episode结束
        if self.locals.get('dones', [False])[0]:
            self.total_episodes += 1
            info = self.locals.get('infos', [{}])[0]
            episode_reward = info.get('episode', {}).get('r', 0)
            episode_length = info.get('episode', {}).get('l', 0)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # 判断成功（根据最终距离）
            distance = info.get('distance_to_target', 1.0)
            success = distance < 0.05
            if success:
                self.success_count += 1

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.model.save(os.path.join(self.log_dir, "best_model.zip"))

            if self.total_episodes % self.save_interval == 0:
                self._save_logs()

            # 打印进度
            avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
            print(f"\nEpisode {self.total_episodes}: reward={episode_reward:.2f}, len={episode_length}, "
                  f"success={success}, avg_reward(10)={avg_reward:.2f}, best={self.best_reward:.2f}")
        return True

    def _save_logs(self):
        logs = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_count': self.success_count,
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward,
            'config': DIRECT_RL_CONFIG
        }
        with open(os.path.join(self.log_dir, "training_logs.json"), 'w') as f:
            json.dump(logs, f, indent=2)

    def _on_training_end(self):
        self._save_logs()
        print(f"\n训练完成。总episodes: {self.total_episodes}, 成功率: {self.success_count/self.total_episodes:.2%}")
        print(f"最佳奖励: {self.best_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description="纯RL训练 (直接速度控制)")
    parser.add_argument('--total-timesteps', type=int, default=50000, help='总训练步数（环境步数）')
    parser.add_argument('--control-steps', type=int, default=25, help='每个RL动作对应的物理步数')
    parser.add_argument('--log-dir', type=str, default="./direct_rl_logs", help='日志保存目录')
    parser.add_argument('--eval-interval', type=int, default=5000, help='评估间隔（环境步数）')
    args = parser.parse_args()

    # 更新配置中的控制步数
    DIRECT_RL_CONFIG["control_steps_per_env_step"] = args.control_steps

    # 1. 初始化物理引擎
    engine = SoftSimEnv()
    scene, solver, sensor_cube, elastoplastic_obj = engine.setup_simulation("box")
    initial_particles = engine.initialize_cube_particles()

    # 2. 创建纯RL运动控制器
    motion_controller = DirectMotionController(
        scene, sensor_cube, elastoplastic_obj, initial_particles,
        output_dir=args.log_dir,
        estimation_interval=25
    )

    # 3. 定义目标形状
    FINAL_TARGET = np.array([
        0.0932, 0.0932, 0.0932,  # scale
        1.0, 1.0,                # shape epsilon
        0.0, 0.0, 0.0,           # translation
        0.0, 0.0, 0.0,           # euler
        0.002375,                # volume
        1.0, 1.0                 # elongation, smoothness
    ], dtype=np.float32)

    # 4. 创建RL环境
    env = DeformationRLEnvDirect(motion_controller, FINAL_TARGET, DIRECT_RL_CONFIG)
    vec_env = DummyVecEnv([lambda: env])

    # 5. 初始化SAC策略
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=DIRECT_RL_CONFIG["learning_rate"],
        buffer_size=DIRECT_RL_CONFIG["buffer_size"],
        learning_starts=DIRECT_RL_CONFIG["learning_starts"],
        batch_size=DIRECT_RL_CONFIG["batch_size"],
        tau=DIRECT_RL_CONFIG["tau"],
        gamma=DIRECT_RL_CONFIG["gamma"],
        train_freq=DIRECT_RL_CONFIG["train_freq"],
        gradient_steps=DIRECT_RL_CONFIG["gradient_steps"],
        policy_kwargs=DIRECT_RL_CONFIG["policy_kwargs"],
        verbose=1,
        tensorboard_log=os.path.join(args.log_dir, "tensorboard")
    )

    # 6. 训练
    callback = DirectRLTrainingCallback(log_dir=args.log_dir, save_interval=50)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    # 7. 保存最终模型
    model.save(os.path.join(args.log_dir, "final_model.zip"))
    print(f"模型已保存至 {args.log_dir}")


if __name__ == "__main__":
    main()