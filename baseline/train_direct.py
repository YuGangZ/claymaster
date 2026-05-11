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
import time

# 添加项目根目录到path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from physical_engine.soft_sim_env import SoftSimEnv
from baseline.motion_controller_direct import DirectMotionController
from baseline.rl_env_direct import DeformationRLEnvDirect
from baseline.config import DIRECT_RL_CONFIG


class DirectRLTrainingCallback(BaseCallback):
    """纯RL训练回调，保存详细训练日志（类似PreciseTrainingCallback）"""
    def __init__(self, log_dir="./direct_rl_logs", save_interval=100, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_interval = save_interval
        os.makedirs(log_dir, exist_ok=True)

        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.total_episodes = 0
        self.best_reward = -float('inf')

        # 详细日志结构
        self.training_logs = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_history': [],
            'timestamps': [],
            'training_config': DIRECT_RL_CONFIG
        }
        self.training_data_file = os.path.join(log_dir, f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _on_training_start(self):
        self.start_time = time.time()
        print("\n" + "="*60)
        print("纯RL训练开始 (直接速度控制)")
        print(f"日志目录: {self.log_dir}")
        print(f"详细日志将保存到: {self.training_data_file}")
        print("="*60)
        self._save_logs()

    def _on_step(self):
        if self.locals.get('dones', [False])[0]:
            self.total_episodes += 1
            info = self.locals.get('infos', [{}])[0]
            episode_reward = info.get('episode', {}).get('r', 0)
            episode_length = info.get('episode', {}).get('l', 0)
            distance = info.get('distance_to_target', 1.0)
            success = distance < 0.05

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            if success:
                self.success_count += 1

            # 保存详细日志
            self.training_logs['episode_rewards'].append(float(episode_reward))
            self.training_logs['episode_lengths'].append(episode_length)
            self.training_logs['success_history'].append(1 if success else 0)
            self.training_logs['timestamps'].append(time.time() - self.start_time)

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.model.save(os.path.join(self.checkpoint_dir, f"best_model_{self.best_reward:.2f}.zip"))

            if self.total_episodes % self.save_interval == 0:
                self._save_logs()
                print(f"📁 训练日志已保存到: {self.training_data_file}")

            # 打印进度
            avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
            print(f"\nEpisode {self.total_episodes}: reward={episode_reward:.2f}, len={episode_length}, "
                  f"success={success}, avg_reward(10)={avg_reward:.2f}, best={self.best_reward:.2f}")

        return True

    def _save_logs(self):
        try:
            # 计算滑动窗口成功率
            if len(self.training_logs['success_history']) > 0:
                window_size = 10
                success_rates = []
                for i in range(len(self.training_logs['success_history'])):
                    start_idx = max(0, i - window_size + 1)
                    window_success = self.training_logs['success_history'][start_idx:i+1]
                    success_rates.append(np.mean(window_success))
                self.training_logs['rolling_success_rate'] = success_rates

            with open(self.training_data_file, 'w') as f:
                json.dump(self.training_logs, f, indent=2, default=self._json_serializer)
        except Exception as e:
            print(f"保存训练日志失败: {e}")

    def _json_serializer(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if 'float' in str(type(obj)) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    def _on_training_end(self):
        self._save_logs()
        elapsed = time.time() - self.start_time
        print(f"\n训练完成。总episodes: {self.total_episodes}, 成功率: {self.success_count/self.total_episodes:.2%}")
        print(f"最佳奖励: {self.best_reward:.2f}")
        print(f"详细日志保存在: {self.training_data_file}")
        print(f"总训练时间: {elapsed:.1f}秒")


def main():
    parser = argparse.ArgumentParser(description="纯RL训练 (直接速度控制)")
    parser.add_argument('--total-timesteps', type=int, default=50000, help='总训练步数（环境步数）')
    parser.add_argument('--control-steps', type=int, default=25, help='每个RL动作对应的物理步数')
    parser.add_argument('--log-dir', type=str, default="./direct_rl_logs", help='日志保存目录')
    args = parser.parse_args()

    DIRECT_RL_CONFIG["control_steps_per_env_step"] = args.control_steps

    # 初始化物理引擎
    engine = SoftSimEnv()
    scene, solver, sensor_cube, elastoplastic_obj = engine.setup_simulation("box")
    initial_particles = engine.initialize_cube_particles()

    # 创建纯RL运动控制器（修改后支持阶段管理）
    motion_controller = DirectMotionController(
        scene, sensor_cube, elastoplastic_obj, initial_particles,
        output_dir=args.log_dir,
        estimation_interval=25,
        predictor_model_path="shape_predictor.pth"   # 确保路径正确
    )

    # 目标形状
    FINAL_TARGET = np.array([
        0.0932, 0.0932, 0.0932,
        1.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.002375,
        1.0, 1.0
    ], dtype=np.float32)

    # 创建环境
    env = DeformationRLEnvDirect(motion_controller, FINAL_TARGET, DIRECT_RL_CONFIG)
    vec_env = DummyVecEnv([lambda: env])

    # SAC策略
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

    callback = DirectRLTrainingCallback(log_dir=args.log_dir, save_interval=50)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(os.path.join(args.log_dir, "final_model.zip"))
    print(f"模型已保存至 {args.log_dir}")

if __name__ == "__main__":
    main()