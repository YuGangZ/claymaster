import os
import time
from datetime import datetime
import numpy as np
import json
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.core.rl_env_wrapper import DeformationRLEnv
from rl.config.rl_config import RL_CONFIG


class PreciseTrainingCallback(BaseCallback):
    """增强版训练回调 - 更详细的日志输出和保存功能"""

    def __init__(self, verbose=0, log_dir="./rl_training_logs", save_interval=100):
        super(PreciseTrainingCallback, self).__init__(verbose)
        self.distance_history = []
        self.uncertainty_history = []
        self.reward_history = []
        self.episode_rewards = []
        self.episode_rewards_raw = []  # 原始奖励记录
        self.current_episode_reward = 0
        self.episode_lengths = []
        self.current_episode_length = 0
        self.best_reward = -float('inf')
        self.start_time = None

        # 新增：日志记录配置
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.save_interval = save_interval

        # 新增：训练日志数据结构
        self.training_logs = {
            'episode_rewards': [],  # 每个episode的总奖励
            'episode_lengths': [],  # 每个episode的步数
            'exploration_rates': [],  # 探索率（如果有）
            'loss_history': [],  # 损失历史
            'success_history': [],  # 成功历史（1=成功，0=失败）
            'timestamps': [],  # 时间戳
            'training_config': RL_CONFIG  # 训练配置
        }

        # 新增：动作历史（用于分析）
        self.action_history = []
        self.state_history = []

        # 新增：统计计数器
        self.success_count = 0
        self.total_episodes = 0

        # 新增：训练数据文件
        self.training_data_file = self.log_dir / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _on_training_start(self) -> None:
        """训练开始时调用"""
        self.start_time = time.time()
        print("\n" + "=" * 60)
        print("🚀 RL训练开始!")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总步数: {self.locals.get('total_timesteps', '未知')}")
        print(f"日志目录: {self.log_dir}")
        print("=" * 60)

        # 保存初始配置
        self._save_training_logs()

    def _on_step(self) -> bool:
        # 累积当前episode的奖励
        if 'reward' in self.locals:
            self.current_episode_reward += self.locals['reward'][0]
            self.current_episode_length += 1

            # 记录奖励历史
            self.reward_history.append(self.locals['reward'][0])

        # 检查episode是否结束
        if 'done' in self.locals and self.locals['done'][0]:
            self.total_episodes += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_rewards_raw.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            # 判断是否成功（根据奖励阈值）
            success = self.current_episode_reward > 5.0  # 成功奖励阈值
            if success:
                self.success_count += 1

            # 更新最佳奖励
            if self.current_episode_reward > self.best_reward:
                self.best_reward = self.current_episode_reward
                # 保存最佳模型检查点
                self._save_best_model()

            # 记录训练日志
            self.training_logs['episode_rewards'].append(float(self.current_episode_reward))
            self.training_logs['episode_lengths'].append(self.current_episode_length)
            self.training_logs['success_history'].append(1 if success else 0)
            self.training_logs['timestamps'].append(time.time() - self.start_time)

            # 记录损失（从模型中获取）
            if hasattr(self.model, 'loss_history'):
                self.training_logs['loss_history'].append(
                    float(self.model.loss_history[-1]) if self.model.loss_history else 0.0)

            # 每完成一个episode打印详细日志
            print(f"\n🎯 Episode {len(self.episode_rewards)} 完成:")
            print(f"   奖励: {self.current_episode_reward:.2f}")
            print(f"   长度: {self.current_episode_length} 步")
            print(f"   是否成功: {'是' if success else '否'}")
            print(f"   平均奖励 (最近10个): {np.mean(self.episode_rewards[-10:]):.2f}" if len(
                self.episode_rewards) >= 10 else "")
            print(f"   最佳奖励: {self.best_reward:.2f}")
            print(
                f"   成功率: {self.success_count}/{self.total_episodes} ({self.success_count / max(1, self.total_episodes):.1%})")

            # 定期保存训练日志
            if len(self.training_logs['episode_rewards']) % self.save_interval == 0:
                self._save_training_logs()
                print(f"📁 训练日志已保存到: {self.training_data_file}")

            # 重置episode计数器
            self.current_episode_reward = 0
            self.current_episode_length = 0

        # 定期打印训练进度 (每100步)
        if self.n_calls % 100 == 0:
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(
                    self.episode_rewards)
                avg_length = np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else np.mean(
                    self.episode_lengths)

                elapsed_time = time.time() - self.start_time
                steps_per_sec = self.n_calls / elapsed_time if elapsed_time > 0 else 0

                print(f"\n📊 训练进度 [{self.n_calls}步 | {elapsed_time:.1f}s | {steps_per_sec:.1f}步/秒]:")
                print(f"   最近10个episode平均奖励: {avg_reward:.2f}")
                print(f"   最近10个episode平均长度: {avg_length:.1f}步")
                print(f"   已完成episode数: {len(self.episode_rewards)}")
                print(f"   最佳奖励: {self.best_reward:.2f}")
                print(f"   总成功率: {self.success_count / max(1, self.total_episodes):.1%}")

        # 每100步打印更详细的信息
        if self.n_calls % 100 == 0:
            print(f"\n{'=' * 50}")
            print(f"📈 检查点 - {self.n_calls}步")
            print(f"{'=' * 50}")

            if len(self.model.ep_info_buffer) > 0:
                recent_rewards = [info['r'] for info in self.model.ep_info_buffer if 'r' in info]
                recent_lengths = [info['l'] for info in self.model.ep_info_buffer if 'l' in info]

                if recent_rewards:
                    print(f"   最近奖励: {recent_rewards}")
                    print(f"   平均奖励: {np.mean(recent_rewards):.2f}")
                if recent_lengths:
                    print(f"   最近长度: {recent_lengths}")
                    print(f"   平均长度: {np.mean(recent_lengths):.1f}")

            # 保存检查点
            self._save_checkpoint()

        return True

    def _save_training_logs(self):
        """保存训练日志到文件"""
        try:
            # 计算滑动窗口成功率
            if len(self.training_logs['success_history']) > 0:
                window_size = 10
                success_rates = []
                for i in range(len(self.training_logs['success_history'])):
                    start_idx = max(0, i - window_size + 1)
                    window_success = self.training_logs['success_history'][start_idx:i + 1]
                    success_rates.append(np.mean(window_success))
                self.training_logs['rolling_success_rate'] = success_rates

            # 计算探索率（基于模型的探索策略）
            if hasattr(self.model, 'exploration_rate'):
                self.training_logs['exploration_rates'] = self.model.exploration_rate

            with open(self.training_data_file, 'w') as f:
                json.dump(self.training_logs, f, indent=2, default=self._json_serializer)
        except Exception as e:
            print(f"保存训练日志失败: {e}")

    def _save_best_model(self):
        """保存最佳模型检查点"""
        try:
            checkpoint_path = self.checkpoint_dir / f"best_model_{self.best_reward:.2f}.zip"
            self.model.save(checkpoint_path)
            print(f"💾 最佳模型已保存: {checkpoint_path}")
        except Exception as e:
            print(f"保存最佳模型失败: {e}")

    def _save_checkpoint(self):
        """保存训练检查点"""
        try:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.n_calls}.zip"
            self.model.save(checkpoint_path)

            # 同时保存训练状态
            checkpoint_data = {
                'total_timesteps': self.n_calls,
                'episode_count': len(self.episode_rewards),
                'best_reward': self.best_reward,
                'success_rate': self.success_count / max(1, self.total_episodes),
                'timestamp': datetime.now().isoformat()
            }

            checkpoint_state_path = self.checkpoint_dir / f"checkpoint_{self.n_calls}_state.json"
            with open(checkpoint_state_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            print(f"保存检查点失败: {e}")

    def _json_serializer(self, obj):
        """JSON序列化辅助函数"""
        if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if 'float' in str(type(obj)) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    def _on_training_end(self) -> None:
        """训练结束时调用"""
        elapsed_time = time.time() - self.start_time

        # 最终保存训练日志
        self._save_training_logs()

        print("\n" + "=" * 60)
        print("✅ RL训练完成!")
        print(f"总训练步数: {self.n_calls}")
        print(f"总训练时间: {elapsed_time:.1f}秒")
        print(f"总episode数: {len(self.episode_rewards)}")
        print(
            f"总成功率: {self.success_count}/{self.total_episodes} ({self.success_count / max(1, self.total_episodes):.1%})")

        if len(self.episode_rewards) > 0:
            print(f"\n最终统计:")
            print(f"   平均奖励: {np.mean(self.episode_rewards):.2f}")
            print(f"   最佳奖励: {self.best_reward:.2f}")
            print(f"   平均长度: {np.mean(self.episode_lengths):.1f}步")
            print(f"   最终10个episode平均奖励: {np.mean(self.episode_rewards[-10:]):.2f}")

        print(f"\n训练日志已保存到: {self.training_data_file}")
        print("=" * 60)

        # 返回训练数据路径，便于后续可视化
        return self.training_data_file


class RLDeformationTrainer:
    """精确维度的RL训练器"""

    def __init__(self, motion_controller, target_shape, log_dir="./rl_training_logs"):
        self.motion_controller = motion_controller
        if len(target_shape) == 16:
            self.target_shape = self._convert_16d_to_14d(target_shape)
        else:
            self.target_shape = target_shape
        self.config = RL_CONFIG
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 创建环境
        self.env = self._create_env()
        self.model = None

        # 可视化器（可选）
        self.visualizer = None
    @staticmethod
    def _convert_16d_to_14d(state_16d):
        """将16维状态转换为14维状态（与mpc_controller.py一致）"""
        if len(state_16d) >= 16:
            return np.array([
                state_16d[0], state_16d[1], state_16d[2],    # scale (3)
                state_16d[3], state_16d[4],                  # shape (2)
                state_16d[5], state_16d[6], state_16d[7],    # translation (3)
                state_16d[8], state_16d[9], state_16d[10],   # rotation (3)
                state_16d[11],                               # volume (1)
                state_16d[12],                               # elongation (1)
                state_16d[14]                                # smoothness (1)
            ], dtype=np.float32)
        else:
            return state_16d  # 如果已经是14维或更少，直接返回
    def _create_env(self):
        """创建RL环境"""

        def make_env():
            env = DeformationRLEnv(
                motion_controller=self.motion_controller,
                target_shape=self.target_shape,
                config=self.config
            )
            return env

        return DummyVecEnv([make_env])

    def initialize_policy(self):
        """初始化策略"""
        self.model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.config["learning_rate"],
            buffer_size=self.config["buffer_size"],
            learning_starts=self.config["learning_starts"],
            batch_size=self.config["batch_size"],
            tau=self.config["tau"],
            gamma=self.config["gamma"],
            train_freq=self.config["train_freq"],
            gradient_steps=self.config["gradient_steps"],
            policy_kwargs=self.config["policy_kwargs"],
            verbose=1,
            tensorboard_log=str(self.log_dir / "tensorboard")
        )

        print("精确维度RL策略初始化完成!")
        print(f"观测空间: {self.env.observation_space.shape}")
        print(f"动作空间: {self.env.action_space.shape}")
        print(f"日志目录: {self.log_dir}")

        return self.model

    def train(self, total_timesteps=50000, visualize=True, save_checkpoints=True):
        """训练RL策略"""
        if self.model is None:
            self.initialize_policy()

        # 创建回调（启用日志记录）
        callback = PreciseTrainingCallback(
            verbose=1,
            log_dir=self.log_dir,
            save_interval=50  # 每50个episode保存一次
        )

        print("开始精确维度RL训练...")
        print(f"总步数: {total_timesteps}")
        print(f"日志保存到: {self.log_dir}")
        print(f"可视化: {'启用' if visualize else '禁用'}")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10
        )

        # 保存最终模型
        final_model_path = self.log_dir / "final_model.zip"
        self.model.save(str(final_model_path))
        print(f"训练完成! 最终模型已保存: {final_model_path}")

        # 训练结束后自动可视化
        if visualize:
            self.visualize_training_results()

        # 返回训练数据路径
        return callback.training_data_file

    def visualize_training_results(self):
        """训练结束后可视化结果"""
        try:
            # 导入可视化器（延迟导入，避免不必要的依赖）
            from rl_visualizer import RLTrainingVisualizer

            # 查找最新的训练日志文件
            log_files = list(self.log_dir.glob("training_data_*.json"))
            if not log_files:
                print("未找到训练日志文件，跳过可视化")
                return

            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)

            # 创建可视化器
            self.visualizer = RLTrainingVisualizer(
                training_log_path=latest_log,
                output_dir=self.log_dir / "visualizations"
            )

            # 生成所有可视化图表
            print("\n" + "=" * 60)
            print("生成训练可视化图表...")
            print("=" * 60)

            # 训练进度图
            self.visualizer.plot_training_progress()

            # 训练报告
            self.visualizer.create_training_report()

            # 如果需要，还可以添加更多可视化
            # self.visualizer.plot_learning_curves_comparison()
            # self.visualizer.plot_action_distribution(actions_history)

            print(f"可视化图表已保存到: {self.log_dir / 'visualizations'}")

        except ImportError as e:
            print(f"无法导入可视化模块: {e}")
            print("请确保 rl_visualizer.py 在同一目录下")
        except Exception as e:
            print(f"可视化失败: {e}")

    def evaluate(self, num_episodes=10, visualize=True):
        """评估策略并生成可视化报告"""
        if self.model is None:
            print("请先加载或训练模型!")
            return

        print(f"\n开始评估策略，运行 {num_episodes} 个episode...")

        successes = 0
        total_rewards = []
        final_distances = []
        avg_uncertainties = []

        # 存储详细的评估数据
        evaluation_data = {
            'episodes': [],
            'actions_history': []
        }

        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_uncertainty = 0
            done = False
            steps = 0

            episode_actions = []
            episode_states = []

            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_actions.append(action[0].copy())  # 保存动作
                episode_states.append(obs[0][:16].copy())  # 保存状态

                if 'uncertainty' in info[0]:
                    episode_uncertainty += np.mean(info[0]['uncertainty'])
                steps += 1

                if done and reward > 5:  # 成功奖励阈值
                    successes += 1

            # 记录最终状态
            if 'distance_to_target' in info[0]:
                final_distances.append(info[0]['distance_to_target'])

            avg_uncertainty = episode_uncertainty / steps if steps > 0 else 0
            avg_uncertainties.append(avg_uncertainty)
            total_rewards.append(episode_reward)

            # 保存episode数据
            evaluation_data['episodes'].append({
                'episode': episode,
                'reward': float(episode_reward),
                'steps': steps,
                'final_distance': info[0].get('distance_to_target', 0),
                'success': (done and reward > 5)
            })

            # 保存动作历史（前3个episode）
            if episode < 3:
                evaluation_data['actions_history'].extend(episode_actions)

            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
                  f"Final Distance = {info[0].get('distance_to_target', 0):.3f}, "
                  f"Avg Uncertainty = {avg_uncertainty:.4f}")

        # 计算评估指标
        success_rate = successes / num_episodes
        avg_reward = np.mean(total_rewards)
        avg_distance = np.mean(final_distances)
        avg_uncertainty = np.mean(avg_uncertainties)

        print(f"\n评估结果:")
        print(f"成功率: {success_rate:.1%}")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"平均最终距离: {avg_distance:.4f}")
        print(f"平均不确定度: {avg_uncertainty:.4f}")

        # 保存评估结果
        eval_results = {
            'success_rate': float(success_rate),
            'average_reward': float(avg_reward),
            'average_final_distance': float(avg_distance),
            'average_uncertainty': float(avg_uncertainty),
            'num_episodes': num_episodes,
            'episode_details': evaluation_data['episodes']
        }

        eval_file = self.log_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)

        print(f"评估结果已保存: {eval_file}")

        # 生成评估可视化
        if visualize and evaluation_data['actions_history']:
            try:
                from rl.eval.rl_visualizer import RLTrainingVisualizer
                visualizer = RLTrainingVisualizer(
                    output_dir=self.log_dir / "evaluation_visualizations"
                )

                # 绘制动作分布
                visualizer.plot_action_distribution(evaluation_data['actions_history'])

                # 如果有episode数据，绘制episode分析
                if len(evaluation_data['episodes']) > 0:
                    episode_data = {
                        'states': episode_states if 'episode_states' in locals() else [],
                        'rewards': [ep['reward'] for ep in evaluation_data['episodes']],
                        'actions': evaluation_data['actions_history'],
                        'target_state': self.target_shape.tolist()
                    }
                    visualizer.plot_episode_analysis(episode_data)

                print(f"评估可视化已保存: {self.log_dir / 'evaluation_visualizations'}")

            except Exception as e:
                print(f"评估可视化失败: {e}")

        return success_rate, avg_reward, avg_distance, avg_uncertainty

    def load_model(self, model_path):
        """加载已有模型"""
        self.model = SAC.load(model_path)
        print(f"模型已加载: {model_path}")
        return self.model


# 新增：独立训练脚本接口
def run_rl_training(motion_controller, target_shape, total_timesteps=50000,
                    log_dir="./rl_training_logs", visualize=True):
    """
    运行RL训练的便捷函数

    Args:
        motion_controller: 运动控制器实例
        target_shape: 目标形状
        total_timesteps: 总训练步数
        log_dir: 日志目录
        visualize: 是否生成可视化

    Returns:
        训练数据文件路径
    """
    # 创建训练器
    trainer = RLDeformationTrainer(
        motion_controller=motion_controller,
        target_shape=target_shape,
        log_dir=log_dir
    )

    # 训练
    training_data_path = trainer.train(
        total_timesteps=total_timesteps,
        visualize=visualize
    )

    # 评估
    print("\n" + "=" * 60)
    print("开始模型评估...")
    print("=" * 60)

    success_rate, avg_reward, avg_distance, avg_uncertainty = trainer.evaluate(
        num_episodes=10,
        visualize=visualize
    )

    print(f"\n最终评估结果:")
    print(f"  成功率: {success_rate:.1%}")
    print(f"  平均奖励: {avg_reward:.2f}")
    print(f"  平均最终距离: {avg_distance:.4f}")

    return training_data_path, trainer