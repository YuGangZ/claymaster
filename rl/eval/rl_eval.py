import numpy as np
import pandas as pd
import json
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from datetime import datetime
from stable_baselines3 import SAC
from rl.core.rl_env_wrapper import DeformationRLEnv
from rl.motion_controller_rl import MotionControllerRL
from physical_engine.soft_sim_env import SoftSimEnv


class RLEvaluator:
    def __init__(self, model_path, output_dir="rl_evaluation_results"):
        """
        RL策略评估器

        Args:
            model_path: 训练好的RL模型路径
            output_dir: 评估结果输出目录
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.model = None
        self.env = None
        self.results = {}

        # 设置绘图样式
        plt.style.use('seaborn-v0_8-darkgrid')
        rcParams['figure.figsize'] = (12, 8)
        rcParams['font.size'] = 12

    def setup_environment(self):
        """设置评估环境"""
        # 初始化物理引擎
        engine = SoftSimEnv()
        scene, solver, sensor_cube, elastoplastic_obj = engine.setup_simulation("box")
        initial_particles = engine.initialize_cube_particles()

        # 目标状态
        target_state = np.array([
            0.0932, 0.0932, 0.0932,  # scale_a1,2,3
            1.0, 1.0,  # shape_epsilon1,2
            0.0, 0.0, 0.0,  # translation_x,y,z
            0.0, 0.0, 0.0,  # euler_rx,ry,rz
            0.003375,  # volume
            1.0, 0.0, 1.0, 1.0  # geometric features
        ], dtype=np.float32)

        # 创建运动控制器
        motion_controller = MotionControllerRL(
            scene, sensor_cube, elastoplastic_obj, initial_particles,
            mpc_controller=None,
            output_dir="rl_eval_data"
        )

        # 创建RL环境
        from rl.config.rl_config import RL_CONFIG
        self.env = DeformationRLEnv(
            motion_controller=motion_controller,
            target_shape=target_state,
            config=RL_CONFIG
        )

        # 加载模型
        self.model = SAC.load(self.model_path)

        print("评估环境设置完成")
        print(f"模型: {self.model_path}")
        print(f"目标状态维度: {target_state.shape}")

    def evaluate_policy(self, num_episodes=20):
        """评估策略性能"""
        if self.model is None or self.env is None:
            self.setup_environment()

        print(f"开始评估，运行 {num_episodes} 个episode...")

        # 存储评估结果
        episode_results = []
        success_count = 0

        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

            obs, info = self.env.reset()
            episode_data = {
                'episode': episode,
                'rewards': [],
                'states': [],
                'actions': [],
                'uncertainties': [],
                'done': False,
                'success': False
            }

            total_reward = 0
            step = 0

            while not episode_data['done']:
                # 预测动作
                action, _states = self.model.predict(obs, deterministic=True)

                # 执行动作
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # 记录数据
                episode_data['rewards'].append(reward)
                episode_data['states'].append(obs[:16].copy())  # 当前状态
                episode_data['actions'].append(action.copy())
                if 'uncertainty' in info:
                    episode_data['uncertainties'].append(info['uncertainty'].copy())

                total_reward += reward
                obs = next_obs
                step += 1

                # 检查终止条件
                episode_data['done'] = terminated or truncated
                if terminated and reward > 5:  # 成功条件
                    episode_data['success'] = True
                    success_count += 1

            # 记录最终状态
            episode_data['total_reward'] = total_reward
            episode_data['steps'] = step
            episode_data['final_state'] = obs[:16].copy() if step > 0 else None

            episode_results.append(episode_data)

            print(f"  总奖励: {total_reward:.2f}, 步数: {step}, 成功: {episode_data['success']}")

        # 计算统计指标
        success_rate = success_count / num_episodes
        avg_reward = np.mean([ep['total_reward'] for ep in episode_results])
        avg_steps = np.mean([ep['steps'] for ep in episode_results])

        # 误差分析
        final_errors = []
        for ep in episode_results:
            if ep['success'] and ep['final_state'] is not None:
                # 计算与目标的距离
                target_state = self.env.target_shape
                error = np.linalg.norm(ep['final_state'] - target_state)
                final_errors.append(error)

        avg_error = np.mean(final_errors) if final_errors else float('inf')

        # 存储结果
        self.results['evaluation_metrics'] = {
            'num_episodes': num_episodes,
            'success_rate': float(success_rate),
            'average_reward': float(avg_reward),
            'average_steps': float(avg_steps),
            'average_final_error': float(avg_error),
            'success_count': success_count
        }

        self.results['episode_details'] = episode_results

        print(f"\n评估完成:")
        print(f"  成功率: {success_rate:.1%}")
        print(f"  平均奖励: {avg_reward:.2f}")
        print(f"  平均步数: {avg_steps:.1f}")
        print(f"  平均最终误差: {avg_error:.4f}")

        return episode_results

    def compare_with_mpc(self, mpc_evaluator):
        """与MPC性能对比"""
        if 'evaluation_metrics' not in self.results:
            print("请先运行RL评估")
            return

        rl_metrics = self.results['evaluation_metrics']

        # 假设MPC评估器有类似结构
        mpc_results = mpc_evaluator.results if hasattr(mpc_evaluator, 'results') else {}

        comparison = {
            'rl_success_rate': rl_metrics['success_rate'],
            'rl_avg_error': rl_metrics['average_final_error'],
            'rl_avg_steps': rl_metrics['average_steps'],
        }

        if 'basic_metrics' in mpc_results:
            mpc_bm = mpc_results['basic_metrics']
            comparison.update({
                'mpc_convergence': mpc_bm['convergence']['converged'],
                'mpc_convergence_step': mpc_bm['convergence']['convergence_step'],
                'mpc_final_error': mpc_bm['total_errors']['final'],
                'mpc_error_improvement': mpc_bm['total_errors']['improvement']
            })

        self.results['mpc_comparison'] = comparison

        # 创建对比报告
        report = self._create_comparison_report(comparison)

        return comparison

    def _create_comparison_report(self, comparison):
        """创建对比报告"""
        report = []
        report.append("=" * 80)
        report.append("RL vs MPC 性能对比报告")
        report.append("=" * 80)

        report.append(f"\n1. RL策略性能:")
        report.append(f"   成功率: {comparison['rl_success_rate']:.1%}")
        report.append(f"   平均最终误差: {comparison['rl_avg_error']:.6f}")
        report.append(f"   平均步数: {comparison['rl_avg_steps']:.1f}")

        if 'mpc_convergence' in comparison:
            report.append(f"\n2. MPC控制器性能:")
            report.append(f"   是否收敛: {'是' if comparison['mpc_convergence'] else '否'}")
            if comparison['mpc_convergence']:
                report.append(f"   收敛步数: {comparison['mpc_convergence_step']}")
            report.append(f"   最终误差: {comparison['mpc_final_error']:.6f}")
            report.append(f"   误差改善率: {comparison['mpc_error_improvement']:.2f}%")

            # 性能对比分析
            report.append(f"\n3. 对比分析:")
            if comparison['rl_success_rate'] > 0.8:
                report.append("   ✓ RL策略在成功率方面表现良好")
            if comparison['rl_avg_error'] < comparison['mpc_final_error']:
                report.append("   ✓ RL策略在最终误差方面优于MPC")
            else:
                report.append("   ○ MPC在最终误差方面表现更好")

        report.append("\n" + "=" * 80)

        # 保存报告
        report_file = self.output_dir / "rl_mpc_comparison_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))

        print(f"对比报告已保存: {report_file}")

        return report

    def save_evaluation_results(self):
        """保存评估结果"""
        # 保存详细结果
        json_file = self.output_dir / "detailed_evaluation_results.json"

        # 转换为可序列化的格式
        serializable_results = {}
        for key, value in self.results.items():
            if key == 'episode_details':
                # 处理episode详情
                serialized_episodes = []
                for ep in value:
                    serialized_ep = {
                        'episode': ep['episode'],
                        'total_reward': float(ep['total_reward']),
                        'steps': ep['steps'],
                        'success': ep['success'],
                        'final_state': ep['final_state'].tolist() if ep['final_state'] is not None else None,
                        'rewards': [float(r) for r in ep['rewards']]
                    }
                    serialized_episodes.append(serialized_ep)
                serializable_results[key] = serialized_episodes
            else:
                serializable_results[key] = value

        with open(json_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"详细评估结果已保存: {json_file}")

        # 保存摘要CSV
        if 'evaluation_metrics' in self.results:
            summary_data = {
                'Metric': ['Success Rate', 'Average Reward', 'Average Steps', 'Average Final Error'],
                'Value': [
                    self.results['evaluation_metrics']['success_rate'],
                    self.results['evaluation_metrics']['average_reward'],
                    self.results['evaluation_metrics']['average_steps'],
                    self.results['evaluation_metrics']['average_final_error']
                ]
            }

            df = pd.DataFrame(summary_data)
            csv_file = self.output_dir / "evaluation_summary.csv"
            df.to_csv(csv_file, index=False)
            print(f"评估摘要已保存: {csv_file}")

    def visualize_evaluation_results(self):
        """可视化评估结果"""
        if 'episode_details' not in self.results:
            print("没有可用的评估数据")
            return

        from rl_visualizer import RLTrainingVisualizer
        visualizer = RLTrainingVisualizer(output_dir=self.output_dir)

        # 创建训练进度可视化（如果有）
        visualizer.create_training_report()

        # 绘制评估结果
        self._plot_evaluation_results()

    def _plot_evaluation_results(self):
        """绘制评估结果图"""
        episode_details = self.results['episode_details']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 每个episode的总奖励
        episode_nums = [ep['episode'] for ep in episode_details]
        total_rewards = [ep['total_reward'] for ep in episode_details]
        success_flags = [ep['success'] for ep in episode_details]

        colors = ['green' if success else 'red' for success in success_flags]
        axes[0, 0].bar(episode_nums, total_rewards, color=colors, alpha=0.7)
        axes[0, 0].axhline(y=np.mean(total_rewards), color='blue', linestyle='--',
                           linewidth=2, label=f'Mean: {np.mean(total_rewards):.2f}')
        axes[0, 0].set_title('Total Reward per Episode', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 成功率分布
        success_rate = np.mean(success_flags)
        axes[0, 1].pie([success_rate, 1 - success_rate],
                       labels=[f'Success ({success_rate:.1%})', f'Failure ({1 - success_rate:.1%})'],
                       colors=['green', 'red'], autopct='%1.1f%%')
        axes[0, 1].set_title('Success Rate Distribution', fontsize=14, fontweight='bold')

        # 3. 步数分布
        steps = [ep['steps'] for ep in episode_details]
        axes[1, 0].hist(steps, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].axvline(x=np.mean(steps), color='red', linestyle='--',
                           linewidth=2, label=f'Mean: {np.mean(steps):.1f}')
        axes[1, 0].set_title('Steps per Episode Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 奖励累积曲线（示例episode）
        if len(episode_details) > 0:
            example_ep = episode_details[0]
            if 'rewards' in example_ep and len(example_ep['rewards']) > 0:
                cumulative_rewards = np.cumsum(example_ep['rewards'])
                axes[1, 1].plot(range(len(cumulative_rewards)), cumulative_rewards,
                                'b-', linewidth=2)
                axes[1, 1].set_title(f'Cumulative Reward (Episode 0)', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Cumulative Reward')
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        plot_kwargs = {
            'dpi': 600,
            'bbox_inches': 'tight',
            'facecolor': 'white'
        }
        plt.savefig(self.output_dir / 'evaluation_results.png', **plot_kwargs)
        plt.close()

        print("✓ Evaluation results plots saved")

    def run_complete_evaluation(self, num_episodes=20, mpc_evaluator=None):
        """运行完整评估流程"""
        print("开始完整RL评估流程...")

        # 1. 评估策略
        self.evaluate_policy(num_episodes)

        # 2. 与MPC对比（如果提供）
        if mpc_evaluator:
            self.compare_with_mpc(mpc_evaluator)

        # 3. 保存结果
        self.save_evaluation_results()

        # 4. 可视化
        self.visualize_evaluation_results()

        print(f"\n评估完成！所有结果已保存至: {self.output_dir}")

        return self.results


def main():
    """主评估函数"""
    # 配置参数
    MODEL_PATH = "sac_deformation_precise.zip"  # 训练好的模型
    NUM_EPISODES = 20

    # 创建评估器
    evaluator = RLEvaluator(MODEL_PATH)

    # 运行完整评估
    results = evaluator.run_complete_evaluation(num_episodes=NUM_EPISODES)

    # 打印总结
    if 'evaluation_metrics' in results:
        metrics = results['evaluation_metrics']
        print("\n" + "=" * 60)
        print("评估总结:")
        print(f"  测试episode数: {metrics['num_episodes']}")
        print(f"  成功率: {metrics['success_rate']:.1%}")
        print(f"  平均奖励: {metrics['average_reward']:.2f}")
        print(f"  平均步数: {metrics['average_steps']:.1f}")
        print(f"  平均最终误差: {metrics['average_final_error']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()