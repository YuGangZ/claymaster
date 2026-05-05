import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path
import pandas as pd
import json


class RLTrainingVisualizer:
    def __init__(self, training_log_path=None, output_dir="rl_training_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 加载训练日志
        self.training_logs = self._load_training_logs(training_log_path)

        # 设置绘图样式
        plt.style.use('seaborn-v0_8-darkgrid')
        rcParams['figure.figsize'] = (12, 8)
        rcParams['font.size'] = 12

        # 保存设置
        self.plot_kwargs = {
            'dpi': 600,
            'bbox_inches': 'tight',
            'facecolor': 'white'
        }

    def _load_training_logs(self, log_path):
        """加载训练日志"""
        if log_path and Path(log_path).exists():
            try:
                with open(log_path, 'r') as f:
                    return json.load(f)
            except:
                print("无法加载训练日志，将使用模拟数据")

        # 模拟训练数据用于演示
        return self._generate_mock_training_data()

    def _generate_mock_training_data(self):
        """生成模拟训练数据"""
        np.random.seed(42)
        episodes = 200

        return {
            'episode_rewards': np.cumsum(np.random.randn(episodes) * 10 + 50),
            'episode_lengths': np.random.randint(40, 60, episodes),
            'losses': np.abs(np.random.randn(episodes) * 0.1),
            'exploration_rate': np.exp(-np.linspace(0, 5, episodes)),
            'success_rate': np.clip(np.linspace(0.1, 0.9, episodes) + np.random.randn(episodes) * 0.1, 0, 1)
        }

    def plot_training_progress(self):
        """绘制训练进度图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        episodes = range(len(self.training_logs['episode_rewards']))

        # 1. 奖励曲线
        axes[0, 0].plot(episodes, self.training_logs['episode_rewards'], 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].fill_between(episodes,
                                np.array(self.training_logs['episode_rewards']) * 0.9,
                                np.array(self.training_logs['episode_rewards']) * 1.1,
                                alpha=0.2, color='blue')
        axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 滑动平均奖励
        window = 10
        if len(self.training_logs['episode_rewards']) > window:
            moving_avg = np.convolve(self.training_logs['episode_rewards'],
                                     np.ones(window) / window, mode='valid')
            axes[0, 1].plot(range(window - 1, len(self.training_logs['episode_rewards'])),
                            moving_avg, 'r-', linewidth=2.5)
        axes[0, 1].set_title(f'Moving Average Reward (window={window})', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 成功率
        axes[0, 2].plot(episodes, self.training_logs['success_rate'], 'g-', linewidth=2)
        axes[0, 2].axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target Success Rate')
        axes[0, 2].fill_between(episodes, 0, self.training_logs['success_rate'],
                                alpha=0.3, color='green')
        axes[0, 2].set_title('Success Rate Over Episodes', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Success Rate')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. 探索率
        axes[1, 0].plot(episodes, self.training_logs['exploration_rate'], 'purple', linewidth=2)
        axes[1, 0].set_title('Exploration Rate Decay', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Exploration Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # 5. 损失函数
        axes[1, 1].plot(episodes, self.training_logs['losses'], 'orange', linewidth=1.5, alpha=0.7)
        if len(self.training_logs['losses']) > 20:
            loss_smooth = np.convolve(self.training_logs['losses'],
                                      np.ones(20) / 20, mode='valid')
            axes[1, 1].plot(range(19, len(self.training_logs['losses'])),
                            loss_smooth, 'orange', linewidth=2.5)
        axes[1, 1].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Episode长度
        axes[1, 2].plot(episodes, self.training_logs['episode_lengths'], 'b-',
                        linewidth=1, alpha=0.5, label='Raw')
        if len(self.training_logs['episode_lengths']) > 10:
            length_smooth = np.convolve(self.training_logs['episode_lengths'],
                                        np.ones(10) / 10, mode='valid')
            axes[1, 2].plot(range(9, len(self.training_logs['episode_lengths'])),
                            length_smooth, 'r-', linewidth=2, label='Smoothed')
        axes[1, 2].set_title('Episode Lengths', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Steps per Episode')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        plt.savefig(self.output_dir / 'training_progress.png', **self.plot_kwargs)
        plt.savefig(self.output_dir / 'training_progress.pdf', **self.plot_kwargs)
        plt.close()

        print("✓ Training progress plots saved")

    def plot_learning_curves_comparison(self, mpc_data=None):
        """绘制RL与MPC学习曲线对比"""
        fig, ax = plt.subplots(figsize=(14, 8))

        episodes = range(len(self.training_logs['episode_rewards']))

        # RL奖励曲线
        ax.plot(episodes, self.training_logs['episode_rewards'],
                'b-', linewidth=2, alpha=0.7, label='RL Episode Reward')

        # 滑动平均
        if len(self.training_logs['episode_rewards']) > 20:
            rl_ma = np.convolve(self.training_logs['episode_rewards'],
                                np.ones(20) / 20, mode='valid')
            ax.plot(range(19, len(self.training_logs['episode_rewards'])),
                    rl_ma, 'b-', linewidth=3, label='RL Moving Average (window=20)')

        # 如果有MPC数据，添加对比
        if mpc_data:
            mpc_steps = np.linspace(0, len(episodes), len(mpc_data['errors']))
            mpc_normalized = np.interp(mpc_steps,
                                       np.linspace(0, len(episodes), len(mpc_data['errors'])),
                                       -mpc_data['errors'] * 100)  # 负误差转为正奖励
            ax.plot(mpc_steps, mpc_normalized, 'r--', linewidth=2.5,
                    label='MPC Performance (Error negated)')

        ax.set_title('RL vs MPC Learning Curves Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Episode / MPC Step')
        ax.set_ylabel('Reward / Performance Metric')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'rl_vs_mpc_comparison.png', **self.plot_kwargs)
        plt.close()

        print("✓ RL vs MPC comparison plot saved")

    def plot_action_distribution(self, actions_history):
        """绘制动作分布"""
        if not actions_history or len(actions_history) == 0:
            print("无动作历史数据")
            return

        actions = np.array(actions_history)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 动作分量分布
        action_labels = ['Δa1', 'Δa2', 'Δa3', 'Δε1', 'Δε2',
                         'Δx', 'Δy', 'Δz', 'Δrx', 'Δry', 'Δrz',
                         'ΔVolume', 'ΔElongation', 'ΔFlatness', 'ΔSmoothness', 'ΔConvexity']

        # 1. 前三个尺度参数的变化分布
        for i in range(3):
            axes[0, 0].hist(actions[:, i], bins=50, alpha=0.6,
                            label=action_labels[i], density=True)
        axes[0, 0].set_title('Scale Parameter Changes Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Change Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 形状参数变化分布
        for i in range(3, 5):
            axes[0, 1].hist(actions[:, i], bins=50, alpha=0.6,
                            label=action_labels[i], density=True)
        axes[0, 1].set_title('Shape Parameter Changes Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Change Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 动作范数分布
        action_norms = np.linalg.norm(actions, axis=1)
        axes[1, 0].hist(action_norms, bins=50, alpha=0.7, color='purple', density=True)
        axes[1, 0].axvline(x=np.mean(action_norms), color='red', linestyle='--',
                           linewidth=2, label=f'Mean: {np.mean(action_norms):.3f}')
        axes[1, 0].set_title('Action Norm Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Action Norm')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 动作相关性热图
        corr_matrix = np.corrcoef(actions.T)
        im = axes[1, 1].imshow(corr_matrix[:8, :8], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 1].set_title('Action Components Correlation (First 8)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(range(8))
        axes[1, 1].set_yticks(range(8))
        axes[1, 1].set_xticklabels(action_labels[:8], rotation=45, ha='right')
        axes[1, 1].set_yticklabels(action_labels[:8])
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'action_distribution.png', **self.plot_kwargs)
        plt.close()

        print("✓ Action distribution plots saved")

    def plot_episode_analysis(self, episode_data):
        """分析单个episode"""
        if not episode_data:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        steps = range(len(episode_data['states']))

        # 1. 状态轨迹
        states = np.array(episode_data['states'])
        axes[0, 0].plot(steps, states[:, 0], 'b-', label='a1', linewidth=2)
        axes[0, 0].plot(steps, states[:, 1], 'r-', label='a2', linewidth=2)
        axes[0, 0].plot(steps, states[:, 2], 'g-', label='a3', linewidth=2)
        axes[0, 0].set_title('Scale Parameters in Episode', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Step in Episode')
        axes[0, 0].set_ylabel('Parameter Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 奖励累积
        cumulative_rewards = np.cumsum(episode_data['rewards'])
        axes[0, 1].plot(steps, cumulative_rewards, 'purple', linewidth=2.5)
        axes[0, 1].set_title('Cumulative Reward in Episode', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Step in Episode')
        axes[0, 1].set_ylabel('Cumulative Reward')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 动作序列
        actions = np.array(episode_data['actions'])
        axes[1, 0].plot(steps[:-1], actions[:, 0], 'b-', label='Action[0]', alpha=0.7)
        axes[1, 0].plot(steps[:-1], actions[:, 1], 'r-', label='Action[1]', alpha=0.7)
        axes[1, 0].plot(steps[:-1], actions[:, 2], 'g-', label='Action[2]', alpha=0.7)
        axes[1, 0].set_title('Action Sequence in Episode', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Step in Episode')
        axes[1, 0].set_ylabel('Action Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 误差收敛
        if 'target_state' in episode_data:
            target = np.array(episode_data['target_state'])
            errors = np.linalg.norm(states - target, axis=1)
            axes[1, 1].plot(steps, errors, 'r-', linewidth=2)
            axes[1, 1].set_title('Error Convergence in Episode', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Step in Episode')
            axes[1, 1].set_ylabel('State Error')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'episode_analysis.png', **self.plot_kwargs)
        plt.close()

        print("✓ Episode analysis plot saved")

    def create_training_report(self):
        """创建训练报告"""
        report = []
        report.append("=" * 80)
        report.append("RL Training Analysis Report")
        report.append("=" * 80)

        if self.training_logs:
            # 基础统计
            final_rewards = self.training_logs['episode_rewards'][-10:] if len(
                self.training_logs['episode_rewards']) >= 10 else self.training_logs['episode_rewards']
            avg_final_reward = np.mean(final_rewards)
            max_reward = np.max(self.training_logs['episode_rewards'])
            min_reward = np.min(self.training_logs['episode_rewards'])

            report.append(f"\n1. 奖励统计:")
            report.append(f"   最终10个episode平均奖励: {avg_final_reward:.2f}")
            report.append(f"   最高奖励: {max_reward:.2f}")
            report.append(f"   最低奖励: {min_reward:.2f}")
            report.append(f"   总episode数: {len(self.training_logs['episode_rewards'])}")

            # 成功率统计
            if 'success_rate' in self.training_logs:
                final_success = self.training_logs['success_rate'][-1] if len(
                    self.training_logs['success_rate']) > 0 else 0
                report.append(f"\n2. 成功率:")
                report.append(f"   最终成功率: {final_success:.1%}")

            # 学习稳定性
            if len(self.training_logs['episode_rewards']) > 20:
                early_avg = np.mean(self.training_logs['episode_rewards'][:20])
                late_avg = np.mean(self.training_logs['episode_rewards'][-20:])
                improvement = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0
                report.append(f"\n3. 学习效果:")
                report.append(f"   前期(前20ep)平均奖励: {early_avg:.2f}")
                report.append(f"   后期(后20ep)平均奖励: {late_avg:.2f}")
                report.append(f"   提升百分比: {improvement:.1f}%")

        report.append("\n" + "=" * 80)

        # 保存报告
        report_file = self.output_dir / "training_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))

        print(f"✓ Training report saved: {report_file}")

        return report