"""
专门记录和可视化训练指标
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


@dataclass
class MetricBuffer:
    """指标缓冲区"""
    max_size: int = 10000
    data: Dict[str, List] = field(default_factory=dict)

    def add(self, key: str, value: Any):
        """添加指标值"""
        if key not in self.data:
            self.data[key] = []

        self.data[key].append(value)

        # 限制缓冲区大小
        if len(self.data[key]) > self.max_size:
            self.data[key] = self.data[key][-self.max_size:]

    def get(self, key: str, default: Any = None) -> List:
        """获取指标值"""
        return self.data.get(key, default or [])

    def clear(self):
        """清空缓冲区"""
        self.data.clear()


class MetricRecorder:
    """指标记录器"""

    def __init__(self, log_dir: Path, experiment_name: str):
        """
        初始化指标记录器

        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name

        # 创建指标目录
        self.metric_dir = log_dir / "metrics"
        self.metric_dir.mkdir(exist_ok=True)

        # 指标缓冲区
        self.buffers = {
            'episode': MetricBuffer(max_size=1000),
            'step': MetricBuffer(max_size=10000),
            'mpc': MetricBuffer(max_size=5000),
            'performance': MetricBuffer(max_size=1000)
        }

        # 统计数据
        self.stats = {}

        # DataFrame缓存
        self.dataframes = {}

        # 当前episode数据
        self.current_episode = {
            'rewards': [],
            'actions': [],
            'states': [],
            'errors': []
        }

    def record_episode_start(self, episode_num: int):
        """记录episode开始"""
        self.current_episode = {
            'episode': episode_num,
            'rewards': [],
            'actions': [],
            'states': [],
            'errors': [],
            'start_time': datetime.now()
        }

    def record_step(self,
                    reward: float,
                    action: Optional[np.ndarray] = None,
                    state: Optional[np.ndarray] = None,
                    error: Optional[float] = None,
                    **kwargs):
        """记录单步数据"""
        # 添加到当前episode
        self.current_episode['rewards'].append(reward)

        if action is not None:
            self.current_episode['actions'].append(action.copy())

        if state is not None:
            self.current_episode['states'].append(state.copy())

        if error is not None:
            self.current_episode['errors'].append(error)

        # 添加到step缓冲区
        step_data = {'reward': reward, 'error': error, **kwargs}
        for key, value in step_data.items():
            if value is not None:
                self.buffers['step'].add(key, value)

    def record_episode_end(self,
                           total_reward: float,
                           success: bool = False,
                           **kwargs):
        """记录episode结束"""
        if not self.current_episode:
            return

        # 计算统计信息
        episode_data = {
            'episode': self.current_episode.get('episode', 0),
            'total_reward': total_reward,
            'average_reward': np.mean(self.current_episode['rewards']) if self.current_episode['rewards'] else 0,
            'std_reward': np.std(self.current_episode['rewards']) if self.current_episode['rewards'] else 0,
            'max_reward': np.max(self.current_episode['rewards']) if self.current_episode['rewards'] else 0,
            'min_reward': np.min(self.current_episode['rewards']) if self.current_episode['rewards'] else 0,
            'steps': len(self.current_episode['rewards']),
            'success': success,
            'average_error': np.mean(self.current_episode['errors']) if self.current_episode['errors'] else 0,
            'final_error': self.current_episode['errors'][-1] if self.current_episode['errors'] else 0,
            'end_time': datetime.now(),
            'duration_seconds': (
                        datetime.now() - self.current_episode.get('start_time', datetime.now())).total_seconds(),
            **kwargs
        }

        # 添加到缓冲区
        for key, value in episode_data.items():
            self.buffers['episode'].add(key, value)

        # 更新统计数据
        self._update_stats(episode_data)

    def record_mpc_data(self,
                        cost: float,
                        iterations: int,
                        control_norm: float,
                        **kwargs):
        """记录MPC数据"""
        mpc_data = {
            'cost': cost,
            'iterations': iterations,
            'control_norm': control_norm,
            'timestamp': datetime.now(),
            **kwargs
        }

        for key, value in mpc_data.items():
            self.buffers['mpc'].add(key, value)

    def record_performance(self,
                           component: str,
                           duration_ms: float,
                           **kwargs):
        """记录性能数据"""
        perf_data = {
            'component': component,
            'duration_ms': duration_ms,
            'timestamp': datetime.now(),
            **kwargs
        }

        for key, value in perf_data.items():
            self.buffers['performance'].add(key, value)

    def _update_stats(self, episode_data: Dict[str, Any]):
        """更新统计数据"""
        episode_num = episode_data['episode']

        # 滑动窗口统计
        window_size = 10
        rewards = self.buffers['episode'].get('total_reward')[-window_size:]

        if rewards:
            self.stats[f'last_{window_size}_episodes'] = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'max_reward': np.max(rewards),
                'min_reward': np.min(rewards),
                'success_rate': np.mean(self.buffers['episode'].get('success')[-window_size:]) if self.buffers[
                    'episode'].get('success') else 0
            }

    def get_latest_stats(self) -> Dict[str, Any]:
        """获取最新统计信息"""
        return self.stats.copy()

    def save_to_csv(self, buffer_type: str = 'episode'):
        """保存指标到CSV文件"""
        if buffer_type not in self.buffers:
            raise ValueError(f"未知缓冲区类型: {buffer_type}")

        buffer = self.buffers[buffer_type]
        if not buffer.data:
            return None

        # 转换为DataFrame
        df = pd.DataFrame(buffer.data)

        # 保存到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.metric_dir / f"{buffer_type}_metrics_{timestamp}.csv"
        df.to_csv(filename, index=False)

        # 缓存DataFrame
        self.dataframes[buffer_type] = df

        return filename

    def plot_training_curves(self, save: bool = True) -> Optional[List[Path]]:
        """绘制训练曲线"""
        if 'episode' not in self.buffers or not self.buffers['episode'].data:
            return None

        episode_data = self.buffers['episode']

        # 确保有足够的数据
        if len(episode_data.get('episode', [])) < 2:
            return None

        saved_files = []

        # 1. 奖励曲线
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 总奖励
        if episode_data.get('total_reward'):
            axes[0, 0].plot(episode_data.get('episode'), episode_data.get('total_reward'), 'b-', alpha=0.5)
            axes[0, 0].set_title('Total Reward per Episode')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True, alpha=0.3)

            # 添加滑动平均
            window = 10
            if len(episode_data.get('total_reward')) >= window:
                rewards = episode_data.get('total_reward')
                moving_avg = pd.Series(rewards).rolling(window=window).mean()
                axes[0, 0].plot(episode_data.get('episode')[window - 1:],
                                moving_avg[window - 1:],
                                'r-', linewidth=2)

        # 2. 成功率
        if episode_data.get('success'):
            success_rate = pd.Series(episode_data.get('success')).rolling(window=20).mean()
            axes[0, 1].plot(episode_data.get('episode')[19:], success_rate[19:], 'g-')
            axes[0, 1].axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Episode长度
        if episode_data.get('steps'):
            axes[0, 2].hist(episode_data.get('steps'), bins=20, alpha=0.7)
            axes[0, 2].axvline(x=np.mean(episode_data.get('steps')),
                               color='r', linestyle='--',
                               label=f'Mean: {np.mean(episode_data.get("steps")):.1f}')
            axes[0, 2].set_title('Episode Length Distribution')
            axes[0, 2].set_xlabel('Steps')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # 4. 误差曲线
        if episode_data.get('final_error'):
            axes[1, 0].plot(episode_data.get('episode'), episode_data.get('final_error'), 'r-', alpha=0.5)
            axes[1, 0].set_title('Final Error per Episode')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Error')
            axes[1, 0].grid(True, alpha=0.3)

            # 对数坐标
            axes[1, 0].set_yscale('log')

        # 5. 持续时间
        if episode_data.get('duration_seconds'):
            axes[1, 1].plot(episode_data.get('episode'), episode_data.get('duration_seconds'), 'purple', alpha=0.5)
            axes[1, 1].set_title('Episode Duration')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Duration (s)')
            axes[1, 1].grid(True, alpha=0.3)

        # 6. 动作统计（如果有）
        if self.current_episode.get('actions'):
            actions = np.array(self.current_episode['actions'])
            if len(actions.shape) > 1:
                action_norms = np.linalg.norm(actions, axis=1)
                axes[1, 2].plot(action_norms, 'orange')
                axes[1, 2].set_title('Action Norm in Last Episode')
                axes[1, 2].set_xlabel('Step in Episode')
                axes[1, 2].set_ylabel('Action Norm')
                axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.metric_dir / f"training_curves_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            saved_files.append(plot_file)
            plt.close()

        return saved_files if save else None

    def plot_performance_analysis(self, save: bool = True) -> Optional[Path]:
        """绘制性能分析图"""
        if 'performance' not in self.buffers or not self.buffers['performance'].data:
            return None

        perf_data = self.buffers['performance']

        if 'component' not in perf_data.data or 'duration_ms' not in perf_data.data:
            return None

        # 按组件分组
        components = {}
        for comp, duration in zip(perf_data.get('component'), perf_data.get('duration_ms')):
            if comp not in components:
                components[comp] = []
            components[comp].append(duration)

        if not components:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. 箱线图
        comp_names = list(components.keys())
        durations = [components[name] for name in comp_names]

        axes[0].boxplot(durations, labels=comp_names)
        axes[0].set_title('Component Execution Time Distribution')
        axes[0].set_ylabel('Duration (ms)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)

        # 2. 累计时间
        total_times = [np.sum(durs) for durs in durations]
        axes[1].bar(comp_names, total_times)
        axes[1].set_title('Total Execution Time per Component')
        axes[1].set_ylabel('Total Time (ms)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.metric_dir / f"performance_analysis_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_file

        return None

    def generate_summary_report(self) -> Dict[str, Any]:
        """生成摘要报告"""
        report = {
            'experiment_name': self.experiment_name,
            'generated_at': datetime.now().isoformat(),
            'total_episodes': len(self.buffers['episode'].get('episode', [])),
            'total_steps': len(self.buffers['step'].get('reward', [])),
            'metrics_summary': {}
        }

        # Episode统计
        if self.buffers['episode'].data:
            ep_data = self.buffers['episode']
            report['metrics_summary']['episodes'] = {
                'mean_reward': np.mean(ep_data.get('total_reward', [0])),
                'mean_steps': np.mean(ep_data.get('steps', [0])),
                'success_rate': np.mean(ep_data.get('success', [0])) * 100,
                'mean_error': np.mean(ep_data.get('final_error', [0]))
            }

        # 性能统计
        if self.buffers['performance'].data:
            perf_data = self.buffers['performance']
            report['metrics_summary']['performance'] = {
                'total_executions': len(perf_data.get('duration_ms', [])),
                'mean_duration_ms': np.mean(perf_data.get('duration_ms', [])),
                'max_duration_ms': np.max(perf_data.get('duration_ms', []))
            }

        # 保存报告
        report_file = self.metric_dir / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        return report

    def clear(self):
        """清空所有数据"""
        for buffer in self.buffers.values():
            buffer.clear()
        self.stats.clear()
        self.dataframes.clear()
        self.current_episode.clear()


# 单例实例
_global_metric_recorder = None


def get_global_metric_recorder(log_dir: Optional[Path] = None,
                               experiment_name: Optional[str] = None) -> MetricRecorder:
    """获取全局指标记录器"""
    global _global_metric_recorder

    if _global_metric_recorder is None:
        if log_dir is None or experiment_name is None:
            raise ValueError("首次调用需要提供log_dir和experiment_name")

        _global_metric_recorder = MetricRecorder(log_dir, experiment_name)

    return _global_metric_recorder


if __name__ == "__main__":
    # 测试指标记录器
    test_dir = Path("./test_metrics")
    recorder = MetricRecorder(test_dir, "test_experiment")

    # 模拟一些数据
    for ep in range(10):
        recorder.record_episode_start(ep)

        for step in range(50):
            recorder.record_step(
                reward=np.random.randn(),
                action=np.random.randn(3),
                state=np.random.randn(16),
                error=np.random.rand() * 0.1
            )

        recorder.record_episode_end(
            total_reward=np.random.randn() * 10,
            success=np.random.rand() > 0.5
        )

    # 保存和绘图
    recorder.save_to_csv('episode')
    recorder.plot_training_curves(save=True)

    # 生成报告
    report = recorder.generate_summary_report()
    print("摘要报告:")
    print(json.dumps(report, indent=2))