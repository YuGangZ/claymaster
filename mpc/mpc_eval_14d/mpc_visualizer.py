import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path
from scipy import signal

# 设置绘图样式
rcParams['figure.figsize'] = (12, 8)
rcParams['font.size'] = 18
rcParams['axes.labelsize'] = 18
rcParams['axes.titlesize'] = 20
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['legend.fontsize'] = 16
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 600
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.format'] = 'png'
sns.set_palette("husl")


class MPCVisualizer:
    def __init__(self, data, results, output_dir):
        self.data = data
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 高质量保存设置
        self.plot_kwargs = {
            'dpi': 600,
            'bbox_inches': 'tight',
            'facecolor': 'white'
        }

    def _smooth_data(self, data, window_size=11, polyorder=2):
        """
        使用Savitzky-Golay滤波器平滑数据

        Args:
            data: 输入数据
            window_size: 窗口大小（必须为奇数）
            polyorder: 多项式阶数

        Returns:
            平滑后的数据
        """
        if len(data) < window_size:
            window_size = len(data) // 2
            if window_size % 2 == 0:
                window_size += 1

        # 确保窗口大小为奇数
        if window_size % 2 == 0:
            window_size += 1

        # 确保窗口大小不超过数据长度
        if window_size > len(data):
            window_size = len(data)
            if window_size % 2 == 0:
                window_size -= 1

        if window_size < 3:
            return data

        try:
            smoothed = signal.savgol_filter(data, window_size, polyorder)
            return smoothed
        except:
            # 如果平滑失败，返回原始数据
            return data

    def _moving_average(self, data, window_size=5):
        """使用移动平均平滑数据"""
        if len(data) < window_size:
            return data

        # 使用卷积实现移动平均
        weights = np.ones(window_size) / window_size
        smoothed = np.convolve(data, weights, mode='valid')

        # 填充两端
        front_padding = np.full(window_size // 2, smoothed[0])
        back_padding = np.full(window_size - len(front_padding) - 1, smoothed[-1])
        return np.concatenate([front_padding, smoothed, back_padding])

    def _adaptive_smooth(self, data, base_window=5, noise_threshold=0.1):
        """
        自适应平滑：对噪声大的区域使用更大的窗口

        Args:
            data: 输入数据
            base_window: 基础窗口大小
            noise_threshold: 噪声阈值

        Returns:
            平滑后的数据
        """
        # 计算梯度
        gradients = np.abs(np.gradient(data))
        avg_gradient = np.mean(gradients)

        # 根据梯度大小调整窗口
        if avg_gradient > noise_threshold:
            window_size = min(15, max(base_window, int(avg_gradient * 50)))
        else:
            window_size = base_window

        # 确保窗口大小为奇数
        if window_size % 2 == 0:
            window_size += 1

        return self._smooth_data(data, window_size)

    def plot_state_trajectory(self):
        """Plot state trajectory with smoothing"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # 提取数据
        steps = [r['step'] for r in self.data]
        states = np.array([r['feature_16d'] for r in self.data])

        # 检查数据长度，决定是否平滑
        if len(steps) < 10:
            print("数据点太少，不进行平滑处理")
            smoothed_states = states
        else:
            print(f"对 {len(states)} 个数据点进行平滑处理")
            # 对每个维度进行自适应平滑
            smoothed_states = np.zeros_like(states)
            for i in range(states.shape[1]):
                smoothed_states[:, i] = self._adaptive_smooth(states[:, i])

        # 尺度参数轨迹
        axes[0].plot(steps, states[:, 0], 'b-', label='a1 (原始)', linewidth=1.5, alpha=0.3)
        axes[0].plot(steps, smoothed_states[:, 0], 'b-', label='a1 (平滑)', linewidth=2.5)

        axes[0].plot(steps, states[:, 1], 'r-', label='a2 (原始)', linewidth=1.5, alpha=0.3)
        axes[0].plot(steps, smoothed_states[:, 1], 'r-', label='a2 (平滑)', linewidth=2.5)

        axes[0].plot(steps, states[:, 2], 'g-', label='a3 (原始)', linewidth=1.5, alpha=0.3)
        axes[0].plot(steps, smoothed_states[:, 2], 'g-', label='a3 (平滑)', linewidth=2.5)

        # 添加目标线
        if 'basic_metrics' in self.results:
            target = self.results['basic_metrics']['target_state']
            axes[0].axhline(y=target[0], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
            axes[0].axhline(y=target[1], color='r', linestyle='--', alpha=0.5, linewidth=1.5)
            axes[0].axhline(y=target[2], color='g', linestyle='--', alpha=0.5, linewidth=1.5)

        axes[0].set_title('Scale Parameters Trajectory', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Simulation Step')
        axes[0].set_ylabel('Parameter Value')
        axes[0].legend(loc='upper right', frameon=True)
        axes[0].grid(True, alpha=0.3, linestyle='--')

        # 形状参数轨迹
        axes[1].plot(steps, states[:, 3], 'b-', label='ε1 (原始)', linewidth=1.5, alpha=0.3)
        axes[1].plot(steps, smoothed_states[:, 3], 'b-', label='ε1 (平滑)', linewidth=2.5)

        axes[1].plot(steps, states[:, 4], 'r-', label='ε2 (原始)', linewidth=1.5, alpha=0.3)
        axes[1].plot(steps, smoothed_states[:, 4], 'r-', label='ε2 (平滑)', linewidth=2.5)

        if 'basic_metrics' in self.results:
            target = self.results['basic_metrics']['target_state']
            axes[1].axhline(y=target[3], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
            axes[1].axhline(y=target[4], color='r', linestyle='--', alpha=0.5, linewidth=1.5)

        axes[1].set_title('Shape Parameters Trajectory', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Simulation Step')
        axes[1].set_ylabel('Parameter Value')
        axes[1].legend(loc='upper right', frameon=True)
        axes[1].grid(True, alpha=0.3, linestyle='--')

        # 几何特征轨迹 - 只绘制Volume(11), Elongation(12), Smoothness(14)
        geo_labels = ['Volume', 'Elongation', 'Smoothness']
        geo_indices = [11, 12, 14]  # 正确的索引
        colors = plt.cm.tab10(range(3))

        for i, (label, idx) in enumerate(zip(geo_labels, geo_indices)):
            # 绘制原始数据（半透明）
            axes[2].plot(steps, states[:, idx],
                         color=colors[i],
                         linewidth=1.0,
                         alpha=0.2)

            # 绘制平滑数据
            axes[2].plot(steps, smoothed_states[:, idx],
                         color=colors[i],
                         label=label,
                         linewidth=2.0,
                         alpha=0.9)

        axes[2].set_title('Geometric Features Trajectory', fontsize=16, fontweight='bold')
        axes[2].set_xlabel('Simulation Step')
        axes[2].set_ylabel('Feature Value')
        axes[2].legend(loc='upper right', fontsize=11, frameon=True)
        axes[2].grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout(pad=3.0)

        # 保存高质量图片
        plt.savefig(self.output_dir / 'state_trajectory.png', **self.plot_kwargs)

        # 额外保存一个只显示平滑数据的版本
        fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12))

        # 尺度参数轨迹（仅平滑）
        axes2[0].plot(steps, smoothed_states[:, 0], 'b-', label='a1', linewidth=2.5)
        axes2[0].plot(steps, smoothed_states[:, 1], 'r-', label='a2', linewidth=2.5)
        axes2[0].plot(steps, smoothed_states[:, 2], 'g-', label='a3', linewidth=2.5)

        if 'basic_metrics' in self.results:
            target = self.results['basic_metrics']['target_state']
            axes2[0].axhline(y=target[0], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
            axes2[0].axhline(y=target[1], color='r', linestyle='--', alpha=0.5, linewidth=1.5)
            axes2[0].axhline(y=target[2], color='g', linestyle='--', alpha=0.5, linewidth=1.5)

        axes2[0].set_title('Scale Parameters Trajectory', fontsize=16, fontweight='bold')
        axes2[0].set_xlabel('Simulation Step')
        axes2[0].set_ylabel('Parameter Value')
        axes2[0].legend(loc='upper right', fontsize=13, frameon=True)
        axes2[0].grid(True, alpha=0.3, linestyle='--')

        # 形状参数轨迹（仅平滑）
        axes2[1].plot(steps, smoothed_states[:, 3], 'b-', label='ε1', linewidth=2.5)
        axes2[1].plot(steps, smoothed_states[:, 4], 'r-', label='ε2', linewidth=2.5)

        if 'basic_metrics' in self.results:
            axes2[1].axhline(y=target[3], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
            axes2[1].axhline(y=target[4], color='r', linestyle='--', alpha=0.5, linewidth=1.5)

        axes2[1].set_title('Shape Parameters Trajectory', fontsize=16, fontweight='bold')
        axes2[1].set_xlabel('Simulation Step')
        axes2[1].set_ylabel('Parameter Value')
        axes2[1].legend(loc='upper right', fontsize=13, frameon=True)
        axes2[1].grid(True, alpha=0.3, linestyle='--')

        # 几何特征轨迹（仅平滑）- 只绘制Volume(11), Elongation(12), Smoothness(14)
        for i, (label, idx) in enumerate(zip(geo_labels, geo_indices)):
            axes2[2].plot(steps, smoothed_states[:, idx],
                          color=colors[i],
                          label=label,
                          alpha=0.9)

        axes2[2].set_title('Geometric Features Trajectory', fontsize=16, fontweight='bold')
        axes2[2].set_xlabel('Simulation Step')
        axes2[2].set_ylabel('Feature Value')
        axes2[2].legend(loc='upper right', fontsize=13, frameon=True)
        axes2[2].grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout(pad=3.0)
        plt.savefig(self.output_dir / 'state_trajectory_smoothed.png', **self.plot_kwargs)
        plt.close(fig2)

        plt.close()
        print("✓ State trajectory plots saved")

    def plot_smoothing_comparison(self):
        """绘制平滑前后的对比图"""
        if len(self.data) < 20:
            print("数据点太少，不绘制平滑对比图")
            return

        steps = [r['step'] for r in self.data]
        states = np.array([r['feature_16d'] for r in self.data])

        # 选择几个代表性的维度进行对比
        dims_to_show = [0, 3, 14]  # a1, ε1, Smoothness（使用正确的索引14）
        dim_names = ['a1', 'ε1', 'Smoothness']

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        for idx, (dim, name) in enumerate(zip(dims_to_show, dim_names)):
            original_data = states[:, dim]
            smoothed_data = self._adaptive_smooth(original_data)

            # 原始数据
            axes[idx, 0].plot(steps, original_data, 'b-', linewidth=2.0)
            axes[idx, 0].set_title(f'{name} - Original', fontsize=14, fontweight='bold')
            axes[idx, 0].set_xlabel('Simulation Step')
            axes[idx, 0].set_ylabel('Value')
            axes[idx, 0].grid(True, alpha=0.3)

            # 平滑后数据
            axes[idx, 1].plot(steps, smoothed_data, 'r-', linewidth=2.0)
            axes[idx, 1].set_title(f'{name} - Smoothed', fontsize=14, fontweight='bold')
            axes[idx, 1].set_xlabel('Simulation Step')
            axes[idx, 1].set_ylabel('Value')
            axes[idx, 1].grid(True, alpha=0.3)

            # 添加统计信息
            stats_text = f'Original: μ={np.mean(original_data):.4f}, σ={np.std(original_data):.4f}\n'
            stats_text += f'Smoothed: μ={np.mean(smoothed_data):.4f}, σ={np.std(smoothed_data):.4f}\n'
            stats_text += f'Max diff: {np.max(np.abs(original_data - smoothed_data)):.6f}'

            axes[idx, 1].text(0.02, 0.98, stats_text, transform=axes[idx, 1].transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.suptitle('Smoothing Comparison', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / 'smoothing_comparison.png', **self.plot_kwargs)
        plt.close()
        print("✓ Smoothing comparison plot saved")