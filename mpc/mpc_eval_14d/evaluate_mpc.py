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
import warnings

warnings.filterwarnings('ignore')

# 设置绘图样式
plt.style.use('seaborn-v0_8-darkgrid')
rcParams['figure.figsize'] = (12, 8)
rcParams['font.size'] = 16
sns.set_palette("husl")


class MPCEvaluator:
    def __init__(self, data_dir, output_dir="mpc_evaluation_results", mpc_history_file='../mpc_control_data/mpc_control_history.json'):
        """
        MPC控制效果评估器

        Args:
            data_dir: MPC控制数据目录（包含superquadric_params子目录）
            output_dir: 评估结果输出目录
            mpc_history_file: MPC控制历史JSON文件路径（可选）
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 存储控制历史文件路径
        self.mpc_history_file = mpc_history_file

        # 加载数据
        self.data = self._load_all_data()
        self.results = {}

        print(f"评估器初始化完成")
        print(f"数据目录: {data_dir}")
        print(f"控制历史文件: {mpc_history_file}")
        print(f"找到 {len(self.data)} 条数据记录")

    def _load_all_data(self):
        """加载所有超二次曲面参数文件"""
        param_files = sorted(glob.glob(
            str(self.data_dir / "superquadric_params" / "superquadric_step*.json")
        ))

        data_records = []
        for file_path in param_files:
            try:
                with open(file_path, 'r') as f:
                    record = json.load(f)

                # 如果缺少 feature_16d，动态构建它
                if 'feature_16d' not in record:
                    # 从 parameters_11d 中提取前11个参数
                    params = record['parameters_11d']
                    feature_16d = [
                        params['scale_a1'],
                        params['scale_a2'],
                        params['scale_a3'],
                        params['shape_epsilon1'],
                        params['shape_epsilon2'],
                        params['translation_x'],
                        params['translation_y'],
                        params['translation_z'],
                        params['euler_rx'],
                        params['euler_ry'],
                        params['euler_rz']
                    ]

                    # 从 geometric_features 中提取后5个参数
                    geo = record['geometric_features']
                    flatness_key = 'flatness' if 'flatness' in geo else 'flatness_top'
                    feature_16d.extend([
                        geo['volume'],
                        geo['elongation'],
                        geo[flatness_key],
                        geo['smoothness'],
                        geo['convexity']
                    ])

                    record['feature_16d'] = feature_16d

                data_records.append(record)
            except Exception as e:
                print(f"加载文件 {file_path} 失败: {e}")

        return data_records

    def compute_basic_metrics(self):
        """计算基础指标"""
        if not self.data:
            raise ValueError("没有找到有效数据")

        # 提取关键数据
        steps = [r['step'] for r in self.data]
        times = [r['time'] for r in self.data]
        states = np.array([r['feature_16d'] for r in self.data])

        # 假设第一个状态是初始状态，最后一个状态是最终状态
        initial_state = states[0]
        final_state = states[-1]

        # 目标状态（需要从MPC控制器获取，这里假设已知）
        # 在实际中，您可能需要从MPC配置中获取
        target_state = np.array([
            0.08, 0.08, 0.08,  # scale_a1,2,3
            1.0, 1.0,  # shape_epsilon1,2
            0.0, 0.0, 0.0,  # translation_x,y,z
            0.0, 0.0, 0.0,  # euler_rx,ry,rz
            0.002308,  # volume
            1.0, 0.0, 1.0, 1.0  # geometric features
        ])

        # 计算误差
        scale_initial_error = np.linalg.norm(initial_state[:3] - target_state[:3])
        scale_final_error = np.linalg.norm(final_state[:3] - target_state[:3])

        # 形状参数误差
        shape_initial_error = np.linalg.norm(initial_state[3:5] - target_state[3:5])
        shape_final_error = np.linalg.norm(final_state[3:5] - target_state[3:5])

        # 几何特征误差
        geo_initial_error = np.linalg.norm(initial_state[11:16] - target_state[11:16])
        geo_final_error = np.linalg.norm(final_state[11:16] - target_state[11:16])

        # 总体误差
        total_initial_error = np.linalg.norm(initial_state - target_state)
        total_final_error = np.linalg.norm(final_state - target_state)

        # 误差改善率
        error_improvement = (total_initial_error - total_final_error) / total_initial_error * 100

        # 收敛速度（达到90%目标的时间）
        errors = np.linalg.norm(states - target_state, axis=1)
        target_error = total_initial_error * 0.1  # 10%的初始误差

        convergence_step = None
        for i, err in enumerate(errors):
            if err <= target_error:
                convergence_step = steps[i]
                convergence_time = times[i]
                break

        metrics = {
            'initial_state': initial_state.tolist(),
            'final_state': final_state.tolist(),
            'target_state': target_state.tolist(),

            'scale_errors': {
                'initial': float(scale_initial_error),
                'final': float(scale_final_error),
                'improvement': float((scale_initial_error - scale_final_error) / scale_initial_error * 100)
            },

            'shape_errors': {
                'initial': float(shape_initial_error),
                'final': float(shape_final_error),
                'improvement': float((shape_initial_error - shape_final_error) / shape_initial_error * 100)
            },

            'geometry_errors': {
                'initial': float(geo_initial_error),
                'final': float(geo_final_error),
                'improvement': float((geo_initial_error - geo_final_error) / geo_initial_error * 100)
            },

            'total_errors': {
                'initial': float(total_initial_error),
                'final': float(total_final_error),
                'improvement': float(error_improvement)
            },

            'convergence': {
                'converged': convergence_step is not None,
                'convergence_step': convergence_step,
                'convergence_time': convergence_time if convergence_step else None,
                'convergence_rate': float(convergence_step / steps[-1] * 100) if convergence_step else None
            },

            'simulation_info': {
                'total_steps': steps[-1],
                'total_time': times[-1],
                'data_points': len(steps)
            }
        }

        self.results['basic_metrics'] = metrics
        return metrics


def main():
    """主评估函数"""
    # 配置评估参数
    DATA_DIR = r"..\realtime_data_demo_cylinder"  # 数据目录
    OUTPUT_DIR = "mpc_evaluation_results"

    # 创建评估器
    evaluator = MPCEvaluator(DATA_DIR, OUTPUT_DIR)

    # 创建状态轨迹图
    from mpc_visualizer import MPCVisualizer
    visualizer = MPCVisualizer(evaluator.data, evaluator.results, OUTPUT_DIR)
    visualizer.control_history = getattr(evaluator, 'control_history', [])
    visualizer.plot_state_trajectory()  # 只调用状态轨迹图函数

    print(f"\n评估完成！状态轨迹图已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()