"""
强化学习专用日志系统
支持：控制台输出、文件记录、TensorBoard、WandB
"""

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import pandas as pd

# 尝试导入可选依赖
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class LogConfig:
    """日志配置数据类"""
    # 基础配置
    experiment_name: str = "rl_experiment"
    log_dir: str = "./logs"
    console_level: int = logging.INFO  # 控制台日志级别
    file_level: int = logging.DEBUG  # 文件日志级别
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    wandb_project: str = "deformation_rl"
    wandb_entity: Optional[str] = None

    # 性能配置
    flush_interval: int = 10  # 每多少步刷新一次日志
    max_log_files: int = 10  # 最大日志文件数

    # 输出配置
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # 模块过滤
    filter_modules: List[str] = None  # 过滤特定模块的日志

    def __post_init__(self):
        if self.filter_modules is None:
            self.filter_modules = ['genesis', 'urllib3']


class RLLogger:
    """强化学习专用日志器"""

    def __init__(self, config: Optional[LogConfig] = None, **kwargs):
        """
        初始化日志器

        Args:
            config: 日志配置
            **kwargs: 覆盖配置参数
        """
        # 合并配置
        self.config = config or LogConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # 创建日志目录
        self.log_dir = Path(self.config.log_dir) / self.config.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 设置时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 初始化组件
        self._init_loggers()
        self._init_writers()
        self._init_metrics()

        # 性能统计
        self.step_count = 0
        self.episode_count = 0
        self.start_time = time.time()

        # 模块特定的logger
        self.module_loggers = {}

        print(f"✅ RLLogger初始化完成 - 实验: {self.config.experiment_name}")
        print(f"   日志目录: {self.log_dir}")

    def _init_loggers(self):
        """初始化Python logging系统"""
        # 根logger配置
        logging.basicConfig(
            level=logging.WARNING,
            format=self.config.log_format,
            datefmt=self.config.date_format
        )

        # 创建主logger
        self.logger = logging.getLogger("rl_main")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # 清除默认handler

        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.config.console_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)8s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 文件handler
        log_file = self.log_dir / f"rl_log_{self.timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.config.file_level)
        file_formatter = logging.Formatter(
            self.config.log_format,
            datefmt=self.config.date_format
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # 添加模块过滤器
        for module in self.config.filter_modules:
            logging.getLogger(module).setLevel(logging.WARNING)

    def _init_writers(self):
        """初始化TensorBoard和WandB"""
        # TensorBoard
        self.tb_writer = None
        if self.config.enable_tensorboard and HAS_TENSORBOARD:
            tb_dir = self.log_dir / "tensorboard" / self.timestamp
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(str(tb_dir))
            self.info(f"TensorBoard日志: {tb_dir}")

        # WandB
        self.wandb_run = None
        if self.config.enable_wandb and HAS_WANDB:
            try:
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    name=f"{self.config.experiment_name}_{self.timestamp}",
                    dir=str(self.log_dir),
                    config=asdict(self.config)
                )
                self.info(f"WandB初始化完成: {self.wandb_run.id}")
            except Exception as e:
                self.warning(f"WandB初始化失败: {e}")

    def _init_metrics(self):
        """初始化指标记录"""
        self.metrics = {
            'training': pd.DataFrame(),
            'evaluation': pd.DataFrame(),
            'mpc': pd.DataFrame(),
            'performance': pd.DataFrame()
        }

        # 创建CSV文件
        self.csv_files = {}
        for metric_type in self.metrics.keys():
            csv_file = self.log_dir / f"{metric_type}_metrics_{self.timestamp}.csv"
            self.csv_files[metric_type] = csv_file

    def get_module_logger(self, module_name: str):
        """获取模块特定logger"""
        if module_name not in self.module_loggers:
            logger = logging.getLogger(f"rl.{module_name}")

            # 配置格式
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(self.config.console_level)
            formatter = logging.Formatter(
                f'%(asctime)s - {module_name:12s} - %(levelname)8s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)

            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False  # 不传播到根logger

            self.module_loggers[module_name] = logger

        return self.module_loggers[module_name]

    # ==================== 基础日志方法 ====================

    def debug(self, msg: str, module: str = "main"):
        """调试日志"""
        if module == "main":
            self.logger.debug(msg)
        else:
            self.get_module_logger(module).debug(msg)

    def info(self, msg: str, module: str = "main"):
        """信息日志"""
        if module == "main":
            self.logger.info(msg)
        else:
            self.get_module_logger(module).info(msg)

    def warning(self, msg: str, module: str = "main"):
        """警告日志"""
        if module == "main":
            self.logger.warning(msg)
        else:
            self.get_module_logger(module).warning(msg)

    def error(self, msg: str, module: str = "main"):
        """错误日志"""
        if module == "main":
            self.logger.error(msg)
        else:
            self.get_module_logger(module).error(msg)

    def critical(self, msg: str, module: str = "main"):
        """严重错误日志"""
        if module == "main":
            self.logger.critical(msg)
        else:
            self.get_module_logger(module).critical(msg)

    # ==================== 结构化日志方法 ====================

    def log_step(self, step_data: Dict[str, Any]):
        """记录训练步骤数据"""
        self.step_count += 1

        # 基础信息
        step_data['step'] = self.step_count
        step_data['timestamp'] = time.time() - self.start_time

        # 添加到DataFrame
        if 'episode_reward' in step_data:
            df = pd.DataFrame([step_data])
            self.metrics['training'] = pd.concat(
                [self.metrics['training'], df], ignore_index=True
            )

            # 定期保存
            if self.step_count % self.config.flush_interval == 0:
                self.save_metrics('training')

        # TensorBoard
        if self.tb_writer:
            for key, value in step_data.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f'training/{key}', value, self.step_count)

        # WandB
        if self.wandb_run:
            wandb.log(step_data, step=self.step_count)

    def log_episode(self, episode_data: Dict[str, Any]):
        """记录episode数据"""
        self.episode_count += 1
        episode_data['episode'] = self.episode_count
        episode_data['total_time'] = time.time() - self.start_time

        self.info(f"Episode {self.episode_count} 完成: "
                  f"奖励={episode_data.get('reward', 0):.2f}, "
                  f"步数={episode_data.get('steps', 0)}",
                  module="rl")

        # 保存到CSV
        df = pd.DataFrame([episode_data])
        self.metrics['evaluation'] = pd.concat(
            [self.metrics['evaluation'], df], ignore_index=True
        )

        # 可视化日志
        if self.episode_count % 10 == 0:
            self.save_metrics('evaluation')

    def log_mpc(self, mpc_data: Dict[str, Any]):
        """记录MPC数据"""
        # 添加时间戳
        mpc_data['mpc_timestamp'] = time.time()

        # 记录到DataFrame
        df = pd.DataFrame([mpc_data])
        self.metrics['mpc'] = pd.concat([self.metrics['mpc'], df], ignore_index=True)

        # 控制台输出（选择性）
        if mpc_data.get('iterations', 0) > 0:
            self.debug(f"MPC优化: {mpc_data.get('iterations', 0)}次迭代, "
                       f"代价={mpc_data.get('cost', 0):.4f}",
                       module="mpc")

    def log_performance(self,
                        component: str,
                        duration: float,
                        details: Optional[Dict] = None):
        """记录性能数据"""
        perf_data = {
            'component': component,
            'duration_ms': duration * 1000,
            'step': self.step_count,
            'timestamp': time.time()
        }

        if details:
            perf_data.update(details)

        df = pd.DataFrame([perf_data])
        self.metrics['performance'] = pd.concat(
            [self.metrics['performance'], df], ignore_index=True
        )

        # 如果耗时过长，发出警告
        if duration > 0.1:  # 超过100ms
            self.warning(f"{component} 耗时过长: {duration * 1000:.1f}ms",
                         module="performance")

    # ==================== 可视化方法 ====================

    def plot_training_curves(self, save: bool = True):
        """绘制训练曲线"""
        if self.metrics['training'].empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        df = self.metrics['training']

        # 1. 奖励曲线
        if 'episode_reward' in df.columns:
            axes[0, 0].plot(df['step'], df['episode_reward'], 'b-', alpha=0.5)
            axes[0, 0].set_title('Episode Reward')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)

        # 2. 成功率曲线
        if 'success' in df.columns:
            window = 20
            if len(df) > window:
                success_rate = df['success'].rolling(window).mean()
                axes[0, 1].plot(df['step'][window - 1:], success_rate[window - 1:], 'g-')
                axes[0, 1].set_title(f'Success Rate (window={window})')
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Success Rate')
                axes[0, 1].grid(True, alpha=0.3)

        # 3. 损失曲线
        if 'loss' in df.columns:
            axes[1, 0].plot(df['step'], df['loss'], 'r-', alpha=0.7)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Episode长度
        if 'episode_length' in df.columns:
            axes[1, 1].hist(df['episode_length'].dropna(), bins=20, alpha=0.7)
            axes[1, 1].set_title('Episode Length Distribution')
            axes[1, 1].set_xlabel('Steps per Episode')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plot_file = self.log_dir / f"training_curves_{self.timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.info(f"训练曲线已保存: {plot_file}")
        else:
            plt.show()

    # ==================== 文件操作方法 ====================

    def save_metrics(self, metric_type: str = 'training'):
        """保存指标到CSV"""
        if metric_type in self.metrics and not self.metrics[metric_type].empty:
            filepath = self.csv_files[metric_type]
            self.metrics[metric_type].to_csv(filepath, index=False)

            # 只记录第一次和定期记录
            if self.step_count % 100 == 0:
                self.debug(f"{metric_type}指标已保存: {filepath}", module="storage")

    def save_config(self, config: Dict[str, Any], filename: str = "experiment_config.json"):
        """保存实验配置"""
        config_file = self.log_dir / filename
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, default=self._json_serializer)
        self.info(f"实验配置已保存: {config_file}")

    def save_model_info(self, model_info: Dict[str, Any], filename: str = "model_info.json"):
        """保存模型信息"""
        info_file = self.log_dir / filename
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, default=self._json_serializer)
        self.info(f"模型信息已保存: {info_file}")

    def generate_report(self):
        """生成实验报告"""
        report = {
            'experiment': self.config.experiment_name,
            'timestamp': self.timestamp,
            'total_steps': self.step_count,
            'total_episodes': self.episode_count,
            'total_time_hours': (time.time() - self.start_time) / 3600,
            'performance_summary': self._get_performance_summary()
        }

        report_file = self.log_dir / f"experiment_report_{self.timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        self.info(f"实验报告已生成: {report_file}")
        return report_file

    def _get_performance_summary(self):
        """获取性能摘要"""
        if self.metrics['performance'].empty:
            return {}

        summary = {}
        for component in self.metrics['performance']['component'].unique():
            comp_data = self.metrics['performance'][
                self.metrics['performance']['component'] == component
                ]
            summary[component] = {
                'count': len(comp_data),
                'mean_duration_ms': comp_data['duration_ms'].mean(),
                'max_duration_ms': comp_data['duration_ms'].max(),
                'min_duration_ms': comp_data['duration_ms'].min()
            }

        return summary

    # ==================== 辅助方法 ====================

    @staticmethod
    def _json_serializer(obj):
        """JSON序列化辅助函数"""
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def get_log_file_path(self, filename: str) -> Path:
        """获取日志文件路径"""
        return self.log_dir / filename

    def close(self):
        """关闭日志器，保存所有数据"""
        # 保存所有指标
        for metric_type in self.metrics.keys():
            self.save_metrics(metric_type)

        # 生成报告
        self.generate_report()

        # 关闭TensorBoard writer
        if self.tb_writer:
            self.tb_writer.close()

        # 关闭WandB
        if self.wandb_run:
            self.wandb_run.finish()

        self.info("日志系统已关闭，所有数据已保存")

    # ==================== 装饰器 ====================

    def log_execution_time(self, component: str = "unknown"):
        """记录函数执行时间的装饰器"""

        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                self.log_performance(
                    component=f"{component}.{func.__name__}",
                    duration=duration
                )

                return result

            return wrapper

        return decorator

    def log_episode_decorator(self):
        """记录episode的装饰器"""

        def decorator(func):
            def wrapper(*args, **kwargs):
                episode_data = func(*args, **kwargs)
                if isinstance(episode_data, dict):
                    self.log_episode(episode_data)
                return episode_data

            return wrapper

        return decorator


# ==================== 便捷函数 ====================

def get_default_logger(experiment_name: str = None) -> RLLogger:
    """获取默认日志器（单例模式）"""
    if not hasattr(get_default_logger, "_instance"):
        config = LogConfig(
            experiment_name=experiment_name or f"experiment_{int(time.time())}",
            log_dir="./experiment_logs"
        )
        get_default_logger._instance = RLLogger(config)

    if experiment_name:
        get_default_logger._instance.config.experiment_name = experiment_name

    return get_default_logger._instance


def setup_logging(experiment_name: str, **kwargs) -> RLLogger:
    """快速设置日志系统"""
    config = LogConfig(experiment_name=experiment_name, **kwargs)
    return RLLogger(config)


if __name__ == "__main__":
    # 测试日志系统
    logger = setup_logging(
        experiment_name="test_logging",
        console_level=logging.INFO,
        enable_tensorboard=False,
        enable_wandb=False
    )

    # 测试各种日志级别
    logger.debug("这是一条调试信息", module="test")
    logger.info("这是一条信息", module="test")
    logger.warning("这是一条警告", module="test")

    # 测试结构化日志
    for i in range(5):
        logger.log_step({
            'step_reward': np.random.randn(),
            'loss': np.random.rand() * 0.1,
            'success': np.random.rand() > 0.5
        })


    # 测试性能日志
    @logger.log_execution_time("test_component")
    def slow_function():
        time.sleep(0.01)
        return "done"


    slow_function()

    # 保存和关闭
    logger.generate_report()
    logger.close()

    print("✅ 日志系统测试完成")