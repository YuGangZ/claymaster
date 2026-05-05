"""
日志系统配置文件
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List


@dataclass
class LoggingPreset:
    """日志预设配置"""
    name: str
    description: str
    config: Dict[str, Any]

    def to_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'config': self.config
        }


# 预设配置
PRESETS = {
    'debug': LoggingPreset(
        name="debug",
        description="调试模式 - 详细日志",
        config={
            'console_level': 'DEBUG',
            'file_level': 'DEBUG',
            'flush_interval': 5,
            'enable_tensorboard': True,
            'enable_wandb': False
        }
    ),

    'training': LoggingPreset(
        name="training",
        description="训练模式 - 平衡性能与信息",
        config={
            'console_level': 'INFO',
            'file_level': 'DEBUG',
            'flush_interval': 10,
            'enable_tensorboard': True,
            'enable_wandb': False
        }
    ),

    'evaluation': LoggingPreset(
        name="evaluation",
        description="评估模式 - 简洁输出",
        config={
            'console_level': 'INFO',
            'file_level': 'INFO',
            'flush_interval': 20,
            'enable_tensorboard': False,
            'enable_wandb': False
        }
    ),

    'production': LoggingPreset(
        name="production",
        description="生产模式 - 最小日志",
        config={
            'console_level': 'WARNING',
            'file_level': 'INFO',
            'flush_interval': 50,
            'enable_tensorboard': False,
            'enable_wandb': True
        }
    ),

    'silent': LoggingPreset(
        name="silent",
        description="静默模式 - 无控制台输出",
        config={
            'console_level': 'ERROR',
            'file_level': 'INFO',
            'flush_interval': 100,
            'enable_tensorboard': False,
            'enable_wandb': False
        }
    )
}


def load_preset(preset_name: str) -> Dict[str, Any]:
    """加载预设配置"""
    if preset_name not in PRESETS:
        available = list(PRESETS.keys())
        raise ValueError(f"未知预设: {preset_name}，可用预设: {available}")

    return PRESETS[preset_name].config.copy()


def create_experiment_config(
        experiment_name: str,
        preset: str = 'training',
        custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """创建实验配置"""
    # 加载预设
    config = load_preset(preset)

    # 基础配置
    base_config = {
        'experiment_name': experiment_name,
        'log_dir': f"./experiments/{experiment_name}",
        'timestamp': None,  # 将在运行时设置
    }

    # 合并配置
    config.update(base_config)

    # 自定义配置覆盖
    if custom_config:
        config.update(custom_config)

    return config


def save_config(config: Dict[str, Any], filepath: Path):
    """保存配置到文件"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_config(filepath: Path) -> Dict[str, Any]:
    """从文件加载配置"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# 模块特定日志级别配置
MODULE_LOG_LEVELS = {
    # 模块: 日志级别
    'rl': 'INFO',  # RL训练
    'mpc': 'DEBUG',  # MPC优化
    'physics': 'WARNING',  # 物理仿真
    'controller': 'INFO',  # 控制器
    'estimator': 'DEBUG',  # 状态估计
    'performance': 'INFO',  # 性能监控
    'storage': 'WARNING',  # 数据存储
}


def get_module_log_level(module_name: str) -> str:
    """获取模块日志级别"""
    return MODULE_LOG_LEVELS.get(module_name, 'INFO')


# 实验命名约定
def generate_experiment_name(
        algorithm: str = "SAC",
        controller: str = "MPC",
        task: str = "deformation",
        timestamp: bool = True
) -> str:
    """生成实验名称"""
    from datetime import datetime

    name_parts = [
        f"{algorithm}",
        f"{controller}",
        f"{task}"
    ]

    name = "_".join(name_parts)

    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{name}_{timestamp_str}"

    return name


if __name__ == "__main__":
    # 测试配置生成
    config = create_experiment_config(
        experiment_name="test_experiment",
        preset="debug",
        custom_config={'enable_wandb': True}
    )

    print("生成的配置:")
    print(json.dumps(config, indent=2))

    # 测试实验命名
    exp_name = generate_experiment_name(
        algorithm="SAC",
        controller="MPC",
        task="box_deformation"
    )
    print(f"\n生成的实验名: {exp_name}")