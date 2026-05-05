"""
实验管理和跟踪系统
"""

import os
import json
import sys
import time
import yaml
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
import pandas as pd
import numpy as np


@dataclass
class ExperimentMetadata:
    """实验元数据"""
    # 基础信息
    experiment_id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # 配置信息
    algorithm: str = "SAC"
    controller: str = "MPC"
    task: str = "deformation"

    # 状态信息
    status: str = "created"  # created, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None

    # 结果信息
    final_reward: Optional[float] = None
    success_rate: Optional[float] = None
    best_model_path: Optional[str] = None

    # 环境信息
    git_hash: Optional[str] = None
    python_version: str = ""
    dependencies: Dict[str, str] = field(default_factory=dict)

    # 自定义标签
    tags: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    """实验跟踪器"""

    def __init__(self,
                 experiments_dir: Path = Path("./experiments"),
                 auto_load: bool = True):
        """
        初始化实验跟踪器

        Args:
            experiments_dir: 实验目录
            auto_load: 是否自动加载已有实验
        """
        self.experiments_dir = experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # 实验数据库
        self.experiments: Dict[str, ExperimentMetadata] = {}

        # 当前实验
        self.current_experiment: Optional[ExperimentMetadata] = None

        # 自动加载已有实验
        if auto_load:
            self.load_all_experiments()

    def generate_experiment_id(self, name: str) -> str:
        """生成实验ID"""
        # 基于名称和时间戳生成唯一ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"{name}_{timestamp}_{name_hash}"

    def create_experiment(self,
                          name: str,
                          description: str = "",
                          **kwargs) -> ExperimentMetadata:
        """创建新实验"""
        # 生成实验ID
        exp_id = self.generate_experiment_id(name)

        # 创建实验目录
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(exist_ok=True)

        # 创建子目录
        (exp_dir / "models").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "results").mkdir(exist_ok=True)

        # 创建元数据
        metadata = ExperimentMetadata(
            experiment_id=exp_id,
            name=name,
            description=description,
            **kwargs
        )

        # 添加环境信息
        metadata.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # 尝试获取git信息
        try:
            import subprocess
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=Path.cwd()
            ).decode().strip()
            metadata.git_hash = git_hash
        except:
            pass

        # 保存元数据
        self._save_metadata(exp_dir, metadata)

        # 添加到数据库
        self.experiments[exp_id] = metadata
        self.current_experiment = metadata

        print(f"✅ 创建实验: {name} (ID: {exp_id})")
        print(f"   目录: {exp_dir}")

        return metadata

    def start_experiment(self, experiment_id: str):
        """开始实验"""
        if experiment_id not in self.experiments:
            raise ValueError(f"实验不存在: {experiment_id}")

        metadata = self.experiments[experiment_id]
        metadata.status = "running"
        metadata.start_time = datetime.now().isoformat()

        # 更新元数据文件
        exp_dir = self.experiments_dir / experiment_id
        self._save_metadata(exp_dir, metadata)

        self.current_experiment = metadata
        print(f"🚀 开始实验: {metadata.name}")

    def complete_experiment(self,
                            final_reward: Optional[float] = None,
                            success_rate: Optional[float] = None,
                            best_model_path: Optional[str] = None):
        """完成实验"""
        if not self.current_experiment:
            raise ValueError("没有正在运行的实验")

        metadata = self.current_experiment
        metadata.status = "completed"
        metadata.end_time = datetime.now().isoformat()

        # 计算持续时间
        if metadata.start_time:
            start_dt = datetime.fromisoformat(metadata.start_time)
            end_dt = datetime.fromisoformat(metadata.end_time)
            metadata.duration_seconds = (end_dt - start_dt).total_seconds()

        # 记录结果
        metadata.final_reward = final_reward
        metadata.success_rate = success_rate
        metadata.best_model_path = best_model_path

        # 更新元数据
        exp_dir = self.experiments_dir / metadata.experiment_id
        self._save_metadata(exp_dir, metadata)

        print(f"✅ 完成实验: {metadata.name}")
        print(f"   持续时间: {metadata.duration_seconds:.1f}秒")
        if final_reward is not None:
            print(f"   最终奖励: {final_reward:.2f}")

    def fail_experiment(self, error_message: str):
        """标记实验失败"""
        if not self.current_experiment:
            raise ValueError("没有正在运行的实验")

        metadata = self.current_experiment
        metadata.status = "failed"
        metadata.end_time = datetime.now().isoformat()

        # 保存错误信息
        exp_dir = self.experiments_dir / metadata.experiment_id
        error_file = exp_dir / "error.log"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"实验失败: {error_message}\n")
            f.write(f"时间: {metadata.end_time}\n")

        # 更新元数据
        self._save_metadata(exp_dir, metadata)

        print(f"❌ 实验失败: {metadata.name}")
        print(f"   错误信息: {error_message}")

    def save_checkpoint(self,
                        checkpoint_data: Dict[str, Any],
                        name: str = "checkpoint"):
        """保存检查点"""
        if not self.current_experiment:
            raise ValueError("没有正在运行的实验")

        exp_dir = self.experiments_dir / self.current_experiment.experiment_id
        checkpoint_dir = exp_dir / "checkpoints"

        # 生成检查点文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = checkpoint_dir / f"{name}_{timestamp}.json"

        # 保存检查点
        checkpoint_data['timestamp'] = timestamp
        checkpoint_data['experiment_id'] = self.current_experiment.experiment_id

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, default=self._json_serializer)

        print(f"💾 保存检查点: {checkpoint_file}")
        return checkpoint_file

    def save_results(self,
                     results: Dict[str, Any],
                     filename: str = "final_results.json"):
        """保存实验结果"""
        if not self.current_experiment:
            raise ValueError("没有正在运行的实验")

        exp_dir = self.experiments_dir / self.current_experiment.experiment_id
        results_dir = exp_dir / "results"

        results_file = results_dir / filename

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)

        print(f"📊 保存结果: {results_file}")
        return results_file

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """记录超参数"""
        if not self.current_experiment:
            raise ValueError("没有正在运行的实验")

        self.current_experiment.hyperparameters.update(hyperparameters)

        # 立即保存
        exp_dir = self.experiments_dir / self.current_experiment.experiment_id
        self._save_metadata(exp_dir, self.current_experiment)

    def add_tag(self, tag: str):
        """添加标签"""
        if not self.current_experiment:
            raise ValueError("没有正在运行的实验")

        if tag not in self.current_experiment.tags:
            self.current_experiment.tags.append(tag)

        # 立即保存
        exp_dir = self.experiments_dir / self.current_experiment.experiment_id
        self._save_metadata(exp_dir, self.current_experiment)

    def get_experiment_dir(self) -> Path:
        """获取当前实验目录"""
        if not self.current_experiment:
            raise ValueError("没有正在运行的实验")

        return self.experiments_dir / self.current_experiment.experiment_id

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """获取实验元数据"""
        return self.experiments.get(experiment_id)

    def list_experiments(self,
                         status: Optional[str] = None,
                         tag: Optional[str] = None) -> List[ExperimentMetadata]:
        """列出实验"""
        experiments = list(self.experiments.values())

        # 状态过滤
        if status:
            experiments = [exp for exp in experiments if exp.status == status]

        # 标签过滤
        if tag:
            experiments = [exp for exp in experiments if tag in exp.tags]

        # 按创建时间排序
        experiments.sort(key=lambda x: x.created_at, reverse=True)

        return experiments

    def generate_report(self,
                        experiment_id: Optional[str] = None,
                        format: str = "markdown") -> str:
        """生成实验报告"""
        if experiment_id:
            metadata = self.get_experiment(experiment_id)
            if not metadata:
                return f"实验不存在: {experiment_id}"
            experiments = [metadata]
        else:
            experiments = self.list_experiments()

        if format == "markdown":
            return self._generate_markdown_report(experiments)
        elif format == "json":
            return self._generate_json_report(experiments)
        else:
            raise ValueError(f"未知格式: {format}")

    def _generate_markdown_report(self, experiments: List[ExperimentMetadata]) -> str:
        """生成Markdown报告"""
        report = "# 实验报告\n\n"

        for exp in experiments:
            report += f"## {exp.name}\n\n"
            report += f"- **ID**: `{exp.experiment_id}`\n"
            report += f"- **状态**: {exp.status}\n"
            report += f"- **创建时间**: {exp.created_at}\n"

            if exp.start_time:
                report += f"- **开始时间**: {exp.start_time}\n"

            if exp.end_time and exp.duration_seconds:
                report += f"- **结束时间**: {exp.end_time}\n"
                report += f"- **持续时间**: {exp.duration_seconds:.1f}秒\n"

            report += f"- **算法**: {exp.algorithm}\n"
            report += f"- **控制器**: {exp.controller}\n"
            report += f"- **任务**: {exp.task}\n"

            if exp.final_reward is not None:
                report += f"- **最终奖励**: {exp.final_reward:.2f}\n"

            if exp.success_rate is not None:
                report += f"- **成功率**: {exp.success_rate:.2%}\n"

            if exp.tags:
                report += f"- **标签**: {', '.join(exp.tags)}\n"

            if exp.description:
                report += f"\n**描述**: {exp.description}\n"

            report += "\n---\n\n"

        return report

    def _generate_json_report(self, experiments: List[ExperimentMetadata]) -> str:
        """生成JSON报告"""
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'total_experiments': len(experiments),
            'experiments': [asdict(exp) for exp in experiments]
        }

        return json.dumps(report_data, indent=2, default=str)

    def _save_metadata(self, exp_dir: Path, metadata: ExperimentMetadata):
        """保存元数据到文件"""
        metadata_file = exp_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)

    def _load_metadata(self, exp_dir: Path) -> Optional[ExperimentMetadata]:
        """从文件加载元数据"""
        metadata_file = exp_dir / "metadata.json"

        if not metadata_file.exists():
            return None

        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 转换为ExperimentMetadata对象
        return ExperimentMetadata(**data)

    def load_all_experiments(self):
        """加载所有实验"""
        self.experiments.clear()

        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            metadata = self._load_metadata(exp_dir)
            if metadata:
                self.experiments[metadata.experiment_id] = metadata

        print(f"📂 加载了 {len(self.experiments)} 个实验")

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
        elif isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# 全局实验跟踪器实例
_global_experiment_tracker = None


def get_global_tracker(experiments_dir: Optional[Path] = None) -> ExperimentTracker:
    """获取全局实验跟踪器"""
    global _global_experiment_tracker

    if _global_experiment_tracker is None:
        if experiments_dir is None:
            experiments_dir = Path("./experiments")

        _global_experiment_tracker = ExperimentTracker(experiments_dir)

    return _global_experiment_tracker


if __name__ == "__main__":
    # 测试实验跟踪器
    tracker = ExperimentTracker(Path("./test_experiments"), auto_load=False)

    # 创建实验
    exp1 = tracker.create_experiment(
        name="SAC_MPC_Deformation",
        description="SAC算法结合MPC控制器的变形控制实验",
        algorithm="SAC",
        controller="MPC",
        task="box_deformation"
    )

    # 开始实验
    tracker.start_experiment(exp1.experiment_id)

    # 记录超参数
    tracker.log_hyperparameters({
        "learning_rate": 3e-4,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005
    })

    # 添加标签
    tracker.add_tag("SAC")
    tracker.add_tag("MPC")
    tracker.add_tag("deformation")

    # 保存检查点
    tracker.save_checkpoint({
        "step": 1000,
        "reward": 25.3,
        "loss": 0.012
    })

    # 模拟完成实验
    time.sleep(1)
    tracker.complete_experiment(
        final_reward=150.5,
        success_rate=0.85,
        best_model_path="models/best_model.zip"
    )

    # 生成报告
    report = tracker.generate_report()
    print(report)