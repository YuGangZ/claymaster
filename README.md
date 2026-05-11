# 粘土塑形控制项目

## 项目概述

本项目是一个基于超二次曲面（Superquadric）的软体变形控制研究项目，主要研究如何通过机械臂与柔软物体的交互，实现对物体形状的精确控制。项目包含**分层强化学习（RL+MPC）**和**RL基线对比**两种方法。

## 环境要求

| 项目 | 当前环境版本 |
|-----|-------------|
| Python | 3.9+ |
| PyTorch | 2.0.1+cu118 |
| CUDA | 11.8 |
| genesis-world | 0.2.1 |

## 依赖包列表

```txt
# 核心依赖
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
scipy>=1.9.0
gymnasium>=0.29.0
stable_baselines3>=2.7.0
genesis-world>=0.2.0

# 可视化与点云处理
open3d>=0.19.0
matplotlib>=3.8.0
mayavi>=4.7.0
trimesh>=3.23.0
opencv-python>=4.11.0
pyqt5>=5.15.0

# 数据分析
pandas>=2.0.0
seaborn>=0.13.0
scikit-learn>=1.3.0

# 其他工具
pyyaml>=6.0
tqdm>=4.66.0
tensorboard>=2.15.0
```

## 项目结构

```
github_upload/
├── rl/                    # 分层强化学习方法（RL + MPC）
│   ├── config/
│   │   └── rl_config.py   # RL配置参数
│   ├── core/
│   │   ├── rl_env_wrapper.py    # 分层RL环境封装
│   │   └── rl_trainer.py        # RL训练器
│   ├── rl_training_logs/   # 训练日志和检查点
│   ├── motion_rl.py        # RL专用运动控制器
│   ├── run_rl_simple.py    # RL运行脚本
│   └── shape_predictor.pth # 预训练预测模型
├── baseline/              # 基线方法（RL）
│   ├── config.py          # 基线配置
│   ├── motion_controller_direct.py  # 直接运动控制器
│   ├── rl_env_direct.py   # 直接RL环境（3维动作）
│   ├── train_direct.py    # 基线训练脚本
│   └── shape_predictor.pth # 预训练模型
├── physical_engine/       # Genesis MPM物理引擎模块
│   ├── soft_sim_env.py    # 软体仿真环境
│   └── state_monitor.py   # MPM状态监控器
├── common/                # 公共模块
│   └── base/
│       ├── control_phase_manager.py # 控制阶段管理器
│       ├── controller_interface.py  # 控制器接口
│       └── motion.py      # 基础运动控制工具
├── data_collect/          # 数据收集模块
│   ├── main_datac.py      # 数据收集主程序
│   ├── motion_datac.py    # 数据收集运动控制器
│   └── realtime_data_random/  # 随机变形实验数据
├── manipulator/           # 真实机械臂控制模块
│   ├── pointcloud_reconstruct/  # 点云重建
│   ├── admittance_control.py    # 导纳控制
│   ├── delta_api.py       # Delta机器人API
│   ├── expert.py          # 专家控制策略
│   └── main_control.py    # 主控制程序
├── model_log_var/         # 形状预测模型（带不确定性估计）
│   ├── data/              # 训练数据处理
│   ├── eval/              # 模型评估
│   ├── shape_predictor.py # 核心神经网络模型
│   ├── train.py           # 模型训练脚本
│   └── shape_predictor_best.pth # 最佳模型权重
└── mpc/                   # 模型预测控制模块
    ├── mpc_eval_14d/      # MPC评估结果
    ├── main_mpc.py        # MPC主程序
    ├── mpc_controller.py  # MPC控制器
    └── mpc_core.py        # MPC核心算法
```

## 核心方法对比

| 方法 | 策略输出 | 底层执行 | 物理引擎 | 特点 |
|-----|---------|---------|---------|------|
| **rl/**（分层RL） | 14维子目标增量 + 感知决策 | MPC/控制器执行 | Genesis MPM | 分层架构，主动感知 |
| **baseline/**（RL+MPC） | 3维速度控制 | 直接物理步进 | Genesis MPM | 端到端控制，无分层 |

## 核心模块详解

### 1. 分层强化学习 (`rl/`)

**核心设计理念**：采用分层控制架构

- **高层策略**：输出15维动作
  - 第0维：感知决策（是否执行主动感知）
  - 第1~14维：子目标增量（14维超二次曲面参数变化）

- **底层控制器**：执行3维速度控制，跟踪高层子目标

**关键组件**：

| 文件 | 功能 |
|-----|------|
| `rl/core/rl_env_wrapper.py` | Gym风格分层RL环境 |
| `rl/core/rl_trainer.py` | RL训练器（支持SAC等算法） |
| `rl/motion_rl.py` | RL专用运动控制器 |
| `rl/config/rl_config.py` | RL超参数配置 |

**观测空间**：45维 = 14维当前状态 + 14维目标状态 + 3维末端位置 + 14维不确定性

**奖励函数设计**：
- 目标距离奖励：鼓励向目标靠近
- 稀疏成功奖励：达到目标时给予奖励
- 时间步惩罚：鼓励快速完成
- 安全约束：防止体积剧烈变化
- 不确定性惩罚：鼓励降低预测不确定性

### 2. 基线方法 (`baseline/`)

**设计理念**：直接输出控制信号的端到端强化学习

- **动作空间**：3维连续速度控制
- **无需分层**：RL策略直接输出执行器速度

**关键组件**：

| 文件 | 功能 |
|-----|------|
| `baseline/rl_env_direct.py` | 直接RL环境（3维动作） |
| `baseline/motion_controller_direct.py` | 直接运动控制器 |
| `baseline/train_direct.py` | 基线训练脚本 |

### 3. 物理引擎模块 (`physical_engine/`)

基于**Genesis物理引擎**（MPM - Material Point Method）实现软体仿真：

**核心功能**：

| 组件 | 功能 |
|-----|------|
| `SoftSimEnv` | 软体仿真环境设置 |
| `MPMMonitor` | MPM状态监控器基类 |
| `ElasticBodyMonitor` | 弹性体（传感器立方体）状态监控 |
| `ElastoPlasticBodyMonitor` | 弹塑性体（被操作物体）状态监控 |
| `ContactMonitor` | 接触检测与状态估计 |

**材料配置**：
- **弹性材料**：Neo-Hookean模型，用于传感器立方体
- **弹塑性材料**：Von Mises屈服准则，用于被操作物体


### 4. 形状预测模型 (`model_log_var/`)

基于深度学习的超二次曲面形状预测模型：

- **多模态特征编码器**：分别对尺度、形状、姿态、几何和控制信号进行编码
- **交叉注意力机制**：捕获不同特征之间的交互关系
- **不确定性估计**：通过对数方差（Log Variance）预测模型输出的不确定性
- **残差融合网络**：融合多模态特征并预测状态变化

**输入输出**：
- 输入：17维（14维当前状态 + 3维控制指令）
- 输出：14维状态变化量 + 14维不确定性估计

### 5. 模型预测控制 (`mpc/`)

基于预测模型的变形控制器：

- **DeformationMPC**：专门设计用于变形控制的MPC算法
- **14维状态表示**：包含尺度（3维）、形状参数（2维）、位置（3维）、姿态（3维）、几何参数（3维）
- **支持相对模式和绝对模式**：灵活的目标设定方式

## 超二次曲面参数格式（14维）

| 维度 | 参数类型 | 说明 |
|-----|---------|------|
| 0-2 | 尺度参数 | 长、宽、高 (scale_a1, scale_a2, scale_a3) |
| 3-4 | 形状参数 | 超二次曲面指数 (epsilon1, epsilon2) |
| 5-7 | 位置 | XYZ坐标 (translation_x, translation_y, translation_z) |
| 8-10 | 姿态 | 欧拉角 (euler_rx, euler_ry, euler_rz) |
| 11-13 | 几何参数 | 体积、伸长率、平滑度 |

## 使用方法

### 训练形状预测模型

```bash
cd model_log_var
python train.py \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --model_path ./shape_predictor_best.pth
```

### 运行分层RL训练（rl/）

```bash
cd rl
python run_rl_simple.py \
    --config config/rl_config.py \
    --log_dir rl_training_logs \
    --num_episodes 10000
```

### 运行基线RL训练（baseline/）

```bash
cd baseline
python train_direct.py \
    --config config.py \
    --log_dir ./logs
```

### 运行MPC控制

```bash
cd mpc
python main_mpc.py \
    --model_path ../model_log_var/shape_predictor_best.pth \
    --horizon 10 \
    --iterations 100
```

### 数据收集

```bash
cd data_collect
python main_datac.py \
    --output_dir ./realtime_data_random \
    --num_samples 1000
```

## 实验结果

### 预训练模型

项目提供了预训练的形状预测模型：
- `model_log_var/shape_predictor_best.pth` - 最佳模型权重
- `rl/shape_predictor.pth` - RL专用模型
- `baseline/shape_predictor.pth` - 基线专用模型

### 实验数据

- `data_collect/realtime_data_random/`：随机变形实验数据
- `mpc/realtime_data_demo_cylinder/`：圆柱体变形演示数据（含200+帧点云和超二次曲面参数）

### 评估结果

- `model_log_var/eval/plots/`：模型评估图表（不确定性可视化）
- `model_log_var/eval/test_results_14dim/`：14维预测结果统计
- `rl/rl_training_logs/`：分层RL训练日志和检查点（21000+步）
- `mpc/mpc_eval_14d/`：MPC控制评估结果

