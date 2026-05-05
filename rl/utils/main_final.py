import os
import sys
import logging

# ============ 强制日志配置 ============
# 设置环境变量
os.environ['GENESIS_VERBOSITY'] = 'warning'

# 配置Python标准logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

# 彻底禁用genesis logger
genesis_logger = logging.getLogger('genesis')
genesis_logger.setLevel(logging.WARNING)
genesis_logger.propagate = False  # 不向上传播
genesis_logger.handlers = [logging.NullHandler()]  # 添加空处理器


# 添加过滤器阻止genesis消息
class GenesisLogFilter(logging.Filter):
    def filter(self, record):
        return not record.name.startswith('genesis')


root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(GenesisLogFilter())
# =====================================

import numpy as np
import argparse
import torch

from physical_engine.soft_sim_env import SoftSimEnv
from motion_controller_mpc import MotionControllerMPC
from mpc_controller import MPCController
from rl_trainer import RLDeformationTrainer

RL_AVAILABLE = True


def create_target_shapes():
    """创建不同的目标形状用于训练"""
    base_shape = np.array([
        1.0, 1.0, 1.0,  # scale_a1,2,3
        0.5, 0.5,  # shape_epsilon1,2
        0.0, 0.0, 0.0,  # translation_x,y,z
        0.0, 0.0, 0.0,  # euler_rx,ry,rz
        1.0,  # volume
        1.0, 0.0, 0.5, 1.0  # elongation, flatness, smoothness, convexity
    ], dtype=np.float32)

    # 确保是16维
    if len(base_shape) != 16:
        # 如果不足16维，用默认值填充
        base_shape = np.pad(base_shape, (0, 16 - len(base_shape)), 'constant', constant_values=0.5)

    # 创建不同的变形目标
    targets = []

    # 压扁目标
    flat_target = base_shape.copy()
    flat_target[2] = 0.5  # z尺度减小
    flat_target[13] = 0.8  # flatness增加
    targets.append(flat_target)

    # 拉长目标
    stretch_target = base_shape.copy()
    stretch_target[0] = 1.5  # x尺度增加
    stretch_target[12] = 1.5  # elongation增加
    targets.append(stretch_target)

    # 对称目标
    symmetric_target = base_shape.copy()
    symmetric_target[14] = 0.9  # smoothness增加
    targets.append(symmetric_target)

    return targets


def main_mpc_mode(args):
    """MPC控制模式"""
    print("=== MPC控制软体塑形仿真 ===")

    # 设置仿真引擎
    engine = SoftSimEnv()
    scene, solver, sensor_cube, elastoplastic_obj = engine.setup_simulation("box")
    initial_particles = engine.initialize_cube_particles()

    # 创建运动控制器（根据控制模式）
    motion_controller = MotionControllerMPC(
        scene, sensor_cube, elastoplastic_obj, initial_particles,
        control_mode=args.control_mode,
        output_dir=args.output,
        model_path=args.model_path,
        device="gpu" if torch.cuda.is_available() else "cpu",
        horizon=5,
        dt=0.05,
        approach_speed=-0.3,  # 更快地接近
        control_interval=50    # 控制间隔为50步
    )

    # 设置目标形状（示例目标）
    TARGET_SHAPE = np.array([
        0.1996, 0.1996, 0.1996,  # scale_a1,2,3
        0.0, 0.0,  # shape_epsilon1,2
        0.0, 0.0, 0.0,  # translation_x,y,z
        0.0, 0.0, 0.0,  # euler_rx,ry,rz
        0.003375,  # volume
        1.0, 0.0, 1.0, 1.0  # elongation, flatness, smoothness, convexity
    ], dtype=np.float32)

    # 确保是16维
    if len(TARGET_SHAPE) != 16:
        TARGET_SHAPE = np.pad(TARGET_SHAPE, (0, 16 - len(TARGET_SHAPE)), 'constant', constant_values=0.5)

    motion_controller.set_target_shape(TARGET_SHAPE)

    # 设置保存间隔
    motion_controller.estimation_interval = args.interval

    print(f"数据保存目录: {args.output}")
    print(f"保存间隔: 每 {args.interval} 步保存一次")
    print(f"控制模式: {args.control_mode}")

    # 主仿真循环
    for i in range(motion_controller.total_steps):
        t = i * scene.dt

        # 获取当前状态
        force_state_update = (i % 10 == 0) or (i < 10)  # 提高更新频率
        current_state = motion_controller.get_system_state_fast(i, t, force_update=force_state_update)

        # 执行超二次曲面估计和实时保存
        motion_controller.estimation_counter += 1
        if motion_controller.estimation_counter >= motion_controller.estimation_interval:
            motion_controller.estimate_and_save_superquadric(current_state)
            motion_controller.estimation_counter = 0

        # 使用阶段管理器计算并应用速度
        vel_array = motion_controller.compute_velocity(current_state)
        motion_controller.apply_velocity(vel_array)

        # 推进仿真
        scene.step()

        # 打印状态
        if motion_controller.should_print_status(i, current_state):
            motion_controller.print_state_info(current_state)

        # 终止条件
        if t > 10.0:  # 仿真时间
            print("=== 仿真时间结束 ===")
            break

    print("=== 仿真完成 ===")

    # 最终数据导出
    motion_controller.finalize_simulation()
    motion_controller.save_custom_format_data()


def main_rl_mode(args):
    """RL训练模式"""
    if not RL_AVAILABLE:
        print("错误: RL模块不可用，请安装相关依赖")
        return

    print("=== RL训练软体塑形仿真 ===")

    # 设置仿真引擎
    engine = SoftSimEnv()
    scene, solver, sensor_cube, elastoplastic_obj = engine.setup_simulation("box")
    initial_particles = engine.initialize_cube_particles()

    # 创建运动控制器
    motion_controller = MotionControllerMPC(
        scene, sensor_cube, elastoplastic_obj, initial_particles,
        control_mode='mpc',  # RL模式下使用MPC作为底层控制器
        output_dir=args.output + "_rl",
        model_path=args.model_path,
        device="cpu",
        horizon=5,
        dt=0.05
    )

    # 设置目标形状
    target_shapes = create_target_shapes()
    target_shape = target_shapes[0]  # 使用第一个目标形状

    print(f"RL训练数据保存目录: {args.output}_rl")
    print(f"目标形状维度: {len(target_shape)}")
    print("控制模式: RL高层策略 + MPC底层执行")

    # 创建RL训练器
    # trainer = RLDeformationTrainer(motion_controller, target_shape)

    print("警告: RL模式需要更新RL训练器以适配新的阶段管理器架构")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='软体塑形控制仿真')

    # 基础参数
    parser.add_argument('--output', '-o', default='control_data',
                        help='输出目录路径')
    parser.add_argument('--interval', '-i', type=int, default=5,
                        help='数据保存间隔（仿真步数）')
    parser.add_argument('--model-path', '-m', default='shape_predictor.pth',
                        help='MPC模型路径')

    # 模式选择
    parser.add_argument('--mode', choices=['mpc', 'rl'], default='rl',
                        help='运行模式: mpc (仅MPC) 或 rl (RL训练)')
    parser.add_argument('--control-mode', choices=['approach_only', 'mpc', 'dummy', 'openloop'],
                        default='dummy', help='控制模式')
    # RL相关参数
    parser.add_argument('--rl-action', choices=['train', 'evaluate', 'deploy'], default='train',
                        help='RL操作模式: train (训练), evaluate (评估), deploy (部署)')
    parser.add_argument('--rl-timesteps', type=int, default=50000,
                        help='RL训练总步数')
    parser.add_argument('--rl-episodes', type=int, default=500,
                        help='RL评估episode数')
    parser.add_argument('--rl-model-path', type=str,
                        help='RL模型路径 (用于评估和部署)')

    args = parser.parse_args()

    if args.mode == 'rl':
        main_rl_mode(args)
    else:
        main_mpc_mode(args)


if __name__ == "__main__":
    main()