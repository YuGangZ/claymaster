import sys
import os
# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from physical_engine.soft_sim_env import SoftSimEnv
from motion_mpc import MotionControllerMPC
from mpc_controller import MPCController
import numpy as np
import argparse


def main():
    print("=== MPC控制软体塑形仿真 ===")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MPC控制软体塑形仿真')
    parser.add_argument('--output', '-o', default='mpc_control_data',
                        help='输出目录路径')
    parser.add_argument('--interval', '-i', type=int, default=5,
                        help='数据保存间隔（仿真步数）')
    parser.add_argument('--model-path', '-m', default='shape_predictor_best.pth',
                        help='MPC模型路径')
    # 新增参数：MPC时域（根据需要调整）
    parser.add_argument('--horizon', type=int, default=10,
                        help='MPC预测时域')
    args = parser.parse_args()

    # 设置仿真引擎
    engine = SoftSimEnv()
    scene, solver, sensor_cube, elastoplastic_obj = engine.setup_simulation("cylinder")
    initial_particles = engine.initialize_cube_particles()

    # 初始化重新设计的MPC控制器
    mpc_controller = MPCController(
        model_path=args.model_path,
        device="cuda",
        horizon=args.horizon,  # 使用参数中的时域
        lr=0.001,  # 调整学习率
        iterations=100  # 调整迭代次数
    )
    total_steps = 4000
    # 目标形状（保持不变）
    TARGET_SHAPE = np.array([
        0.065, 0.065, 0.065,  # scale_a1,2,3
        1.0, 1.0,  # shape_epsilon1,2
        0.0, 0.0, 0.0,  # translation_x,y,z
        0.0, 0.0, 0.0,  # euler_rx,ry,rz
        0.00019,  # volume
        1.0, 0.0, 1.0, 1.0  # elongation, flatness, smoothness, convexity
    ])

    mpc_controller.set_target(TARGET_SHAPE)

    # 创建运动控制器（集成新的MPC）
    motion_controller = MotionControllerMPC(
        scene, sensor_cube, elastoplastic_obj, initial_particles,
        mpc_controller=mpc_controller,  # 这里会自动适配新控制器
        output_dir=args.output
    )

    # 设置保存间隔
    motion_controller.estimation_interval = args.interval

    print(f"数据保存目录: {args.output}")
    print(f"保存间隔: 每 {args.interval} 步保存一次")
    print(f"MPC时域: {args.horizon}")
    print("控制模式: MPC闭环控制")

    # 主仿真循环
    for i in range(total_steps):
        t = i * scene.dt

        # 获取当前状态
        force_state_update = (i % 20 == 0) or (i < 10)
        current_state = motion_controller.get_system_state(i, t, force_update=force_state_update)

        # 更新运动阶段（MPC控制）
        motion_controller.update_motion_phase(current_state, t)

        # 执行超二次曲面估计和实时保存
        motion_controller.estimation_counter += 1
        if motion_controller.estimation_counter >= motion_controller.estimation_interval:
            motion_controller.estimate_and_save_superquadric(current_state)
            motion_controller.estimation_counter = 0

        # 计算并应用MPC控制速度
        vel_array = motion_controller.calculate_mpc_velocity(current_state)
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

    # 打印MPC控制统计
    print("\n=== MPC控制统计 ===")
    status = mpc_controller.get_status()
    print(f"总控制步数: {status['step']}")
    print(f"最终控制指令: {status['last_control']}")
    print(f"MPC时域: {status['mpc_status']['horizon'] if 'mpc_status' in status else 'N/A'}")

    if hasattr(scene, 'viewer') and scene.viewer is not None:
        scene.viewer.stop()


if __name__ == "__main__":
    main()