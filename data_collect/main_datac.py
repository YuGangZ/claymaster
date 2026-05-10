import numpy as np
if not hasattr(np, 'typing'):
    np.typing = np._typing
import sys
import os
# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from physical_engine.soft_sim_env import SoftSimEnv
from motion_datac import MotionControllerDataCollection
import argparse
import torch
import gc
def main():
    print("=== 开始立方体多阶段运动仿真 ===")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MPM仿真实时数据保存')
    parser.add_argument('--output', '-o', default='realtime_data_random',
                        help='输出目录路径')
    parser.add_argument('--interval', '-i', type=int, default=25,
                        help='数据保存间隔（仿真步数）')
    parser.add_argument('--action-duration', '-d', type=int, default=100,
                        help='每个随机动作的持续步数')
    parser.add_argument('--total-steps', '-s', type=int, default=5000,
                        help='总仿真步数')
    parser.add_argument('--traditional', '-t', action='store_true',
                        help='使用传统三阶段运动模式（默认使用随机动作模式）')
    args = parser.parse_args()

    elastoplastic_shape = "cylinder"
    # elastoplastic_shape = "mesh"
    mesh_file_path = None#"test_bunny.stl"
    # 设置仿真引擎
    engine = SoftSimEnv()
    scene, solver, sensor_cube, elastoplastic_obj = engine.setup_simulation(elastoplastic_shape, mesh_file_path)

    # 初始化立方体粒子
    initial_particles = engine.initialize_cube_particles()

    # 创建运动控制器 - 确保输出目录相对于data_collect目录
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    motion_controller = MotionControllerDataCollection(
        scene, sensor_cube, elastoplastic_obj, initial_particles,total_steps=args.total_steps,
        output_dir=output_dir
    )

    # 设置参数
    motion_controller.estimation_interval = args.interval
    motion_controller.action_duration = args.action_duration
    motion_controller.use_traditional_mode = args.traditional

    # 控制标志
    motion_controller.sampling_enabled = False  # 初始不进行采样
    motion_controller.first_contact_step = None  # 记录首次接触的步数
    motion_controller.has_contact_ever_established = False  # 接触是否曾经建立过

    print(f"数据保存目录: {args.output}")
    print(f"保存间隔: 每 {args.interval} 步保存一次")
    if motion_controller.use_traditional_mode:
        print("运行模式: 传统三阶段运动")
    else:
        print("运行模式: 随机动作模式")
        print(f"动作持续时间: 每 {args.action_duration} 步切换动作")

    print("=== 采样触发条件: 等待接触建立 ===")

    # 主仿真循环
    for i in range(motion_controller.total_steps):
        t = i * scene.dt
        # 获取当前状态
        force_state_update = (i % 20 == 0) or (i < 10)
        current_state = motion_controller.get_system_state(i, t, force_update=force_state_update)

        # 获取接触信息
        contact_info = current_state.get('contact', {})
        is_contact = contact_info.get('contact_detected', False)
        penetration_depth = contact_info.get('penetration_depth', 0)

        # 更新采样启用状态：首次检测到接触时启用
        if not motion_controller.has_contact_ever_established and is_contact:
            motion_controller.sampling_enabled = True
            motion_controller.first_contact_step = i
            motion_controller.has_contact_ever_established = True
            print(f"\n=== 接触建立！开始实时采样和估计 ===")
            print(f"首次接触步数: {i}, 穿透深度: {penetration_depth:.4f}m")

            # 如果是随机模式，接触建立后启用随机动作
            if not motion_controller.use_traditional_mode:
                motion_controller.random_action_enabled = True
                motion_controller._select_random_action()
                print(f"=== 启动随机动作模式 ===")

        # 更新运动阶段
        motion_controller.update_motion_phase(current_state, t)

        # 一旦接触建立，进行超二次曲面估计
        motion_controller.estimation_counter += 1
        if motion_controller.sampling_enabled and motion_controller.estimation_counter >= motion_controller.estimation_interval:
            # 执行超二次曲面估计和保存
            estimation_result = motion_controller.estimate_and_save_superquadric(current_state)
            if estimation_result:
                print(f"步骤 {i}: 数据保存成功")
            motion_controller.estimation_counter = 0

        # 计算并应用速度
        vel_array = motion_controller.calculate_velocity()
        motion_controller.apply_velocity(vel_array)
        if i % 200 == 0:  # 每100步清理一次
            clear_gpu_memory()
        # 推进仿真
        scene.step()

        # 打印状态（包含采样状态信息）
        if motion_controller.should_print_status(i, current_state):
            # 添加采样状态信息
            print(f"采样状态: {'已启用' if motion_controller.sampling_enabled else '等待接触'}")
            print(f"接触状态: {'已建立' if motion_controller.has_contact_ever_established else '未建立'}")
            if is_contact:
                print(f"接触深度: {penetration_depth:.4f}m")
            motion_controller.print_state_info(current_state)

        if i >= motion_controller.total_steps - 1:
            print("=== 达到最大步数，仿真结束 ===")
            break

    print("=== 仿真完成 ===")

    # 执行最后一次超二次曲面估计（如果采样已启用）
    if motion_controller.sampling_enabled:
        print("\n=== 执行最终超二次曲面估计 ===")
        final_state = motion_controller.get_system_state(
            motion_controller.total_steps,
            motion_controller.total_steps * scene.dt,
            force_update=True
        )
        final_params = motion_controller.estimate_and_save_superquadric(final_state)

        if final_params:
            print("\n最终11维参数:")
            params = final_params['parameters_11d']
            print(f"尺度: a1={params['scale_a1']:.6f}, a2={params['scale_a2']:.6f}, a3={params['scale_a3']:.6f}")
            print(f"形状: ε1={params['shape_epsilon1']:.6f}, ε2={params['shape_epsilon2']:.6f}")
            print(
                f"位移: tx={params['translation_x']:.6f}, ty={params['translation_y']:.6f}, tz={params['translation_z']:.6f}")
            print(f"欧拉角: rx={params['euler_rx']:.6f}, ry={params['euler_ry']:.6f}, rz={params['euler_rz']:.6f}")

    # 打印统计信息
    if motion_controller.sampling_enabled:
        print(f"\n=== 采样统计 ===")
        print(f"首次接触步数: {motion_controller.first_contact_step}")
        print(f"保存的数据点数量: {len(motion_controller.superquadric_params_history)}")
        print(f"数据采集步数: {motion_controller.total_steps - motion_controller.first_contact_step}")
        print(f"数据采集效率: {len(motion_controller.superquadric_params_history) / ((motion_controller.total_steps - motion_controller.first_contact_step) / motion_controller.estimation_interval):.2%}")


    if not motion_controller.use_traditional_mode and motion_controller.action_history:
        print("\n=== 动作统计 ===")
        print(f"总动作次数: {len(motion_controller.action_history)}")
        action_types = {}
        for action in motion_controller.action_history:
            action_name = action['action']
            action_types[action_name] = action_types.get(action_name, 0) + 1

        print("动作分布:")
        for action, count in action_types.items():
            desc = motion_controller.action_space[action]['desc']
            print(f"  {desc}: {count}次")


def clear_gpu_memory():
    if torch.cuda.is_available():
        # 强制垃圾回收
        gc.collect()

        # 清空CUDA缓存
        torch.cuda.empty_cache()

        # 重置内存统计
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        print("GPU内存已强制清理")

if __name__ == "__main__":
    main()
