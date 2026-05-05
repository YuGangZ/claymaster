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
from physical_engine.soft_sim_env import SoftSimEnv
from motion_rl import MotionManagementRL
from core.rl_trainer import RLDeformationTrainer
from config.rl_config import RL_CONFIG
from common.base.control_phase_manager import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control-mode', choices=['mpc', 'dummy'], default='mpc')
    parser.add_argument('--control-steps', type=int, default=25)
    parser.add_argument('--rl-timesteps', type=int, default=5000)
    parser.add_argument('--simulation-steps', type=int, default=1000)
    args = parser.parse_args()

    # 1. 初始化物理引擎
    engine = SoftSimEnv()
    scene, solver, sensor_cube, elastoplastic_obj = engine.setup_simulation("box")
    initial_particles = engine.initialize_cube_particles()

    # 2. 创建运动控制器
    motion_controller = MotionManagementRL(
        scene, sensor_cube, elastoplastic_obj, initial_particles,
        control_mode=args.control_mode,
        output_dir=f"hierarchical_rl_{args.control_mode}",
        control_interval=args.control_steps,
        estimation_interval=25,
        use_rl=True  # 启用RL模式
    )

    # 3. 设置最终目标形状
    FINAL_TARGET = np.array([
        0.0932, 0.0932, 0.0932,  # scale_a1,2,3
        1.0, 1.0,  # shape_epsilon1,2
        0.0, 0.0, 0.0,  # translation_x,y,z
        0.0, 0.0, 0.0,  # euler_rx,ry,rz
        0.002375,  # volume
        1.0, 0.0, 1.0, 1.0  # elongation, flatness, smoothness, convexity
    ], dtype=np.float32)

    motion_controller.set_target_shape(FINAL_TARGET)

    # 4. 创建RL训练器和环境
    print("初始化RL训练器...")
    RL_CONFIG.update({
        "control_steps_per_rl_step": args.control_steps,
        "action_is_delta": True,  # RL输出Δstate
        "use_hierarchical": True,  # 启用分层控制
    })

    trainer = RLDeformationTrainer(motion_controller, FINAL_TARGET)
    trainer.initialize_policy()

    # 5. 训练RL策略
    print(f"开始RL训练，总步数: {args.rl_timesteps}")
    trainer.train(total_timesteps=args.rl_timesteps, visualize=False)

    # 6. 获取训练好的RL环境
    # 注意：这里假设训练后环境中的模型已经更新
    rl_env = trainer.env.envs[0] if hasattr(trainer.env, 'envs') else trainer.env

    # 7. 将RL环境设置到运动控制器
    motion_controller.set_rl_env(rl_env)

    # 8. 主仿真循环
    print("\n" + "=" * 60)
    print("开始分层RL-MPC控制仿真")
    print(f"仿真总步数: {args.simulation_steps}")
    print(f"控制间隔: {args.control_steps}步")
    print("=" * 60)

    for step in range(args.simulation_steps):
        # 获取当前系统状态
        current_state = motion_controller.get_system_state(
            step, step * scene.dt,
            force_update=(step % 5 == 0)  # 每5步强制更新一次
        )

        # 更新运动阶段（这会处理接触检测和阶段转移）
        motion_controller.update_motion_phase(current_state, scene.time)

        # 计算控制速度（这会触发状态估计、RL计算、MPC计算的循环）
        vel_array = motion_controller.calculate_control_velocity(current_state)

        # 应用速度到物理引擎
        motion_controller.apply_velocity(vel_array)

        # 物理步进
        scene.step()

        # 定期打印状态
        if motion_controller.should_print_status(step, current_state):
            motion_controller.print_state_info(current_state)

        # 检查是否达到最大步数或完成目标
        if hasattr(motion_controller, 'current_16d_state') and motion_controller.current_16d_state is not None:
            current_16d = motion_controller.current_16d_state
            target_16d = FINAL_TARGET
            distance = np.linalg.norm(current_16d - target_16d)

            if distance < 0.05:  # 成功阈值
                print(f"\n🎉 成功达到目标形状！距离: {distance:.4f}")
                print(f"总步数: {step}")
                break

    # 9. 保存结果
    print("\n仿真完成，保存数据...")
    motion_controller.finalize_simulation()
    motion_controller.save_control_history()
    motion_controller.save_custom_format_data()

    print(f"\n所有数据已保存到: {motion_controller.output_dir}")


if __name__ == "__main__":
    main()