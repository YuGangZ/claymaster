import json
import os
import numpy as np
from common.base.motion import BaseMotionController, MotionControllerUtils
from common.base.control_phase_manager import PhaseManager, ApproachPhase, AutonomousControlPhase
from common.controller.dummy_controller import DummyController
from common.controller.open_loop_controller import OpenLoopController
from mpc.mpc_controller import MPCController


class MotionManagementRL(BaseMotionController):
    """Motion controller for RL and MPC control with phase management."""

    def __init__(self, scene, sensor_cube, elastoplastic_obj, initial_particles,
                 control_mode='dummy', output_dir="rl_control_data", **controller_kwargs):
        super().__init__(scene, sensor_cube, elastoplastic_obj, initial_particles, output_dir)

        # Control mode and parameters
        self.control_mode = control_mode
        self.use_rl = controller_kwargs.get('use_rl', False)

        # 控制参数（确保与基类不冲突）
        self.control_interval = controller_kwargs.get('control_interval', 25)
        self.estimation_interval = controller_kwargs.get('estimation_interval', 25)

        # 阶段管理参数
        self.current_phase_index = 0
        self.phases_initialized = False

        # Target state for RL/MPC
        self.target_16d_state = None

        # 状态估计跟踪
        self.last_estimation_step = -1
        self.force_estimation_next_step = False

        # RL/MPC specific
        self.current_16d_state = None
        self.control_history = []
        self.print_interval = 50

        # 控制缓存
        self.cached_control = np.zeros(3)
        self.control_counter = 0
        self.last_control_update_step = -1

        # 初始化阶段管理器（延迟到实际使用前）
        self.phase_manager = None
        self._phase_manager_kwargs = controller_kwargs

        # 打印计数器重置
        self.print_counter = 0

    def _initialize_phase_manager(self):
        """延迟初始化阶段管理器，确保所有参数已设置"""
        if self.phase_manager is not None:
            return

        manager = PhaseManager()

        # Phase 1: Approach phase
        approach_speed = self._phase_manager_kwargs.get('approach_speed', self.v_down)
        approach = ApproachPhase(approach_speed=approach_speed)
        manager.add_phase(approach)

        # Phase 2: Autonomous control phase
        use_rl = self._phase_manager_kwargs.get('use_rl', False)
        controller = self._create_controller(self._phase_manager_kwargs)

        control_phase = AutonomousControlPhase(
            controller,
            control_interval=self.control_interval,
            use_rl=use_rl,
            # 注意：这里暂时不设置rl_env，需要在外部设置
        )
        manager.add_phase(control_phase)

        self.phase_manager = manager
        self.phases_initialized = True

        print(f"阶段管理器已初始化: {self.control_mode}模式, RL={use_rl}")

    def set_rl_env(self, rl_env):
        """设置RL环境到阶段管理器"""
        if self.phase_manager and len(self.phase_manager.phases) > 1:
            # 找到自主控制阶段并设置RL环境
            for phase in self.phase_manager.phases:
                if isinstance(phase, AutonomousControlPhase):
                    phase.set_rl_env(rl_env)
                    phase.use_rl = True  # 确保启用RL
                    print(f"RL环境已设置到自主控制阶段")

    def _create_controller(self, controller_kwargs):
        """Create controller based on control mode."""
        if self.control_mode == 'mpc':
            model_path = controller_kwargs.get('model_path', 'shape_predictor.pth')
            device = controller_kwargs.get('device', 'cpu')
            horizon = controller_kwargs.get('horizon', 5)

            controller = MPCController(
                model_path=model_path,
                device=device,
                horizon=horizon
            )

        elif self.control_mode == 'dummy':
            gain = controller_kwargs.get('gain', 0.2)
            max_speed = controller_kwargs.get('max_speed', 0.1)
            controller = DummyController(gain=gain, max_speed=max_speed)

        elif self.control_mode == 'openloop':
            trajectory = controller_kwargs.get('trajectory', 'spiral')
            controller = OpenLoopController(trajectory=trajectory)

        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")

        return controller


    def set_target_shape(self, target_16d_state):
        """Set target shape for RL/MPC control."""
        self.target_16d_state = target_16d_state

        # 确保阶段管理器已初始化
        if not self.phases_initialized:
            self._initialize_phase_manager()

        # 传递目标到阶段管理器中的控制器
        if self.phase_manager:
            for phase in self.phase_manager.phases:
                if isinstance(phase, AutonomousControlPhase) and phase.controller:
                    phase.controller.set_target(target_16d_state)

        print(f"Target shape set: {target_16d_state[:3]}...")

    def update_motion_phase(self, current_state, t):
        """更新运动阶段，处理阶段转移和接触检测"""
        # 确保阶段管理器已初始化
        if not self.phases_initialized:
            self._initialize_phase_manager()

        # 更新接触状态
        contact_info = current_state.get('contact', {})

        if contact_info.get('contact_detected', False) and not self.contact_established:
            self.contact_established = True
            print(f"=== 接触建立！穿透深度: {contact_info.get('penetration_depth', 0):.4f}m ===")
            # ✅ 强制下次进行状态估计
            self.force_estimation_next_step = True

        # 强制更新缓存状态
        self.cached_state = current_state
        self.last_state_time = current_state['step']

        # 使用阶段管理器更新阶段
        if self.phase_manager:
            current_phase = self.phase_manager.phases[self.phase_manager.current_phase_index]

            # 检查是否需要阶段转移
            if current_phase.should_transition(current_state):
                if self.phase_manager.current_phase_index < len(self.phase_manager.phases) - 1:
                    old_phase = type(current_phase).__name__
                    self.phase_manager.current_phase_index += 1
                    new_phase = type(self.phase_manager.phases[self.phase_manager.current_phase_index]).__name__

                    print(f"\n=== 阶段转移 ===")
                    print(f"{old_phase} -> {new_phase}")
                    print(f"=== ========= ===")

                    # ✅ 阶段转移后强制状态估计
                    if new_phase == "AutonomousControlPhase":
                        self.force_estimation_next_step = True

            # 如果是自主控制阶段，检查状态估计
            current_phase = self.phase_manager.phases[self.phase_manager.current_phase_index]
            if isinstance(current_phase, AutonomousControlPhase):
                # ✅ 修改条件：增加对None状态的检查
                need_estimation = (
                        self.current_16d_state is None or  # 尚无有效状态
                        current_state['step'] - self.last_estimation_step > self.estimation_interval or
                        self.force_estimation_next_step
                )

                if need_estimation:
                    # 执行状态估计
                    estimation_result = self._perform_superquadric_estimation(current_state)
                    if estimation_result:
                        # 计算16D特征向量
                        self.current_16d_state = MotionControllerUtils.get_16d_feature_vector(
                            estimation_result['param_dict'],
                            estimation_result['geometric_features']
                        )

                        # 更新控制器状态
                        if current_phase.controller and hasattr(current_phase.controller, 'current_state'):
                            current_phase.controller.current_state = self.current_16d_state

                        self.last_estimation_step = current_state['step']
                        self.force_estimation_next_step = False

                        # 保存数据
                        self._save_estimation_data(estimation_result, current_state)

    def _save_estimation_data(self, estimation_result, current_state):
        """保存估计数据"""
        # 保存点云
        point_cloud_filename = self.save_point_cloud(
            estimation_result['surface_points'],
            estimation_result['step'],
            estimation_result['time']
        )

        # 计算16D特征向量
        feature_16d = MotionControllerUtils.get_16d_feature_vector(
            estimation_result['param_dict'],
            estimation_result['geometric_features']
        )

        # 创建参数记录
        estimation_record = {
            'step': estimation_result['step'],
            'time': estimation_result['time'],
            'contact_info': estimation_result['contact_info'],
            'parameters_11d': estimation_result['param_dict'],
            'geometric_features': estimation_result['geometric_features'],
            'feature_16d': feature_16d,
            'point_cloud_file': point_cloud_filename
        }

        # 添加弹性体状态
        if 'elastic' in current_state and 'state' in current_state['elastic']:
            elastic_state = current_state['elastic']['state']
            if 'center' in elastic_state:
                estimation_record['elastic'] = {
                    'state': {
                        'center': elastic_state['center'],
                        'step': estimation_result['step'],
                        'time': estimation_result['time']
                    }
                }

        # 保存参数
        param_filename = self.save_superquadric_params(
            estimation_record,
            estimation_result['step'],
            estimation_result['time']
        )

        self.superquadric_params_history.append(estimation_record)

        print(f"=== 状态估计完成 (步骤 {estimation_result['step']}) ===")
        print(f"16维状态: {[f'{x:.3f}' for x in feature_16d[:5]]}...")

    def calculate_velocity(self):
        """计算速度 - 与基类接口兼容"""
        if not self.cached_state:
            # 获取最新状态
            current_state = self.get_system_state(
                self.scene.step_counter,
                self.scene.time,
                force_update=True
            )
            return self.calculate_control_velocity(current_state)

        return self.calculate_control_velocity(self.cached_state)

    def calculate_control_velocity(self, current_state):
        """计算控制速度"""
        if not self.phases_initialized:
            self._initialize_phase_manager()

        # 使用阶段管理器更新阶段并计算速度
        if self.phase_manager:
            # 调用阶段管理器的update方法，它会处理阶段转移
            vel_array = self.phase_manager.update(current_state, self)

            # 缓存控制
            if vel_array.shape[0] > 0:
                self.cached_control = vel_array[0, :]

            # 记录控制历史
            control_record = {
                'step': current_state['step'],
                'time': current_state['time'],
                'control': vel_array[0, :].tolist() if vel_array.shape[0] > 0 else [0, 0, self.v_down],
                'control_mode': self.control_mode,
                'use_rl': self.use_rl,
                'state': self.current_16d_state
            }
            self.control_history.append(control_record)

            return vel_array
        else:
            # 如果没有阶段管理器，返回下降速度
            vel_array = np.zeros((len(self.initial_particles), 3))
            vel_array[:, 2] = self.v_down
            return vel_array

    def estimate_and_save_superquadric(self, current_state):
        """执行超二次曲面估计并保存结果"""
        ems_params = {
            'OutlierRatio': 0.05,
            'MaxIterationEM': 20,
            'ToleranceEM': 1e-3,
            'RelativeToleranceEM': 2e-1,
            'MaxOptiIterations': 2,
            'Sigma': 1e-3,
            'MaxiSwitch': 2,
            'AdaptiveUpperBound': True,
            'Rescale': True
        }

        estimation_result = self._perform_superquadric_estimation(current_state, ems_params)
        if estimation_result is None:
            return None

        # 计算16D特征向量
        self.current_16d_state = MotionControllerUtils.get_16d_feature_vector(
            estimation_result['param_dict'],
            estimation_result['geometric_features']
        )

        self.last_estimation_step = current_state['step']

        # 保存数据
        self._save_estimation_data(estimation_result, current_state)

        return {
            'sq': estimation_result['sq'],
            'param_dict': estimation_result['param_dict'],
            'geometric_features': estimation_result['geometric_features'],
            'feature_16d': self.current_16d_state
        }

    def print_state_info(self, current_state):
        """打印状态信息"""
        # 获取当前阶段信息
        phase_info = ""
        if self.phase_manager:
            phase_index = self.phase_manager.current_phase_index
            if phase_index < len(self.phase_manager.phases):
                current_phase = self.phase_manager.phases[phase_index]
                phase_info = f"  阶段: {type(current_phase).__name__}"

                # 如果是ApproachPhase，显示步数
                if isinstance(current_phase, ApproachPhase):
                    phase_info += f" (步数: {current_phase.step_counter})"
                # 如果是AutonomousControlPhase，显示控制计数器
                elif isinstance(current_phase, AutonomousControlPhase):
                    phase_info += f" (控制计数器: {current_phase.control_counter})"

        # 基础信息
        elastic = current_state.get('elastic', {})
        contact = current_state.get('contact', {})

        print(f"[STEP {current_state['step']}] t={current_state['time']:.3f}")

        if phase_info:
            print(phase_info)

        print(f"  控制模式: {self.control_mode}{' (RL)' if self.use_rl else ''}")

        # 显示当前控制
        control_norm = np.linalg.norm(self.cached_control)
        print(f"  当前控制: {self.cached_control}, 范数: {control_norm:.4f}")

        if 'state' in elastic:
            state = elastic['state']
            print(f"  弹性体: 中心=({state['center'][0]:.3f}, {state['center'][1]:.3f}, {state['center'][2]:.3f})")

        if contact.get('contact_detected', False):
            print(f"  接触: 穿透深度={contact.get('penetration_depth', 0):.4f}m")
        else:
            print(f"  接触: 无接触")

        print("-" * 80)

    def should_print_status(self, step, current_state):
        """确定是否应该打印状态"""
        self.print_counter += 1

        # 使用固定的打印间隔
        if step < 100:
            return self.print_counter % 80 == 0
        else:
            return self.print_counter % 40 == 0

    def save_control_history(self):
        """保存控制历史到文件"""
        try:
            history_file = os.path.join(self.output_dir, f"{self.control_mode}_control_history.json")
            serializable_history = []
            for record in self.control_history:
                serializable_record = {
                    'step': record['step'],
                    'time': record['time'],
                    'control': record['control'],
                    'control_mode': record['control_mode'],
                    'use_rl': record['use_rl']
                }
                if 'state' in record and record['state'] is not None:
                    serializable_record['state'] = record['state']
                if 'cached' in record:
                    serializable_record['cached'] = record['cached']
                serializable_history.append(serializable_record)

            with open(history_file, 'w') as f:
                json.dump(serializable_history, f, indent=2, default=MotionControllerUtils.json_serializer)

            print(f"控制历史已保存: {history_file}")
            return history_file
        except Exception as e:
            print(f"保存控制历史失败: {e}")
            return None

    def save_custom_format_data(self):
        """保存自定义格式数据"""
        return MotionControllerUtils.save_custom_format_data(
            self.output_dir,
            self.superquadric_params_history,
            data_type="mpc" if self.control_mode == 'mpc' else "general"
        )

    def finalize_simulation(self):
        """完成仿真"""
        try:
            # 保存自定义格式数据
            MotionControllerUtils.save_custom_format_data(
                self.output_dir,
                self.superquadric_params_history,
                data_type="mpc" if self.control_mode == 'mpc' else "general"
            )

            # 保存控制历史
            self.save_control_history()

            print(f"RL/MPC仿真数据导出完成: {self.output_dir}")
        except Exception as e:
            print(f"导出数据时出错: {e}")