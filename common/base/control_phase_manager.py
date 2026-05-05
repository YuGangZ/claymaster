import numpy as np
from abc import ABC, abstractmethod


class ControlPhase(ABC):
    """控制阶段抽象基类"""

    @abstractmethod
    def compute_velocity(self, current_state, motion_controller) -> np.ndarray:
        pass

    @abstractmethod
    def should_transition(self, current_state) -> bool:
        pass


class ApproachPhase(ControlPhase):
    """下压接近阶段"""

    def __init__(self, approach_speed=-0.2, contact_threshold=0.01):
        self.approach_speed = approach_speed
        self.contact_threshold = contact_threshold
        self.step_counter = 0

    def compute_velocity(self, current_state, motion_controller):
        # 简单向下运动
        self.step_counter += 1
        vel_array = np.zeros((motion_controller.initial_particles.shape[0], 3))
        vel_array[:, 2] = self.approach_speed

        if self.step_counter % 10 == 0:
            print(f"  下压阶段: 速度 = {self.approach_speed:.3f} m/s, 步数 = {self.step_counter}")

        return vel_array

    def should_transition(self, current_state):
        # 检测是否接触
        contact_info = current_state.get('contact', {})
        contact_detected = contact_info.get('contact_detected', False)

        if contact_detected:
            penetration = contact_info.get('penetration_depth', 0)
            print(f"  接触检测: 穿透深度 = {penetration:.4f} m")

        return contact_detected


class AutonomousControlPhase(ControlPhase):
    def __init__(self, controller, control_interval=25, use_rl=False, rl_env=None):
        self.controller = controller
        self.control_interval = control_interval
        self.use_rl = use_rl  # 是否使用RL
        self.rl_env = rl_env  # RL环境（可选）
        self.rl_sub_target = None  # RL提供的子目标（MPC模式用）
        self.current_control = np.zeros(3)
        self.control_counter = 0
        self.last_control_time = 0
        self.last_rl_computation_time = 0
        self.last_estimation_time = 0
        self.initial_estimation_done = False

        # 判断控制模式
        self.is_direct_rl_mode = (controller is None and use_rl)  # RL直接控制模式
        self.is_mpc_mode = (controller is not None)  # MPC模式（可能带RL子目标）

    def set_rl_env(self, rl_env):
        """设置RL环境"""
        self.rl_env = rl_env

    def set_rl_sub_target(self, sub_target):
        """由外部调用，设置RL子目标（MPC模式用）"""
        self.external_rl_sub_target = sub_target
        print(f"  外部设置的RL子目标: {sub_target[:3]}...")

    def compute_velocity(self, current_state, motion_controller):
        self.control_counter += 1

        # 1. 检查是否需要重新计算（每control_interval步）
        need_recomputation = (self.control_counter - self.last_control_time) >= self.control_interval

        # 2. 首次进入自主控制阶段时强制计算
        if self.control_counter == 1 and not self.initial_estimation_done:
            need_recomputation = True
            print("🔄 首次进入自主控制阶段，执行初始状态估计...")

        # ============ 状态估计 ============
        if need_recomputation and hasattr(motion_controller, 'estimate_and_save_superquadric'):
            print(f"  Step {current_state['step']}: 执行状态估计...")
            estimation_result = motion_controller.estimate_and_save_superquadric(current_state)
            if estimation_result and estimation_result.get('feature_16d') is not None:
                motion_controller.current_16d_state = estimation_result.get('feature_16d')
                print(f"    状态估计完成")
                self.last_estimation_time = self.control_counter
                self.initial_estimation_done = True
            else:
                print(f"    状态估计失败，使用上一状态")

        # ============ 根据模式计算控制 ============
        if self.is_direct_rl_mode:
            # RL直接控制模式
            if hasattr(motion_controller, 'current_rl_action') and hasattr(motion_controller, 'rl_control_enabled'):
                if motion_controller.rl_control_enabled:
                    # 直接使用RL输出的控制指令
                    rl_action = motion_controller.current_rl_action
                    self.current_control = rl_action.copy()

                    if need_recomputation and self.control_counter % 10 == 0:
                        print(f"  RL直接控制: 动作 = {rl_action}")
                else:
                    # RL控制未启用，保持静止
                    self.current_control = np.zeros(3)
            else:
                # 没有RL动作，保持静止
                self.current_control = np.zeros(3)

        elif self.is_mpc_mode and self.controller is not None:
            # MPC模式（可能带RL子目标）
            if need_recomputation:
                print(f"  Step {current_state['step']}: MPC计算控制...")

                # 获取当前状态
                current_16d_state = motion_controller.current_16d_state
                if current_16d_state is None:
                    print(f"    警告: current_16d_state为None，使用零状态")
                    current_16d_state = np.zeros(16)

                # ============ RL计算子目标（如果启用） ============
                if self.use_rl and self.rl_env and current_16d_state is not None:
                    try:
                        print(f"  Step {current_state['step']}: RL计算子目标...")

                        # 获取当前观测
                        current_state_14d = self._convert_16d_to_14d(current_16d_state)
                        self.rl_env.current_state = current_state_14d
                        obs = self.rl_env._get_observation()

                        # 调用RL策略计算动作
                        if hasattr(self.rl_env, 'model') and self.rl_env.model is not None:
                            rl_action, _ = self.rl_env.model.predict(obs, deterministic=True)

                            # 将RL动作转换为子目标
                            if hasattr(self.rl_env, 'action_scale'):
                                delta_shape = rl_action * self.rl_env.action_scale
                                self.rl_sub_target = current_state_14d + delta_shape

                                # 传递给控制器
                                if hasattr(self.controller, 'set_sub_target'):
                                    # 将14D转换回16D
                                    sub_target_16d = np.zeros(16, dtype=np.float32)
                                    sub_target_16d[0:3] = self.rl_sub_target[0:3]
                                    sub_target_16d[3:5] = self.rl_sub_target[3:5]
                                    sub_target_16d[5:8] = self.rl_sub_target[5:8]
                                    sub_target_16d[8:11] = self.rl_sub_target[8:11]
                                    sub_target_16d[11] = self.rl_sub_target[11]
                                    sub_target_16d[12] = self.rl_sub_target[12]
                                    sub_target_16d[13] = 0.0
                                    sub_target_16d[14] = self.rl_sub_target[13]
                                    sub_target_16d[15] = 1.0

                                    self.controller.set_sub_target(sub_target_16d)
                                    print(f"    RL计算完成: 子目标 = {self.rl_sub_target[:3]}...")
                                    self.last_rl_computation_time = self.control_counter
                    except Exception as e:
                        print(f"    RL计算失败: {e}")

                # ============ MPC计算控制 ============
                # 如果没有RL子目标，使用最终目标
                if not self.use_rl or self.rl_sub_target is None:
                    target_state = motion_controller.target_16d_state
                    if target_state is not None and hasattr(self.controller, 'set_target'):
                        self.controller.set_target(target_state)

                # 计算控制
                contact_info = current_state.get('contact', {})
                self.current_control = self.controller.control(
                    current_state=current_16d_state,
                    contact_info=contact_info
                )

                print(
                    f"    MPC计算完成: 控制 = {self.current_control}, 范数 = {np.linalg.norm(self.current_control):.4f}")

                # 更新时间戳
                self.last_control_time = self.control_counter

        else:
            # 其他情况，保持静止
            self.current_control = np.zeros(3)

        # 应用控制到所有粒子
        vel_array = np.zeros((motion_controller.initial_particles.shape[0], 3))
        vel_array[:, :] = self.current_control

        # 打印当前状态信息（每隔几步）
        if self.control_counter % 20 == 0:
            mode_str = "RL直接控制" if self.is_direct_rl_mode else "MPC控制"
            print(f"  自主控制阶段[{mode_str}]: 步数={self.control_counter}")

        return vel_array

    def should_transition(self, current_state):
        return False  # 自主控制阶段不主动转移

    def _convert_16d_to_14d(self, state_16d):
        """将16维状态转换为14维状态"""
        return np.array([
            state_16d[0], state_16d[1], state_16d[2],  # scale (3)
            state_16d[3], state_16d[4],  # shape (2)
            state_16d[5], state_16d[6], state_16d[7],  # translation (3)
            state_16d[8], state_16d[9], state_16d[10],  # rotation (3)
            state_16d[11],  # volume (1)
            state_16d[12],  # elongation (1)
            state_16d[14]  # smoothness (1)
        ], dtype=np.float32)


class PhaseManager:
    """阶段管理器"""

    def __init__(self):
        self.phases = []
        self.current_phase_index = 0

    def add_phase(self, phase):
        self.phases.append(phase)

    def update(self, current_state, motion_controller):
        if not self.phases:
            raise ValueError("没有添加任何控制阶段")

        current_phase = self.phases[self.current_phase_index]

        # 检查是否需要阶段转移
        should_transition = current_phase.should_transition(current_state)

        if should_transition and self.current_phase_index < len(self.phases) - 1:
            old_phase = type(current_phase).__name__
            self.current_phase_index += 1
            new_phase = type(self.phases[self.current_phase_index]).__name__

            print(f"\n=== 阶段转移 ===")
            print(f"{old_phase} -> {new_phase}")
            print(f"=== ========= ===")

            current_phase = self.phases[self.current_phase_index]

        # 计算速度
        return current_phase.compute_velocity(current_state, motion_controller)

    def reset(self):
        """重置到初始阶段"""
        self.current_phase_index = 0
        # 重置所有阶段的状态
        for phase in self.phases:
            if isinstance(phase, ApproachPhase):
                phase.step_counter = 0
            elif isinstance(phase, AutonomousControlPhase):
                phase.control_counter = 0
                phase.current_control = np.zeros(3)
                phase.last_control_time = 0
                phase.last_rl_computation_time = 0
                phase.last_estimation_time = 0
                phase.initial_estimation_done = False
                phase.rl_sub_target = None