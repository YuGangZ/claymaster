import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DeformationRLEnv(gym.Env):
    """
    软体变形强化学习环境
    - 高层RL策略输出14维目标形状变化
    - 底层控制器（MPC/Dummy等）执行3维速度控制
    """

    def __init__(self, motion_controller,
                target_shape: np.ndarray,
                config: dict):
        super().__init__()

        # 保存运动控制器
        self.motion_ctrl = motion_controller
        self.target_shape = target_shape
        self.config = config

        # 维度定义
        self.state_dim = config["state_dim"]  # 14
        self.target_dim = self.state_dim
        self.uncertainty_dim = config["uncertainty_dim"]  # 14
        self.control_dim = config["control_dim"]  # 3
        self.max_steps = config["max_episode_steps"]
        self.action_scale = config["action_scale"]  # 14维缩放因子

        # ✅ 定义动作缓冲（RL输出的原始动作）
        self.action_buffer = np.zeros(self.state_dim, dtype=np.float32)

        # Gymnasium空间定义
        # 观测空间: 14当前 + 14目标 + 3末端 + 14不确定度 = 45维
        obs_dim = self.state_dim + self.target_dim + self.control_dim + self.uncertainty_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 动作空间: 15维混合动作（感知决策 + 连续子目标）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.config["action_dim"],), dtype=np.float32
        )
        # 感知决策阈值
        self.perception_threshold = 0.5

        # 状态跟踪
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.uncertainty_estimate = np.ones(self.uncertainty_dim, dtype=np.float32) * 0.01
        self.prediction_errors = []  # 预测误差历史，用于不确定度估计

        # Gymnasium元数据
        self.metadata = {'render_modes': ['human']}
        self.render_mode = None

    def reset(self, seed=None, options=None):
        """重置环境 - Gymnasium标准接口"""
        super().reset(seed=seed)

        # 重置仿真场景
        self.motion_ctrl.scene.reset()

        # 重置阶段管理器
        if hasattr(self.motion_ctrl, 'phase_manager'):
            self.motion_ctrl.phase_manager.reset()

        self.current_step = 0
        self.prediction_errors.clear()
        # 重置初始体积记录
        if hasattr(self, '_initial_volume'):
            delattr(self, '_initial_volume')
        # 获取初始状态
        self.current_state = self._get_current_shape()
        self.uncertainty_estimate = self._estimate_uncertainty()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, rl_action):
        """
        执行RL动作
        Args:
            rl_action: 15维归一化动作 [-1, 1]
                第0维 -> 感知决策 (经过sigmoid)
                第1~14维 -> 子目标增量
        """
        # 1. 解码混合动作
        action_array = np.array(rl_action, dtype=np.float32).flatten()
        # 感知决策概率
        perception_prob = 1.0 / (1.0 + np.exp(-action_array[0]))  # sigmoid
        do_perception = perception_prob > self.perception_threshold

        # 子目标增量 (第1~14维)
        subgoal_delta = action_array[1:15] * self.action_scale[1:15]  # 注意action_scale长度15

        # 2. 如果触发感知，执行主动感知过程
        if do_perception:
            self._perform_active_perception()
            # 感知后重新获取当前形状（更新状态）
            self.current_state = self._get_current_shape()
            self.uncertainty_estimate = self._estimate_uncertainty()
            # 感知动作本身不产生形状变化，直接返回
            reward, terminated = self._compute_reward(self.current_state)
            self.current_step += 1
            truncated = self.current_step >= self.max_steps
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        # 3. 否则执行重塑动作
        target_state = self.current_state + subgoal_delta
        achieved_state = self._execute_control_cycle(target_state)

        # 4. 计算奖励
        reward, terminated = self._compute_reward(achieved_state)

        # 5. 更新状态
        self.current_state = achieved_state
        self.uncertainty_estimate = self._estimate_uncertainty()
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _perform_active_perception(self):
        """
        执行论文中描述的 roll-and-restore 主动感知过程
        通过运动控制器触发，更新当前形状估计和不确定性
        """
        if hasattr(self.motion_ctrl, 'perform_active_perception'):
            # 假设 motion_controller 实现了该方法
            self.motion_ctrl.perform_active_perception()
        else:
            # 简化实现：模拟感知过程，降低不确定性
            self.uncertainty_estimate *= 0.5  # 不确定性减半
            print("⚠️ 主动感知执行（简化版）")

    def _execute_control_cycle(self, target_state):
        """执行控制循环 - 仅在自主控制阶段设置RL子目标"""
        # ✅ 修正：不要在这里设置RL子目标，因为子目标已经在AutonomousControlPhase内部计算
        # 只在自主控制阶段执行控制循环
        if hasattr(self.motion_ctrl.phase_manager, 'phases'):
            current_phase = self.motion_ctrl.phase_manager.phases[self.motion_ctrl.phase_manager.current_phase_index]

            # 检查当前阶段是否是自主控制阶段且使用RL
            if hasattr(current_phase, 'use_rl') and current_phase.use_rl:
                # ✅ 不要设置子目标，因为子目标会在AutonomousControlPhase.compute_velocity()中计算
                print(f"✅ RL控制阶段，子目标将在compute_velocity中计算")
            else:
                print(f"当前不是RL控制阶段，跳过RL子目标设置")

        control_steps = self.config.get("control_steps_per_rl_step", 25)

        for step in range(control_steps):
            # 获取当前系统状态
            current_state_data = self.motion_ctrl.get_system_state(
                step, step * self.motion_ctrl.scene.dt,
                force_update=(step % 5 == 0)
            )

            # 通过阶段管理器计算并应用速度
            vel_array = self.motion_ctrl.calculate_control_velocity(current_state_data)
            self.motion_ctrl.apply_velocity(vel_array)
            self.motion_ctrl.scene.step()

        return self._get_current_shape()

    def _get_observation(self) -> np.ndarray:
        """构建45维观测向量"""
        # 14维当前形状 + 14维目标形状 + 3维末端位置 + 14维不确定度
        ee_pos = self._get_end_effector_position()

        obs = np.concatenate([
            self.current_state,  # 14维当前形状
            self.target_shape,  # 14维目标形状
            ee_pos,  # 3维末端位置
            self.uncertainty_estimate  # 14维不确定度
        ], dtype=np.float32)

        return obs

    def _get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器3维位置"""
        try:
            # 方法1: 从监控器获取弹性体中心位置
            if hasattr(self.motion_ctrl, 'elastic_monitor'):
                elastic_state = self.motion_ctrl.elastic_monitor.get_current_state()
                if elastic_state is not None:
                    return elastic_state['center'].astype(np.float32)

            # 方法2: 通过物理引擎获取粒子位置并计算中心
            if hasattr(self.motion_ctrl.sensor_cube, 'get_particles'):
                particles = self.motion_ctrl.sensor_cube.get_particles()
                if hasattr(particles, 'cpu'):
                    particles = particles.cpu().numpy()
                particles = particles.reshape(-1, 3)
                center = np.mean(particles, axis=0)
                return center[:3].astype(np.float32)

            # 方法3: 如果实体有获取位置的方法（可能是不同的名称）
            for method_name in ['get_center', 'get_position', 'center', 'position']:
                if hasattr(self.motion_ctrl.sensor_cube, method_name):
                    pos = getattr(self.motion_ctrl.sensor_cube, method_name)()
                    if pos is not None:
                        if hasattr(pos, 'cpu'):
                            pos = pos.cpu().numpy()
                        return pos.flatten()[:3].astype(np.float32)

            print(f"⚠️ 无法获取末端位置，返回零向量")
            return np.zeros(3, dtype=np.float32)

        except Exception as e:
            print(f"⚠️ 获取末端位置失败: {e}")
            return np.zeros(3, dtype=np.float32)

    def _get_current_shape(self) -> np.ndarray:
        """获取当前14维形状特征"""
        try:
            if hasattr(self.motion_ctrl, 'current_16d_state') and self.motion_ctrl.current_16d_state is not None:
                state_16d = np.array(self.motion_ctrl.current_16d_state, dtype=np.float32)

                # 确保是16维
                if len(state_16d) != 16:
                    print(f"⚠️ 警告: current_16d_state 不是16维，实际维度: {len(state_16d)}")
                    # 尝试补全到16维
                    if len(state_16d) < 16:
                        padded = np.zeros(16, dtype=np.float32)
                        padded[:len(state_16d)] = state_16d
                        state_16d = padded

                # 转换为14维：去除第14维(flatness)和第16维(convexity)
                indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]  # 保留的索引
                state_14d = state_16d[indices]

                # 检查是否有效
                if np.all(np.abs(state_14d[:3]) < 1e-6):  # 尺度接近0
                    print(f"⚠️ 警告: 状态尺度接近0: {state_14d[:3]}")
                    # 返回默认值或上一次有效值
                    if hasattr(self, '_last_valid_state'):
                        return self._last_valid_state.copy()
                    else:
                        return np.ones(self.state_dim, dtype=np.float32) * 0.08  # 默认值
                else:
                    self._last_valid_state = state_14d.copy()
                    return state_14d
            else:
                print(f"⚠️ 警告: current_16d_state 为None或不存在")
                return np.ones(self.state_dim, dtype=np.float32) * 0.08  # 默认值
        except Exception as e:
            print(f"⚠️ 获取当前形状失败: {e}")
            return np.ones(self.state_dim, dtype=np.float32) * 0.08  # 默认值

    def _estimate_uncertainty(self) -> np.ndarray:
        """基于预测误差历史估计16维不确定度"""
        if len(self.prediction_errors) > 0:
            recent_errors = np.array(self.prediction_errors[-5:], dtype=np.float32)  # 最近5次
            uncertainty = np.var(recent_errors, axis=0)
        else:
            uncertainty = np.ones(self.uncertainty_dim, dtype=np.float32) * 0.01

        # 确保16维
        if len(uncertainty) == self.uncertainty_dim:
            return uncertainty
        elif len(uncertainty) > self.uncertainty_dim:
            return uncertainty[:self.uncertainty_dim]
        else:
            padded = np.zeros(self.uncertainty_dim, dtype=np.float32)
            padded[:len(uncertainty)] = uncertainty
            return padded

    def _compute_reward(self, achieved_state: np.ndarray) -> tuple[float, bool]:
        """
        计算奖励和终止标志
        Returns:
            reward: 标量奖励值
            terminated: 是否成功达到目标
        """
        weights = self.config["reward_weights"]

        # 1. 形状距离奖励（主要奖励）
        dist_old = np.linalg.norm(self.current_state - self.target_shape)
        dist_new = np.linalg.norm(achieved_state - self.target_shape)
        reward_goal = (dist_old - dist_new) * weights["goal"]

        # 2. 稀疏成功奖励
        success_threshold = 0.05
        success = dist_new < success_threshold
        reward_success = weights["success"] if success else 0.0

        # 3. 时间步惩罚
        reward_time = weights["time"]

        # 4. 安全约束惩罚（剧烈体积变化）
        vol_idx = 11
        reward_safety = 0.0
        # 获取当前体积和初始体积
        if len(achieved_state) > vol_idx:
            current_volume = achieved_state[vol_idx]

            # 检查是否有记录的初始体积
            if not hasattr(self, '_initial_volume'):
                # 如果是第一次，将当前体积作为初始体积
                self._initial_volume = current_volume

            # 计算体积偏差率
            if self._initial_volume > 1e-6:  # 避免除以零
                volume_deviation_rate = abs(current_volume - self._initial_volume) / self._initial_volume

                # 当偏差率超过20%时施加惩罚
                if volume_deviation_rate > 0.2:
                    reward_safety = weights["safety"]
                    print(f"⚠️ 安全约束触发：体积偏差率 {volume_deviation_rate:.1%} > 20%")

        # 5. 不确定度惩罚
        avg_uncertainty = np.mean(self.uncertainty_estimate)
        reward_uncertainty = avg_uncertainty * weights["uncertainty"]

        total_reward = reward_goal + reward_success + reward_time + reward_safety + reward_uncertainty

        return total_reward, success


    def _get_info(self) -> dict:
        """返回环境信息字典"""
        return {
            'step': self.current_step,
            'distance_to_target': float(np.linalg.norm(self.current_state - self.target_shape)),
            'current_state': self.current_state.copy(),
            'uncertainty': self.uncertainty_estimate.copy(),
            'control_mode': self.motion_ctrl.control_mode
        }

    def render(self):
        """渲染/打印状态"""
        if self.render_mode == 'human':
            dist = np.linalg.norm(self.current_state - self.target_shape)
            avg_uncert = np.mean(self.uncertainty_estimate)
            print(f"[渲染] Step: {self.current_step}, Distance: {dist:.4f}, Uncertainty: {avg_uncert:.4f}")

    def close(self):
        """关闭环境资源"""
        try:
            if hasattr(self, 'motion_ctrl'):
                self.motion_ctrl.finalize_simulation()
        except Exception as e:
            print(f"⚠️ 关闭环境时出错: {e}")