import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DeformationRLEnv(gym.Env):
    """
    软体变形强化学习环境
    - 高层RL策略输出14维目标形状变化
    - 底层控制器（MPC/Dummy等）执行3维速度控制

    适配14维状态：scale(3), shape(2), trans(3), rot(3), volume, elongation, smoothness
    """

    def __init__(self, motion_controller,
                target_shape: np.ndarray,
                config: dict):
        super().__init__()

        self.motion_ctrl = motion_controller
        # target_shape 为 14 维
        self.target_shape = np.array(target_shape, dtype=np.float32)
        self.config = config

        # 维度定义
        self.state_dim = config["state_dim"]            # 14
        self.uncertainty_dim = config["uncertainty_dim"]# 14
        self.control_dim = config["control_dim"]        # 3
        self.max_steps = config["max_episode_steps"]

        # action_scale 必须为 15 维 (感知 + 14个子目标增量)
        self.action_scale = np.array(config["action_scale"], dtype=np.float32)
        if len(self.action_scale) != 15:
            raise ValueError("action_scale must be length 15 (1 perception + 14 subgoal increments)")

        self.action_buffer = np.zeros(self.state_dim, dtype=np.float32)

        # 观测空间: 14(current) + 14(target) + 3(ee_pos) + 14(uncertainty) = 45
        obs_dim = self.state_dim + self.state_dim + self.control_dim + self.uncertainty_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 动作空间: 15维混合动作 (感知 + 连续子目标)
        self.action_space = spaces.Dict({
            "do_perception": spaces.MultiBinary(1),  # 伯努利动作：0/1
            "subgoal_delta": spaces.Box(-1, 1, shape=(14,), dtype=np.float32)
        })
        self.perception_threshold = 0.5  # 感知动作的sigmoid阈值

        # 状态跟踪
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.uncertainty_estimate = np.ones(self.uncertainty_dim, dtype=np.float32) * 0.01
        self.prediction_errors = []          # 存储预测误差，用于更新不确定性
        self.last_predicted_state = None     # 上一次预测的目标状态（用于计算误差）

        # 用于缓存上一个有效形状，避免尺度退化
        self._last_valid_state = None

        # Gymnasium元数据
        self.metadata = {'render_modes': ['human']}
        self.render_mode = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        """重置环境 - Gymnasium标准接口"""
        super().reset(seed=seed)

        # 重置物理场景
        self.motion_ctrl.scene.reset()

        # 重置阶段管理器（回到ApproachPhase）
        if hasattr(self.motion_ctrl, 'phase_manager'):
            self.motion_ctrl.phase_manager.reset()

        self.current_step = 0
        self.prediction_errors.clear()
        self.last_predicted_state = None
        if hasattr(self, '_initial_volume'):
            delattr(self, '_initial_volume')
        self._last_valid_state = None

        # 获取初始形状特征（可能触发一次估计）
        self.current_state = self._get_current_shape()
        self.uncertainty_estimate = self._estimate_uncertainty()

        return self._get_observation(), self._get_info()

    def step(self, rl_action):
        """
        执行RL动作
        Args:
            rl_action: 15维归一化动作 [-1, 1]
                第0维 -> 感知决策 (经过sigmoid)
                第1~14维 -> 子目标增量
        """
        action_array = np.array(rl_action, dtype=np.float32).flatten()

        # 感知决策概率 (sigmoid)
        perception_prob = 1.0 / (1.0 + np.exp(-action_array[0]))
        do_perception = perception_prob > self.perception_threshold

        # 子目标增量 (1~14维)
        subgoal_delta = action_array[1:15] * self.action_scale[1:15]

        if do_perception:
            # 执行主动感知，更新形状估计和不确定性，不推进物理步进
            self._perform_active_perception()
            # 重新获取感知后的形状
            self.current_state = self._get_current_shape()
            self.uncertainty_estimate = self._estimate_uncertainty()
            # 感知动作不产生奖励和终止
            reward, terminated = self._compute_reward(self.current_state)
            self.current_step += 1
            truncated = self.current_step >= self.max_steps
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        # 执行重塑动作：目标 = 当前状态 + 增量
        target_state = self.current_state + subgoal_delta
        achieved_state = self._execute_control_cycle(target_state)

        # 记录预测误差，用于更新不确定性
        if self.last_predicted_state is not None:
            error = achieved_state - self.last_predicted_state
            self.prediction_errors.append(error)
        self.last_predicted_state = target_state.copy()

        # 计算奖励和终止条件
        reward, terminated = self._compute_reward(achieved_state)

        # 更新状态和不确定性
        self.current_state = achieved_state
        self.uncertainty_estimate = self._estimate_uncertainty()
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _perform_active_perception(self):
        """
        调用运动控制器的主动感知流程，执行一次完整的状态重新估计。
        不推进物理仿真步数，仅更新形状状态和不确定性。
        如果控制器没有实现该方法，退化为简单减小不确定度。
        """
        if hasattr(self.motion_ctrl, 'perform_active_perception'):
            self.motion_ctrl.perform_active_perception()
        else:
            # 简化版：将不确定性减半
            self.uncertainty_estimate *= 0.5

    def _execute_control_cycle(self, target_state):
        """执行控制循环，底层由阶段管理器自动处理RL子目标计算"""
        control_steps = self.config.get("control_steps_per_rl_step", 25)

        for step in range(control_steps):
            # 获取当前物理状态（适当时强制更新）
            current_state_data = self.motion_ctrl.get_system_state(
                step, step * self.motion_ctrl.scene.dt,
                force_update=(step % 5 == 0)
            )

            # 通过阶段管理器计算速度并应用
            vel_array = self.motion_ctrl.calculate_control_velocity(current_state_data)
            self.motion_ctrl.apply_velocity(vel_array)
            self.motion_ctrl.scene.step()

        return self._get_current_shape()

    def _get_observation(self) -> np.ndarray:
        """构建45维观测向量"""
        ee_pos = self._get_end_effector_position()

        return np.concatenate([
            self.current_state,           # 14
            self.target_shape,            # 14
            ee_pos,                       # 3
            self.uncertainty_estimate     # 14
        ], dtype=np.float32)

    def _get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器3维位置（弹性体中心）"""
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

            # 方法3: 尝试其他获取位置的方法
            for method_name in ['get_center', 'get_position', 'center', 'position']:
                if hasattr(self.motion_ctrl.sensor_cube, method_name):
                    pos = getattr(self.motion_ctrl.sensor_cube, method_name)()
                    if pos is not None:
                        if hasattr(pos, 'cpu'):
                            pos = pos.cpu().numpy()
                        return pos.flatten()[:3].astype(np.float32)

            # 回退：返回零向量
            return np.zeros(3, dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    def _get_current_shape(self) -> np.ndarray:
        """
        获取当前14维形状特征。
        优先使用 motion_ctrl 已缓存的 current_14d_state；
        若不存在，尝试主动执行一次状态估计；
        若仍失败，返回默认值并记录警告。
        """
        # 1. 优先从缓存读取
        if hasattr(self.motion_ctrl, 'current_14d_state') and self.motion_ctrl.current_14d_state is not None:
            state = np.array(self.motion_ctrl.current_14d_state, dtype=np.float32)
            if len(state) != 14:
                # 维度异常处理：截断或填充
                resized = np.zeros(self.state_dim, dtype=np.float32)
                resized[:min(len(state), self.state_dim)] = state[:self.state_dim]
                state = resized
            # 有效性检查：尺度不能全零
            if np.all(np.abs(state[:3]) < 1e-6):
                if self._last_valid_state is not None:
                    return self._last_valid_state.copy()
                else:
                    return np.ones(self.state_dim, dtype=np.float32) * 0.08
            self._last_valid_state = state.copy()
            return state

        # 2. 尝试主动估计一次 (可能发生在接触前)
        if hasattr(self.motion_ctrl, 'estimate_and_save_superquadric'):
            try:
                # 获取当前物理状态（强制更新）
                sys_state = self.motion_ctrl.get_system_state(
                    step=self.motion_ctrl.scene.step_counter,
                    time=self.motion_ctrl.scene.time,
                    force_update=True
                )
                result = self.motion_ctrl.estimate_and_save_superquadric(sys_state)
                if result and 'feature_14d' in result:
                    state = result['feature_14d'].astype(np.float32)
                    if len(state) == self.state_dim:
                        self._last_valid_state = state.copy()
                        return state
            except Exception as e:
                print(f"主动形状估计失败: {e}")

        # 3. 最终回退
        return np.ones(self.state_dim, dtype=np.float32) * 0.08

    def _estimate_uncertainty(self) -> np.ndarray:
        """基于最近的预测误差估计14维不确定性"""
        if len(self.prediction_errors) > 0:
            # 取最近至多5次误差的方差
            recent = np.array(self.prediction_errors[-5:], dtype=np.float32)
            uncertainty = np.var(recent, axis=0)
            # 确保维度一致
            if len(uncertainty) < self.uncertainty_dim:
                padded = np.zeros(self.uncertainty_dim, dtype=np.float32)
                padded[:len(uncertainty)] = uncertainty
                uncertainty = padded
            elif len(uncertainty) > self.uncertainty_dim:
                uncertainty = uncertainty[:self.uncertainty_dim]
            return uncertainty
        else:
            return np.ones(self.uncertainty_dim, dtype=np.float32) * 0.01

    def _compute_reward(self, achieved_state: np.ndarray) -> tuple[float, bool]:
        """
        计算奖励和终止标志
        Returns:
            reward: 标量奖励值
            terminated: 是否成功达到目标
        """
        weights = self.config["reward_weights"]

        # 形状距离奖励
        dist_old = np.linalg.norm(self.current_state - self.target_shape)
        dist_new = np.linalg.norm(achieved_state - self.target_shape)
        reward_goal = (dist_old - dist_new) * weights["goal"]

        # 稀疏成功奖励
        success_threshold = 0.05
        success = dist_new < success_threshold
        reward_success = weights["success"] if success else 0.0

        # 时间步惩罚
        reward_time = weights["time"]

        # 安全约束：体积剧烈变化惩罚
        vol_idx = 11  # volume 在14维向量中的索引
        reward_safety = 0.0
        if len(achieved_state) > vol_idx:
            current_volume = achieved_state[vol_idx]
            if not hasattr(self, '_initial_volume'):
                self._initial_volume = current_volume
            if self._initial_volume > 1e-6:
                deviation = abs(current_volume - self._initial_volume) / self._initial_volume
                if deviation > 0.2:
                    reward_safety = weights["safety"]

        # 不确定性惩罚
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
            print(f"关闭环境时出错: {e}")