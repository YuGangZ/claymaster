# baseline/rl_env_direct.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class DeformationRLEnvDirect(gym.Env):
    """
    纯RL环境：直接输出3维速度控制
    观测：45维 = 14(current) + 14(target) + 3(ee_pos) + 14(uncertainty)
    动作：3维连续速度
    不确定性：来自 ShapePredictor 模型的预测方差
    """
    def __init__(self, motion_controller, target_shape, config):
        super().__init__()
        self.motion_ctrl = motion_controller
        self.target_shape = np.array(target_shape, dtype=np.float32)
        self.config = config

        self.state_dim = config["state_dim"]
        self.control_dim = config["control_dim"]
        self.max_steps = config["max_episode_steps"]
        self.control_steps = config["control_steps_per_env_step"]

        obs_dim = self.state_dim + self.state_dim + self.control_dim + self.state_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        low = config.get("action_low", -0.3)
        high = config.get("action_high", 0.3)
        self.action_space = spaces.Box(low=low, high=high, shape=(self.control_dim,), dtype=np.float32)

        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.previous_state = None
        self.uncertainty_estimate = np.ones(self.state_dim, dtype=np.float32) * 0.01

        self.prediction_error_buffer = deque(maxlen=10)
        self._estimation_step_interval = 5
        self._steps_since_last_estimation = 0

        self._initial_volume = None
        self._last_valid_state = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.motion_ctrl.scene.reset()
        if hasattr(self.motion_ctrl, 'reset'):
            self.motion_ctrl.reset()

        self.current_step = 0
        self._steps_since_last_estimation = 0
        self.prediction_error_buffer.clear()
        self._initial_volume = None
        self._last_valid_state = None

        self.current_state = self._get_current_shape(force_estimation=True)
        self.previous_state = self.current_state.copy()
        self.uncertainty_estimate = self._get_prediction_uncertainty(self.current_state, np.zeros(3))
        return self._get_observation(), self._get_info()

    def step(self, action):
        prev_state = self.current_state.copy()
        prev_uncertainty = self.uncertainty_estimate.copy()

        # 获取基于当前状态-动作的预测不确定性
        pred_uncertainty = self._get_prediction_uncertainty(prev_state, action)

        # 执行连续多个物理步
        for phys_step in range(self.control_steps):
            vel_array = np.zeros((self.motion_ctrl.initial_particles.shape[0], 3))
            vel_array[:, :] = action
            self.motion_ctrl.apply_velocity(vel_array)
            self.motion_ctrl.scene.step()

            self._steps_since_last_estimation += 1
            if self._steps_since_last_estimation >= self._estimation_step_interval:
                self._update_shape_estimate()
                self._steps_since_last_estimation = 0

        self._update_shape_estimate()
        new_state = self.current_state.copy()

        # 记录预测误差（如果之前有预测值）
        if self.previous_state is not None:
            pred_error = new_state - self.previous_state
            self.prediction_error_buffer.append(pred_error)

        self.previous_state = new_state
        # 使用模型预测的不确定性作为当前不确定性估计
        self.uncertainty_estimate = pred_uncertainty

        reward, terminated = self._compute_reward(prev_state, new_state)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_prediction_uncertainty(self, state, action):
        """调用运动控制器中的 ShapePredictor 获取不确定性"""
        if hasattr(self.motion_ctrl, 'predict_next_state_and_uncertainty'):
            _, uncertainty = self.motion_ctrl.predict_next_state_and_uncertainty(state, action)
            return uncertainty
        else:
            # 降级方案
            return np.ones(self.state_dim, dtype=np.float32) * 0.01

    def _update_shape_estimate(self):
        """执行一次形状估计，更新 current_state"""
        try:
            sys_state = self.motion_ctrl.get_system_state(
                step=self.motion_ctrl.scene.step_counter,
                time=self.motion_ctrl.scene.time,
                force_update=True
            )
            if sys_state.get('contact', {}).get('contact_detected', False):
                result = self.motion_ctrl.estimate_and_save_superquadric(sys_state)
                if result and 'feature_14d' in result:
                    state = result['feature_14d'].astype(np.float32)
                    if len(state) == self.state_dim and np.all(np.abs(state[:3]) > 1e-6):
                        self._last_valid_state = state.copy()
                        self.current_state = state
                        return
        except Exception as e:
            print(f"形状估计失败: {e}")

        if self._last_valid_state is not None:
            self.current_state = self._last_valid_state.copy()

    def _get_current_shape(self, force_estimation=False):
        if force_estimation:
            sys_state = self.motion_ctrl.get_system_state(
                step=self.motion_ctrl.scene.step_counter,
                time=self.motion_ctrl.scene.time,
                force_update=True
            )
            if sys_state.get('contact', {}).get('contact_detected', False):
                result = self.motion_ctrl.estimate_and_save_superquadric(sys_state)
                if result and 'feature_14d' in result:
                    state = result['feature_14d'].astype(np.float32)
                    if len(state) == self.state_dim and np.all(np.abs(state[:3]) > 1e-6):
                        self._last_valid_state = state.copy()
                        return state
        if self._last_valid_state is not None:
            return self._last_valid_state.copy()
        default = np.ones(self.state_dim, dtype=np.float32) * 0.08
        default[3:5] = 1.0
        default[11] = 0.002375
        return default

    def _get_observation(self):
        ee_pos = self._get_end_effector_position()
        return np.concatenate([
            self.current_state,
            self.target_shape,
            ee_pos,
            self.uncertainty_estimate
        ], dtype=np.float32)

    def _get_end_effector_position(self):
        try:
            if hasattr(self.motion_ctrl, 'elastic_monitor'):
                elastic_state = self.motion_ctrl.elastic_monitor.get_current_state()
                if elastic_state is not None:
                    return elastic_state['center'].astype(np.float32)
            if hasattr(self.motion_ctrl.sensor_cube, 'get_particles'):
                particles = self.motion_ctrl.sensor_cube.get_particles()
                if hasattr(particles, 'cpu'):
                    particles = particles.cpu().numpy()
                particles = particles.reshape(-1, 3)
                center = np.mean(particles, axis=0)
                return center[:3].astype(np.float32)
        except Exception:
            pass
        return np.zeros(3, dtype=np.float32)

    def _compute_reward(self, prev_state, new_state):
        weights = self.config["reward_weights"]

        dist_old = np.linalg.norm(prev_state - self.target_shape)
        dist_new = np.linalg.norm(new_state - self.target_shape)
        reward_goal = (dist_old - dist_new) * weights["goal"]

        success = dist_new < 0.05
        reward_success = weights["success"] if success else 0.0
        reward_time = weights["time"]

        reward_safety = 0.0
        vol_idx = 11
        if len(new_state) > vol_idx:
            current_volume = new_state[vol_idx]
            if self._initial_volume is None:
                self._initial_volume = current_volume
            if self._initial_volume > 1e-6:
                deviation = abs(current_volume - self._initial_volume) / self._initial_volume
                if deviation > 0.2:
                    reward_safety = weights["safety"]

        avg_uncertainty = np.mean(self.uncertainty_estimate)
        reward_uncertainty = avg_uncertainty * weights["uncertainty"]

        total_reward = reward_goal + reward_success + reward_time + reward_safety + reward_uncertainty
        return total_reward, success

    def _get_info(self):
        return {
            'step': self.current_step,
            'distance_to_target': float(np.linalg.norm(self.current_state - self.target_shape)),
            'current_state': self.current_state.copy(),
            'uncertainty': self.uncertainty_estimate.copy(),
        }

    def render(self):
        print(f"[Step {self.current_step}] dist={np.linalg.norm(self.current_state - self.target_shape):.4f}")