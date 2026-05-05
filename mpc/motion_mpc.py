import numpy as np
from common.base.motion import BaseMotionController, MotionControllerUtils


class MotionControllerMPC(BaseMotionController):
    """Motion controller integrated with MPC control."""

    def __init__(self, scene, sensor_cube, elastoplastic_obj, initial_particles,
                 mpc_controller, output_dir="mpc_control_data"):
        super().__init__(scene, sensor_cube, elastoplastic_obj, initial_particles, output_dir)

        self.mpc_controller = mpc_controller
        self.mpc_enabled = False

        # MPC specific
        self.current_16d_state = None
        self.control_history = []
        self.mpc_update_interval = 25
        self.mpc_step_counter = 0
        self.cached_mpc_control = np.zeros(3)

        # 恢复原来的打印控制参数
        self.print_interval = 50  # MPC使用固定的打印间隔

    def _setup_monitors(self):
        """Setup monitors with MPC-specific sampling ratio."""
        super()._setup_monitors(elastoplastic_sampling_ratio=1.0)

    def update_motion_phase(self, current_state, t):
        """Update motion phase - MPC control."""
        contact_info = current_state.get('contact', {})

        if contact_info.get('contact_detected', False) and not self.contact_established:
            self.contact_established = True
            self.mpc_enabled = True
            print(f"=== 接触建立！穿透深度: {contact_info.get('penetration_depth', 0):.4f}m，启用MPC控制 ===")

    def calculate_velocity(self):
        """Calculate velocity (for compatibility)."""
        return self.calculate_mpc_velocity(self.cached_state)

    def calculate_mpc_velocity(self, current_state):
        """Calculate MPC control velocity."""
        if not self.mpc_enabled or self.current_16d_state is None:
            vel_array = np.zeros((len(self.initial_particles), 3))
            vel_array[:, 2] = self.v_down

            control_record = {
                'step': current_state['step'],
                'time': current_state['time'],
                'control': [0.0, 0.0, self.v_down],
                'mpc_enabled': False,
                'state': self.current_16d_state
            }
            self.control_history.append(control_record)
            return vel_array

        if self.mpc_step_counter >= self.mpc_update_interval:
            self.mpc_step_counter = 0

            latest_state_estimate = self._get_latest_state_estimate(current_state)
            if latest_state_estimate is not None:
                self.current_16d_state = latest_state_estimate
                print(f"🔄 MPC优化前更新状态到步骤 {current_state['step']}")

            try:
                contact_info = current_state.get('contact', {})
                mpc_contact_info = {
                    'contact_detected': contact_info.get('contact_detected', False),
                    'penetration_depth': contact_info.get('penetration_depth', 0)
                }

                control = self.mpc_controller.control(self.current_16d_state, mpc_contact_info)
                self.cached_mpc_control = control

                control_record = {
                    'step': current_state['step'],
                    'time': current_state['time'],
                    'control': control.tolist(),
                    'mpc_enabled': True,
                    'state': self.current_16d_state,
                    'mpc_step': self.mpc_step_counter
                }
                self.control_history.append(control_record)

                print(f"\n🎯 MPC控制更新 (步骤 {current_state['step']}):")
                print(f"  使用状态: {[f'{x:.3f}' for x in self.current_16d_state[:3]]}...")
                print(f"  新控制指令: {control}, 范数: {np.linalg.norm(control):.4f}")

            except Exception as e:
                print(f"❌ MPC控制计算失败: {e}，使用上一控制指令")

        self.mpc_step_counter += 1

        vel_array = np.zeros((len(self.initial_particles), 3))
        vel_array[:, 0] = self.cached_mpc_control[0]
        vel_array[:, 1] = self.cached_mpc_control[1]
        vel_array[:, 2] = self.cached_mpc_control[2]

        return vel_array

    def _get_latest_state_estimate(self, current_state):
        """Get latest state estimate."""
        try:
            if self.current_16d_state is None:
                print("⚠️ 初始状态为空，强制状态估计")
                return self._force_state_estimation(current_state)

            contact_info = current_state.get('contact', {})
            if not contact_info.get('contact_detected', False):
                return None

            estimation_record = self.estimate_and_save_superquadric(current_state)
            if estimation_record and 'feature_16d' in estimation_record:
                return estimation_record['feature_16d']

            return None
        except Exception as e:
            print(f"获取最新状态估计失败: {e}")
            return None

    def _force_state_estimation(self, current_state):
        """Force state estimation without saving."""
        ems_params = {
            'OutlierRatio': 0.05,
            'MaxIterationEM': 20,
            'ToleranceEM': 1e-3,
            'RelativeToleranceEM': 2e-1,
            'MaxOptiIterations': 2,
            'Sigma': 0.0,
            'MaxiSwitch': 2,
            'AdaptiveUpperBound': True,
            'Rescale': False
        }

        estimation_result = self._perform_superquadric_estimation(current_state, ems_params)
        if estimation_result is None:
            return None

        return MotionControllerUtils.get_16d_feature_vector(
            estimation_result['param_dict'],
            estimation_result['geometric_features']
        )

    def estimate_and_save_superquadric(self, current_state):
        """Execute superquadric estimation and save results."""
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

        # Calculate 16D feature vector for MPC
        self.current_16d_state = MotionControllerUtils.get_16d_feature_vector(
            estimation_result['param_dict'],
            estimation_result['geometric_features']
        )

        # Save point cloud
        point_cloud_filename = self.save_point_cloud(
            estimation_result['surface_points'],
            estimation_result['step'],
            estimation_result['time']
        )

        # Create parameter record
        estimation_record = {
            'step': estimation_result['step'],
            'time': estimation_result['time'],
            'contact_info': estimation_result['contact_info'],
            'parameters_11d': estimation_result['param_dict'],
            'geometric_features': estimation_result['geometric_features'],
            'feature_16d': self.current_16d_state,
            'point_cloud_file': point_cloud_filename
        }

        # Add elastic state if available
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

        # Save parameters
        param_filename = self.save_superquadric_params(
            estimation_record,
            estimation_result['step'],
            estimation_result['time']
        )

        self.superquadric_params_history.append(estimation_record)

        print(f"=== 状态估计完成 (步骤 {estimation_result['step']}) ===")
        print(f"16维状态: {[f'{x:.3f}' for x in self.current_16d_state[:5]]}...")

        return estimation_record

    def print_state_info(self, current_state):
        """Print state information."""
        MotionControllerUtils.print_state_info(
            current_state,
            mpc_enabled=self.mpc_enabled,
            current_16d_state=self.current_16d_state,
            use_traditional_mode=False
        )

    def should_print_status(self, step, current_state):
        """Determine if status should be printed - 保持与main_mpc.py兼容的接口"""
        return MotionControllerUtils.should_print_status(
            self.print_counter, step,
            print_interval_fast=self.print_interval,
            print_interval_slow=self.print_interval,
            initial_threshold=10000
        )

    def save_control_history(self):
        """Save control history to file."""
        MotionControllerUtils.save_control_history(self.output_dir, self.control_history)

    def save_custom_format_data(self):
        """Save training data format."""
        return MotionControllerUtils.save_custom_format_data(
            self.output_dir,
            self.superquadric_params_history,
            data_type="mpc"
        )

    def finalize_simulation(self):
        """Finalize simulation."""
        try:
            # Save custom format data for MPC
            MotionControllerUtils.save_custom_format_data(
                self.output_dir,
                self.superquadric_params_history,
                data_type="mpc"
            )

            # Save control history
            MotionControllerUtils.save_control_history(self.output_dir, self.control_history)

            print("MPC仿真数据导出完成")
        except Exception as e:
            print(f"导出数据时出错: {e}")
