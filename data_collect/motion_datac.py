import math
import os
import random
import numpy as np
from common.base.motion import BaseMotionController, MotionControllerUtils


class MotionControllerDataCollection(BaseMotionController):
    """Motion controller for data collection with random actions."""

    def __init__(self, scene, sensor_cube, elastoplastic_obj, initial_particles,
                total_steps=6000, output_dir="realtime_data"):
        super().__init__(scene, sensor_cube, elastoplastic_obj, initial_particles, output_dir)

        # Random action parameters
        self.random_action_enabled = False
        self.action_duration = 100
        self.action_counter = 0
        self.current_action = None
        self.use_traditional_mode = False
        self.target_height = 0.2

        # Traditional mode parameters
        self.motion_phase = "向下运动"
        self.current_radius = 0.0
        self.current_angle = 0.0
        self.target_z_position = None
        self.spiral_target_radius = 0.2
        self.spiral_radial_speed = 0.16
        self.spiral_angular_speed = 2.6
        self.circle_angular_speed = 1.4

        # Action space
        self.action_space = {
            'move_x_positive': {'vel': [0.1, 0, 0], 'desc': 'X轴正方向移动'},
            'move_x_negative': {'vel': [-0.1, 0, 0], 'desc': 'X轴负方向移动'},
            'move_y_positive': {'vel': [0, 0.1, 0], 'desc': 'Y轴正方向移动'},
            'move_y_negative': {'vel': [0, -0.1, 0], 'desc': 'Y轴负方向移动'},
            'move_z_positive': {'vel': [0, 0, 0.05], 'desc': 'Z轴正方向移动'},
            'move_z_negative': {'vel': [0, 0, -0.05], 'desc': 'Z轴负方向移动'},
            'move_xy_positive': {'vel': [0.1, 0.1, 0], 'desc': 'XY平面正向移动'},
            'move_xy_negative': {'vel': [-0.1, -0.1, 0], 'desc': 'XY平面负向移动'},
            'move_xy_pn': {'vel': [0.1, -0.1, 0], 'desc': 'XY平面正向移动'},
            'move_xy_np': {'vel': [-0.1, 0.1, 0], 'desc': 'XY平面负向移动'},
            'move_xz_positive': {'vel': [0.1, 0, 0.05], 'desc': 'XZ平面正向移动'},
            'move_xz_negative': {'vel': [-0.1, 0, -0.05], 'desc': 'XZ平面负向移动'},
            'move_yz_positive': {'vel': [0, 0.1, 0.05], 'desc': 'YZ平面正向移动'},
            'move_yz_negative': {'vel': [0, -0.1, -0.05], 'desc': 'YZ平面负向移动'},
            'move_xyz_positive': {'vel': [0.07, 0.07, 0.03], 'desc': 'XYZ空间正向移动'},
            'move_xyz_negative': {'vel': [-0.07, -0.07, -0.03], 'desc': 'XYZ空间负向移动'},
        }

        self.action_weights = {
            'move_x_positive': 0.0, 'move_x_negative': 0.0,
            'move_y_positive': 0.0, 'move_y_negative': 1.0,
            'move_z_positive': 0.0, 'move_z_negative': 0.0,
            'move_xy_positive': 0.0, 'move_xy_negative': 0.0,
            'move_xy_pn': 0.0, 'move_xy_np': 0.0,
            'move_xz_positive': 0.0, 'move_xz_negative': 0.0,
            'move_yz_positive': 0.0, 'move_yz_negative': 0.0,
            'move_xyz_positive': 0.0, 'move_xyz_negative': 0.0,
        }
        # self.action_weights = {
        #     'move_x_positive': 1.0, 'move_x_negative': 1.0,
        #     'move_y_positive': 1.0, 'move_y_negative': 1.0,
        #     'move_z_positive': 1.0, 'move_z_negative': 1.0,
        #     'move_xy_positive': 10.0, 'move_xy_negative': 10.0,
        #     'move_xz_positive': 10.0, 'move_xz_negative': 1.0,
        #     'move_yz_positive': 10.0, 'move_yz_negative': 1.0,
        #     'move_xyz_positive': 1.0, 'move_xyz_negative': 1.0,
        # }
        # History
        self.action_history = []
        self.total_steps = total_steps
        self.point_cloud_history = []
        self.initial_sq = None

        # 恢复原来的打印控制参数
        self.print_interval_fast = 80
        self.print_interval_slow = 40

    def update_motion_phase(self, current_state, t):
        """Update motion phase."""
        contact_info = current_state.get('contact', {})

        if contact_info.get('contact_detected', False):
            self.contact_established = True
            if not hasattr(self.scene, 'contact_established_step'):
                self.scene.contact_established_step = current_state['step']
                print(f"=== 接触建立！穿透深度: {contact_info.get('penetration_depth', 0):.4f}m ===")

                if not self.use_traditional_mode:
                    self.random_action_enabled = True
                    self._select_random_action()
                    print(f"=== 开始随机动作模式 ===")

        if self.use_traditional_mode:
            self._update_traditional_motion_phase(current_state, t)
        else:
            self._update_random_motion_phase(current_state, t)

    def _update_traditional_motion_phase(self, current_state, t):
        """Traditional mode motion phase update."""
        elastic_data = current_state.get('elastic', {})
        if 'state' in elastic_data:
            cube_min_z = elastic_data['state']['bounding_box']['min'][2]

            if self.motion_phase == "向下运动" and cube_min_z <= self.target_height and self.contact_established:
                self.motion_phase = "螺旋运动"
                self.current_angle = 0.0
                self.target_z_position = elastic_data['state']['center'][2]
                print(f"=== 开始螺旋运动 ===")

            elif self.motion_phase == "螺旋运动" and self.current_radius >= self.spiral_target_radius:
                self.motion_phase = "圆周运动"
                self.current_radius = self.spiral_target_radius
                print(f"=== 开始圆周运动 ===")

    def _update_random_motion_phase(self, current_state, t):
        """Random mode motion phase update."""
        if self.random_action_enabled:
            self.action_counter += 1
            if self.action_counter >= self.action_duration:
                self._select_random_action()
                self.action_counter = 0

    def _select_random_action(self,):
        """Select random action."""
        available_actions = list(self.action_space.keys())
        available_weights = [self.action_weights[action] for action in available_actions]

        if self.random_action_enabled and len(self.action_history) < 6:
            upward_actions = ['move_z_positive', 'move_xz_positive', 'move_yz_positive', 'move_xyz_positive']
            filtered_data = [(a, w) for a, w in zip(available_actions, available_weights) if a not in upward_actions]
            if filtered_data:
                available_actions, available_weights = zip(*filtered_data)

        selected_action = random.choices(available_actions, weights=available_weights, k=1)[0]
        self.current_action = selected_action

        action_record = {
            'step': getattr(self.scene, 'step_counter', 0),
            'time': getattr(self.scene, 'step_counter', 0) * self.scene.dt,
            'action': selected_action,
            'description': self.action_space[selected_action]['desc'],
            'velocity': self.action_space[selected_action]['vel']
        }
        self.action_history.append(action_record)

        print(f"执行动作: {self.action_space[selected_action]['desc']}")
        print(f"速度向量: {self.action_space[selected_action]['vel']}")

    def calculate_velocity(self):
        """Calculate current velocity."""
        if self.use_traditional_mode:
            return self._calculate_traditional_velocity()
        else:
            return self._calculate_random_velocity()

    def _calculate_traditional_velocity(self):
        """Traditional mode velocity calculation."""
        if self.motion_phase == "向下运动":
            vel_array = np.zeros((len(self.initial_particles), 3))
            vel_array[:, 2] = self.v_down

        elif self.motion_phase == "螺旋运动":
            self.current_angle += self.spiral_angular_speed * self.scene.dt
            self.current_radius = min(self.current_radius + self.spiral_radial_speed * self.scene.dt,
                                      self.spiral_target_radius)

            v_x = (self.spiral_radial_speed * math.cos(self.current_angle) -
                   self.spiral_angular_speed * self.current_radius * math.sin(self.current_angle))
            v_y = (self.spiral_radial_speed * math.sin(self.current_angle) +
                   self.spiral_angular_speed * self.current_radius * math.cos(self.current_angle))
            v_z = self._calculate_z_velocity_control()

            vel_array = np.zeros((len(self.initial_particles), 3))
            vel_array[:, 0] = v_x
            vel_array[:, 1] = v_y
            vel_array[:, 2] = v_z

        elif self.motion_phase == "圆周运动":
            self.current_angle += self.circle_angular_speed * self.scene.dt

            v_x = -self.circle_angular_speed * self.spiral_target_radius * math.sin(self.current_angle)
            v_y = self.circle_angular_speed * self.spiral_target_radius * math.cos(self.current_angle)
            v_z = self._calculate_z_velocity_control()

            vel_array = np.zeros((len(self.initial_particles), 3))
            vel_array[:, 0] = v_x
            vel_array[:, 1] = v_y
            vel_array[:, 2] = v_z

        else:
            vel_array = np.zeros((len(self.initial_particles), 3))

        return vel_array

    def _calculate_random_velocity(self):
        """Random mode velocity calculation."""
        if not self.random_action_enabled:
            vel_array = np.zeros((len(self.initial_particles), 3))
            vel_array[:, 2] = self.v_down
            return vel_array

        if self.current_action and self.current_action in self.action_space:
            action_vel = self.action_space[self.current_action]['vel']
            vel_array = np.zeros((len(self.initial_particles), 3))
            vel_array[:, 0] = action_vel[0]
            vel_array[:, 1] = action_vel[1]
            vel_array[:, 2] = action_vel[2]
            return vel_array
        else:
            return np.zeros((len(self.initial_particles), 3))

    def _calculate_z_velocity_control(self):
        """Calculate Z-direction velocity control."""
        if self.target_z_position is None:
            return 0.0

        current_state = self.elastic_monitor.get_current_state()
        if current_state is None:
            return 0.0

        current_z = current_state['center'][2]
        z_error = self.target_z_position - current_z

        k_p = 3.0
        v_z = k_p * z_error
        max_v_z = 0.05
        return np.clip(v_z, -max_v_z, max_v_z)

    def estimate_and_save_superquadric(self, current_state):
        """Execute superquadric estimation and save results."""
        estimation_result = self._perform_superquadric_estimation(current_state)
        if estimation_result is None:
            return None

        # Get additional data specific to data collection mode
        elastic_state = self.elastic_monitor.get_current_state()
        delta_elastic = self.calculate_elastic_displacement(elastic_state)

        current_vel_array = self.calculate_velocity()

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
            'control_velocity': {
                'vel_x': float(current_vel_array[0, 0]),
                'vel_y': float(current_vel_array[0, 1]),
                'vel_z': float(current_vel_array[0, 2])
            },
            'delta_elastic': delta_elastic,
            'point_cloud_file': point_cloud_filename
        }

        # Save parameters
        param_filename = self.save_superquadric_params(
            estimation_record,
            estimation_result['step'],
            estimation_result['time']
        )

        # Update history
        self.superquadric_params_history.append(estimation_record)

        print(
            f"=== 实时数据保存完成 (步骤 {estimation_result['step']}, 接触深度: {estimation_result['contact_info'].get('penetration_depth', 0):.4f}m) ===")
        print(f"点云文件: {point_cloud_filename}")
        print(f"参数文件: {param_filename}")

        return estimation_record

    def print_state_info(self, current_state):
        """Print state information."""
        MotionControllerUtils.print_state_info(
            current_state,
            motion_phase=self.motion_phase if self.use_traditional_mode else None,
            random_action_enabled=self.random_action_enabled,
            current_action=self.current_action,
            action_space=self.action_space,
            action_duration=self.action_duration,
            action_counter=self.action_counter,
            use_traditional_mode=self.use_traditional_mode
        )

    def should_print_status(self, step, current_state):
        """Determine if status should be printed - 保持与main_datac.py兼容的接口"""
        return MotionControllerUtils.should_print_status(
            self.print_counter, step,
            print_interval_fast=self.print_interval_fast,
            print_interval_slow=self.print_interval_slow
        )

    def save_action_history(self):
        """Save action history - 保持与main_datac.py兼容的接口"""
        MotionControllerUtils.save_action_history(self.output_dir, self.action_history)

    def save_custom_format_data(self):
        """Save custom format data - 保持与main_datac.py兼容的接口"""
        return MotionControllerUtils.save_custom_format_data(
            self.output_dir,
            self.superquadric_params_history,
            data_type="general"
        )

    def export_summary_data(self):
        """导出汇总数据 - 保持与main_datac.py兼容的接口"""
        # 这个函数在main_datac.py中没有被调用，但为了完整性保留
        import json
        import pandas as pd

        if not self.superquadric_params_history:
            print("没有超二次曲面参数可导出")
            return

        summary_data = {
            'simulation_info': {
                'start_time': self.simulation_start_time,
                'total_steps': len(self.superquadric_params_history),
                'output_directory': self.output_dir,
                'initial_geometric_features': self.initial_sq
            },
            'superquadric_estimations': self.superquadric_params_history,
            'point_cloud_files': [
                {
                    'step': pc['step'],
                    'time': pc['time'],
                    'filename': pc['filename'],
                    'num_points': len(pc['points'])
                }
                for pc in self.point_cloud_history
            ]
        }

        summary_file = os.path.join(self.output_dir, "simulation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"汇总数据已导出: {summary_file}")

    def finalize_simulation(self):
        """Finalize simulation."""
        try:
            # Save custom format data
            MotionControllerUtils.save_custom_format_data(
                self.output_dir,
                self.superquadric_params_history,
                data_type="general"
            )

            # Save action history in random mode
            if not self.use_traditional_mode and self.action_history:
                MotionControllerUtils.save_action_history(self.output_dir, self.action_history)

            print("所有监控数据已导出")
        except Exception as e:
            print(f"导出数据时出错: {e}")