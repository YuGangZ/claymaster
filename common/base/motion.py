"""
Shared utilities for motion controllers.
This module contains common functionality between data collection and MPC motion controllers.
"""
import json
import math
import numpy as np
from datetime import datetime
from physical_engine.state_monitor import ElasticBodyMonitor, ElastoPlasticBodyMonitor, ContactMonitor
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from superquadric_estimator.metric import calculate_geometric_features
from superquadric_estimator.ems_recovery import EMS_recovery
import genesis as gs


class MotionControllerUtils:
    """Utility class containing shared functionality for motion controllers."""

    @staticmethod
    def create_output_directories(output_dir):
        """Create output directory structure."""
        directories = [
            output_dir,
            os.path.join(output_dir, "point_clouds"),
            os.path.join(output_dir, "superquadric_params"),
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")
        return directories

    @staticmethod
    def setup_monitors(sensor_cube, elastoplastic_obj, elastic_sampling_ratio=0.1, elastoplastic_sampling_ratio=0.15):
        """Setup monitors for elastic and elastoplastic bodies."""
        elastic_monitor = ElasticBodyMonitor(
            sensor_cube, name="弹性立方体", sampling_ratio=elastic_sampling_ratio, record_full_state=True
        )
        elastoplastic_monitor = ElastoPlasticBodyMonitor(
            elastoplastic_obj, name="弹塑体", sampling_ratio=elastoplastic_sampling_ratio, record_full_state=True
        )
        contact_monitor = ContactMonitor(elastic_monitor, elastoplastic_monitor)

        elastic_monitor.initialize()
        elastoplastic_monitor.initialize()
        print("MPM材料监控器已初始化")

        return elastic_monitor, elastoplastic_monitor, contact_monitor

    @staticmethod
    def get_standard_11d_params(sq):
        """Extract standard 11D superquadric parameters."""
        params = {
            'scale_a1': sq.scale[0],
            'scale_a2': sq.scale[1],
            'scale_a3': sq.scale[2],
            'shape_epsilon1': sq.shape[0],
            'shape_epsilon2': sq.shape[1],
            'translation_x': sq.translation[0],
            'translation_y': sq.translation[1],
            'translation_z': sq.translation[2],
            'euler_rx': sq.euler[2],
            'euler_ry': sq.euler[1],
            'euler_rz': sq.euler[0],
        }

        param_vector = [
            params['scale_a1'], params['scale_a2'], params['scale_a3'],
            params['shape_epsilon1'], params['shape_epsilon2'],
            params['translation_x'], params['translation_y'], params['translation_z'],
            params['euler_rx'], params['euler_ry'], params['euler_rz']
        ]

        return params, param_vector

    @staticmethod
    def get_14d_feature_vector(sq_params, geometric_features):
        """Form 14D feature vector."""
        sq_vector = [
            sq_params['scale_a1'], sq_params['scale_a2'], sq_params['scale_a3'],
            sq_params['shape_epsilon1'], sq_params['shape_epsilon2'],
            sq_params['translation_x'], sq_params['translation_y'], sq_params['translation_z'],
            sq_params['euler_rx'], sq_params['euler_ry'], sq_params['euler_rz']
        ]

        geometric_vector = [
            geometric_features.get('volume', 0),
            geometric_features.get('elongation', 1.0),
            geometric_features.get('smoothness', 0.5)
        ]

        return sq_vector + geometric_vector

    @staticmethod
    def save_point_cloud(points, output_dir, step, time):
        """Save point cloud data as PLY file."""
        try:
            header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""

            filename = f"pointcloud_step{step:06d}_t{time:.3f}.ply"
            filepath = os.path.join(output_dir, "point_clouds", filename)

            with open(filepath, 'w') as f:
                f.write(header)
                for point in points:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

            return filename
        except Exception as e:
            print(f"保存点云失败: {e}")
            return None

    @staticmethod
    def save_superquadric_params(params_record, output_dir, step, time):
        """Save superquadric parameters."""
        try:
            def convert_to_serializable(obj):
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj

            serializable_record = convert_to_serializable(params_record)

            filename = f"superquadric_step{step:06d}_t{time:.3f}.json"
            filepath = os.path.join(output_dir, "superquadric_params", filename)

            with open(filepath, 'w') as f:
                json.dump(serializable_record, f, indent=2, ensure_ascii=False)

            return filename
        except Exception as e:
            print(f"保存超二次曲面参数失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def calculate_geometric_features(sq, point_cloud):
        """Calculate geometric features."""
        try:
            features = calculate_geometric_features(sq, point_cloud)
            return {
                'volume': features.get('volume', np.prod(sq.scale) * 8),
                'elongation': features.get('elongation', 1.0),
                'smoothness': features.get('smoothness', 0.5),
            }
        except Exception as e:
            print(f"几何特征计算失败: {e}")
            return {
                'volume': np.prod(sq.scale) * 8,
                'elongation': 1.0,
                'smoothness': 0.5,
            }

    @staticmethod
    def apply_velocity(sensor_cube, vel_array, frame=0):
        """Apply velocity to the sensor cube."""
        vel_tensor = gs.Tensor(vel_array)
        sensor_cube.set_vel(frame, vel_tensor)
        sensor_cube.process_input()

    @staticmethod
    def json_serializer(obj):
        """JSON serializer helper function."""
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    @staticmethod
    def should_print_status(print_counter, step, print_interval_fast=80, print_interval_slow=40, initial_threshold=100):
        """Determine if status should be printed."""
        print_counter += 1
        if step < initial_threshold:
            return print_counter % print_interval_fast == 0
        else:
            return print_counter % print_interval_slow == 0

    @staticmethod
    def print_state_info(current_state, mpc_enabled=False, current_14d_state=None, motion_phase=None,
                        random_action_enabled=False, current_action=None, action_space=None,
                        action_duration=None, action_counter=None, use_traditional_mode=True):
        """Print state information."""
        elastic = current_state.get('elastic', {})
        elastoplastic = current_state.get('elastoplastic', {})
        contact = current_state.get('contact', {})

        print(f"[STEP {current_state['step']}] t={current_state['time']:.3f}")

        if mpc_enabled:
            print("  控制模式: MPC闭环控制")
            if current_14d_state is not None:
                print(f"  当前状态: {[f'{x:.3f}' for x in current_14d_state[:3]]}...")
        elif use_traditional_mode and motion_phase:
            print(f"  阶段: {motion_phase}")
        elif not use_traditional_mode:
            if random_action_enabled and current_action:
                if action_space and current_action in action_space:
                    action_desc = action_space[current_action]['desc']
                    print(f"  动作模式: 随机动作 - {action_desc}")
                    if action_duration and action_counter is not None:
                        print(f"  动作剩余时间: {action_duration - action_counter}步")
                else:
                    print(f"  动作模式: 随机动作 - {current_action}")
            else:
                print(f"  动作模式: 向下运动")

        if 'state' in elastic:
            state = elastic['state']
            print(f"  弹性体: 中心=({state['center'][0]:.3f}, {state['center'][1]:.3f}, {state['center'][2]:.3f})")
        if 'state' in elastoplastic:
            state = elastoplastic['state']
            print(f"  弹塑性体: 中心=({state['center'][0]:.3f}, {state['center'][1]:.3f}, {state['center'][2]:.3f})")

        if contact.get('contact_detected', False):
            print(f"  接触: 穿透={contact.get('penetration_depth', 0):.4f}m")
        else:
            print(f"  接触: 无接触")

        print("-" * 80)


class BaseMotionController:
    """Base class with common functionality for all motion controllers."""

    def __init__(self, scene, sensor_cube, elastoplastic_obj, initial_particles, output_dir):
        self.scene = scene
        self.sensor_cube = sensor_cube
        self.elastoplastic_obj = elastoplastic_obj
        self.initial_particles = initial_particles
        self.output_dir = output_dir

        # Common parameters
        self.v_down = -0.2
        self.contact_established = False

        # Data collection
        self.superquadric_params_history = []
        self.estimation_interval = 50
        self.estimation_counter = 0
        self.simulation_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # State caching
        self.last_state_time = -1
        self.cached_state = None

        # Print control
        self.print_counter = 0

        # Initialize
        MotionControllerUtils.create_output_directories(output_dir)
        self._setup_monitors()

    def _setup_monitors(self, elastoplastic_sampling_ratio=1.0):
        """Setup monitors with customizable sampling ratio."""
        self.elastic_monitor, self.elastoplastic_monitor, self.contact_monitor = \
            MotionControllerUtils.setup_monitors(
                self.sensor_cube, self.elastoplastic_obj,
                elastic_sampling_ratio=0.1,
                elastoplastic_sampling_ratio=elastoplastic_sampling_ratio
            )

    def get_system_state(self, step, time, force_update=False):
        """Quickly get system state with caching."""
        if not force_update and self.last_state_time == step:
            return self.cached_state

        elastic_state = self.elastic_monitor.get_current_state()
        elastoplastic_state = self.elastoplastic_monitor.get_current_state()
        contact_info = self.contact_monitor.detect_contact(elastic_state, elastoplastic_state)

        current_state = {
            'step': step,
            'time': time,
            'elastic': {'state': elastic_state} if elastic_state else {},
            'elastoplastic': {'state': elastoplastic_state} if elastoplastic_state else {},
            'contact': contact_info
        }

        self.cached_state = current_state
        self.last_state_time = step
        return current_state

    def apply_velocity(self, vel_array, frame=0):
        """Apply velocity to the sensor cube."""
        MotionControllerUtils.apply_velocity(self.sensor_cube, vel_array, frame)

    def calculate_elastic_displacement(self, current_elastic_state):
        """Calculate elastic body displacement."""
        if current_elastic_state is None:
            return {'delta_x': 0, 'delta_y': 0, 'delta_z': 0}

        current_center = current_elastic_state['center']

        if not hasattr(self, 'elastic_initial_center'):
            initial_state = self.elastic_monitor.initial_state
            if initial_state and 'center' in initial_state:
                self.elastic_initial_center = initial_state['center']
            else:
                self.elastic_initial_center = current_center

        return {
            'delta_x': float(current_center[0] - self.elastic_initial_center[0]),
            'delta_y': float(current_center[1] - self.elastic_initial_center[1]),
            'delta_z': float(current_center[2] - self.elastic_initial_center[2])
        }

    def _perform_superquadric_estimation(self, current_state, ems_params=None):
        """Perform the core superquadric estimation (common part)."""
        if ems_params is None:
            ems_params = {
                'OutlierRatio': 0.05,
                'MaxIterationEM': 20,
                'ToleranceEM': 1e-3,
                'RelativeToleranceEM': 1e-1,
                'MaxOptiIterations': 3,
                'Sigma': 1e-3,
                'MaxiSwitch': 2,
                'AdaptiveUpperBound': True,
                'Rescale': True
            }

        contact_info = current_state.get('contact', {})
        if not contact_info.get('contact_detected', False):
            return None

        elastoplastic_state = self.elastoplastic_monitor.get_current_state()
        if elastoplastic_state is None or len(elastoplastic_state['surface_positions']) < 10:
            return None

        surface_points = elastoplastic_state['surface_positions']
        sq, probabilities = EMS_recovery(surface_points, **ems_params)

        param_dict, param_vector = MotionControllerUtils.get_standard_11d_params(sq)
        geometric_features = MotionControllerUtils.calculate_geometric_features(sq, surface_points)

        return {
            'sq': sq,
            'param_dict': param_dict,
            'geometric_features': geometric_features,
            'surface_points': surface_points,
            'contact_info': contact_info,
            'step': current_state['step'],
            'time': current_state['time']
        }

    def save_point_cloud(self, points, step, time):
        """Save point cloud data."""
        return MotionControllerUtils.save_point_cloud(points, self.output_dir, step, time)

    def save_superquadric_params(self, params_record, step, time):
        """Save superquadric parameters."""
        return MotionControllerUtils.save_superquadric_params(params_record, self.output_dir, step, time)

    def should_print_status(self, step):
        """Determine if status should be printed."""
        return MotionControllerUtils.should_print_status(
            self.print_counter, step,
            print_interval_fast=80,
            print_interval_slow=40
        )
