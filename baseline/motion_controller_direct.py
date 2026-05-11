# baseline/motion_controller_direct.py
import numpy as np
import torch
import os
from common.base.motion import BaseMotionController, MotionControllerUtils
from superquadric_estimator.ems_recovery import EMS_recovery


class DirectMotionController(BaseMotionController):
    def __init__(self, scene, sensor_cube, elastoplastic_obj, initial_particles,
                output_dir="direct_rl_data", estimation_interval=25,
                predictor_model_path="shape_predictor.pth"):
        super().__init__(scene, sensor_cube, elastoplastic_obj, initial_particles, output_dir)

        # 控制参数
        self.estimation_interval = estimation_interval
        self.last_estimation_step = -1
        self.current_14d_state = None          # 当前14维形状
        self.current_uncertainty = np.ones(14, dtype=np.float32) * 0.01
        self.force_estimation = True
        self.last_estimation_result = None

        # 阶段管理（模仿分层方法的 ApproachPhase -> RL 直接控制）
        self.approach_speed = -0.2
        self.contact_established = False
        self.rl_control_enabled = False
        self.current_rl_action = np.zeros(3)

        # 加载 ShapePredictor 模型（用于不确定性预测）
        self.predictor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(predictor_model_path):
            self._load_predictor(predictor_model_path)
        else:
            print(f"[警告] ShapePredictor 模型文件不存在: {predictor_model_path}，不确定性将使用默认值")

        # 历史记录
        if not hasattr(self, 'control_history'):
            self.control_history = []
        self.print_counter = 0

    # -------------------- 模型加载与不确定性预测 --------------------
    def _load_predictor(self, model_path):
        """加载预训练的 ShapePredictor 模型"""
        try:
            from model_log_var.shape_predictor import ShapePredictor
            self.predictor = ShapePredictor(input_dim=17, output_dim=14)
            state_dict = torch.load(model_path, map_location=self.device)
            self.predictor.load_state_dict(state_dict)
            self.predictor.eval().to(self.device)
            print(f"[DirectMotionController] ShapePredictor 模型加载成功: {model_path}")
        except Exception as e:
            print(f"[DirectMotionController] 加载 ShapePredictor 失败: {e}")
            self.predictor = None

    def predict_next_state_and_uncertainty(self, current_state_14d, control_3d):
        """
        使用 ShapePredictor 预测下一状态变化和不确定性。
        返回 (delta_state, uncertainty) 均为 (14,) 的 numpy 数组。
        """
        if self.predictor is None:
            return np.zeros(14, dtype=np.float32), np.ones(14, dtype=np.float32) * 0.01
        with torch.no_grad():
            state_t = torch.from_numpy(current_state_14d).float().unsqueeze(0).to(self.device)
            control_t = torch.from_numpy(control_3d).float().unsqueeze(0).to(self.device)
            delta, log_var = self.predictor(state_t, control_t)
            delta_np = delta.squeeze(0).cpu().numpy().astype(np.float32)
            uncertainty_np = torch.exp(log_var).squeeze(0).cpu().numpy().astype(np.float32)
            uncertainty_np = np.clip(uncertainty_np, 1e-4, 0.1)
            return delta_np, uncertainty_np

    # -------------------- 阶段管理与速度计算 --------------------
    def get_system_state(self, step, time, force_update=False):
        """获取系统状态（复用父类）"""
        return super().get_system_state(step, time, force_update)

    def apply_velocity(self, vel_array):
        """应用速度（复用父类）"""
        super().apply_velocity(vel_array)

    def compute_control_velocity(self, current_state):
        """
        根据当前阶段计算控制速度。
        阶段1：未接触 -> 向下运动 (0,0,approach_speed)
        阶段2：接触后 -> 使用 current_rl_action 作为速度指令
        """
        contact_info = current_state.get('contact', {})
        contact_detected = contact_info.get('contact_detected', False)

        if not self.contact_established and contact_detected:
            self.contact_established = True
            self.rl_control_enabled = True
            print("\n=== 接触检测，切换到RL直接控制模式 ===")

        if not self.contact_established:
            vel = np.array([0.0, 0.0, self.approach_speed])
        else:
            if self.rl_control_enabled:
                vel = self.current_rl_action.copy()
            else:
                vel = np.zeros(3)

        vel_array = np.zeros((self.initial_particles.shape[0], 3))
        vel_array[:, :] = vel
        return vel_array

    def _perform_superquadric_estimation(self, current_state, ems_params=None):
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

        # 获取11维参数
        param_dict, param_vector = MotionControllerUtils.get_standard_11d_params(sq)

        # 手动计算几何特征，确保返回字典
        from superquadric_estimator.metric import calculate_geometric_features as sq_metric
        features_14d, geom_features = sq_metric(sq, surface_points)

        # 确保 geom_features 是字典且包含所需字段
        if not isinstance(geom_features, dict):
            geom_features = {}
        required_keys = ['volume', 'elongation', 'smoothness']
        for key in required_keys:
            if key not in geom_features:
                if key == 'volume':
                    geom_features[key] = np.prod(sq.scale) * 8
                elif key == 'elongation':
                    geom_features[key] = 1.0
                else:
                    geom_features[key] = 0.5

        return {
            'sq': sq,
            'param_dict': param_dict,
            'geometric_features': geom_features,
            'surface_points': surface_points,
            'contact_info': contact_info,
            'step': current_state['step'],
            'time': current_state['time']
        }

    def estimate_and_save_superquadric(self, current_state):
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

        # 计算14D特征向量
        self.current_14d_state = np.array(
            MotionControllerUtils.get_14d_feature_vector(
                estimation_result['param_dict'],
                estimation_result['geometric_features']
            ),
            dtype=np.float32
        )
        if self.current_14d_state.shape[0] != 14:
            self.current_14d_state = np.zeros(14, dtype=np.float32)

        self.last_estimation_step = current_state['step']

        # 保存估计数据
        self._save_estimation_data(estimation_result, current_state)

        return {
            'sq': estimation_result['sq'],
            'param_dict': estimation_result['param_dict'],
            'geometric_features': estimation_result['geometric_features'],
            'feature_14d': self.current_14d_state
        }

    def _save_estimation_data(self, estimation_result, current_state):
        """
        保存估计数据
        """
        point_cloud_filename = self.save_point_cloud(
            estimation_result['surface_points'],
            estimation_result['step'],
            estimation_result['time']
        )
        feature_14d = MotionControllerUtils.get_14d_feature_vector(
            estimation_result['param_dict'],
            estimation_result['geometric_features']
        )
        estimation_record = {
            'step': estimation_result['step'],
            'time': estimation_result['time'],
            'contact_info': estimation_result['contact_info'],
            'parameters_11d': estimation_result['param_dict'],
            'geometric_features': estimation_result['geometric_features'],
            'feature_14d': feature_14d,
            'point_cloud_file': point_cloud_filename
        }
        self.save_superquadric_params(estimation_record, estimation_result['step'], estimation_result['time'])
        self.superquadric_params_history.append(estimation_record)

    # -------------------- 重置 --------------------
    def reset(self):
        """重置控制器状态（包括阶段标志）"""
        self.current_14d_state = None
        self.current_uncertainty = np.ones(14, dtype=np.float32) * 0.01
        self.last_estimation_step = -1
        self.force_estimation = True
        self.last_estimation_result = None
        self.contact_established = False
        self.rl_control_enabled = False
        self.current_rl_action = np.zeros(3)
        self.superquadric_params_history.clear()
        self.control_history.clear()
        print("[DirectMotionController] Reset")