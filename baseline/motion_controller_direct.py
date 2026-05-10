# baseline/motion_controller_direct.py
import numpy as np
import torch
import os
from common.base.motion import BaseMotionController, MotionControllerUtils

class DirectMotionController(BaseMotionController):
    def __init__(self, scene, sensor_cube, elastoplastic_obj, initial_particles,
                 output_dir="direct_rl_data", estimation_interval=25,
                 predictor_model_path="shape_predictor.pth"):
        super().__init__(scene, sensor_cube, elastoplastic_obj, initial_particles, output_dir)
        self.estimation_interval = estimation_interval
        self.last_estimation_step = -1
        self.current_14d_state = None          # 当前估计的14维形状
        self.current_uncertainty = np.ones(14, dtype=np.float32) * 0.01  # 不确定性
        self.force_estimation = True
        self.last_estimation_result = None

        # 加载 ShapePredictor 模型
        self.predictor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(predictor_model_path):
            self._load_predictor(predictor_model_path)
        else:
            print(f"[警告] ShapePredictor 模型文件不存在: {predictor_model_path}，将使用默认不确定性")

        # 控制历史记录
        if not hasattr(self, 'control_history'):
            self.control_history = []
        self.print_counter = 0

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
        使用 ShapePredictor 模型预测下一状态及其不确定性。
        输入：
            current_state_14d: np.ndarray, shape (14,)
            control_3d: np.ndarray, shape (3,)
        输出：
            pred_delta: np.ndarray, shape (14,)  预测的状态变化量
            pred_uncertainty: np.ndarray, shape (14,)  预测的不确定性（对数方差）
        """
        if self.predictor is None:
            # 降级方案：返回零变化和默认不确定性
            return np.zeros(14, dtype=np.float32), np.ones(14, dtype=np.float32) * 0.01

        with torch.no_grad():
            state_t = torch.from_numpy(current_state_14d).float().unsqueeze(0).to(self.device)
            control_t = torch.from_numpy(control_3d).float().unsqueeze(0).to(self.device)
            delta, log_var = self.predictor(state_t, control_t)
            delta_np = delta.squeeze(0).cpu().numpy().astype(np.float32)
            uncertainty_np = torch.exp(log_var).squeeze(0).cpu().numpy().astype(np.float32)
            # 限制不确定性范围，避免过大或过小
            uncertainty_np = np.clip(uncertainty_np, 1e-4, 0.1)
            return delta_np, uncertainty_np

    def get_system_state(self, step, time, force_update=False):
        """获取当前系统状态（复用父类）"""
        return super().get_system_state(step, time, force_update)

    def apply_velocity(self, vel_array):
        """应用速度到传感器立方体（复用父类）"""
        super().apply_velocity(vel_array)

    def estimate_and_save_superquadric(self, current_state):
        """
        执行超二次曲面估计并更新 current_14d_state 和 current_uncertainty。
        返回包含估计结果的字典，如果估计失败则返回 None。
        """
        contact_info = current_state.get('contact', {})
        if not contact_info.get('contact_detected', False):
            return None

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

        from common.base.motion import MotionControllerUtils
        self.current_14d_state = MotionControllerUtils.get_14d_feature_vector(
            estimation_result['param_dict'],
            estimation_result['geometric_features']
        ).astype(np.float32)

        self.current_uncertainty = np.ones(14, dtype=np.float32) * 0.01

        self.last_estimation_step = current_state['step']
        self._save_estimation_data(estimation_result, current_state)

        estimation_result['feature_14d'] = self.current_14d_state
        estimation_result['uncertainty'] = self.current_uncertainty
        self.last_estimation_result = estimation_result

        return {
            'sq': estimation_result['sq'],
            'param_dict': estimation_result['param_dict'],
            'geometric_features': estimation_result['geometric_features'],
            'feature_14d': self.current_14d_state,
            'uncertainty': self.current_uncertainty
        }

    def _save_estimation_data(self, estimation_result, current_state):
        point_cloud_filename = self.save_point_cloud(
            estimation_result['surface_points'],
            estimation_result['step'],
            estimation_result['time']
        )
        feature_14d = self.current_14d_state
        estimation_record = {
            'step': estimation_result['step'],
            'time': estimation_result['time'],
            'contact_info': estimation_result['contact_info'],
            'parameters_11d': estimation_result['param_dict'],
            'geometric_features': estimation_result['geometric_features'],
            'feature_14d': feature_14d,
            'uncertainty': self.current_uncertainty.tolist(),
            'point_cloud_file': point_cloud_filename
        }
        self.save_superquadric_params(estimation_record, estimation_result['step'], estimation_result['time'])
        self.superquadric_params_history.append(estimation_record)

    def reset(self):
        self.current_14d_state = None
        self.current_uncertainty = np.ones(14, dtype=np.float32) * 0.01
        self.last_estimation_step = -1
        self.force_estimation = True
        self.last_estimation_result = None
        self.superquadric_params_history.clear()
        self.control_history.clear()
        print("[DirectMotionController] Reset")