import torch
import numpy as np
from model_log_var.shape_predictor import ShapePredictor
from mpc.mpc_core import DeformationMPC


class MPCController:
    def __init__(self, model_path, device='gpu', horizon=10, lr=0.05, iterations=100):
        """
        Args:
            model_path: 动力学模型路径
            device: 运行设备
            horizon: 预测时域
            lr: 学习率
            iterations: 优化迭代次数
        """
        # 设备设置
        self.device = torch.device(
            "cuda" if (device == 'gpu' and torch.cuda.is_available()) else "cpu"
        )

        # 加载动力学模型
        self.dynamics_model = ShapePredictor()
        try:
            self.dynamics_model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.dynamics_model.eval().to(self.device)

            # 冻结模型参数
            for param in self.dynamics_model.parameters():
                param.requires_grad_(False)

            print(f"[控制器] 动力学模型加载成功: {model_path}")
            print(f"[控制器] 模型参数量: {sum(p.numel() for p in self.dynamics_model.parameters())}")

            # 验证模型
            self._validate_model()

        except Exception as e:
            print(f"[控制器错误] 模型加载失败: {e}")
            raise

        # 初始化重新设计的MPC
        self.mpc = DeformationMPC(
            dynamics_model=self.dynamics_model,
            horizon=horizon,
            lr=lr,
            iterations=iterations,
        )

        # 运行时状态
        self.current_state = None
        self.target_state = None  # 最终目标（14维）
        self.current_target = None  # 当前子目标（14维）
        self.last_control = np.zeros(3)
        self.step_count = 0

    def _validate_model(self):
        """验证动力学模型输出"""
        print("[验证] 测试动力学模型...")

        # 测试状态和控制
        test_state = torch.tensor(
            [0.08, 0.08, 0.08] + [0.0] * 11,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        test_control = torch.tensor(
            [0.05, 0.0, 0.0],
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            delta, uncertainty = self.dynamics_model(test_state, test_control)

            print(f"[验证] 输入状态: {test_state[0][:3].cpu().numpy()}")
            print(f"[验证] 输入控制: {test_control[0].cpu().numpy()}")
            print(f"[验证] 输出变化: {delta[0][:3].cpu().numpy()}")
            print(f"[验证] 变化范数: {torch.norm(delta[0][:3]).item():.6f}")

            # 检查是否符合训练数据统计
            expected_range = 0.05  # 训练数据显示变化范围约±0.06
            actual_norm = torch.norm(delta[0][:3]).item()

            if actual_norm < 1e-4:
                print("[警告] 模型预测变化过小")
            elif actual_norm > expected_range:
                print("[警告] 模型预测变化过大")
            else:
                print("[验证] 模型输出在合理范围内")

    def set_target(self, target_state_16d, relative_mode=True):
        """设置最终目标状态 - 仅用于非RL模式

        Args:
            target_state_16d: 16维最终目标状态
            relative_mode: True表示相对模式（只关注比例），False表示绝对模式
        """
        assert len(target_state_16d) == 16, f"最终目标必须是16维"

        self.relative_mode = relative_mode
        target_state_14d = self._convert_16d_to_14d(target_state_16d)

        if relative_mode:
            self.target_ratios = np.array([1.0, 1.0, 1.0])
            print(f"[控制器] 设置相对最终目标：尺度比例 {self.target_ratios}（球体）")
            self.mpc.set_reference(target_state_14d)
        else:
            self.mpc.set_reference(target_state_14d)
            print(f"[控制器] 设置绝对最终目标：尺度参数 {target_state_14d[:3]}")

        # 存储最终目标
        self.target_state = target_state_14d.copy()
        self.current_target = self.target_state.copy()  # 默认当前目标为最终目标

    def set_sub_target(self, sub_target_14d: np.ndarray) -> None:
        """设置子目标 - 由RL调用，14维"""
        assert len(sub_target_14d) == 14, f"RL子目标必须是14维"

        self.current_target = sub_target_14d.copy()
        self.mpc.set_reference(self.current_target)
        print(f"[控制器] RL子目标已设置，前3维: {self.current_target[:3]}")

    def get_current_target(self) -> np.ndarray:
        """获取当前跟踪的目标（14维）"""
        if self.current_target is not None:
            return self.current_target
        elif self.target_state is not None:
            return self.target_state
        else:
            return None

    def control(self, current_state, contact_info=None):
        """计算控制指令 - 输入为16维状态，转换为14维"""
        # 输入验证
        measured_state_16d = current_state
        if isinstance(measured_state_16d, list):
            measured_state_16d = np.array(measured_state_16d, dtype=np.float32)
        assert len(measured_state_16d) == 16, f"测量状态必须是16维"

        if np.any(np.isnan(measured_state_16d)):
            print("[错误] 测量状态包含NaN")
            return np.zeros(3)

        # 将16维状态转换为14维
        measured_state_14d = self._convert_16d_to_14d(measured_state_16d)

        # 更新当前状态
        self.current_state = measured_state_14d.copy()
        self.step_count += 1
        print(f"\n[控制步骤 {self.step_count}]")
        print(f"  状态尺度: {self.current_state[:3]}")

        # 获取当前目标
        current_target = self.get_current_target()
        if current_target is not None:
            scale_error = current_target[:3] - self.current_state[:3]
            print(f"  尺度误差: {scale_error}")
            print(f"  误差范数: {np.linalg.norm(scale_error):.6f}")

        # 确定实际使用的目标（优先级：子目标 > 相对模式 > 最终目标）
        if self.current_target is not None:
            # 使用子目标
            pass  # MPC的目标已在set_sub_target中设置
        elif hasattr(self, 'relative_mode') and self.relative_mode:
            # 相对模式：动态计算相对目标
            current_scales = measured_state_14d[:3]
            scale_mean = np.mean(current_scales)
            relative_target = measured_state_14d.copy()
            relative_target[:3] = scale_mean * self.target_ratios
            self.mpc.set_reference(relative_target, relative_mode=True)
        elif self.target_state is not None:
            # 使用最终目标
            self.mpc.set_reference(self.target_state)

        # MPC优化
        try:
            control = self.mpc.solve(
                current_state=self.current_state,
                contact_info=contact_info
            )
            control_norm = np.linalg.norm(control)
            print(f"  MPC控制: {control}")
            print(f"  控制范数: {control_norm:.6f}")

            # 检查控制是否合理
            if control_norm < 0.005:
                print("[警告] 控制量过小，可能无法产生明显变化")

            self.last_control = control.copy()

        except Exception as e:
            print(f"[控制错误] MPC求解失败: {e}")
            control = np.zeros(3)
            self.last_control = control

        return control

    def _convert_16d_to_14d(self, state_16d):
        """将16维状态转换为14维状态"""
        return np.array([
            state_16d[0], state_16d[1], state_16d[2],  # scale (3)
            state_16d[3], state_16d[4],  # shape (2)
            state_16d[5], state_16d[6], state_16d[7],  # translation (3)
            state_16d[8], state_16d[9], state_16d[10],  # rotation (3)
            state_16d[11],  # volume (1)
            state_16d[12],  # elongation (1)
            state_16d[14]  # smoothness (1) - 总共14维
        ])

    def get_status(self):
        """获取控制器状态"""
        mpc_status = self.mpc.get_status()

        return {
            'step': self.step_count,
            'current_state': self.current_state,
            'target_state': self.target_state,
            'current_target': self.current_target,
            'last_control': self.last_control,
            'mpc_status': mpc_status,
            'last_cost': self.mpc.optimization_history[-1]['cost'] if self.mpc.optimization_history else None
        }

    def reset(self):
        """重置控制器"""
        self.current_state = None
        self.target_state = None
        self.current_target = None
        self.last_control = np.zeros(3)
        self.step_count = 0
        self.mpc.optimization_history.clear()
        print("[控制器] 已重置")