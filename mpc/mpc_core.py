import numpy as np
import torch


class DeformationMPC:
    def __init__(self, dynamics_model, horizon=5, lr=0.05, iterations=100, state_scale=1.0):
        """
        初始化MPC控制器
        Args:
            dynamics_model: 动力学模型 (在原始尺度上训练)
            horizon: 预测时域步数
            lr: 梯度下降学习率
            iterations: 优化迭代次数
            state_scale: 状态尺度放大因子，用于解决小数值梯度消失问题
        """
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.lr = lr
        self.iterations = iterations
        self.state_dim = 14
        self.control_dim = 3

        # 状态放大因子
        self.state_scale = state_scale

        # 目标状态 - 两种模式
        self.relative_mode = True  # 默认为绝对模式
        self.target_state = None  # 原始尺度目标（绝对模式）
        self.target_state_scaled = None  # 放大尺度目标（绝对模式）
        self.target_numpy = None  # numpy格式目标（绝对模式）
        self.target_ratios = np.array([1.0, 1.0, 1.0])  # 目标比例（相对模式）

        # 控制约束
        self.control_bounds = {
            'lower': torch.tensor([-0.3, -0.3, -0.2], dtype=torch.float32),
            'upper': torch.tensor([0.3, 0.3, 0.2], dtype=torch.float32)
        }

        # 权重矩阵
        self.Q = torch.from_numpy(self._create_weight_matrix()).float()
        self.R = 0.001 * torch.eye(self.control_dim)
        self.optimization_history = []
        self._cost_components_cache = None

        print(f"[MPC] 初始化完成: 状态放大{state_scale:.0f}倍, 时域={horizon}, 迭代={iterations}")

    def _create_weight_matrix(self):
        """创建状态权重矩阵"""
        Q = np.eye(self.state_dim)
        # 尺度参数权重
        Q[0:3, 0:3] = 100.0 * np.eye(3)
        # 形状参数权重
        Q[3:5, 3:5] = 10.0 * np.eye(2)
        # 位姿参数权重
        Q[5:8, 5:8] = 0.000 * np.eye(3)  # 位置
        Q[8:11, 8:11] = 0.000 * np.eye(3)  # 旋转
        # 几何特征权重
        Q[11:14, 11:14] = np.diag([0.0,  # 体积
                                0.0,  # 细长性
                                0])  # 平滑度
        return Q

    def set_reference(self, target_state, relative_mode=True):
        """设置目标状态（支持相对模式和绝对模式）"""
        self.relative_mode = relative_mode

        if relative_mode:
            # 相对模式：只关心比例，不关心绝对数值
            self.target_ratios = np.array([1.0, 1.0, 1.0])  # a1:a2:a3=1:1:1（球体）
            print(f"[MPC] 设置为相对模式：目标比例 {self.target_ratios}")
            # 清空绝对目标，避免误用
            self.target_state = None
            self.target_state_scaled = None
            self.target_numpy = None
        else:
            # 绝对模式：保持原有逻辑
            self.target_numpy = np.array(target_state, dtype=np.float32)
            self.target_state = torch.tensor(target_state, dtype=torch.float32).unsqueeze(0)
            self.target_state_scaled = self.target_state * self.state_scale
            print(f"[MPC] 设置为绝对模式：目标尺度 {target_state[:3]}")

        return True

    def _scale_state(self, state):
        """将状态转换到放大空间"""
        return state * self.state_scale

    def _unscale_state(self, state_scaled):
        """将状态从放大空间转换回原始空间"""
        return state_scaled / self.state_scale

    def _predict_state_trajectory(self, initial_state, controls):
        """
        预测状态轨迹
        """
        device = controls.device
        if isinstance(initial_state, np.ndarray):
            current_state = torch.from_numpy(initial_state).float().unsqueeze(0).to(device)
        else:
            current_state = initial_state.float().unsqueeze(0).to(device)

        current_state_scaled = self._scale_state(current_state)
        states_scaled = [current_state_scaled]

        for k in range(self.horizon):
            u_t = controls[:, k, :]

            try:
                # 转回原始尺度供模型使用
                current_state_original = self._unscale_state(current_state_scaled)
                delta_state, _ = self.dynamics_model(current_state_original, u_t)
                # 检查异常值
                if torch.any(torch.isnan(delta_state)) or torch.any(torch.isinf(delta_state)):
                    delta_state = torch.zeros_like(delta_state)

                # 转换到放大空间
                delta_state_scaled = self._scale_state(delta_state)

                # 在放大空间中更新状态
                current_state_scaled = current_state_scaled + delta_state_scaled
                states_scaled.append(current_state_scaled)

            except Exception as e:
                print(f"[警告] 第{k}步预测失败: {e}")
                states_scaled.append(states_scaled[-1].clone())

        return states_scaled

    def _compute_relative_scale_cost(self, states_scaled):
        """使用相对误差的尺度代价"""
        device = states_scaled[0].device
        scale_cost = torch.tensor(0.0, device=device)

        for k in range(1, len(states_scaled)):
            current_state = self._unscale_state(states_scaled[k])
            current_scales = current_state[:, :3]

            # 1. 相对比例误差（更稳定）
            scale_mean = torch.mean(current_scales, dim=1, keepdim=True)
            relative_error = (current_scales - scale_mean) / (scale_mean + 1e-6)

            # 2. 使用Huber损失（对异常值更鲁棒）
            huber_loss = torch.where(
                torch.abs(relative_error) < 1.0,
                0.5 * relative_error ** 2,
                torch.abs(relative_error) - 0.5
            )

            scale_cost += torch.sum(huber_loss) * 10.0  # 小权重即可

        return scale_cost

    def _compute_relative_shape_cost(self, states_scaled):
        """计算相对模式下的形状代价（ε1, ε2接近1）"""
        device = states_scaled[0].device
        shape_cost = torch.tensor(0.0, device=device)

        for k in range(1, len(states_scaled)):
            current_state = self._unscale_state(states_scaled[k])
            current_shapes = current_state[:, 3:5]  # ε1, ε2

            # 目标形状参数：对于球体，ε1=1, ε2=1
            shape_error = current_shapes - 1.0
            shape_cost += torch.sum(shape_error ** 2) * 20.0

        return shape_cost

    def _compute_absolute_cost(self, states_scaled):
        """计算绝对模式下的代价"""
        device = states_scaled[0].device

        # 获取放大目标状态
        target_scaled = self.target_state_scaled.to(device)

        # 3. 计算各代价分量
        state_cost_total = torch.tensor(0.0, device=device)
        scale_cost_total = torch.tensor(0.0, device=device)
        shape_cost_total = torch.tensor(0.0, device=device)

        for k in range(1, self.horizon + 1):
            # 状态误差（放大空间）
            state_error_scaled = states_scaled[k] - target_scaled

            # 分离尺度误差，计算尺度代价
            scale_error_scaled = state_error_scaled[:, :3]  # 前3维是尺度参数
            Q_scale = self.Q[:3, :3].to(device)
            scale_cost = torch.mean(
                torch.sum(scale_error_scaled * torch.matmul(scale_error_scaled, Q_scale), dim=1)
            )
            scale_cost_total += scale_cost

            # 形状代价
            shape_error_scaled = state_error_scaled[:, 3:5]  # 只取形状参数
            Q_shape = self.Q[3:5, 3:5].to(device)
            shape_cost = torch.mean(
                torch.sum(shape_error_scaled * torch.matmul(shape_error_scaled, Q_shape), dim=1)
            )
            shape_cost_total += shape_cost

            # 其他参数代价（保持原始权重）
            other_error_scaled = state_error_scaled[:, 5:]
            Q_other = self.Q[5:, 5:].to(device)
            if other_error_scaled.numel() > 0:
                other_cost = torch.mean(
                    torch.sum(other_error_scaled * torch.matmul(other_error_scaled, Q_other), dim=1)
                )
                state_cost_total += other_cost

        return scale_cost_total, shape_cost_total, state_cost_total

    def _compute_cost_add(self, u_sequence, states_scaled):
        """
        改进的混合代价函数：基于已经预测的状态轨迹计算新增代价
        """
        device = u_sequence.device

        # 初始化新增代价
        additional_cost = torch.tensor(0.0, device=device)

        for k in range(1, len(states_scaled)):
            # 获取原始尺度状态
            curr_state = self._unscale_state(states_scaled[k])

            # 从状态中获取体积（第12维）
            curr_vol = curr_state[:, 11]  # 体积
            init_vol = self._unscale_state(states_scaled[0])[:, 11]  # 初始体积

            if torch.any(init_vol > 0):
                vol_change = (curr_vol - init_vol) / init_vol
                # 目标：轻微压缩（1-5%）
                target_compression = -0.03  # 3%压缩
                J_contact = torch.sum((vol_change - target_compression) ** 2) * 300.0
                additional_cost += J_contact

        return additional_cost

    def _compute_cost(self, u_sequence, initial_state, contact_info=None):
        """
        在放大空间中计算代价函数
        """
        device = u_sequence.device

        # 转换初始状态到放大空间
        if isinstance(initial_state, np.ndarray):
            initial_tensor = torch.from_numpy(initial_state).float().unsqueeze(0).to(device)
        else:
            initial_tensor = initial_state.float().unsqueeze(0).to(device)

        # 1. 预测放大空间中的状态轨迹
        states_scaled = self._predict_state_trajectory(initial_state, u_sequence)

        # 根据不同模式计算代价
        if self.relative_mode:
            # 相对模式：关注比例而非绝对数值
            scale_cost = self._compute_relative_scale_cost(states_scaled)
            shape_cost = self._compute_relative_shape_cost(states_scaled)
            state_cost_total = torch.tensor(0.0, device=device)  # 相对模式下忽略位姿等代价
        else:
            # 绝对模式：原有逻辑
            if self.target_state_scaled is None:
                raise ValueError("绝对模式下必须先设置目标状态")
            scale_cost, shape_cost, state_cost_total = self._compute_absolute_cost(states_scaled)

        # 4. 控制代价（原始尺度）
        control_cost = torch.sum(u_sequence ** 2) * 0.10

        # 5. 平滑代价
        smooth_cost = torch.tensor(0.0, device=device)
        if self.horizon > 1:
            control_diff = u_sequence[:, 1:, :] - u_sequence[:, :-1, :]
            smooth_cost = torch.sum(control_diff ** 2) * 0.50

        # 6. 新增代价
        additional_cost = self._compute_cost_add(u_sequence, states_scaled)

        # 7. 总代价
        total_cost = scale_cost + shape_cost + state_cost_total + control_cost + smooth_cost# + additional_cost

        # 8. 记录代价分量供外部打印
        if hasattr(self, '_print_cost_debug') and self._print_cost_debug:
            self._cost_components_cache = {
                'scale': scale_cost.item() if isinstance(scale_cost, torch.Tensor) else scale_cost,
                'shape': shape_cost.item() if isinstance(shape_cost, torch.Tensor) else shape_cost,
                'other': state_cost_total.item() if isinstance(state_cost_total, torch.Tensor) else state_cost_total,
                'control': control_cost.item() if isinstance(control_cost, torch.Tensor) else control_cost,
                'smooth': smooth_cost.item() if isinstance(smooth_cost, torch.Tensor) else smooth_cost,
                'additional': additional_cost.item() if isinstance(additional_cost, torch.Tensor) else additional_cost,
                'mode': 'relative' if self.relative_mode else 'absolute'
            }

        return total_cost

    def _contact_aware_penalty(self, states_scaled, contact_info):
        """接触感知惩罚（在放大空间中）"""
        device = states_scaled[0].device
        penalty = torch.tensor(0.0, device=device)

        if not contact_info:
            return penalty

        min_distance = contact_info.get('min_distance', 0.0)
        if not isinstance(min_distance, torch.Tensor):
            min_distance = torch.tensor(min_distance, device=device)

        if min_distance < 0.02:
            # 将放大状态转换回原始尺度计算体积变化
            initial_volume = self._unscale_state(states_scaled[0])[:, 11]  # 第12维是体积
            final_volume = self._unscale_state(states_scaled[-1])[:, 11]
            volume_change = torch.abs(final_volume - initial_volume)

            distance_penalty = 0.5 * torch.relu((0.02 - min_distance) / 0.02) ** 2
            contact_penalty = distance_penalty * volume_change * 10.0

            if not torch.isnan(contact_penalty):
                penalty = penalty + contact_penalty

        return penalty

    def solve(self, current_state, contact_info=None):
        if self.relative_mode:
            # 相对模式：不需要绝对目标状态
            print(f"[MPC] 相对模式求解")
        elif self.target_state_scaled is None:
            raise ValueError("绝对模式下必须先设置目标状态")

        device = next(self.dynamics_model.parameters()).device

        # 1. 准备数据
        if isinstance(current_state, np.ndarray):
            current_tensor = torch.from_numpy(current_state).float().unsqueeze(0).to(device)
        else:
            current_tensor = current_state.float().unsqueeze(0).to(device)

        current_scaled = self._scale_state(current_tensor)

        # 2. 计算目标方向（根据模式）
        if self.relative_mode:
            # 相对模式：计算当前尺度参数的比例误差
            current_scales = current_tensor[:, :3]
            scale_mean = torch.mean(current_scales, dim=1, keepdim=True)
            scale_std = torch.std(current_scales, dim=1)
            relative_std = scale_std / (scale_mean + 1e-8)

            print(f"[相对模式] 当前尺度: {current_scales[0].cpu().numpy()}")
            print(f"[相对模式] 平均值: {scale_mean.item():.4f}, 相对标准差: {relative_std.item():.4f}")

            # 形状参数误差
            current_shapes = current_tensor[:, 3:5]
            shape_error = current_shapes - 1.0  # 目标ε1=1, ε2=1
            shape_error_norm = torch.norm(shape_error).item()

            # 基于比例误差和形状误差的引导
            if relative_std > 0.05 or shape_error_norm > 0.1:
                # 误差较大，需要更强的控制
                if shape_error_norm > 0.1:
                    # 形状误差主导：沿形状误差方向
                    if shape_error_norm > 0.001:
                        # 将2D形状误差扩展到3D控制空间
                        shape_direction_2d = shape_error / shape_error_norm
                        shape_direction_3d = torch.zeros(1, 3, device=device)
                        shape_direction_3d[0, :2] = shape_direction_2d[0, :2]  # X, Y方向
                        # Z方向设为0，因为形状参数ε主要受平面变形影响
                        base_control = shape_direction_3d * 0.15
                    else:
                        base_control = torch.randn(1, self.control_dim, device=device) * 0.05
                else:
                    # 比例误差主导：随机探索以改善比例
                    base_control = torch.randn(1, self.control_dim, device=device) * 0.1
            else:
                # 误差较小，轻微调整
                base_control = torch.randn(1, self.control_dim, device=device) * 0.02

            scale_direction = None
            scale_direction_norm = 0.0
            target_scaled = None

        else:
            # 绝对模式：保持原有逻辑
            target_scaled = self.target_state_scaled.to(device)
            target_original = self.target_state.to(device)

            # 计算目标方向
            scale_direction_original = target_original[:, :3] - current_tensor[:, :3]
            scale_direction_norm = torch.norm(scale_direction_original)
            shape_error_original = target_original[:, 3:5] - current_tensor[:, 3:5]
            shape_error_norm = torch.norm(shape_error_original).item()

            if scale_direction_norm > 0.001:
                scale_direction = scale_direction_original / scale_direction_norm
                # 基于误差大小的基础控制量
                error_magnitude = min(0.3, scale_direction_norm.item() * 3.0)
                base_control = scale_direction * error_magnitude
            else:
                base_control = torch.zeros(1, self.control_dim, device=device)

            # 形状方向
            shape_direction_2d = torch.zeros_like(shape_error_original)
            if shape_error_norm > 0.001:
                shape_direction_2d = shape_error_original / shape_error_norm

            shape_direction_3d = torch.zeros(1, 3, device=device)
            shape_direction_3d[0, :2] = shape_direction_2d[0, :2]  # X, Y方向

        # 3. 初始化控制序列
        u_sequence = base_control.repeat(1, self.horizon, 1) + \
                     (torch.randn(1, self.horizon, self.control_dim, device=device) * 0.08)
        u_sequence.requires_grad_(True)

        # 4. 控制约束
        u_min = self.control_bounds['lower'].to(device)
        u_max = self.control_bounds['upper'].to(device)

        # 5. 优化器设置
        optimizer = torch.optim.Adam([u_sequence], lr=self.lr, betas=(0.9, 0.999))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=10, verbose=False, min_lr=1e-6
        )

        best_cost = float('inf')
        best_u_sequence = u_sequence.clone().detach()

        # 6. 启用诊断打印
        self._print_cost_debug = True
        self._print_predict_diagnosis = True

        # 7. 优化循环
        for i in range(self.iterations):
            optimizer.zero_grad()

            try:
                # 计算原始代价
                total_cost = self._compute_cost(u_sequence, current_state, contact_info)

                # 相对模式：添加比例对齐引导
                if self.relative_mode:
                    current_scales = current_tensor[:, :3]
                    scale_mean = torch.mean(current_scales, dim=1, keepdim=True)

                    # 鼓励控制使尺度参数向平均值靠拢
                    first_control = u_sequence[:, 0, :]
                    control_norm = torch.norm(first_control, dim=1)

                    if torch.any(control_norm > 1e-6):
                        # 计算当前尺度差异的方向
                        scale_diff = current_scales - scale_mean
                        scale_diff_norm = torch.norm(scale_diff, dim=1)

                        if scale_diff_norm > 0.001:
                            # 归一化差异方向
                            scale_diff_dir = scale_diff / scale_diff_norm.unsqueeze(1)

                            # 我们希望控制的方向与差异方向有一定的相关性
                            # 但不是简单的对齐，而是探索性调整
                            alignment = torch.sum(first_control * scale_diff_dir, dim=1) / control_norm
                            alignment_loss = -0.1 * torch.mean(alignment)  # 负号鼓励一定程度的探索
                            total_cost = total_cost + alignment_loss

                # 绝对模式：原有的对齐引导
                else:
                    # 原有的尺度对齐
                    if scale_direction_norm > 0.001:
                        first_control = u_sequence[:, 0, :]
                        control_alignment = torch.sum(first_control * scale_direction, dim=1)
                        control_norm = torch.norm(first_control, dim=1)

                        if torch.any(control_norm > 1e-6):
                            normalized_alignment = control_alignment / control_norm
                            alignment_loss = -0.2 * torch.mean(normalized_alignment)
                            total_cost = total_cost + alignment_loss

                if not total_cost.requires_grad:
                    raise RuntimeError("total_cost不包含梯度图！")

                total_cost.backward()

                # 梯度裁剪（更宽松）
                if u_sequence.grad is not None:
                    torch.nn.utils.clip_grad_norm_([u_sequence], max_norm=20.0)

                optimizer.step()
                scheduler.step(total_cost)

            except Exception as e:
                print(f"[错误] 优化步骤失败: {e}")
                if u_sequence.grad is not None:
                    u_sequence.grad.zero_()
                continue

            # 8. 约束投影
            with torch.no_grad():
                u_sequence.data.clamp_(u_min, u_max)

            # 9. 记录最佳解
            try:
                current_cost = total_cost.item()
            except:
                current_cost = float('inf')

            if current_cost < best_cost and not np.isnan(current_cost):
                best_cost = current_cost
                best_u_sequence = u_sequence.clone().detach()

            # 10. 定期打印进度
            if i % 10 == 0 or i == self.iterations - 1:
                current_control = u_sequence[0, 0, :].detach().cpu().numpy()
                control_norm = np.linalg.norm(current_control)
                current_lr = optimizer.param_groups[0]['lr']

                # 模式信息
                mode_str = "相对" if self.relative_mode else "绝对"

                # 代价分量信息
                cost_info = ""
                if hasattr(self, '_cost_components_cache'):
                    comp = self._cost_components_cache
                    cost_info = (f"尺度:{comp['scale']:.4f}, 形状:{comp['shape']:.4f}, "
                                 f"控制:{comp['control']:.4f}, 新增:{comp['additional']:.4f}")

                print(f"[{mode_str}MPC进度] 迭代{i:3d}: 总代价={current_cost:.6f}, "
                      f"控制范数={control_norm:.4f}, {cost_info}")

        # 11. 禁用诊断打印
        self._print_cost_debug = False
        self._print_predict_diagnosis = False

        # 12. 获取最终控制指令
        final_control = best_u_sequence[0, 0, :].cpu().detach().numpy()
        final_norm = np.linalg.norm(final_control)

        print(f"\n[MPC] 优化结束:")
        print(f"  模式: {'相对' if self.relative_mode else '绝对'}")
        print(f"  最优代价: {best_cost:.6f}")
        print(f"  最终控制: {final_control}, 范数: {final_norm:.4f}")

        # 相对模式的额外诊断信息
        if self.relative_mode:
            current_scales = current_tensor[:, :3].cpu().numpy()
            scale_mean = np.mean(current_scales)
            scale_std = np.std(current_scales)
            relative_std = scale_std / (scale_mean + 1e-8)

            print(f"  当前尺度: {current_scales[0]}")
            print(f"  平均值: {scale_mean:.4f}, 相对标准差: {relative_std:.4f}")

        # 13. 如果控制量太小，应用启发式控制
        if final_norm < 0.02 or best_cost == float('inf'):
            print("[警告] 控制量过小或优化失败，应用启发式控制")

            if self.relative_mode:
                # 相对模式的启发式控制
                current_scales = current_tensor[:, :3].cpu().numpy()
                scale_std = np.std(current_scales)

                if scale_std > 0.01:
                    # 比例差异较大，施加随机探索
                    final_control = np.random.randn(3) * 0.15
                    print("   基于比例误差的随机探索")
                else:
                    # 比例较好，轻微调整
                    final_control = np.random.randn(3) * 0.05
                    print("   轻微调整")
            else:
                # 绝对模式的启发式控制（原有逻辑）
                if scale_direction_norm > 0.001:
                    error_magnitude = min(0.3, scale_direction_norm.item() * 5.0)
                    direction_component = scale_direction[0].cpu().numpy() * error_magnitude * 0.8
                    random_component = np.random.randn(3) * 0.1 * error_magnitude
                    final_control = direction_component + random_component
                else:
                    final_control = np.random.randn(3) * 0.1

            # 应用约束
            final_control = np.clip(final_control,
                                    self.control_bounds['lower'].numpy(),
                                    self.control_bounds['upper'].numpy())
            final_norm = np.linalg.norm(final_control)
            print(f"  启发式控制: {final_control}, 范数: {final_norm:.4f}")

        # 14. 记录优化历史
        self.optimization_history.append({
            'cost': best_cost,
            'controls': best_u_sequence.squeeze(0).cpu().detach().numpy(),
            'success': best_cost < float('inf'),
            'final_control_norm': final_norm,
            'state_scale': self.state_scale,
            'mode': 'relative' if self.relative_mode else 'absolute'
        })

        return final_control

    def get_status(self):
        """获取MPC状态信息"""
        return {
            'horizon': self.horizon,
            'state_dim': self.state_dim,
            'control_dim': self.control_dim,
            'mode': 'relative' if self.relative_mode else 'absolute',
            'target_set': self.target_state_scaled is not None or self.relative_mode,
            'state_scale': self.state_scale,
            'history_length': len(self.optimization_history),
            'control_bounds': {
                'lower': self.control_bounds['lower'].numpy(),
                'upper': self.control_bounds['upper'].numpy()
            }
        }
