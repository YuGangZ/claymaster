import numpy as np
import json
from scipy.spatial import ConvexHull


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


class MPMMonitor:
    def __init__(self, entity, name, sampling_ratio=0.1, record_full_state=False):
        self.entity = entity
        self.name = name
        self.sampling_ratio = sampling_ratio
        self.record_full_state = record_full_state
        self.surface_particles_indices = None
        self.history = []
        self.initial_state = None

    def initialize(self):
        """初始化监控器，记录初始状态"""
        initial_data = self.get_current_state()
        self.initial_state = {
            'positions': initial_data['positions'].copy(),
            'center': initial_data['center'].copy(),
            'volume': self.estimate_volume(initial_data['positions'])
        }

        # 记录初始的MPM状态
        if 'mpm_state' in initial_data:
            self.initial_state['mpm_state'] = initial_data['mpm_state']

        return initial_data

    def get_current_state(self):
        """获取当前状态 - 包含MPM状态信息"""
        try:
            # 获取基本粒子位置
            particles = self.entity.get_particles()
            if hasattr(particles, 'cpu'):
                particles = particles.cpu().numpy()
            particles = particles.reshape(-1, 3)

            # 尝试获取完整的MPM状态
            mpm_state = None
            if self.record_full_state:
                try:
                    state = self.entity.get_state()
                    mpm_state = {}

                    # 提取各种MPM状态量
                    if hasattr(state, 'pos') and state.pos is not None:
                        mpm_state['pos'] = state.pos[0].cpu().numpy() if hasattr(state.pos, 'cpu') else state.pos[0]

                    if hasattr(state, 'vel') and state.vel is not None:
                        mpm_state['vel'] = state.vel[0].cpu().numpy() if hasattr(state.vel, 'cpu') else state.vel[0]

                    if hasattr(state, 'F') and state.F is not None:
                        F_tensor = state.F[0].cpu().numpy() if hasattr(state.F, 'cpu') else state.F[0]
                        mpm_state['F'] = F_tensor.reshape(-1, 3, 3)  # 变形梯度

                        # 计算基于变形梯度的力学量
                        mpm_state.update(self._calculate_mechanical_metrics(F_tensor))

                    if hasattr(state, 'C') and state.C is not None:
                        C_tensor = state.C[0].cpu().numpy() if hasattr(state.C, 'cpu') else state.C[0]
                        mpm_state['C'] = C_tensor.reshape(-1, 3, 3)  # 仿射矩阵

                    if hasattr(state, 'Jp') and state.Jp is not None:
                        Jp_tensor = state.Jp[0].cpu().numpy() if hasattr(state.Jp, 'cpu') else state.Jp[0]
                        mpm_state['Jp'] = Jp_tensor.flatten()  # 体积比

                except Exception as e:
                    print(f"获取{self.name}完整MPM状态失败: {e}")

            # 识别表面粒子
            self.surface_particles_indices = self.identify_surface_particles(particles)
            surface_particles = particles[self.surface_particles_indices]

            state_info = {
                'positions': particles,
                'surface_positions': surface_particles,
                'count': len(particles),
                'surface_count': len(surface_particles),
                'center': np.mean(particles, axis=0),
                'bounding_box': {
                    'min': np.min(particles, axis=0),
                    'max': np.max(particles, axis=0)
                }
            }

            # 添加MPM状态信息
            if mpm_state is not None:
                state_info['mpm_state'] = mpm_state

            return state_info
        except Exception as e:
            print(f"获取{self.name}状态失败: {e}")
            return None

    def _calculate_mechanical_metrics(self, F_tensor):
        """基于变形梯度计算力学量"""
        F_reshaped = F_tensor.reshape(-1, 3, 3)
        metrics = {}

        try:
            # 计算每个粒子的变形梯度行列式 (体积变化)
            det_F = np.linalg.det(F_reshaped)
            metrics['det_F'] = det_F
            metrics['volume_change_ratio'] = det_F - 1.0  # J - 1

            # 计算左右柯西-格林张量
            C = np.matmul(F_reshaped.transpose(0, 2, 1), F_reshaped)  # F^T F
            metrics['C_tensor'] = C

            # 计算主拉伸
            eigenvalues = np.linalg.eigvals(C)
            principal_stretches = np.sqrt(eigenvalues)
            metrics['principal_stretches'] = principal_stretches

            # 计算等效应变 (工程应变)
            strain_engineering = 0.5 * (C - np.eye(3))
            metrics['strain_engineering'] = strain_engineering

            # 计算冯米塞斯等效应变
            deviatoric_strain = strain_engineering - (1 / 3) * np.trace(strain_engineering, axis1=1, axis2=2)[:, None,
                                                            None] * np.eye(3)
            von_mises_strain = np.sqrt(1.5 * np.sum(deviatoric_strain ** 2, axis=(1, 2)))
            metrics['von_mises_strain'] = von_mises_strain

        except Exception as e:
            print(f"计算力学量失败: {e}")

        return metrics

    def identify_surface_particles(self, particles):
        """识别表面粒子"""
        try:
            if len(particles) < 4:
                return np.arange(len(particles))

            # 使用凸包算法识别表面粒子
            hull = ConvexHull(particles)
            surface_indices = hull.vertices

            # 如果表面粒子太多，进行采样
            if len(surface_indices) > len(particles) * self.sampling_ratio:
                surface_indices = np.random.choice(
                    surface_indices,
                    size=int(len(particles) * self.sampling_ratio),
                    replace=False
                )

            return surface_indices
        except Exception as e:
            print(f"识别{self.name}表面粒子失败: {e}")
            return np.arange(len(particles))

    def calculate_deformation_metrics(self, current_state):
        """计算形变指标 - 增强版"""
        if self.initial_state is None:
            return {}

        current_center = current_state['center']
        initial_center = self.initial_state['center']

        # 计算位移
        center_displacement = np.linalg.norm(current_center - initial_center)

        # 计算体积变化
        current_volume = self.estimate_volume(current_state['positions'])
        initial_volume = self.initial_state['volume']
        volume_change_ratio = (current_volume - initial_volume) / initial_volume if initial_volume > 0 else 0

        # 计算最大粒子位移
        particle_displacements = np.linalg.norm(
            current_state['positions'] - self.initial_state['positions'],
            axis=1
        )
        max_particle_displacement = np.max(particle_displacements) if len(particle_displacements) > 0 else 0

        # 基础形变指标
        deformation_metrics = {
            'center_displacement': center_displacement,
            'volume_change_ratio': volume_change_ratio,
            'max_particle_displacement': max_particle_displacement,
            'avg_particle_displacement': np.mean(particle_displacements) if len(particle_displacements) > 0 else 0
        }

        # 添加基于MPM状态的力学指标
        if 'mpm_state' in current_state and 'mpm_state' in self.initial_state:
            mpm_metrics = self._calculate_mpm_based_metrics(current_state)
            deformation_metrics.update(mpm_metrics)

        return deformation_metrics

    def _calculate_mpm_based_metrics(self, current_state):
        """基于MPM状态计算力学指标"""
        metrics = {}
        try:
            mpm_state = current_state['mpm_state']

            # 基于变形梯度的体积变化
            if 'volume_change_ratio' in mpm_state:
                metrics['mpm_volume_change'] = np.mean(mpm_state['volume_change_ratio'])
                metrics['mpm_max_volume_change'] = np.max(mpm_state['volume_change_ratio'])

            # 基于冯米塞斯应变
            if 'von_mises_strain' in mpm_state:
                metrics['von_mises_strain_mean'] = np.mean(mpm_state['von_mises_strain'])
                metrics['von_mises_strain_max'] = np.max(mpm_state['von_mises_strain'])

            # 基于主拉伸
            if 'principal_stretches' in mpm_state:
                principal_stretches = mpm_state['principal_stretches']
                metrics['max_principal_stretch'] = np.max(principal_stretches)
                metrics['min_principal_stretch'] = np.min(principal_stretches)

        except Exception as e:
            print(f"计算MPM基础指标失败: {e}")

        return metrics

    def estimate_volume(self, particles):
        """估计粒子云体积"""
        try:
            if len(particles) < 4:
                return 0
            hull = ConvexHull(particles)
            return hull.volume
        except:
            # 如果凸包失败，使用边界框估计
            bbox_min = np.min(particles, axis=0)
            bbox_max = np.max(particles, axis=0)
            return np.prod(bbox_max - bbox_min)


class ElasticBodyMonitor(MPMMonitor):
    """弹性体监控器"""

    def __init__(self, entity, name="弹性体", sampling_ratio=0.1, record_full_state=True):
        super().__init__(entity, name, sampling_ratio, record_full_state)
        self.stress_history = []

    def _calculate_stress_metrics(self, F_tensor, material_params=None):
        """计算应力指标 - 基于Neo-Hookean模型"""
        if material_params is None:
            material_params = {'E': 1e4, 'nu': 0.49}  # 默认参数

        try:
            F_reshaped = F_tensor.reshape(-1, 3, 3)
            mu = material_params['E'] / (2 * (1 + material_params['nu']))
            k = material_params['E'] / (3 * (1 - 2 * material_params['nu']))

            stresses = []
            for F in F_reshaped:
                J = np.linalg.det(F)
                # Neo-Hookean应力
                stress = mu * (F @ F.T - np.eye(3)) + (k * (J - 1)) * np.eye(3)
                stresses.append(stress)

            stresses = np.array(stresses)

            # 计算冯米塞斯等效应力
            deviatoric_stress = stresses - (1 / 3) * np.trace(stresses, axis1=1, axis2=2)[:, None, None] * np.eye(3)
            von_mises_stress = np.sqrt(1.5 * np.sum(deviatoric_stress ** 2, axis=(1, 2)))
            metrics = {
                'cauchy_stress': stresses,
                'von_mises_stress': von_mises_stress,
                'max_von_mises': np.max(von_mises_stress),
                'mean_von_mises': np.mean(von_mises_stress)
            }

            # 添加应力统计信息
            if len(von_mises_stress) > 0:
                metrics.update({
                    'stress_percentile_25': np.percentile(von_mises_stress, 25),
                    'stress_percentile_50': np.percentile(von_mises_stress, 50),
                    'stress_percentile_75': np.percentile(von_mises_stress, 75),
                    'stress_variance': np.var(von_mises_stress)
                })

            return metrics
        except Exception as e:
            print(f"计算应力失败: {e}")
            return {}


class ElastoPlasticBodyMonitor(MPMMonitor):
    """弹塑性体监控器"""

    def __init__(self, entity, name="弹塑性体", sampling_ratio=1.0, record_full_state=True):
        super().__init__(entity, name, sampling_ratio, record_full_state)
        self.plastic_deformation_history = []

    def _calculate_plastic_metrics(self, current_state):
        """计算塑性变形指标"""
        metrics = {
            'permanent_deformation_estimate': self.estimate_permanent_deformation(current_state)
        }

        # 基于MPM状态计算塑性指标
        if 'mpm_state' in current_state:
            mpm_state = current_state['mpm_state']

            # 使用Jp（塑性体积比）作为塑性变形指标
            if 'Jp' in mpm_state:
                Jp = mpm_state['Jp']
                metrics.update({
                    'plastic_volume_ratio_mean': np.mean(Jp),
                    'plastic_volume_ratio_max': np.max(Jp),
                    'plastic_volume_ratio_min': np.min(Jp),
                    'plastic_volume_ratio_std': np.std(Jp)
                })

            # 基于变形梯度的塑性估计
            if 'F' in mpm_state:
                F = mpm_state['F']
                det_F = np.linalg.det(F.reshape(-1, 3, 3))
                plastic_indicator = np.where(det_F < 0.95, 1.0, 0.0)  # 简单塑性判断
                metrics['plastic_fraction'] = np.mean(plastic_indicator)

                # 添加变形梯度统计
                metrics.update({
                    'det_F_mean': np.mean(det_F),
                    'det_F_max': np.max(det_F),
                    'det_F_min': np.min(det_F),
                    'det_F_std': np.std(det_F)
                })

        return metrics

    def estimate_permanent_deformation(self, current_state):
        """估计永久变形"""
        if self.initial_state is None:
            return 0

        # 基于中心位移估计永久变形
        current_center = current_state['center']
        initial_center = self.initial_state['center']
        vertical_displacement = abs(current_center[2] - initial_center[2])

        # 简单假设：如果垂直位移超过阈值，则认为有永久变形
        permanent_threshold = 0.01
        return max(0, vertical_displacement - permanent_threshold)


class ContactMonitor:
    def __init__(self, body1_monitor, body2_monitor):
        self.body1_monitor = body1_monitor
        self.body2_monitor = body2_monitor
        self.contact_history = []
        self.debug_counter = 0

    def detect_contact(self, state1, state2):
        """接触检测"""
        if state1 is None or state2 is None:
            print("接触检测失败: state1 或 state2 为 None")
            return {'contact_detected': False}

        # 基础信息
        count1 = state1.get('count', 0)
        count2 = state2.get('count', 0)
        center1 = state1.get('center', [0, 0, 0])
        center2 = state2.get('center', [0, 0, 0])

        center_distance = np.linalg.norm(np.array(center1) - np.array(center2))

        # 方法1：基于粒子距离的接触检测
        positions1 = state1['positions']
        positions2 = state2['positions']

        distance_contact, min_distance, close_pairs = self._detect_distance_contact(positions1, positions2)

        # 方法2：基于MPM状态的接触检测
        mpm_contact, stress_info = self._detect_mpm_contact(state1, state2)

        # 综合判断 - 使用距离接触或MPM接触
        contact_detected = distance_contact or mpm_contact

        # 计算穿透深度（基于粒子间距离）
        penetration_depth = self._calculate_penetration_depth(min_distance, close_pairs)

        contact_info = {
            'contact_detected': contact_detected,
            'min_distance': min_distance,
            'center_distance': center_distance,
            'distance_contact': distance_contact,
            'mpm_contact': mpm_contact,
            'close_pairs': close_pairs,
            'estimated_force': self.estimate_contact_force(penetration_depth) if penetration_depth > 0 else 0,
            'penetration_depth': penetration_depth
        }

        # 定期详细输出
        self.debug_counter += 1
        if self.debug_counter % 10 == 0:
            print(f"[接触详细] {contact_info}")

        return contact_info

    def _calculate_penetration_depth(self, min_distance, close_pairs):
        """基于粒子间距离计算穿透深度"""
        try:
            # 如果有粒子对距离小于0，表示穿透
            if min_distance < 0:
                return abs(min_distance)

            # 如果距离很小但为正，也可能表示轻微接触
            elif min_distance < 0.01 and close_pairs > 2:
                # 轻微接触，返回一个小的穿透深度
                return 0.001

            else:
                return 0
        except:
            return 0

    def _detect_distance_contact(self, positions1, positions2):
        """使用空间划分优化距离检测"""
        from scipy.spatial import KDTree
        tree1 = KDTree(positions1)
        tree2 = KDTree(positions2)

        # 批量查询最近距离
        distances, _ = tree1.query(positions2, distance_upper_bound=0.03)

        # 过滤无效距离（大于上限的）
        valid_distances = distances[distances < 0.03]

        if len(valid_distances) == 0:
            min_distance = float('inf')
            close_pairs = 0
        else:
            min_distance = np.min(valid_distances)
            close_pairs = np.sum(valid_distances < 0.01)  # 更合理的阈值

        return close_pairs > 2, min_distance, close_pairs

    def _detect_mpm_contact(self, state1, state2):
        """基于MPM状态的接触检测"""
        mpm_contact = False
        stress_info = "无应力数据"

        # 检查弹性体的应力状态
        if 'mpm_state' in state1:
            mpm_state = state1['mpm_state']
            if 'von_mises_strain' in mpm_state:
                strain = mpm_state['von_mises_strain']
                max_strain = np.max(strain)
                avg_strain = np.mean(strain)

                stress_info = f"最大应变: {max_strain:.4f}, 平均: {avg_strain:.4f}"

                # 如果检测到显著应变，认为有接触
                if max_strain > 0.01:  # 应变阈值
                    mpm_contact = True

        return mpm_contact, stress_info

    def estimate_contact_force(self, penetration_depth):
        """估计接触力"""
        if penetration_depth <= 0:
            return 0

        # 使用非线性接触模型
        stiffness = 1e5  # 刚度系数 (N/m)
        return stiffness * penetration_depth ** 1.5  # 非线性接触
