import numpy as np
from scipy.spatial import ConvexHull, KDTree
from scipy.special import beta as beta_function
from sklearn.decomposition import PCA


def calculate_geometric_features(sq, point_cloud):
    features = {}

    a1, a2, a3 = sq.scale
    epsilon1, epsilon2 = sq.shape

    geom_mean = (a1 * a2 * a3) ** (1 / 3)
    if geom_mean > 1e-8:
        a1_norm = a1 / geom_mean
        a2_norm = a2 / geom_mean
        a3_norm = a3 / geom_mean
    else:
        a1_norm, a2_norm, a3_norm = a1, a2, a3

    features['a1'] = a1_norm
    features['a2'] = a2_norm
    features['a3'] = a3_norm
    features['epsilon1'] = epsilon1
    features['epsilon2'] = epsilon2

    euler = sq.euler
    features['rot_x'] = euler[2]
    features['rot_y'] = euler[1]
    features['rot_z'] = euler[0]

    features['tx'] = sq.translation[0]
    features['ty'] = sq.translation[1]
    features['tz'] = sq.translation[2]

    beta = calculate_volume_coefficient(epsilon1, epsilon2)
    # 使用归一化尺度计算体积，保持尺度不变性
    volume = beta * a1_norm * a2_norm * a3_norm * (geom_mean ** 3)
    volume = beta * a1 * a2 * a3
    features['volume'] = volume

    if a1_norm * a2_norm > 1e-8:
        elongation = a3_norm / np.sqrt(a1_norm * a2_norm)
    else:
        elongation = 1.0
    features['elongation'] = elongation

    # r_o = alpha_shape_area / convex_hull_area
    # 基于触觉点云投影到局部切平面
    roughness = calculate_roughness_paper_version(point_cloud)
    features['smoothness'] = roughness

    features_14d = np.array([
        features['a1'], features['a2'], features['a3'],
        features['epsilon1'], features['epsilon2'],
        features['rot_x'], features['rot_y'], features['rot_z'],
        features['tx'], features['ty'], features['tz'],
        features['volume'], features['elongation'], features['smoothness']
    ])

    return features_14d, features


def calculate_volume_coefficient(epsilon1, epsilon2):
    eps1 = float(np.clip(epsilon1, 0.05, 5.0))
    eps2 = float(np.clip(epsilon2, 0.05, 5.0))

    if abs(eps1 - 1.0) < 0.01 and abs(eps2 - 1.0) < 0.01:
        return 4.1887902047863905  # 4π/3

    try:
        beta1 = beta_function(eps1 / 2, eps1 * eps2 / 2)
        beta2 = beta_function(eps2 / 2, eps2 / 2)
        coefficient = (4.0 * np.pi / 3.0) * eps1 * eps2 * beta1 * beta2
        return float(np.clip(coefficient, 0.1, 20.0))
    except:
        # 备选近似
        return 4.18879 * (eps1 * eps2) ** (-0.15)


def calculate_roughness_paper_version(point_cloud, alpha=1.2):
    """
    计算表面粗糙度 r_o
    r_o = S̄_α(P') / C_on(P')

    其中：
    - P'：点云投影到局部切平面的2D点集
    - S̄_α：alpha-shape的面积
    - C_on：凸包面积

    参数:
        point_cloud: 3D点云 (N, 3)
        alpha: alpha-shape参数，论文设为1.2mm

    返回:
        roughness: [0, 1]范围内的粗糙度值
    """
    try:
        if len(point_cloud) < 10:
            return 0.5

        # 步骤1：使用PCA找到局部切平面（投影到前两个主成分）
        pca = PCA(n_components=3)
        pca.fit(point_cloud)

        # 投影到前两个主成分构成的平面（切平面）
        # 第三个主成分方向为法向量
        tangent_vectors = pca.components_[0:2]  # 两个主方向
        normal_vector = pca.components_[2]  # 法向量

        # 将点云投影到切平面（2D坐标）
        centered_pc = point_cloud - np.mean(point_cloud, axis=0)
        projected_2d = centered_pc @ tangent_vectors.T  # (N, 2)

        # 步骤2：计算凸包面积 C_on(P')
        try:
            hull = ConvexHull(projected_2d)
            convex_hull_area = hull.volume  # 在2D中，volume=area
        except:
            # 点共线或点太少，使用边界框近似
            bbox = np.max(projected_2d, axis=0) - np.min(projected_2d, axis=0)
            convex_hull_area = bbox[0] * bbox[1]

        if convex_hull_area < 1e-8:
            return 0.5

        # 步骤3：计算alpha-shape面积 S̄_α(P')
        alpha_shape_area = calculate_alpha_shape_area(projected_2d, alpha)

        # 步骤4：计算粗糙度 = alpha-shape面积 / 凸包面积
        # 越粗糙的表面，alpha-shape越"破碎"，面积越大
        roughness = alpha_shape_area / convex_hull_area

        # 归一化到[0, 1]（理论上粗糙度可以>1，但通常<2）
        roughness = min(roughness, 2.0) / 2.0

        return float(roughness)

    except Exception as e:
        print(f"粗糙度计算错误: {e}")
        return 0.5


def calculate_alpha_shape_area(points_2d, alpha):
    """
    计算2D点集的alpha-shape面积

    简化实现：使用滚球算法（rolling ball algorithm）概念
    在2D中，alpha-shape是半径为1/alpha的圆滚动形成的轮廓

    这里使用基于Delaunay三角化的简化方法
    """
    try:
        from scipy.spatial import Delaunay

        if len(points_2d) < 3:
            return 0.0

        # 使用Delaunay三角化
        tri = Delaunay(points_2d)

        # 筛选满足alpha条件的边（简化：使用 circumradius 条件）
        alpha_area = 0.0

        for simplex in tri.simplices:
            # 获取三角形的三个顶点
            p1, p2, p3 = points_2d[simplex]

            # 计算三角形面积
            triangle_area = 0.5 * abs(
                (p2[0] - p1[0]) * (p3[1] - p1[1]) -
                (p3[0] - p1[0]) * (p2[1] - p1[1])
            )

            # 计算外接圆半径
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)

            # 使用海伦公式计算外接圆半径
            if a * b * c > 1e-8:
                circum_r = (a * b * c) / (4 * triangle_area + 1e-8)

                # alpha条件：外接圆半径 < 1/alpha
                if circum_r < (1.0 / alpha + 1e-8):
                    alpha_area += triangle_area

        return alpha_area

    except ImportError:
        # 如果没有Delaunay，使用凸包面积的近似
        try:
            hull = ConvexHull(points_2d)
            return hull.volume * 0.8  # 近似：alpha-shape略小于凸包
        except:
            return 0.0
    except Exception as e:
        print(f"Alpha-shape计算错误: {e}")
        return 0.0


def get_default_14d_features():
    """返回默认的14维特征值"""
    return np.array([
        1.0, 1.0, 1.0,  # a1, a2, a3
        1.0, 1.0,  # epsilon1, epsilon2（接近球体）
        0.0, 0.0, 0.0,  # rot_x, rot_y, rot_z
        0.0, 0.0, 0.0,  # tx, ty, tz
        4.18879,  # volume（单位球体积）
        1.0,  # elongation（球体为1）
        0.5  # smoothness（中等粗糙度）
    ])