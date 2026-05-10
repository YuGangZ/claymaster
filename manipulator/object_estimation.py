import numpy as np
import sys
sys.path.append(r"C:\Users\23142\Desktop\project\perception\EMS")
from superquadric_estimator.ems_recovery import EMS_recovery
from superquadric_estimator.utilities import read_ply, showPoints
from mayavi import mlab
from sklearn.cluster import DBSCAN
import trimesh
from trimesh import smoothing
import os


def hierarchical_ems(
        point_cloud,
        max_layer=5,
        outlier_ratio=0.05,
        eps=1.7,
        min_points=60,
        ems_kwargs=None  # 添加参数默认值处理
):
    """层次化超二次曲面拟合算法"""
    # 处理可选参数
    ems_params = {
        'OutlierRatio': outlier_ratio,
        'MaxIterationEM': 20,
        'ToleranceEM': 1e-3,
        'RelativeToleranceEM': 2e-1,
        'MaxOptiIterations': 2,
        'Sigma': 0.3,
        'MaxiSwitch': 2,
        'AdaptiveUpperBound': True,
        'Rescale': False
    }
    # 合并用户自定义参数
    if ems_kwargs is not None:
        ems_params.update(ems_kwargs)

    # 初始化数据结构
    segmented_points = {layer: [] for layer in range(max_layer + 1)}
    outliers = {layer: [] for layer in range(max_layer + 1)}
    segmented_points[0] = [point_cloud]
    quadrics = []

    for layer in range(max_layer):
        for cluster_idx in range(len(segmented_points[layer])):
            cluster = segmented_points[layer][cluster_idx]

            # 执行EMS恢复算法
            params, probabilities = EMS_recovery(cluster, **ems_params)
            quadrics.append(params)

            # 分割内点和离群点
            inliers_mask = probabilities > 0.1
            inliers = cluster[inliers_mask]
            current_outliers = cluster[~inliers_mask]

            # 更新当前层的分割结果
            segmented_points[layer][cluster_idx] = inliers

            # 对离群点进行聚类分析
            if len(current_outliers) > 0.2 * len(cluster):
                dbscan = DBSCAN(eps=eps, min_samples=min_points).fit(current_outliers)
                unique_labels = np.unique(dbscan.labels_)

                # 处理有效聚类
                for label in unique_labels:
                    if label == -1:
                        continue  # 跳过噪声点
                    mask = dbscan.labels_ == label
                    segmented_points[layer + 1].append(current_outliers[mask])

                # 记录噪声点
                noise_mask = dbscan.labels_ == -1
                outliers[layer].append(current_outliers[noise_mask])
            else:
                outliers[layer].append(current_outliers)

    return segmented_points, outliers, quadrics


def generate_superquadric_mesh(sq, resolution=100):
    """生成超二次曲面网格"""
    # 参数提取
    epsilon1, epsilon2 = sq.shape
    a, b, c = sq.scale
    translation = sq.translation
    rotation = sq.RotM

    # 参数化采样
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(-np.pi / 2, np.pi / 2, resolution)
    theta, phi = np.meshgrid(theta, phi)

    # 计算顶点坐标（优化计算效率）
    cos_phi = np.abs(np.cos(phi)) ** epsilon1
    sin_phi = np.abs(np.sin(phi)) ** epsilon1
    cos_theta = np.abs(np.cos(theta)) ** epsilon2
    sin_theta = np.abs(np.sin(theta)) ** epsilon2

    x = a * np.sign(np.cos(phi)) * cos_phi * np.sign(np.cos(theta)) * cos_theta
    y = b * np.sign(np.cos(phi)) * cos_phi * np.sign(np.sin(theta)) * sin_theta
    z = c * np.sign(np.sin(phi)) * sin_phi

    # 坐标变换
    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    vertices = vertices @ rotation.T + translation

    # 生成面片（优化内存效率）
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = i * resolution + j
            v1 = v0 + 1
            v2 = (i + 1) * resolution + j
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def save_superquadrics(quadrics, output_path, merge=True):
    """
    保存超二次曲面模型
    :param quadrics: 超二次曲面参数列表
    :param output_path: 输出文件完整路径
    :param merge: 是否合并为一个文件
    """
    # 提取目录路径
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 生成有效网格
    valid_meshes = []
    for i, sq in enumerate(quadrics):
        try:
            mesh = generate_superquadric_mesh(sq)
            valid_meshes.append(mesh)
        except Exception as e:
            print(f"Mesh generation failed for quadric {i}: {str(e)}")

    # 合并或单独保存
    if merge and valid_meshes:
        merged = trimesh.util.concatenate(valid_meshes)
        merged.export(output_path)
    elif not merge:
        # 生成基础文件名
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        for i, mesh in enumerate(valid_meshes):
            mesh_path = os.path.join(output_dir, f"{base_name}_{i}.stl")
            mesh.export(mesh_path)


if __name__ == "__main__":
    # 数据加载
    pc = read_ply(r"C:\Users\23142\Desktop\project\perception\control0807\control_data\loop1\scaled.ply")

    # 执行分层拟合
    segments, outliers, quadrics = hierarchical_ems(
        pc,
        max_layer=1,
        eps=1.7,
        min_points=20
    )

    # 保存结果
    # save_superquadrics(quadrics, "output/superquadrics.stl")

    # # # 可视化
    fig = mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))
    for q in quadrics:
        q.showSuperquadric(arclength=0.2)
    showPoints(pc, scale_factor=0.04)
    mlab.show()