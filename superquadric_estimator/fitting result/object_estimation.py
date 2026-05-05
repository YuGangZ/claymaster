import numpy as np
from EMS.EMS_recovery import EMS_recovery
from EMS.utilities import read_ply, showPoints
from mayavi import mlab
from sklearn.cluster import DBSCAN
import trimesh
from trimesh import smoothing
import os
import datetime
from EMS.evaluate import evaluate_all


def hierarchical_ems(
        point_cloud,
        max_layer=5,
        outlier_ratio=0.9,
        eps=1.7,
        min_points=60,
        ems_kwargs=None,
        calculate_rmse_flag=True
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
        'Rescale': True
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

            # 恢复原始聚类触发条件：内点概率和<80%原始点数
            if np.sum(probabilities) < (0.8 * len(cluster)):
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
    # # 计算RMSE
    # if calculate_rmse_flag and quadrics:
    #     rmse = calculate_rmse(point_cloud, quadrics)
    #     print(f"RMSE between point cloud and estimated superquadrics: {rmse:.6f}")
    #     for idx, sq in enumerate(quadrics):
    #         res = classify_shape(sq)
    #         print(f"[Quadric {idx}]  best_match={res['best']}  score={res['best_score']:.3f}")
    #         print(f"              all_scores={res['scores']}")

    return segmented_points, outliers, quadrics


# def generate_superquadric_mesh(sq, resolution=100):
#     """生成超二次曲面网格（修复空洞问题）"""
#     # 参数提取
#     epsilon1, epsilon2 = sq.shape
#     a, b, c = sq.scale
#     translation = sq.translation
#     rotation = sq.RotM
#
#     # 参数化采样 - 排除两极
#     theta = np.linspace(0, 2 * np.pi, resolution)
#     phi = np.linspace(-np.pi / 2 + 1e-5, np.pi / 2 - 1e-5, resolution - 2)  # 排除两极
#     theta, phi = np.meshgrid(theta, phi)
#
#     # 计算中间部分顶点
#     cos_phi = np.abs(np.cos(phi)) ** epsilon1
#     sin_phi = np.abs(np.sin(phi)) ** epsilon1
#     cos_theta = np.abs(np.cos(theta)) ** epsilon2
#     sin_theta = np.abs(np.sin(theta)) ** epsilon2
#
#     x = a * np.sign(np.cos(phi)) * cos_phi * np.sign(np.cos(theta)) * cos_theta
#     y = b * np.sign(np.cos(phi)) * cos_phi * np.sign(np.sin(theta)) * sin_theta
#     z = c * np.sign(np.sin(phi)) * sin_phi
#
#     # 创建极点
#     north_pole = np.array([0, 0, c]).reshape(1, 3)
#     south_pole = np.array([0, 0, -c]).reshape(1, 3)
#
#     # 合并所有顶点
#     vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
#     vertices = np.vstack([south_pole, vertices, north_pole])
#
#     # 应用变换
#     vertices = vertices @ rotation.T + translation
#
#     # 生成面片
#     faces = []
#     n_phi = resolution - 2  # 纬度层数（排除两极）
#     n_theta = resolution  # 经度点数
#
#     # 1. 生成中间部分的面片
#     for i in range(n_phi - 1):  # 纬度索引
#         for j in range(n_theta):  # 经度索引
#             j_next = (j + 1) % n_theta
#
#             # 当前四边形的四个顶点
#             v0 = i * n_theta + j + 1  # +1 跳过南极
#             v1 = i * n_theta + j_next + 1
#             v2 = (i + 1) * n_theta + j + 1
#             v3 = (i + 1) * n_theta + j_next + 1
#
#             # 添加两个三角形
#             faces.append([v0, v1, v2])
#             faces.append([v1, v3, v2])
#
#     # 2. 生成南极三角形扇
#     south_idx = 0  # 南极索引
#     first_ring_start = 1  # 第一圈起始索引
#     for j in range(n_theta):
#         j_next = (j + 1) % n_theta
#         v0 = south_idx
#         v1 = first_ring_start + j
#         v2 = first_ring_start + j_next
#         faces.append([v0, v2, v1])  # 注意顶点顺序保证法线朝外
#
#     # 3. 生成北极三角形扇
#     north_idx = len(vertices) - 1  # 北极索引
#     last_ring_start = (n_phi - 1) * n_theta + 1  # 最后一圈起始索引
#     for j in range(n_theta):
#         j_next = (j + 1) % n_theta
#         v0 = north_idx
#         v1 = last_ring_start + j
#         v2 = last_ring_start + j_next
#         faces.append([v0, v1, v2])  # 注意顶点顺序保证法线朝外
#
#     # 创建网格对象
#     mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
#     mesh.fix_normals()
#     smoothing.filter_laplacian(mesh, iterations=1)
#
#     # 确保网格是水密的
#     if not mesh.is_watertight:
#         mesh.fill_holes()
#
#     return mesh
def generate_superquadric_mesh(sq, resolution=100):
    """生成超二次曲面网格（确保封闭）"""
    # 参数提取
    epsilon1, epsilon2 = sq.shape
    a, b, c = sq.scale
    translation = sq.translation
    rotation = sq.RotM

    # 参数化采样
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(-np.pi / 2, np.pi / 2, resolution)
    theta, phi = np.meshgrid(theta, phi)

    # 计算顶点
    cos_phi = np.abs(np.cos(phi)) ** epsilon1
    sin_phi = np.abs(np.sin(phi)) ** epsilon1
    cos_theta = np.abs(np.cos(theta)) ** epsilon2
    sin_theta = np.abs(np.sin(theta)) ** epsilon2

    x = a * np.sign(np.cos(phi)) * cos_phi * np.sign(np.cos(theta)) * cos_theta
    y = b * np.sign(np.cos(phi)) * cos_phi * np.sign(np.sin(theta)) * sin_theta
    z = c * np.sign(np.sin(phi)) * sin_phi

    # 应用变换
    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    vertices = vertices @ rotation.T + translation

    # 生成面片 - 使用更简单的方法
    faces = []
    n_theta = resolution
    n_phi = resolution

    for i in range(n_phi - 1):
        for j in range(n_theta):
            # 当前四边形的四个顶点
            v0 = i * n_theta + j
            v1 = i * n_theta + (j + 1) % n_theta
            v2 = (i + 1) * n_theta + j
            v3 = (i + 1) * n_theta + (j + 1) % n_theta

            # 添加两个三角形
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    # 创建网格
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    # 强制确保封闭性
    if not mesh.is_watertight:
        print("执行网格修复...")
        # 移除无效几何
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()

        # 填充孔洞
        mesh.fill_holes()

        # 如果仍然不封闭，使用凸包作为最后手段
        if not mesh.is_watertight:
            print("使用凸包修复...")
            mesh = mesh.convex_hull

    print(f"修复后网格状态: 顶点={len(mesh.vertices)}, 面片={len(mesh.faces)}, 封闭={mesh.is_watertight}")
    return mesh

def postprocess_merged_mesh(merged_mesh, subdivision=2, laplacian_iter=5):
    """
    对合并后的网格进行细分和平滑
    :param subdivision: 细分次数（0=不细分）
    :param laplacian_iter: 拉普拉斯平滑迭代次数
    """
    # 网格细分
    if subdivision > 0:
        try:
            for _ in range(subdivision):
                merged_mesh = merged_mesh.subdivide_loop()
        except Exception as e:
            print(f"细分失败，改用自适应细分: {str(e)}")
            # 使用更稳健的自适应细分
            merged_mesh = merged_mesh.subdivide_to_size(
                max_edge=0.1 * merged_mesh.scale,
                max_iter=subdivision * 2
            )

    # 拉普拉斯平滑 - 增加迭代次数
    smoothing.filter_laplacian(merged_mesh, iterations=laplacian_iter)

    # 修复网格问题
    merged_mesh.update_faces(merged_mesh.nondegenerate_faces())
    merged_mesh.update_faces(merged_mesh.unique_faces())
    merged_mesh.fill_holes()
    merged_mesh.remove_unreferenced_vertices()

    # 检查顶点中是否有无效值
    if np.any(~np.isfinite(merged_mesh.vertices)):
        valid_mask = np.all(np.isfinite(merged_mesh.vertices), axis=1)
        merged_mesh.update_vertices(valid_mask)

    # 确保网格是水密的
    if not merged_mesh.is_watertight:
        merged_mesh.fill_holes()

    return merged_mesh


def save_superquadrics(quadrics, output_dir="output_models", merge=True):
    """保存超二次曲面模型"""
    os.makedirs(output_dir, exist_ok=True)

    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 生成有效网格
    valid_meshes = []
    for i, sq in enumerate(quadrics):
        try:
            mesh = generate_superquadric_mesh(sq)
            valid_meshes.append(mesh)
            print(f"Generated mesh {i} with {len(mesh.vertices)} vertices")
        except Exception as e:
            print(f"Mesh generation failed for quadric {i}: {str(e)}")

    if not valid_meshes:
        print("No valid meshes to save.")
        return

    # 合并或单独保存
    if merge and valid_meshes:
        # 合并顶点和面片
        merged_vertices = np.vstack([m.vertices for m in valid_meshes])
        merged_faces = np.vstack([
            m.faces + offset
            for offset, m in zip(
                np.cumsum([0] + [len(m.vertices) for m in valid_meshes[:-1]]),
                valid_meshes
            )
        ])

        merged_mesh = trimesh.Trimesh(
            vertices=merged_vertices,
            faces=merged_faces,
            process=True
        )

        # 添加后处理
        merged_mesh = postprocess_merged_mesh(merged_mesh, subdivision=2, laplacian_iter=5)

        # 生成描述性文件名
        bbox_center = merged_mesh.bounding_box.center_mass.round(2)
        bbox_size = merged_mesh.bounding_box.extents.round(2)

        center_str = f"center_{bbox_center[0]:.1f}_{bbox_center[1]:.1f}_{bbox_center[2]:.1f}"
        size_str = f"size_{bbox_size[0]:.1f}_{bbox_size[1]:.1f}_{bbox_size[2]:.1f}"
        filename = f"merged_{len(valid_meshes)}parts_{center_str}_{size_str}_{timestamp}.stl"
        filename = filename.replace(" ", "").replace("[", "").replace("]", "")

        # 保存文件
        merged_mesh.export(os.path.join(output_dir, filename))
        print(f"Saved merged model: {filename}")
        return os.path.join(output_dir, filename)

    elif not merge:
        saved_files = []
        for i, mesh in enumerate(valid_meshes):
            bbox_center = mesh.bounding_box.center_mass.round(2)
            filename = f"quadric_{i}_center_{bbox_center[0]:.1f}_{bbox_center[1]:.1f}_{bbox_center[2]:.1f}_{timestamp}.stl"
            filepath = os.path.join(output_dir, filename)
            mesh.export(filepath)
            print(f"Saved: {filename}")
            saved_files.append(filepath)
        return saved_files

    return None



if __name__ == "__main__":
    # 数据加载
    pc = read_ply(r"C:\Users\23142\Desktop\compare\EMS\_cylinder.ply")
    # # 诊断点云
    # print("=== 点云诊断 ===")
    # print(f"点云点数: {len(pc)}")
    # print(f"点云边界框: {np.min(pc, axis=0)} ~ {np.max(pc, axis=0)}")
    #
    # # 检查点云是否封闭（简单方法）
    # from scipy.spatial import ConvexHull
    #
    # try:
    #     hull = ConvexHull(pc)
    #     print(f"点云凸包顶点数: {len(hull.vertices)}")
    #     print(f"点云凸包体积: {hull.volume:.6f}")
    # except:
    #     print("点云无法形成凸包，可能不封闭")
    # 执行分层拟合
    segments, outliers, quadrics = hierarchical_ems(
        pc,
        max_layer=1,
        eps=1.7,
        min_points=60
    )
    # print("\n=== 超二次曲面诊断 ===")
    for i, sq in enumerate(quadrics):
        print(f"超二次曲面 {i}:")
        print(f"  形状参数: {sq.shape}")
        print(f"  尺度参数: {sq.scale}")
    #     print(f"  平移: {sq.translation}")
    #
    #     # 检查单个超二次曲面网格
    #     try:
    #         single_mesh = generate_superquadric_mesh(sq)
    #         print(f"  单个网格是否封闭: {single_mesh.is_watertight}")
    #         print(f"  单个网格顶点数: {len(single_mesh.vertices)}")
    #         print(f"  单个网格面片数: {len(single_mesh.faces)}")
    #     except Exception as e:
    #         print(f"  单个网格生成失败: {e}")


    metrics = evaluate_all(pc, quadrics)

    print(f"RMSE      : {metrics['RMSE']:.6f}")
    print(f"MAE       : {metrics['MAE']:.6f}")
    print(f"SSD       : {metrics['SSD']:.6f}")
    print(f"Hausdorff : {metrics['Hausdorff']:.6f}")
    # print(f"VolumeIoU : {metrics['VolumeIoU']:.6f}")
    for idx, cls in enumerate(metrics['shape_classification']):
        print(f"超二次曲面 {idx}:")
        print(f"  最佳匹配: {cls['best']} (得分: {cls['best_score']:.6f})")
        print("  所有形状得分:")
        for shape_name, score in cls['scores'].items():
            print(f"    {shape_name}: {score:.6f}")
    # # 保存结果
    # output_file = save_superquadrics(quadrics, output_dir="asuperquadric_models", merge=True)
    #
    # if output_file:
    #     print(f"Final model saved to: {output_file}")

    # 可视化
    fig = mlab.figure(size=(800, 642), bgcolor=(1, 1, 1))
    for q in quadrics:
        q.showSuperquadric(arclength=0.2)
        # q.showSuperquadric(arclength=0.2)
    # 动态计算合适的 scale_factor
    bbox_size = np.max(pc, axis=0) - np.min(pc, axis=0)
    scale_factor = np.min(bbox_size) * 0.01  # 对象最小尺寸的 4%
    # print(f"使用 scale_factor: {scale_factor:.4f}")

    showPoints(pc, scale_factor=scale_factor)
    mlab.show()