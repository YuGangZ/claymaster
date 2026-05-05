import numpy as np
import trimesh
from trimesh.proximity import closest_point
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


def generate_superquadric_mesh(sq, resolution=100):
    from EMS.object_estimation import generate_superquadric_mesh as _gen
    return _gen(sq, resolution)


# ------------------------------------------------------------------------------
# 1.  RMSE（Root-Mean-Square Error）
# ------------------------------------------------------------------------------
def calculate_rmse(point_cloud, quadrics):
    """点云 vs 所有超二次曲面集合的最小 RMSE"""
    point_cloud = point_cloud.astype(np.float64)
    min_dist = np.full(len(point_cloud), np.inf)

    # 复用 EMS 距离函数
    from EMS.EMS_recovery import Distance
    for sq in quadrics:
        x = np.hstack([sq.shape, sq.scale, sq.euler, sq.translation])
        dist = Distance(point_cloud, x)  # 已 ≥ 0
        min_dist = np.minimum(min_dist, dist)

    return float(np.sqrt(np.mean(min_dist ** 2)))


# ------------------------------------------------------------------------------
# 2.  MAE（Mean Absolute Error）
# ------------------------------------------------------------------------------
def calculate_mae(point_cloud, quadrics):
    """点云 vs 所有超二次曲面集合的最小 MAE"""
    point_cloud = point_cloud.astype(np.float64)
    min_abs = np.full(len(point_cloud), np.inf)

    from EMS.EMS_recovery import Distance
    for sq in quadrics:
        x = np.hstack([sq.shape, sq.scale, sq.euler, sq.translation])
        dist = np.abs(Distance(point_cloud, x))
        min_abs = np.minimum(min_abs, dist)

    return float(np.mean(min_abs))


# ------------------------------------------------------------------------------
# 3.  Shape Similarity & Classification
# ------------------------------------------------------------------------------
def shape_similarity(sq, target_type: str) -> float:
    """
    单个超二次曲面与理想模板形状之间的归一化相似度
    target_type ∈ ["sphere","ellipsoid","cube","cuboid","cylinder"]
    返回 0~1，1=完全匹配
    """
    est_shape = np.array(sq.shape)
    est_scale = np.array(sq.scale)

    templates = {
        "sphere": {"shape": [1.0, 1.0], "scale_ratio": [1, 1, 1]},
        "ellipsoid": {"shape": [1.0, 1.0], "scale_ratio": None},
        "cube": {"shape": [0.1, 0.1], "scale_ratio": [1, 1, 1]},
        "cuboid": {"shape": [0.1, 0.1], "scale_ratio": None},
        "cylinder": {"shape": [0.1, 1.0], "scale_ratio": [1, 1, 2]}
    }
    if target_type not in templates:
        raise ValueError(f"Unsupported target type: {target_type}")

    template = templates[target_type]
    ideal_shape = np.array(template["shape"])
    ideal_ratio = template["scale_ratio"]

    # 形状得分
    shape_dist = np.linalg.norm(est_shape - ideal_shape)
    max_shape_dist = np.linalg.norm([2.0, 2.0] - ideal_shape)
    shape_score = 1 - (shape_dist / max_shape_dist)

    # 尺度得分
    if ideal_ratio is None:
        scale_score = 1.0
    else:
        scale_ratio = np.sort(est_scale / np.max(est_scale))
        ideal_ratio = np.sort(np.array(ideal_ratio) / np.max(ideal_ratio))
        scale_score = 1 - np.linalg.norm(scale_ratio - ideal_ratio) / np.linalg.norm(ideal_ratio)

    return 0.7 * shape_score + 0.3 * scale_score


def classify_shape(sq) -> dict:
    """
    对单个 sq 输出与所有基本形状的相似度，并返回最可能类别
    return {"scores":{...}, "best":str, "best_score":float}
    """
    classes = ["sphere", "ellipsoid", "cube", "cuboid", "cylinder"]
    scores = {c: shape_similarity(sq, c) for c in classes}
    best_cls = max(scores, key=scores.get)
    return {"scores": scores, "best": best_cls, "best_score": scores[best_cls]}


# ------------------------------------------------------------------------------
# 4.  Symmetric Surface Distance (SSD)
# ------------------------------------------------------------------------------
def symmetric_surface_distance(mesh_a: trimesh.Trimesh,
                               mesh_b: trimesh.Trimesh,
                               sample_num: int = 50000):
    """双向平均最近面距离（兼容新版 trimesh）"""
    pts_a, _ = trimesh.sample.sample_surface(mesh_a, sample_num)
    pts_b, _ = trimesh.sample.sample_surface(mesh_b, sample_num)

    # ---- A→B ----
    # 修复：closest_point 返回的是元组，我们需要第一个元素（点坐标）
    closest_result_b = closest_point(mesh_b, pts_a)
    cp_b = closest_result_b[0] if isinstance(closest_result_b, tuple) else closest_result_b
    cp_b = np.asarray(cp_b)  # 确保是 ndarray
    dist_ab = np.linalg.norm(pts_a - cp_b, axis=1).mean()

    # ---- B→A ----
    closest_result_a = closest_point(mesh_a, pts_b)
    cp_a = closest_result_a[0] if isinstance(closest_result_a, tuple) else closest_result_a
    cp_a = np.asarray(cp_a)
    dist_ba = np.linalg.norm(pts_b - cp_a, axis=1).mean()

    return float((dist_ab + dist_ba) / 2)


def ssd_to_superquadrics(point_cloud, quadrics, n_sample: int = 50000):
    """点云凸包 vs 重建 sq 合并网格 的 SSD"""
    pc_mesh = trimesh.PointCloud(point_cloud).convex_hull
    recon_mesh = merge_sq_meshes(quadrics)
    return symmetric_surface_distance(pc_mesh, recon_mesh, n_sample)


# ------------------------------------------------------------------------------
# 5.  Hausdorff Distance
# ------------------------------------------------------------------------------
def hausdorff_distance(point_cloud, quadrics):
    """双向 Hausdorff 距离"""
    recon_mesh = merge_sq_meshes(quadrics)
    recon_pts, _ = trimesh.sample.sample_surface(recon_mesh, 50000)

    # 直接解包，但只取第一个值（距离）
    d_ab, _, _ = directed_hausdorff(point_cloud, recon_pts)
    d_ba, _, _ = directed_hausdorff(recon_pts, point_cloud)

    return float(max(d_ab, d_ba))


# ------------------------------------------------------------------------------
# 6.  Chamfer Distance
# ------------------------------------------------------------------------------
def chamfer_distance(point_cloud, quadrics, n_samples=50000):
    """
    计算点云与重建超二次曲面之间的倒角距离

    Chamfer Distance 定义:
    CD = (1/|S1|) * Σ_{x∈S1} min_{y∈S2} ||x-y||^2 + (1/|S2|) * Σ_{y∈S2} min_{x∈S1} ||y-x||^2

    对于表面点云评估，这是非常合适的指标。
    """
    # 从重建网格采样点
    recon_mesh = merge_sq_meshes(quadrics)
    recon_samples, _ = trimesh.sample.sample_surface(recon_mesh, n_samples)

    # 构建KDTree加速最近邻搜索
    tree_pc = cKDTree(point_cloud)
    tree_recon = cKDTree(recon_samples)

    # 计算点云到重建表面的距离
    dist_pc_to_recon, _ = tree_recon.query(point_cloud)
    cd_pc_to_recon = np.mean(dist_pc_to_recon ** 2)

    # 计算重建表面到点云的距离
    dist_recon_to_pc, _ = tree_pc.query(recon_samples)
    cd_recon_to_pc = np.mean(dist_recon_to_pc ** 2)

    # Chamfer Distance 是两部分的和
    chamfer_dist = cd_pc_to_recon + cd_recon_to_pc

    print(f"Chamfer Distance: {chamfer_dist:.6f}")
    print(f"  点云→重建: {cd_pc_to_recon:.6f}")
    print(f"  重建→点云: {cd_recon_to_pc:.6f}")

    return float(chamfer_dist)


# ------------------------------------------------------------------------------
# 7.  Volume IoU
# ------------------------------------------------------------------------------
def volume_iou(point_cloud, quadrics, voxel_resolution=64):
    """使用体素化方法计算 Volume IoU"""
    pc_mesh = trimesh.PointCloud(point_cloud).convex_hull
    recon_mesh = merge_sq_meshes(quadrics)

    print(f"点云凸包是否封闭: {pc_mesh.is_watertight}")
    print(f"点云凸包体积: {pc_mesh.volume:.6f}")
    print(f"重建网格是否封闭: {recon_mesh.is_watertight}")
    print(f"重建网格体积: {recon_mesh.volume:.6f}")

    try:
        # 尝试布尔运算
        inter = pc_mesh.intersection(recon_mesh)
        vol_inter = float(inter.volume) if inter.is_watertight else 0.0
        vol_union = float(pc_mesh.volume + recon_mesh.volume - vol_inter)
        method = "布尔运算"
    except Exception as e:
        print(f"布尔运算失败: {e}")
        print("使用体素化方法计算 IoU...")

        # 使用体素化方法
        try:
            # 体素化点云凸包
            pc_voxel = pc_mesh.voxelized(voxel_resolution)

            # 体素化重建网格
            recon_voxel = recon_mesh.voxelized(voxel_resolution)

            # 确保体素矩阵存在且形状相同
            if (hasattr(pc_voxel, 'matrix') and hasattr(recon_voxel, 'matrix') and
                    pc_voxel.matrix.shape == recon_voxel.matrix.shape):

                # 计算体素交集和并集
                inter_matrix = pc_voxel.matrix & recon_voxel.matrix
                union_matrix = pc_voxel.matrix | recon_voxel.matrix

                # 转换为标量
                inter_count = float(np.sum(inter_matrix))
                union_count = float(np.sum(union_matrix))

                # 计算体积
                voxel_volume = float(pc_voxel.scale ** 3)
                vol_inter = inter_count * voxel_volume
                vol_union = union_count * voxel_volume
                method = f"体素化 (分辨率: {voxel_resolution})"
            else:
                # 回退到近似计算
                vol_inter = float(min(pc_mesh.volume, recon_mesh.volume) * 0.8)
                vol_union = float(pc_mesh.volume + recon_mesh.volume - vol_inter)
                method = "近似估计"

        except Exception as ve:
            print(f"体素化也失败: {ve}")
            # 最终回退
            vol_inter = float(min(pc_mesh.volume, recon_mesh.volume) * 0.5)
            vol_union = float(pc_mesh.volume + recon_mesh.volume - vol_inter)
            method = "保守估计"

    iou = float(vol_inter / (vol_union + 1e-12))
    print(f"Volume IoU ({method}): {iou:.6f}")

    return iou


# ------------------------------------------------------------------------------
# 通用：合并 List[sq] → 一个 trimesh
# ------------------------------------------------------------------------------
def merge_sq_meshes(quadrics):
    valid = []
    for sq in quadrics:
        try:
            valid.append(generate_superquadric_mesh(sq))
        except Exception as e:
            print('[merge] skip one sq mesh:', e)
    if not valid:
        raise RuntimeError('No valid superquadric mesh to compute metrics.')

    v_all = np.vstack([m.vertices for m in valid])
    f_all = np.vstack([
        m.faces + offset for offset, m in zip(
            np.cumsum([0] + [len(m.vertices) for m in valid[:-1]]), valid
        )
    ])
    merged = trimesh.Trimesh(vertices=v_all, faces=f_all, process=True)
    merged.fill_holes()
    return merged


# ------------------------------------------------------------------------------
# 一键评测
# ------------------------------------------------------------------------------
def evaluate_all(point_cloud, quadrics):
    """
    返回 dict，包含 RMSE/MAE/SSD/Hausdorff/Chamfer Distance
    额外再把每个 sq 的 best shape 也返回，方便打印
    """
    metrics = {
        'RMSE': calculate_rmse(point_cloud, quadrics),
        'MAE': calculate_mae(point_cloud, quadrics),
        'SSD': ssd_to_superquadrics(point_cloud, quadrics),
        'Hausdorff': hausdorff_distance(point_cloud, quadrics),
        'Chamfer': chamfer_distance(point_cloud, quadrics),
        # 'VolumeIoU': volume_iou(point_cloud, quadrics),  # 注释掉，因为对表面点云不太适用
        # 分类结果
        'shape_classification': [classify_shape(sq) for sq in quadrics]
    }

    # 添加体积信息作为参考
    pc_mesh = trimesh.PointCloud(point_cloud).convex_hull
    recon_mesh = merge_sq_meshes(quadrics)
    metrics['point_cloud_volume'] = float(pc_mesh.volume)
    metrics['reconstructed_volume'] = float(recon_mesh.volume)

    print(f"\n=== 体积参考信息 ===")
    print(f"点云凸包体积: {metrics['point_cloud_volume']:.6f}")
    print(f"重建网格体积: {metrics['reconstructed_volume']:.6f}")

    return metrics