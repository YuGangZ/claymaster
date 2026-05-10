import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree


class ZLayerPointCloudFilter:
    def __init__(self, voxel_size=16.0, radius_multiplier=1.5):
        """
        参数:
        voxel_size: XY平面的体素网格大小
        radius_multiplier: 二次过滤半径的乘数因子（相对于体素大小）
        """
        self.voxel_size = voxel_size
        self.radius_multiplier = radius_multiplier
        self.original_point_count = 0
        self.filtered_point_count = 0
        self.avg_point_distance = 0.0

    def filter(self, input_path, output_path="filtered.ply", visualize=False):
        # ——————————————————————————————————————————————
        # 1. 读取点云
        # ——————————————————————————————————————————————
        pcd = o3d.io.read_point_cloud(input_path)
        points = np.asarray(pcd.points)
        self.original_point_count = len(points)

        # 计算平均点间距
        diffs = points[1:] - points[:-1]
        self.avg_point_distance = np.mean(np.linalg.norm(diffs, axis=1))
        print("平均点间距：", self.avg_point_distance)

        # ——————————————————————————————————————————————
        # 2. 计算所有点在 XY 平面上的体素索引
        # ——————————————————————————————————————————————
        xmin, ymin = points[:, 0].min(), points[:, 1].min()
        grid = dict()

        for idx, (x, y, z) in enumerate(points):
            i = int(np.floor((x - xmin) / self.voxel_size))
            j = int(np.floor((y - ymin) / self.voxel_size))
            key = (i, j)

            if key not in grid:
                grid[key] = []
            grid[key].append((z, idx))

        # ——————————————————————————————————————————————
        # 3. 在每个单元中挑选出 z 最小的点
        # ——————————————————————————————————————————————
        filtered_indices = []
        for cell_pts in grid.values():
            min_z, min_idx = min(cell_pts, key=lambda x: x[0])
            filtered_indices.append(min_idx)

        # ——————————————————————————————————————————————
        # 4. 二次过滤：去除残余孤立噪声
        # ——————————————————————————————————————————————
        filtered_points = points[filtered_indices]
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        filtered_xy = np.asarray(temp_pcd.points)[:, :2]
        kdtree_2d = cKDTree(filtered_xy)

        r = self.voxel_size * self.radius_multiplier
        keep_mask = np.ones(len(filtered_indices), dtype=bool)

        for local_idx, pt_idx in enumerate(filtered_indices):
            x, y, z = points[pt_idx]
            neighbors = kdtree_2d.query_ball_point([x, y], r)

            for nb_local in neighbors:
                if nb_local == local_idx:
                    continue
                nb_global_idx = filtered_indices[nb_local]
                if points[nb_global_idx][2] < z:
                    keep_mask[local_idx] = False
                    break

        final_indices = np.array(filtered_indices)[keep_mask]
        self.filtered_point_count = len(final_indices)

        # ——————————————————————————————————————————————
        # 5. 构造并保存最终点云
        # ——————————————————————————————————————————————
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(points[final_indices])

        if output_path:
            o3d.io.write_point_cloud(output_path, final_pcd)

        # 打印统计信息
        print("原始点云点数：", self.original_point_count)
        print("初次体素最低层保留点数：", len(filtered_indices))
        print("二次半径过滤后保留点数：", self.filtered_point_count)

        # 可视化
        if visualize:
            o3d.visualization.draw_geometries([pcd], window_name="原始融合点云")
            o3d.visualization.draw_geometries([final_pcd], window_name="过滤后仅保留最低Z层")

        return final_pcd


# 使用示例
if __name__ == "__main__":
    # 创建过滤器实例
    filter = ZLayerPointCloudFilter(voxel_size=16.0, radius_multiplier=1.5)

    # 执行过滤
    filtered_pcd = filter.filter(
        input_path="Y+8_X0000_Y0008_Z-345_172933.ply",
        output_path="filtered.ply",
        visualize=True
    )