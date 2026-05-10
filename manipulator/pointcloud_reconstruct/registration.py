# test.py
import open3d as o3d
import numpy as np
import os
import re
from tqdm import tqdm
import sys



class PointCloudFuser:
    def __init__(self, scan_folder):
        self.scan_folder = scan_folder
        self.global_pcd = o3d.geometry.PointCloud()
        self.transform_cache = {}
        self.tool_offset = np.array([0, 0, -40])  # 传感器相对于末端的偏移
        self.initial_position = None
        self.initial_centroid = None  # 初始质心坐标
        self._validate_input()
        self.ppmm = 0.029

    def _parse_centroid(self, filename):
        """从质心文件解析坐标"""
        centroid_file = filename.replace('.ply', '_centroid.txt')
        path = os.path.join(self.scan_folder, centroid_file)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            content = f.read().strip()
            cx, cy = map(float, content.split(','))
            return (cx, cy)

    def _parse_position(self, filename):
        """从文件名解析机械臂位姿（支持浮点数和负数）"""
        pattern = r'X([-+]?\d+\.?\d*)_Y([-+]?\d+\.?\d*)_Z([-+]?\d+\.?\d*)'
        match = re.search(pattern, filename)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3))
            return (x, y, z)
        return None

    def _get_transform_matrix(self, position, filename):
        """生成包含旋转补偿的坐标变换矩阵，动态计算物体半径"""
        # 解析当前帧质心
        current_centroid = self._parse_centroid(filename)
        delta_mm = [0.0, 0.0]
        # 计算质心位移（像素单位）
        if self.initial_centroid and current_centroid:
            delta_px = np.array(current_centroid) - np.array(self.initial_centroid)
            # 转换为实际位移（毫米），考虑传感器坐标系到世界坐标系的转换
            delta_mm[0] = delta_px[0] * self.ppmm
            delta_mm[1] = delta_px[1] * self.ppmm * (-1)  # Y轴方向需要取反

            # 调试输出
            print(f"Centroid displacement: {delta_mm} mm")
        else:
            delta_mm = np.zeros(2)
        # 计算基础平移量
        translation = np.array(position) + self.tool_offset
        # 补偿物体位移：实际位移 = 机械臂位移 - 传感器同物体相对位移
        translation[:2] += delta_mm  # 仅补偿XY方向

        # 动态计算物体半径：从Z坐标推算 r = (z - (-360)) / 2
        # 注意：根据单位，若Z为毫米，此处r同样为毫米；可根据需求转换为米
        z = position[2]
        r = (z + 400) / 2

        # 初始化旋转矩阵
        rotation = np.eye(3)

        if self.initial_position is not None:
            dx = position[0] - self.initial_position[0]
            dy = position[1] - self.initial_position[1]
            ds = np.hypot(dx, dy)

            if ds > 1e-6:
                # 计算滚动角度 θ = ds / r
                theta = ds / r
                axis = np.array([dy, dx, 0.0]) / ds
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * theta)
                rotation = R

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform


    def _preprocess_pointcloud(self, pcd):
        """点云预处理（优化法线估计参数）"""
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05, max_nn=30))  # 调整搜索半径
        return pcd

    def _pairwise_registration(self, source, target):
        """ICP精配准（增加配准鲁棒性）"""
        reg = o3d.pipelines.registration.registration_icp(
            source, target, max_correlation_distance=0.01,  # 调整最大对应距离
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=500))  # 增加迭代次数
        return reg.transformation


    def _validate_input(self):
        """增强型输入验证"""
        pcd_files = [f for f in os.listdir(self.scan_folder) if f.endswith('.ply')]
        if not pcd_files:
            raise ValueError(f"No PLY files found in {self.scan_folder}")

        # 检查前三个文件
        for f in pcd_files[:3]:
            path = os.path.join(self.scan_folder, f)
            pcd = o3d.io.read_point_cloud(path)
            if len(pcd.points) == 0:
                raise ValueError(f"Empty point cloud in {f}")
            print(f"File {f} contains {len(pcd.points)} points")

    def _adaptive_key_frame_selection(self, files, min_interval=3, max_interval=10):
        """增强型关键帧选择（带缓存验证）"""
        # 仅选择已成功处理的文件
        valid_files = [f for f in files if f in self.transform_cache]

        # 检查有效文件数量
        if len(valid_files) < 2:
            raise RuntimeError(f"Insufficient valid frames: {len(valid_files)} available")

        # 计算位移量
        positions = [self._parse_position(f) for f in valid_files]
        displacements = [np.linalg.norm(np.array(p1) - np.array(p0))
                        for p0, p1 in zip(positions[:-1], positions[1:])]

        # 动态调整间隔
        avg_displacement = np.mean(displacements)
        interval = max(min_interval, min(max_interval, int(5 / avg_displacement)))

        # 返回关键帧并打印调试信息
        key_frames = valid_files[::interval]
        print(f"Selected {len(key_frames)} key frames from {len(valid_files)} valid files")
        print("Key frames:", key_frames)
        return key_frames

    def fuse_clouds(self):
        """增强型融合流程"""
        # 第一阶段：带调试输出的粗配准
        print("Stage 1: Coarse registration with debugging")
        pcd_files = sorted([f for f in os.listdir(self.scan_folder) if f.endswith('.ply')],
                        key=lambda x: re.search(r'_(\d{6})\.', x).group(1))

        for idx, f in enumerate(tqdm(pcd_files)):
            position = self._parse_position(f)
            if position is None:
                print(f"Skipping {f} due to parse failure")
                continue
            # 初始化基准坐标系
            if self.initial_position is None:
                self.initial_position = position
                # 获取初始质心坐标
                self.initial_centroid = self._parse_centroid(f)
                print(f"Initial centroid: {self.initial_centroid}")
            # 加载并记录原始点云信息
            try:
                raw_pcd = o3d.io.read_point_cloud(os.path.join(self.scan_folder, f))
                # print(f"Raw {f}: {len(raw_pcd.points)} points")
            except:
                print(f"Failed to load {f}")
                continue

            # 预处理并记录处理结果
            processed_pcd = self._preprocess_pointcloud(raw_pcd)
            print(f"Processed {f}: {len(processed_pcd.points)} points")
            # 生成并验证变换矩阵
            transform = self._get_transform_matrix(position, f)
            print(f"Transform matrix for {f}:\n{transform}")


            # —— 新增：首次记录 initial_position，并缓存本帧变换 ——
            if self.initial_position is None:
                self.initial_position = position
            self.transform_cache[f] = transform

            # 应用变换并合并
            processed_pcd.transform(transform)
            # self.global_pcd += processed_pcd
            self._icp_and_merge(processed_pcd)
            print(f"Global points after {f}: {len(self.global_pcd.points)}")


        # 第二阶段：自适应关键帧选择
        print("\nStage 2: Adaptive key frame selection")
        key_frames = self._adaptive_key_frame_selection(pcd_files)

        # 第三阶段：带安全验证的多分辨率 ICP
        print("\nStage 3: Multi-resolution ICP refinement")
        for i in tqdm(range(1, len(key_frames))):
            source_file = key_frames[i]
            target_file = key_frames[i - 1]

            # 缓存有效性验证
            if source_file not in self.transform_cache or target_file not in self.transform_cache:
                print(f"Skip {source_file} or {target_file}: not in transform cache")
                continue

            # —— 新增：加载并预处理点云对象 ——
            source_pcd = o3d.io.read_point_cloud(os.path.join(self.scan_folder, source_file))
            target_pcd = o3d.io.read_point_cloud(os.path.join(self.scan_folder, target_file))
            source_pcd = self._preprocess_pointcloud(source_pcd)
            target_pcd = self._preprocess_pointcloud(target_pcd)

            # —— 新增：应用粗配准变换 ——
            source_pcd.transform(self.transform_cache[source_file])
            target_pcd.transform(self.transform_cache[target_file])

            # 调用多分辨率 ICP
            delta_transform = self._multi_resolution_icp(source_pcd, target_pcd)
            self._update_transforms(source_file, delta_transform)

        # 最终融合与优化
        print("\nFinal fusion and optimization")
        self.global_pcd = o3d.geometry.PointCloud()
        for f in tqdm(pcd_files):
            pcd = o3d.io.read_point_cloud(os.path.join(self.scan_folder, f))
            pcd.transform(self.transform_cache[f])
            self.global_pcd += pcd

        # 全局优化和后处理
        # self.global_pcd = self._global_optimization()
        return self.global_pcd


    def _multi_resolution_icp(self, source, target, resolutions=None):
        """多分辨率ICP配准（增加法线检查）"""
        # 确保输入点云包含法线
        if resolutions is None:
            resolutions = [0.001, 0.0005, 0.0002]
        assert source.has_normals(), "Source point cloud missing normals"
        assert target.has_normals(), "Target point cloud missing normals"

        current_transform = np.eye(4)
        for radius in resolutions:
            # 降采样并保持法线
            source_down = source.voxel_down_sample(radius)
            target_down = target.voxel_down_sample(radius)

            # 重新估计法线（降采样后需要重新计算）
            source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius * 2, max_nn=30))
            target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius * 2, max_nn=30))

            reg = o3d.pipelines.registration.registration_icp(
                source_down, target_down, radius * 2,
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=100))
            current_transform = reg.transformation
        return current_transform

    def _global_optimization(self):
        labels = np.array(self.global_pcd.cluster_dbscan(eps=0.05, min_points=100))
        valid_idx = np.where(labels >= 0)[0]
        if len(valid_idx) > 0:
            self.global_pcd = self.global_pcd.select_by_index(valid_idx)
        # 否则保持原始点云，不剔除
        return self.global_pcd

    def _update_transforms(self, filename, delta_transform):
        """变换矩阵更新（增加变换矩阵验证）"""
        if np.linalg.det(delta_transform[:3, :3]) < 0.5:
            print(f"Warning: Invalid transformation for {filename}")
            return
        self.transform_cache[filename] = delta_transform @ self.transform_cache[filename]

    def save_result(self, output_path):
        """保存结果（增加压缩选项）"""
        o3d.io.write_point_cloud(output_path, self.global_pcd,
                                write_ascii=False, compressed=True)
        print(f"Final point cloud saved to {output_path}")

    # 在 class PointCloudFuser 中，新增一个方法：
    def _icp_and_merge(self, new_pcd):
        """
        将 new_pcd 与 self.global_pcd 做 ICP 精配准，并融合到 global_pcd。
        """
        if len(self.global_pcd.points) == 0:
            # 第一次直接赋值
            self.global_pcd = new_pcd
            return

        # 预处理
        source = new_pcd.voxel_down_sample(0.005)
        target = self.global_pcd.voxel_down_sample(0.005)
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.05, max_nn=30))
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.05, max_nn=30))

        # 做 ICP
        reg = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance=0.02,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        new_pcd.transform(reg.transformation)

        # 合并并体素下采样去重
        self.global_pcd += new_pcd
        self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=0.005)


if __name__ == "__main__":
    # 改进后的使用示例
    fuser = PointCloudFuser(r"C:\Users\23142\Desktop\delta_perception\delta_based\Scan_20250518-215741")
    fused_pcd = fuser.fuse_clouds()
    fuser.save_result("improved_fused_result.ply")

    # 增强可视化
    o3d.visualization.draw_geometries([fused_pcd],
                                    window_name="Fused Point Cloud",
                                    width=1920,
                                    height=1080,
                                    mesh_show_back_face=True)