# expert.py
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
import time
import os
from is_sphere_stl import is_sphere_stl
from is_sphere_img import SphereContactDetector
from is_sphere_pc import evaluate_local_sphere_surface

# 常量定义
SPHERICITY_THRESHOLD = 0.92
DEVIATION_THRESHOLD = 3.0
MAX_CYCLES = 10
FIXED_TRANSLATION = 3.0  # 固定平移距离(mm)
FIXED_PRESS_DEPTH = 3.0  # 固定按压深度(mm)


class ExpertSystem:
    def __init__(self, initial_pointcloud, log_file='expert_system.log'):
        self.config = {
            'sphericity_threshold': SPHERICITY_THRESHOLD,
            'max_cycles': MAX_CYCLES
        }

        # 状态跟踪
        self.cycle_count = 0
        self.press_history = []
        self.log_file = log_file

        # 存储感知数据（明确触觉数据格式）
        self.current_pointcloud = initial_pointcloud
        self.current_mask = None  # 触觉掩膜 (二值图像)
        self.current_superquadrics = None
        self.current_tactile_img = None  # 触觉图像 (灰度图)
        self.stl_file_path = None

        # 初始化球体检测器
        self.sphere_contact_detector = SphereContactDetector()


    def log(self, message):
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')

        print(log_entry)

    def is_sphere(self):
        """综合三类数据判断是否为球体"""
        print(f"[ExpertSystem] 开始球体检测...")
        results = {
            'point_cloud': False,
            'stl': False,
            'image': False
        }

        # 1. STL模型检查
        if self.stl_file_path:
            print(f"[ExpertSystem] 检查STL模型: {self.stl_file_path}")
            results['stl'], _ = is_sphere_stl(self.stl_file_path)
            print(f"[ExpertSystem] STL模型球体检测结果: {results['stl']}")

        # 2. 点云局部球面检查
        print("[ExpertSystem] 检查点云球面...")
        pc_result = evaluate_local_sphere_surface(self.current_pointcloud)
        results['point_cloud'] = pc_result['is_sphere_surface'] and pc_result['confidence'] > 0.7
        print(f"[ExpertSystem] 点云球面检测结果: {results['point_cloud']} (置信度: {pc_result['confidence']:.2f})")

        # 3. 触觉图像检查
        if self.current_tactile_img is not None and self.current_mask is not None:
            print("[ExpertSystem] 检查触觉图像...")
            results['image'], _ = self.sphere_contact_detector.is_sphere_contact(self.current_mask)
            print(f"[ExpertSystem] 触觉图像球体检测结果: {results['image']}")
        else:
            print("[ExpertSystem] 警告: 缺少触觉图像或掩码数据")

        # 综合判断
        confirmations = sum(results.values())
        print(f"[ExpertSystem] 综合球体检测结果: {confirmations >= 2} (确认数: {confirmations}/3)")
        return confirmations >= 2

    def find_pressing_direction(self):
        """基于点云几何特征确定按压方向"""
        print("[ExpertSystem] 计算按压方向...")
        # 计算点云重心
        centroid = np.mean(self.current_pointcloud, axis=0)
        print(f"[ExpertSystem] 点云重心: {centroid}")

        # 找到最远点（假设为凸点）
        max_distance = -np.inf
        press_point = None

        for point in self.current_pointcloud:
            dist = np.linalg.norm(point - centroid)
            if dist > max_distance:
                max_distance = dist
                press_point = point

        if press_point is None:
            print("[ExpertSystem] 错误: 无法找到按压点")
            return None, None

        print(f"[ExpertSystem] 找到按压点: {press_point} (距离重心: {max_distance:.2f}mm)")

        # 从重心指向凸点
        press_direction = press_point - centroid
        press_direction /= np.linalg.norm(press_direction)
        print(f"[ExpertSystem] 按压方向向量: {press_direction}")

        return press_direction, press_point


    def check_termination(self):
        # 1. 球体判断
        if self.is_sphere():
            self.log("Termination: Sphere confirmed by multiple modalities")
            return True

        # 2. 循环次数检查
        if self.cycle_count >= self.config['max_cycles']:
            self.log(f"Termination: Max cycles reached ({self.cycle_count}/{self.config['max_cycles']})")
            return True

        return False


    def process_cycle(self, pointcloud, mask=None, tactile_img=None, stl_file_path=None):
        """处理单个感知-行动循环"""
        print("\n[ExpertSystem] === 开始新的处理周期 ===")
        # 更新当前感知数据
        self.current_pointcloud = pointcloud
        self.current_mask = mask
        self.current_tactile_img = tactile_img
        self.stl_file_path = stl_file_path
        self.log(f"New data received: points={len(pointcloud)}")

        # 记录额外数据
        if tactile_img is not None:
            self.log(f"Tactile image received: size={tactile_img.shape}")
        if stl_file_path is not None:
            self.log(f"STL file path received: {stl_file_path}")

        # 检查终止条件
        if self.check_termination():
            self.log("Cycle terminated - target shape achieved")
            return None

        # 确定按压方向
        press_direction, press_point = self.find_pressing_direction()
        if press_direction is None:
            self.log("No suitable pressing direction found")
            return None

        # 保存按压状态
        self.press_history.append((press_point, press_direction))
        self.cycle_count += 1

        # 返回按压指令
        press_command = {
            'direction': press_direction.tolist(),
            'translation': FIXED_TRANSLATION,
            'press_depth': FIXED_PRESS_DEPTH,
            'press_point': press_point.tolist()  # 添加按压点位置信息
        }

        self.log(
            f"Press command generated: direction={press_direction}, translation={FIXED_TRANSLATION}mm, press_depth={FIXED_PRESS_DEPTH}mm")

        print("[ExpertSystem] === 处理周期完成 ===")
        return press_command


# 与主控制系统的接口类
class ExpertSystemInterface:
    def __init__(self, reconstructor, robot_arm):
        self.reconstructor = reconstructor
        self.robot_arm = robot_arm
        self.expert_system = None
        self.current_pointcloud = None
        self.current_mask = None
        self.current_tactile_img = None
        self.stl_file_path = None


    def log(self, message):
        """记录日志"""
        print(f"[ExpertInterface] {message}")

    def update_perception_data(self, pointcloud, mask, tactile_img=None, stl_file_path=None):
        """更新感知数据"""
        print(f"[ExpertInterface] 更新感知数据: 点云点数={len(pointcloud)}")
        if tactile_img is not None:
            print(f"[ExpertInterface] 触觉图像尺寸: {tactile_img.shape}")
        if stl_file_path is not None:
            print(f"[ExpertInterface] STL文件路径: {stl_file_path}")

        self.current_pointcloud = pointcloud
        self.current_mask = mask
        self.current_tactile_img = tactile_img
        self.stl_file_path = stl_file_path

        # 如果是第一次接收数据，初始化专家系统（使用真实数据）
        if self.expert_system is None:
            print("[ExpertInterface] 使用真实点云初始化专家系统...")
            self.expert_system = ExpertSystem(initial_pointcloud=pointcloud)

    def generate_press_command(self):
        """生成按压指令"""
        if self.expert_system is None:
            self.log("Expert system not initialized")
            return None

        print("[ExpertInterface] 生成按压指令...")
        press_command = self.expert_system.process_cycle(
            self.current_pointcloud,
            self.current_mask,
            self.current_tactile_img,
            self.stl_file_path
        )

        if press_command:
            print(f"[ExpertInterface] 生成的按压指令: {press_command}")
        else:
            print("[ExpertInterface] 未生成按压指令")

        return press_command

    def check_shaping_complete(self):
        """检查塑形是否完成"""
        if self.expert_system is None:
            return False
        return self.expert_system.check_termination()