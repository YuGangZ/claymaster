# main_control.py
import sys
import time
import cv2
import keyboard
import threading
import os
import re
import queue
import numpy as np
from delta_api import *
from admittance_control import AdmittanceController
from manipulator.pointcloud_reconstruct.reconstruct_by_delta import ContactReconstructor
from registration import PointCloudFuser
from manipulator.pointcloud_reconstruct.point_cloud_filter import ZLayerPointCloudFilter
from manipulator.object_estimation import hierarchical_ems, save_superquadrics
from expert import ExpertSystemInterface
import open3d as o3d


class RobotArmController:
    # 控制参数常量
    Z_LOWER_LIMIT = -350.0  # Z轴下限
    CONTROL_INTERVAL = 0.01  # 控制周期(s)
    MAX_SCAN_WAIT_TIME = 20.0  # 最大扫描等待时间(s)

    # 系统状态枚举
    class State:
        IDLE = 0
        SCANNING = 2
        EMERGENCY_STOP = 3
        PROCESSING_DATA = 5  # 数据处理状态
        EXPERT_ANALYSIS = 6

    # 机械臂状态跟踪器
    class ArmStatus:
        def __init__(self):
            self.position_reached = threading.Event()
            self.target_position = None
            self.position_tolerance = 0.5  # mm
            self.last_update_time = time.time()

        def update_position(self, x, y, z):
            current_time = time.time()
            # 避免过于频繁的更新检查
            if current_time - self.last_update_time < 0.05:
                return

            self.last_update_time = current_time
            if self.target_position and self.position_reached.is_set() is False:
                dx = abs(x - self.target_position[0])
                dy = abs(y - self.target_position[1])
                dz = abs(z - self.target_position[2])
                if dx < self.position_tolerance and dy < self.position_tolerance and dz < self.position_tolerance:
                    self.position_reached.set()

    def __init__(self, reconstructor=None, port='COM7', baudrate=115200):
        self.port = port
        self.speed = 0.4
        self.zero_z = -281
        self.baudrate = baudrate
        self.running = False
        self.current_x = 0
        self.current_y = 0
        self.current_z = -281
        self.shaping_complete = False  # 塑形完成标志
        self.fused_pointcloud = None  # 存储融合后的点云
        self.filtered_pointcloud = None  # 存储滤波后的点云
        self.scaled_pointcloud = None  # 存储缩放后的点云
        self.superquadrics = None  # 存储超二次曲面参数
        self.current_scan_folder = None  # 当前扫描文件夹
        self.loop_index = 1  # 循环索引计数器

        # 线程同步机制
        self.lock = threading.Lock()
        self.state_cond = threading.Condition(self.lock)
        self.system_state = self.State.IDLE

        # 点云处理
        self.reconstructor = reconstructor
        self.pointcloud_queue = queue.Queue(maxsize=5)

        # 视触觉传感器相关
        self.contact_mask = None  # 最新的掩膜图像
        self.last_mask_update = time.time()
        self.contact_area_threshold = 5000  # 掩膜像素面积阈值

        # 初始化线程
        self.pointcloud_thread = threading.Thread(target=self._pointcloud_processing_loop, daemon=True)
        self.mask_thread = threading.Thread(target=self._mask_update_loop, daemon=True)  # 掩膜更新线程

        # 指令队列
        self.command_queue = queue.Queue()
        self.command_thread = threading.Thread(target=self._command_execution_loop, daemon=True)

        if reconstructor:
            self.sensor_thread = threading.Thread(
                target=self.reconstructor.run,
                daemon=True
            )
            self.sensor_thread.start()
            print("触觉传感器线程已启动")

        # 状态跟踪器
        self.arm_status = self.ArmStatus()

        # 运动控制锁和计时器
        self.motion_lock = threading.Lock()  # 运动指令互斥锁
        self.last_motion_time = 0  # 上次运动指令时间
        self.MIN_MOTION_INTERVAL = 0.3  # 最小运动指令间隔(s)

        # 共享坐标系统
        self.shared_target_x = None  # 由扫描流程设置的X坐标
        self.shared_target_y = None  # 由扫描流程设置的Y坐标
        self.shared_target_z = None  # 由导纳控制设置的Z坐标

        # 扫描点完成计数器
        self.scan_points_completed = 0
        self.total_scan_points = 4  # 默认4个扫描点

        # 专家系统接口
        self.expert_interface = ExpertSystemInterface(
            reconstructor=self.reconstructor,
            robot_arm=self
        )

        # 专家专用的导纳控制器 (用于按压操作)
        self.expert_admittance_ctrl = AdmittanceController(
            Kd=0.1,
            Bd=0.8,
            Md=0.08,
            target_force=21.9,
            init_position=self.zero_z,
            target_position=-335.0,
            max_vel=5.0,
            max_acc=20.0
        )

        # 扫描专用的导纳控制器 (用于保持无按压状态)
        self.scan_admittance_ctrl = AdmittanceController(
            Kd=0.05,  # 较小的刚度
            Bd=0.5,  # 适中的阻尼
            Md=0.05,  # 较小的质量
            target_force=0.5,  # 很小的目标力，仅保持接触
            init_position=self.zero_z,
            target_position=self.zero_z,
            max_vel=2.0,  # 较慢的速度
            max_acc=10.0
        )

        # 导纳控制线程相关变量
        self.admittance_running = threading.Event()
        self.admittance_target_z = self.zero_z  # 导纳控制计算的目标Z
        self.admittance_target_z_lock = threading.Lock()

        # 启动导纳控制线程
        self.admittance_thread = threading.Thread(target=self._admittance_control_loop, daemon=True)
        self.admittance_thread.start()

    # ================== 公共方法 ==================
    def start(self):
        """启动系统"""
        self.running = True
        # 创建总数据目录
        self.base_data_dir = "control_data"
        os.makedirs(self.base_data_dir, exist_ok=True)

        # 初始化机械臂
        open_port(self.port, self.baudrate, self._status_callback)
        ping(1)
        set_speed_rate(self.speed)
        set_zero_position(0, 0, self.zero_z)
        move_zero()
        time.sleep(2)
        # 初始化共享坐标为当前位置
        with self.lock:
            self.shared_target_x = self.current_x
            self.shared_target_y = self.current_y
            self.shared_target_z = self.current_z
        set_control_signal(1)

        # 启动线程
        self.pointcloud_thread.start()
        self.mask_thread.start()  # 启动掩膜更新线程
        self.command_thread.start()

        # 用户交互线程
        self.keyboard_thread = threading.Thread(target=self._keyboard_handler)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

        print("系统初始化完成，等待用户输入...")
        print("操作指南:\n M - 输入目标坐标\n G - 启动扫描流程\n S - 实时可视化开关\n ESC - 紧急停止")

    def cleanup(self):
        """正常关闭"""
        print("\n正在关闭系统...")
        self.running = False
        self.admittance_running.clear()  # 停止导纳控制线程

        # 等待线程结束
        self.set_system_state(self.State.IDLE)

        # 通知点云线程退出
        self.pointcloud_queue.put("SHUTDOWN")
        # 🔽 通知触觉传感器线程退出
        if self.reconstructor:
            self.reconstructor.running = False
            self.reconstructor.cleanup()
        set_control_signal(5)
        move_zero()
        close_port()

        print("系统已安全关闭")

    def emergency_stop(self):
        """紧急停止"""
        print("\n!!! 紧急停止 !!!")
        with self.state_cond:
            self.system_state = self.State.EMERGENCY_STOP
            self.state_cond.notify_all()

        self.running = False
        self.admittance_running.clear()  # 停止导纳控制线程
        set_control_signal(5)
        move_zero()
        close_port()

        # 清空处理队列
        while not self.pointcloud_queue.empty():
            try:
                self.pointcloud_queue.get_nowait()
            except queue.Empty:
                break

        # 通知点云线程退出
        self.pointcloud_queue.put("SHUTDOWN")

        if self.reconstructor:
            self.reconstructor.cleanup()

    # ================== 状态控制 ==================
    def set_system_state(self, new_state):
        """线程安全的状态设置方法"""
        with self.state_cond:
            # 状态转换检查
            if self.system_state == self.State.EMERGENCY_STOP:
                return

            # 状态转换规则
            valid_transitions = {
                self.State.IDLE: [self.State.SCANNING],
                self.State.SCANNING: [self.State.PROCESSING_DATA],
                self.State.PROCESSING_DATA: [self.State.EXPERT_ANALYSIS],
                self.State.EXPERT_ANALYSIS: [self.State.IDLE, self.State.SCANNING],
            }

            # 检查是否为有效转换
            if new_state not in valid_transitions.get(self.system_state, []):
                print(f"无效状态转换: {self.system_state} -> {new_state}")
                return

            # 状态转换日志
            state_names = {
                0: "IDLE", 2: "SCANNING",
                3: "EMERGENCY_STOP", 5: "PROCESSING_DATA", 6: "EXPERT_ANALYSIS"
            }
            print(
                f"状态变更: {state_names.get(self.system_state, 'UNKNOWN')} -> {state_names.get(new_state, 'UNKNOWN')}")

            self.system_state = new_state
            self.state_cond.notify_all()

    # ================== 运动控制 ==================
    def safe_linear_movement(self, x, y, z):
        """线程安全的运动指令调用"""
        current_time = time.time()
        # 检查指令间隔
        if current_time - self.last_motion_time < self.MIN_MOTION_INTERVAL:
            return False  # 跳过过频指令

        with self.motion_lock:
            try:
                # 记录目标位置
                self.arm_status.target_position = (x, y, z)
                self.arm_status.position_reached.clear()

                # 执行运动指令
                linear_movement(x, y, z)
                self.last_motion_time = current_time
                return True
            except Exception as e:
                print(f"运动指令失败: {str(e)}")
                return False

    def _execute_move_command(self, target_pos):
        """执行移动命令"""
        x, y, z = target_pos
        self.safe_linear_movement(x, y, z)

    # ================== 掩膜更新循环 ==================
    def _mask_update_loop(self):
        """掩膜图像更新循环"""
        print("掩膜更新线程启动")

        # 等待传感器准备好
        sensor_ready = False
        while self.running and not sensor_ready:
            try:
                if hasattr(self.reconstructor, 'latest_C') and self.reconstructor.latest_C is not None:
                    sensor_ready = True
                    print("掩膜更新: 传感器已就绪")
            except AttributeError:
                pass
            if not sensor_ready:
                print("掩膜更新: 等待传感器初始化...")
                time.sleep(0.5)

        last_update = time.time()
        update_interval = 1.0 / 30  # 降低更新频率以减少负载

        while self.running:
            try:
                # 检查传感器是否仍然可用
                if not hasattr(self.reconstructor, 'get_mask_image'):
                    print("掩膜更新: 传感器方法不可用")
                    time.sleep(1.0)
                    continue

                # 获取掩膜
                mask_image = self.reconstructor.get_mask_image()
                if mask_image is not None:
                    with self.lock:
                        self.contact_mask = mask_image
                        self.last_mask_update = time.time()
                    # print(f"掩膜更新成功，面积: {np.sum(mask_image)} 像素")
                else:
                    print("警告：从传感器获取掩膜失败")

                # 控制更新频率
                next_update = last_update + update_interval
                sleep_time = max(0.0, next_update - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_update = time.time()

            except Exception as e:
                print(f"掩膜更新错误: {str(e)}")
                time.sleep(0.5)

    def _admittance_control_loop(self):
        """导纳控制线程 - 持续运行，计算Z轴调整量（基于图像掩膜）"""
        last_update = time.time()

        while self.running:
            try:
                # 精确控制更新频率
                current_time = time.time()
                elapsed = current_time - last_update
                if elapsed < self.CONTROL_INTERVAL:
                    time.sleep(self.CONTROL_INTERVAL - elapsed)
                    current_time = time.time()

                dt = current_time - last_update
                last_update = current_time

                # 获取当前掩膜和位置数据
                with self.lock:
                    mask = self.contact_mask
                    mask_time = current_time - self.last_mask_update
                    current_z = self.current_z

                if mask_time > 0.1:
                    print(f"警告：掩膜更新延迟 {mask_time:.3f}s")
                    continue

                # 检查掩膜是否有效
                if mask is None:
                    print("警告：未获取到掩膜图像")
                    continue

                # 计算掩膜区域大小
                contact_area = np.sum(mask)  # 掩膜中非零像素数量
                # print(f"当前接触面积: {contact_area} 像素")

                # 使用接触面积作为力的替代指标
                simulated_force = contact_area / 2000.0  # 示例：每100像素对应1N力

                # 更新导纳控制器
                new_z = self.scan_admittance_ctrl.update(
                    current_force=simulated_force,
                    target_force=0.5,  # 很小的目标力，仅保持接触
                    dt=dt
                )

                # 安全限位
                new_z = max(new_z, self.Z_LOWER_LIMIT)

                # 更新导纳控制计算的目标Z
                with self.admittance_target_z_lock:
                    self.admittance_target_z = new_z

            except Exception as e:
                print(f"导纳控制线程异常: {str(e)}")
                time.sleep(0.1)

    def move_with_admittance_control(self, target_x, target_y, start_z, step_size=1.0):
        """
        带导纳控制的移动函数
        从当前位置移动到目标XY位置，每隔step_size使用导纳控制调整Z
        """
        # 获取当前位置
        with self.lock:
            current_x, current_y, current_z = self.current_x, self.current_y, self.current_z

        # 计算移动方向和距离
        dx = target_x - current_x
        dy = target_y - current_y
        distance = np.sqrt(dx ** 2 + dy ** 2)

        if distance == 0:
            return current_z  # 已经在目标位置

        # 计算单位方向向量
        dir_x = dx / distance
        dir_y = dy / distance

        # 计算步数
        num_steps = int(distance / step_size)

        # 初始化导纳控制器
        self.scan_admittance_ctrl.init_position = start_z
        self.scan_admittance_ctrl.target_position = start_z

        # 逐步移动并使用导纳控制调整Z
        for step in range(num_steps + 1):  # +1 确保到达目标点
            # 计算当前步的XY位置
            progress = min(step * step_size, distance)
            step_x = current_x + dir_x * progress
            step_y = current_y + dir_y * progress

            # 获取导纳控制计算的目标Z位置
            with self.admittance_target_z_lock:
                target_z = self.admittance_target_z

            # 移动到当前步的位置
            self.safe_linear_movement(step_x, step_y, target_z)

            # 短暂等待确保位置稳定
            time.sleep(0.1)

            # 获取当前掩膜面积用于状态显示
            with self.lock:
                mask = self.contact_mask
                contact_area = np.sum(mask) if mask is not None else 0

            # 打印状态信息
            print(f"[导纳移动] 步骤 {step}/{num_steps} | "
                  f"X={step_x:.1f} Y={step_y:.1f} Z={target_z:.1f} | "
                  f"接触面积={contact_area} 像素")

        # 确保最终到达目标点
        self.safe_linear_movement(target_x, target_y, target_z)
        time.sleep(0.2)  # 等待位置稳定

        return target_z

    # ================== 扫描流程 ==================
    def start_scan_sequence(self):
        """启动扫描流程 - 实现'一去一回'采集方式"""
        with self.state_cond:
            if self.system_state != self.State.IDLE:
                print("警告：系统忙，无法启动新扫描")
                return
            self.system_state = self.State.SCANNING
            # 重置扫描点计数器
            self.scan_points_completed = 0

        try:
            # 创建当前循环目录
            loop_dir = os.path.join(self.base_data_dir, f"loop{self.loop_index}")
            os.makedirs(loop_dir, exist_ok=True)
            print(f"创建循环目录: {loop_dir}")

            # 获取当前位置作为原点
            with self.lock:
                origin_x, origin_y, origin_z = self.current_x, self.current_y, self.current_z

            # 设置共享目标坐标
            with self.lock:
                self.shared_target_x, self.shared_target_y = origin_x, origin_y

            # 定义扫描路径
            directions = [
                (8, 0, 'X+8'), (-8, 0, 'X-8'),
                (0, 8, 'Y+8'), (0, -8, 'Y-8')
            ]
            self.total_scan_points = len(directions) + 1  # 4个方向点+1个中心点

            # 1. 中心点扫描
            print("开始中心点扫描")
            self.command_queue.put(("SCAN_POINT", (loop_dir, (origin_x, origin_y), 'Center')))

            # 2. 方向点扫描（每次扫描后返回中心点）
            for dx, dy, desc in directions:
                target_x = origin_x + dx
                target_y = origin_y + dy

                # 使用导纳控制移动到方向点
                self.command_queue.put(("MOVE_WITH_ADMITTANCE", (target_x, target_y, origin_z, desc)))

                # 扫描方向点
                self.command_queue.put(("SCAN_POINT", (loop_dir, (target_x, target_y), desc)))

                # 使用导纳控制返回中心点
                self.command_queue.put(("MOVE_WITH_ADMITTANCE", (origin_x, origin_y, origin_z, f"Return from {desc}")))

            # 3. 所有点扫描完成后，保持在中心点并立即启动点云处理
            self.command_queue.join()  # 等待队列清空
            self.pointcloud_queue.put(loop_dir)
            print(f"所有扫描点完成，启动点云处理: {loop_dir}")
            self.set_system_state(self.State.PROCESSING_DATA)

            print(f"扫描任务已启动，共{self.total_scan_points}个扫描点")

        except Exception as e:
            print(f"扫描流程异常: {str(e)}")
            self.set_system_state(self.State.IDLE)

    def _execute_move_with_admittance(self, task_data):
        """执行带导纳控制的移动"""
        target_x, target_y, start_z, desc = task_data

        print(f"[导纳移动] 开始移动: {desc}")
        print(f"[导纳移动] 从当前位置移动到: X={target_x:.1f}, Y={target_y:.1f}")

        # 执行带导纳控制的移动
        final_z = self.move_with_admittance_control(target_x, target_y, start_z)

        print(f"[导纳移动] 移动完成: {desc}")
        print(f"[导纳移动] 最终位置: X={target_x:.1f}, Y={target_y:.1f}, Z={final_z:.1f}")

    def _execute_scan_point(self, task_data):
        """扫描点执行方法"""
        folder, target_xy, desc = task_data
        target_x, target_y = target_xy

        print(f"开始扫描点: {desc}")

        # 设置目标XY（Z保持不变）
        with self.lock:
            self.shared_target_x, self.shared_target_y = target_x, target_y
            current_z = self.current_z

        # 短暂等待确保位置稳定
        time.sleep(0.5)

        # 直接保存数据
        self.save_scan_data(folder, target_x, target_y, current_z, desc)
        self.scan_points_completed += 1
        print(f"扫描点完成 ({self.scan_points_completed}/{self.total_scan_points})")

    def save_scan_data(self, folder, x, y, z, direction):
        """保存扫描数据"""
        # 将浮点坐标转为整数
        x_int = int(round(x))
        y_int = int(round(y))
        z_int = int(round(z))
        # 生成标准化文件名: [方向]_X[坐标]_Y[坐标]_Z[坐标]_[时间]
        filename = f"{direction}_X{x_int:04d}_Y{y_int:04d}_Z{z_int:04d}_{time.strftime('%H%M%S')}"
        raw_path = os.path.join(folder, filename)

        if self.reconstructor:
            # 保存原始数据
            self.reconstructor.save_data(raw_path)
            print(f"扫描数据保存至: {raw_path}.*")

    # ================== 点云处理 ==================
    def _pointcloud_processing_loop(self):
        """点云处理线程"""
        while self.running:
            try:
                folder = self.pointcloud_queue.get(timeout=1.0)
                if folder == "SHUTDOWN":
                    break
                # 设置当前扫描文件夹
                self.current_scan_folder = folder
                print(f"开始处理扫描数据: {folder}")

                # 1. 点云融合
                print(f"开始点云融合: {folder}")
                fuser = PointCloudFuser(folder)
                fused_pcd = fuser.fuse_clouds()
                fused_path = os.path.join(folder, "fused.ply")
                o3d.io.write_point_cloud(fused_path, fused_pcd)
                self.fused_pointcloud = fused_pcd
                print(f"点云融合完成，保存至: {fused_path}")

                # 2. 点云滤波
                print(f"开始点云滤波: {folder}")
                filter = ZLayerPointCloudFilter(voxel_size=16.0, radius_multiplier=1.5)
                filtered_path = os.path.join(folder, "filtered.ply")
                filtered_pcd = filter.filter(fused_path, filtered_path, visualize=False)
                self.filtered_pointcloud = filtered_pcd
                print(f"点云滤波完成，保存至: {filtered_path}")

                # === 点云缩放 ===
                # 从Open3D点云转换为numpy数组
                pc_points = np.asarray(filtered_pcd.points)

                # 缩放点云，初始点云体积太大
                scale_factor = 0.001
                scaled_points = pc_points * scale_factor  # 单位毫米，可理解为从微米到毫米，主要原因是视触觉传感器点云尺度有误

                # 创建缩放后的点云对象
                scaled_pcd = o3d.geometry.PointCloud()
                scaled_pcd.points = o3d.utility.Vector3dVector(scaled_points)
                self.scaled_pointcloud = scaled_pcd
                # 保存缩放后的点云用于调试
                scaled_path = os.path.join(folder, "scaled.ply")
                o3d.io.write_point_cloud(scaled_path, scaled_pcd)
                print(f"缩放后点云保存至: {scaled_path}")

                # 3. 超二次曲面估计
                print(f"开始超二次曲面估计: {folder}")
                pc = np.asarray(scaled_pcd.points)
                # === 新增数据验证 ===
                # 移除NaN值
                pc = pc[~np.isnan(pc).any(axis=1)]

                # 检查点云有效性
                if len(pc) < 10:
                    print(f"错误：有效点数量不足 ({len(pc)}点)")
                    self.pointcloud_queue.task_done()
                    return

                # 检查点云是否共面（使用PCA）
                cov_matrix = np.cov(pc, rowvar=False)
                eigenvalues = np.linalg.eigvals(cov_matrix)
                if np.min(eigenvalues) < 1e-6:  # 最小特征值接近0
                    print("警告：点云可能共面，添加微小噪声")
                    pc += np.random.normal(0, 1e-5, pc.shape)
                # === 结束新增 ===
                segments, outliers, self.superquadrics = hierarchical_ems(
                    pc,
                    max_layer=1,
                    outlier_ratio=0.05,
                    eps=1.7,
                    min_points=60,
                    ems_kwargs={
                        'ToleranceEM': 1e-2,  # 增大收敛容差
                        'Sigma': 0.5,  # 增大初始sigma
                        'MaxOptiIterations': 5  # 增加优化迭代
                    }
                )

                # 4. 保存超二次曲面模型
                sq_path = os.path.join(folder, "superquadrics.stl")
                save_superquadrics(self.superquadrics, output_path=sq_path, merge=True)
                print(f"超二次曲面模型已保存至: {sq_path}")

                # 5. 点云处理完成后直接启动专家分析
                self.set_system_state(self.State.EXPERT_ANALYSIS)
                print(f"[状态] 点云处理完成，进入专家分析: {folder}")
                # 将专家分析加入命令队列
                self.command_queue.put(("EXPERT_ANALYSIS", None))
                print("[点云处理] 已添加专家分析任务到命令队列")
            except queue.Empty:
                pass
            except Exception as e:
                print(f"点云处理错误: {str(e)}")

    def _expert_analysis(self):
        print("\n[主控] === 开始专家分析 ===")

        # 1. 获取触觉数据（中心点数据）
        print("[主控] 获取中心点触觉数据...")
        tactile_img, contact_mask = self._get_center_tactile_data()

        if tactile_img is None or contact_mask is None:
            print("[主控] 错误: 无法获取中心点触觉数据!")
            return

        print(f"[主控] 获取到触觉图像: {tactile_img.shape if tactile_img is not None else '无'}")
        print(f"[主控] 获取到接触掩码: {contact_mask.shape if contact_mask is not None else '无'}")

        # 2. 准备专家系统输入
        stl_path = os.path.join(self.current_scan_folder, "superquadrics.stl")
        print(f"[主控] STL文件路径: {stl_path}")

        # 3. 更新专家系统感知数据
        print("[主控] 更新专家系统感知数据...")
        self.expert_interface.update_perception_data(
            pointcloud=np.asarray(self.scaled_pointcloud.points),
            mask=contact_mask,
            tactile_img=tactile_img,
            stl_file_path=stl_path
        )

        # 4. 检查塑形完成状态
        print("[主控] 检查塑形完成状态...")
        if self.expert_interface.check_shaping_complete():
            print("[主控] 专家系统判断塑形完成!")
            self.shaping_complete = True
            self.set_system_state(self.State.IDLE)
            return

        # 5. 生成按压指令
        print("[主控] 请求专家系统生成按压指令...")
        press_cmd = self.expert_interface.generate_press_command()

        if not press_cmd:
            print("[主控] 错误: 未获得有效的按压指令!")
            return

        # 6. 执行按压
        print("[主控] 执行按压指令...")
        self._execute_press_command(press_cmd)  # 执行按压并自动更新中心

        # 7. 启动新一轮扫描
        self.loop_index += 1
        self.set_system_state(self.State.IDLE)
        time.sleep(0.2)
        print("[主控] === 专家分析完成 ===")
        if not self.shaping_complete:
            print(f"形状未完成，准备启动第{self.loop_index}轮扫描")
            # 通过主线程调度避免死锁
            threading.Thread(
                target=self._delayed_start_scan,
                daemon=True
            ).start()

    def _delayed_start_scan(self):
        """延迟启动扫描避免死锁"""
        time.sleep(0.5)  # 确保状态切换完成
        if self.system_state == self.State.IDLE:
            self.start_scan_sequence()
        else:
            print(f"状态错误: 期望IDLE, 实际{self.system_state}")

    def _execute_press_command(self, command):
        print(f"[主控] 执行按压指令: {command}")
        # 获取当前位置（线程安全）
        with self.lock:
            center_x, center_y, center_z = self.current_x, self.current_y, self.current_z

        # 设置按压参数
        direction = np.array(command['direction'])
        translation = command['translation']
        press_depth = command['press_depth']

        # 计算目标XY位置
        target_x = center_x + direction[0] * translation
        target_y = center_y + direction[1] * translation

        # 移动到按压起始点（XY变化，Z保持）
        print(f"[力控按压] 移动到起始点: X={target_x:.1f}, Y={target_y:.1f}, Z={center_z:.1f}")
        if not self.safe_linear_movement(target_x, target_y, center_z):
            print("错误: 移动指令失败")
            return

        # 等待位置稳定
        time.sleep(0.5)

        # 螺旋运动参数
        circles = 2  # 螺旋圈数
        steps_per_circle = 8  # 每圈步数
        radius = 8.0  # 螺旋半径(mm)
        total_steps = circles * steps_per_circle

        # 创建导纳控制器实例（专用于按压过程）
        target_contact_area = 10000  # 目标接触面积（像素）
        adm_ctrl = AdmittanceController(
            Kd=0.5,
            Bd=0.5,
            Md=0.05,
            target_force=target_contact_area / 2000.0,  # 转换为模拟力
            init_position=center_z,
            target_position=center_z - press_depth,  # 允许下压深度
            max_vel=2.0,  # 较慢的速度保证稳定
            max_acc=10.0
        )
        print(f"[力控按压] 导纳控制目标接触面积: {target_contact_area} 像素")

        # 高频导纳控制线程
        running = threading.Event()
        running.set()
        target_z = center_z  # 初始目标Z位置
        target_z_lock = threading.Lock()

        def admittance_control_loop():
            """高频导纳控制线程 (100Hz)"""
            last_update = time.time()
            control_interval = 0.01  # 100Hz

            while running.is_set() and self.running:
                # 精确控制更新频率
                current_time = time.time()
                elapsed = current_time - last_update
                if elapsed < control_interval:
                    time.sleep(control_interval - elapsed)
                    current_time = time.time()

                dt = current_time - last_update
                last_update = current_time

                try:
                    # 获取当前掩膜和接触面积
                    with self.lock:
                        mask = self.contact_mask
                        mask_time = current_time - self.last_mask_update

                    if mask_time > 0.1 or mask is None:
                        continue

                    # 计算当前接触面积
                    contact_area = np.sum(mask)
                    simulated_force = contact_area / 2000.0  # 转换为模拟力

                    # 更新导纳控制器，使用接触面积作为控制输入
                    new_z = adm_ctrl.update(
                        current_force=simulated_force,
                        target_force=target_contact_area / 2000.0,
                        dt=dt
                    )

                    # 安全限位
                    new_z = max(new_z, self.Z_LOWER_LIMIT)

                    # 更新共享目标Z位置
                    with target_z_lock:
                        nonlocal target_z
                        target_z = new_z

                except Exception as e:
                    print(f"导纳控制异常: {str(e)}")

        # 启动导纳控制线程
        control_thread = threading.Thread(target=admittance_control_loop, daemon=True)
        control_thread.start()
        print("[力控按压] 启动高频导纳控制线程 (100Hz)")

        # 执行螺旋运动
        print(f"[力控按压] 开始螺旋导纳控制接触...")
        start_time = time.time()
        last_control_time = start_time

        for step in range(total_steps):
            # 极坐标计算当前螺旋角度和XY位置
            angle = 2 * np.pi * (step / steps_per_circle)
            current_x = target_x + radius * np.cos(angle)
            current_y = target_y + radius * np.sin(angle)

            # 获取导纳控制计算的目标Z位置
            with target_z_lock:
                current_target_z = target_z

            # 发送运动指令（螺旋轨迹+导纳调整的Z）
            self.safe_linear_movement(current_x, current_y, current_target_z)

            # 获取当前接触面积用于显示
            with self.lock:
                mask = self.contact_mask
                contact_area = np.sum(mask) if mask is not None else 0

            # 打印状态信息
            current_time = time.time()
            dt = current_time - last_control_time
            last_control_time = current_time

            print(f"[螺旋力控] 步骤{step + 1}/{total_steps} | "
                  f"X={current_x:.1f} Y={current_y:.1f} Z={current_target_z:.1f} | "
                  f"接触面积={contact_area} 像素 | "
                  f"dt={dt * 1000:.1f}ms")

            # 短暂等待
            time.sleep(0.05)  # 50ms/点

        # 停止导纳控制线程
        running.clear()
        control_thread.join(timeout=0.1)
        print("[力控按压] 导纳控制线程已停止")

        # 获取最终的Z位置
        with target_z_lock:
            final_z = target_z

        print(f"[力控按压] 按压完成，保持当前高度: Z={final_z:.1f}")

        # X、Y回正到按压起始点
        print(f"[力控按压] X、Y回正到起始点: X={target_x:.1f}, Y={target_y:.1f}")
        self.safe_linear_movement(target_x, target_y, final_z)

        # 等待位置稳定
        time.sleep(0.5)

        # 更新扫描中心（保持当前按压后的高度）
        new_center = (target_x, target_y, final_z)
        self.update_scan_center(new_center)
        print(f"[主控] 按压操作完成，更新扫描中心: {new_center}")

    def update_scan_center(self, new_center):
        self.shared_target_x, self.shared_target_y, self.shared_target_z = new_center
        print(f"更新扫描中心: {new_center}")

    def _get_center_tactile_data(self):
        """获取中心点触觉数据（原始图像和掩码图像）"""
        print("[主控] 获取中心点触觉数据...")
        if not self.current_scan_folder:
            print("警告：未设置当前扫描文件夹")
            return None, None

        print(f"[主控] 扫描文件夹内容: {os.listdir(self.current_scan_folder)}")

        # 查找所有以"Center"开头并以"_raw.png"或"_mask.png"结尾的文件
        center_files = []
        for file in os.listdir(self.current_scan_folder):
            if file.startswith("Center") and (file.endswith("_raw.png") or file.endswith("_mask.png")):
                # 提取基础文件名（去掉后缀）
                base_name = re.split(r'_(raw|mask)\.png', file)[0]
                if base_name not in center_files:
                    center_files.append(base_name)

        if not center_files:
            print("警告：未找到中心点扫描数据")
            return None, None

        print(f"[主控] 找到的中心点基础文件名: {center_files}")
        base_name = center_files[0]

        # 构建完整路径
        raw_path = os.path.join(self.current_scan_folder, f"{base_name}_raw.png")
        mask_path = os.path.join(self.current_scan_folder, f"{base_name}_mask.png")

        print(f"[主控] 原始图像路径: {raw_path}")
        print(f"[主控] 掩码图像路径: {mask_path}")

        # 读取图像
        tactile_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        contact_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if tactile_img is None:
            print(f"无法读取原始图像: {raw_path}")
        if contact_mask is None:
            print(f"无法读取掩码图像: {mask_path}")
        if tactile_img is None or contact_mask is None:
            return None, None

        print(f"[主控] 成功读取触觉图像: {tactile_img.shape}, 掩码图像: {contact_mask.shape}")
        return tactile_img, contact_mask

    # ================== 命令执行线程 ==================
    def _command_execution_loop(self):
        """专用的命令执行线程"""
        while self.running:
            try:
                # 从队列获取命令
                command = self.command_queue.get(timeout=0.1)

                if command[0] == "MOVE":
                    self._execute_move_command(command[1])
                elif command[0] == "SCAN_POINT":
                    self._execute_scan_point(command[1])
                elif command[0] == "EXPERT_ANALYSIS":
                    self._expert_analysis()
                elif command[0] == "MOVE_WITH_ADMITTANCE":
                    self._execute_move_with_admittance(command[1])

                self.command_queue.task_done()
            except queue.Empty:
                pass

    # ================== 回调函数 ==================
    def _status_callback(self, x, y, z, s1, s2, s3, precur, pretemp):
        """机械臂状态回调"""
        with self.lock:
            self.current_x, self.current_y, self.current_z = float(x), float(y), float(z)
            self.arm_status.update_position(float(x), float(y), float(z))
            self.global_info = f"X：{x}, Y：{y}, Z：{z}, S1：{s1}, S2：{s2}, S3：{s3}, 电流：{precur}mA, 温度：{pretemp}℃"

    # ================== 键盘处理 ==================
    def _keyboard_handler(self):
        """优化后的键盘处理线程"""
        print("操作指南:\n M - 输入目标坐标\n G - 启动扫描流程\n S - 实时可视化开关\n ESC - 紧急停止")

        # 按键状态跟踪
        key_states = {'esc': False, 's': False, 'm': False, 'g': False}

        while self.running:
            try:
                # 1. 使用更紧凑的轮询周期
                time.sleep(0.02)  # 20ms

                # 2. 非阻塞式按键检测
                for key in ['esc', 's', 'm', 'g']:
                    if keyboard.is_pressed(key):
                        if not key_states[key]:  # 首次检测到按下
                            key_states[key] = True

                            if key == 'esc':
                                self.emergency_stop()
                            elif key == 'm':
                                threading.Thread(
                                    target=self._manual_position_input,
                                    daemon=True
                                ).start()
                            elif key == 'g' and self.system_state == self.State.IDLE:
                                threading.Thread(
                                    target=self.start_scan_sequence,
                                    daemon=True
                                ).start()
                    else:
                        key_states[key] = False  # 重置状态

            except Exception as e:
                print(f"键盘处理错误: {str(e)}")

    def _manual_position_input(self):
        """手动输入目标坐标"""
        try:
            print("\n请输入目标坐标 (格式: X Y Z)")
            user_input = input("> ").strip()
            if not user_input:
                return

            # 解析坐标
            coords = user_input.split()
            if len(coords) != 3:
                print("错误：需要输入三个坐标值 (X Y Z)")
                return

            x = float(coords[0])
            y = float(coords[1])
            z = float(coords[2])

            # 检查Z坐标是否超出下限
            if z < self.Z_LOWER_LIMIT:
                print(f"警告：Z坐标 ({z}) 低于安全下限 ({self.Z_LOWER_LIMIT})，已自动调整")
                z = self.Z_LOWER_LIMIT

            # 移动到目标位置
            print(f"移动至: X={x}, Y={y}, Z={z}")
            self.safe_linear_movement(x, y, z)

            # 更新共享坐标
            with self.lock:
                self.shared_target_x = x
                self.shared_target_y = y
                self.shared_target_z = z

        except ValueError:
            print("错误：输入的坐标必须是数字")
        except Exception as e:
            print(f"坐标输入错误: {str(e)}")


if __name__ == "__main__":
    sensor = ContactReconstructor()
    controller = RobotArmController(sensor)
    try:
        controller.start()

        # 等待塑形完成
        while controller.running and not controller.shaping_complete:
            time.sleep(1)

        if controller.shaping_complete:
            print("塑形任务成功完成！")

    except KeyboardInterrupt:
        pass
    finally:
        controller.cleanup()