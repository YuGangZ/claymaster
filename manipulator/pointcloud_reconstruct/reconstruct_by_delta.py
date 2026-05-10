# reconstruct_by_delta.py
import cv2
import numpy as np
import yaml
import time
import os
import sys
sys.path.append(r"C:\Users\23142\Desktop\project\perception\pointcloud_reconstruct")
from gs_reconstruct import Reconstructor
from scipy.ndimage import binary_erosion


def height2pointcloud(H, ppmm):
    """
    将高度图 H 转换为以米为单位的点云，并将图像中心移至原点。
    :param H: np.ndarray (H, W) 高度图（像素单位）
    :param ppmm: float 每毫米像素数
    :return: np.ndarray (N, 3) 点云（米单位）
    """
    h, w = H.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    xx = (xx - w/2 + 0.5) / ppmm
    yy = (yy - h/2 + 0.5) / ppmm
    zz = H / ppmm
    pts = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    return pts


def erode_contact_mask(C):
    """腐蚀接触掩码以增强鲁棒性"""
    erode_size = max(C.shape[0] // 24, 3)
    structure = np.ones((erode_size, erode_size), dtype=bool)
    return binary_erosion(C, structure=structure)

def calculate_centroid(boolean_image):
    """计算给定二维布尔图像中所有True点的质心。"""
    y_coords, x_coords = np.where(boolean_image)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    x_centroid = np.mean(x_coords)
    y_centroid = np.mean(y_coords)
    return x_centroid, y_centroid


class ContactReconstructor:
    def __init__(
        self,
        cam_id=0,
        config_path=r"C:\Users\23142\Desktop\project\perception\pointcloud_reconstruct\configs/mygelsight.yaml",
        calib_model=r"C:\Users\23142\Desktop\project\perception\pointcloud_reconstruct\model/nnmodel_0512_100.pth",
    ):
        # 加载配置
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        # 读取参数
        self.ppmm = self.config["ppmm"]
        self.crop_y1, self.crop_y2 = self.config["crop_y1"], self.config["crop_y2"]
        self.crop_x1, self.crop_x2 = self.config["crop_x1"], self.config["crop_x2"]
        self.save_h, self.save_w = self.config["imgh"], self.config["imgw"]

        # 摄像头初始化
        self.cap = cv2.VideoCapture(cam_id)
        # if not self.cap.isOpened():
        #     print(f"无法打开摄像头 {cam_id}，尝试默认后端...")
        #     self.cap = cv2.VideoCapture(cam_id)  # 回退到默认后端
        #
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {cam_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["raw_imgw"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["raw_imgh"])
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 重建模块
        self.recon = Reconstructor(calib_model, device="cpu")
        self._init_background()

        # 状态
        self.current_frame = None
        self.latest_pc = None
        self.latest_C = None
        self.running = True
        self.initial_centroid_px = None
        self.current_centroid_px = None

    def _init_background(self):
        bgs = []
        for _ in range(70):
            ret, frame = self.cap.read()
            if not ret:
                continue
            bgs.append(self._preprocess(frame))
        bg = np.median(bgs, axis=0).astype(np.uint8)
        self.recon.load_bg(bg)
        cv2.imwrite("background.png", bg)
        print("背景图已保存为 background.png")

    def _preprocess(self, frame):
        mtx = np.array(self.config["camera_matrix"])
        dist = np.array(self.config["dist_coeffs"])
        und = cv2.undistort(frame, mtx, dist)
        crop = und[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]
        return cv2.resize(crop, (self.save_w, self.save_h), interpolation=cv2.INTER_AREA)

    def update(self):
        ret, frame = self.cap.read()
        # cv2.imshow("Raw Frame", frame)
        if not ret:
            print("摄像头读取失败，检查连接或权限")
            return
        proc = self._preprocess(frame)
        self.current_frame = proc
        cv2.imshow("Tactile View", proc)

        _, H, C = self.recon.get_surface_info(proc, self.ppmm)
        if H is not None:
            # 缓存接触掩膜用于后续保存
            self.latest_C = C  # 新增行

            # C_eroded = erode_contact_mask(C)
            centroid = calculate_centroid(C)
            if centroid is not None:             
                # 记录初始质心（第一次进入）         
                if self.initial_centroid_px is None:             
                    self.initial_centroid_px = centroid         
                self.current_centroid_px = centroid

                all_pts = height2pointcloud(H, self.ppmm)
                self.latest_pc = all_pts[C.ravel()]
        key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     self.running = False


    def save_data(self, prefix):
        """保存触觉图像和点云数据"""
        # 图像
        imgp = f"{prefix}_raw.png"
        cv2.imwrite(imgp, self.current_frame)
        # 点云
        plyp = None
        if self.latest_pc is not None:
            import open3d as o3d
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(self.latest_pc)
            plyp = f"{prefix}.ply"
            o3d.io.write_point_cloud(plyp, pc)

        # === 新增：保存接触掩膜图像 ===
        if hasattr(self, 'latest_C') and self.latest_C is not None:
            maskp = f"{prefix}_mask.png"
            # 将二值掩膜转换为0-255图像
            mask_img = (self.latest_C * 255).astype(np.uint8)
            cv2.imwrite(maskp, mask_img)

        # 保存质心坐标         
        if self.current_centroid_px is not None:             
            cx, cy = self.current_centroid_px             
            centroid_file = f"{prefix}_centroid.txt"             
            with open(centroid_file, 'w') as f:                 
                f.write(f"{cx:.2f},{cy:.2f}\n")
        print(f"Saved: {imgp}", f", {plyp}" if plyp else "", f", centroid at {self.current_centroid_px}")


    def run(self):
        try:
            while self.running:
                self.update()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Resources released.")


    def cleanup(self):
        """确保正确释放资源"""
        print("释放摄像头资源...")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.running = False
        print("资源释放完成")

if __name__ == "__main__":
    sensor = ContactReconstructor()
    sensor.run()