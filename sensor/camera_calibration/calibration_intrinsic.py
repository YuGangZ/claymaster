import cv2
import numpy as np
import glob
import os

# 校准参数
os.environ['DISPLAY'] = ':0'
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
w, h, l = 3, 5, 3  # 棋盘格尺寸和方块边长
path = r"C:\Users\23142\Desktop\perception\1_camera_calibration\chessboard_data"

# 准备棋盘格角点
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objp = objp * l
objpoints = []
imgpoints = []

# 读取图片
images = glob.glob(path + '/*.jpg')
# images = glob.glob(path + 'WIN_20250103_16_54_50_Pro.jpg')
if len(images) == 0:
    raise FileNotFoundError(f"未在路径 {path} 中找到图片，请检查路径！")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"无法加载图片: {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测棋盘角点
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if ret:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 640, 480)
        cv2.imshow('findCorners', img)
        cv2.waitKey(500)
    else:
        print(f"未能检测到角点: {fname}")

cv2.destroyAllWindows()

# 校准相机
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("相机内参矩阵：", mtx)
    print("畸变系数：", dist)
else:
    print("未能找到足够的棋盘角点进行相机校准，请检查图片和参数！")


# import cv2
# import numpy as np
# import glob
# import os
#
# # 参数配置
# CHECKERBOARD = (3, 5)  # 棋盘格实际内角点数量（行列数-1）
# SQUARE_SIZE = 3  # 棋盘格方块实际尺寸（单位：毫米）
# CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
# SUBPIX_WIN_SIZE = (5, 5)  # 亚像素细化窗口尺寸
# CALIB_FLAGS = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5  # 畸变模型配置
#
# # 准备棋盘格三维坐标
# objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE
#
# # 数据收集
# objpoints = []  # 三维点
# imgpoints = []  # 二维点
# valid_images = []  # 有效图像路径
# reproject_errors = []  # 重投影误差
# path = r"C:\Users\23142\Desktop\perception\1_camera_calibration\chessboard_data"
#
# # 读取并处理图像
# images = glob.glob(path + '/*.jpg')
# if not images:
#     raise FileNotFoundError("未找到标定图片！请检查路径")
#
# for fname in images:
#     img = cv2.imread(fname)
#     if img is None:
#         print(f"警告：无法读取图像 {fname}")
#         continue
#
#     # 预处理
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯降噪
#
#     # 角点检测
#     ret, corners = cv2.findChessboardCornersSB(
#         gray, CHECKERBOARD,
#         flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
#     )
#
#     if not ret:
#         print(f"未检测到角点：{os.path.basename(fname)}")
#         continue
#
#     # 亚像素优化
#     corners_refined = cv2.cornerSubPix(
#         gray, corners, SUBPIX_WIN_SIZE, (-1, -1),
#         criteria=CALIB_CRITERIA
#     )
#
#     # 收集数据
#     objpoints.append(objp)
#     imgpoints.append(corners_refined)
#     valid_images.append(fname)
#
#     # 可视化验证
#     vis = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners_refined, ret)
#     cv2.putText(vis, f"Points: {len(corners_refined)}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#     cv2.imshow('Calibration', vis)
#     if cv2.waitKey(500) == 27:  # ESC键退出
#         break
#
# cv2.destroyAllWindows()
#
# # 标定验证
# if len(objpoints) < 10:
#     raise ValueError(f"有效图像不足（{len(objpoints)}张），至少需要10张有效图像")
#
# # 执行标定
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
#     objpoints, imgpoints, gray.shape[::-1],
#     None, None, flags=CALIB_FLAGS, criteria=CALIB_CRITERIA
# )
#
# # 计算重投影误差
# mean_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(
#         objpoints[i], rvecs[i], tvecs[i], mtx, dist
#     )
#     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#     reproject_errors.append(error)
#     mean_error += error
#
# mean_error /= len(objpoints)
# print(f"\n标定结果：")
# print(f"内参矩阵：\n{np.round(mtx, 5)}")
# print(f"畸变系数：\n{np.round(dist, 5)}")
# print(f"平均重投影误差：{mean_error:.5f} 像素")
#
# # 误差分析
# threshold = 0.2  # 像素误差阈值
# good_samples = [i for i, e in enumerate(reproject_errors) if e <= threshold]
# print(f"\n有效图像：{len(good_samples)}/{len(objpoints)} (误差 ≤ {threshold}像素)")
#
# # 保存标定结果
# np.savez("camera_params.npz", mtx=mtx, dist=dist,
#         mean_error=mean_error, used_images=valid_images)
#
# # 验证标定效果
# test_img = cv2.imread(r"C:\Users\23142\Pictures\Camera Roll\WIN_20250505_12_19_11_Pro.jpg")
# undistorted = cv2.undistort(test_img, mtx, dist)
# cv2.imshow('raw', test_img)
# cv2.imshow('undistortion', undistorted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()