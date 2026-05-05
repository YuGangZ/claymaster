# clean_rotation_data_fixed.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os


def wrap_angle_diff(current, next_angle):
    """计算周期性的角度差（处理-π到π的跳变）"""
    diff = next_angle - current
    diff = np.remainder(diff + np.pi, 2 * np.pi) - np.pi
    return diff


def clean_rotation_data_standalone():
    """
    基于已经清洗过尺度和形状的数据，进行旋转数据清洗
    不依赖元数据，直接对现有数据进行过滤
    """

    # 加载尺度和形状清洗后的数据
    X = np.load('training_data_X_clean.npy')
    y = np.load('training_data_y_clean.npy')

    print("=" * 60)
    print("基于尺度和形状清洗后的数据进行旋转数据清洗")
    print("=" * 60)
    print(f"输入数据: X形状={X.shape}, y形状={y.shape}")

    # 提取旋转参数索引
    rotation_indices = [8, 9, 10]  # euler_rx, euler_ry, euler_rz

    # 计算旋转变化量
    current_rotations = X[:, rotation_indices]
    next_rotations = y[:, rotation_indices]

    # 计算每个轴的旋转变化（考虑周期性）
    delta_rx = wrap_angle_diff(current_rotations[:, 0], next_rotations[:, 0])
    delta_ry = wrap_angle_diff(current_rotations[:, 1], next_rotations[:, 1])
    delta_rz = wrap_angle_diff(current_rotations[:, 2], next_rotations[:, 2])

    # 计算每个轴的变化量绝对值
    abs_delta_rx = np.abs(delta_rx)
    abs_delta_ry = np.abs(delta_ry)
    abs_delta_rz = np.abs(delta_rz)

    # 计算最大变化量（三个轴中的最大值）
    max_rotation_per_sample = np.maximum.reduce([abs_delta_rx, abs_delta_ry, abs_delta_rz])

    print("\n=== 旋转变化统计 ===")
    print(f"各轴旋转变化统计:")
    print(
        f"  ΔRX: 均值={np.mean(abs_delta_rx):.6f}, 标准差={np.std(abs_delta_rx):.6f}, 最大值={np.max(abs_delta_rx):.6f}")
    print(
        f"  ΔRY: 均值={np.mean(abs_delta_ry):.6f}, 标准差={np.std(abs_delta_ry):.6f}, 最大值={np.max(abs_delta_ry):.6f}")
    print(
        f"  ΔRZ: 均值={np.mean(abs_delta_rz):.6f}, 标准差={np.std(abs_delta_rz):.6f}, 最大值={np.max(abs_delta_rz):.6f}")

    # 查看分布情况
    thresholds = [0.1, 0.2, 0.3, 0.5, 1.0, np.pi]
    print(f"\n不同阈值下的样本分布（基于最大变化量）:")
    for thresh in thresholds:
        ratio = np.mean(max_rotation_per_sample <= thresh) * 100
        count = np.sum(max_rotation_per_sample <= thresh)
        print(f"  最大变化 ≤ {thresh:.2f} 弧度: {ratio:.1f}% ({count}个样本)")

    # ==================== 多级过滤策略 ====================
    print("\n=== 过滤策略 ===")

    # 策略1：基于最大变化量的简单阈值
    threshold_max_rotation = 1.5  # 0.3弧度 ≈ 17度
    mask_max_rotation = max_rotation_per_sample <= threshold_max_rotation

    # 策略2：基于各轴独立阈值的过滤
    # 考虑到RY的异常分布，可以对不同轴使用不同阈值
    threshold_rx = 0.5  # RX阈值可以稍大
    threshold_ry = 0.3  # RY阈值较严格
    threshold_rz = 0.5  # RZ阈值可以稍大

    mask_axis_independent = (
            (abs_delta_rx <= threshold_rx) &
            (abs_delta_ry <= threshold_ry) &
            (abs_delta_rz <= threshold_rz)
    )

    # 策略3：组合过滤（两种方法都满足）
    mask_combined = mask_max_rotation & mask_axis_independent

    print(f"过滤策略统计:")
    print(
        f"  策略1（最大变化≤{threshold_max_rotation}弧度）: 保留 {np.sum(mask_max_rotation)}/{len(X)} 样本 ({np.mean(mask_max_rotation) * 100:.1f}%)")
    print(
        f"  策略2（各轴独立阈值）: 保留 {np.sum(mask_axis_independent)}/{len(X)} 样本 ({np.mean(mask_axis_independent) * 100:.1f}%)")
    print(f"  策略3（组合过滤）: 保留 {np.sum(mask_combined)}/{len(X)} 样本 ({np.mean(mask_combined) * 100:.1f}%)")

    # 让用户选择过滤策略
    print("\n请选择过滤策略:")
    print(f"  1: 仅基于最大变化量（阈值={threshold_max_rotation}弧度）")
    print(f"  2: 基于各轴独立阈值（RX≤{threshold_rx}, RY≤{threshold_ry}, RZ≤{threshold_rz}弧度）")
    print(f"  3: 组合过滤（两者都满足）")

    choice = input("请输入选择 (1/2/3，默认3): ").strip()
    if choice == "1":
        clean_mask = mask_max_rotation
        print(f"选择策略1，将保留 {np.sum(clean_mask)} 个样本")
    elif choice == "2":
        clean_mask = mask_axis_independent
        print(f"选择策略2，将保留 {np.sum(clean_mask)} 个样本")
    else:
        clean_mask = mask_combined
        print(f"选择策略3，将保留 {np.sum(clean_mask)} 个样本")

    # ==================== 应用过滤 ====================
    X_clean = X[clean_mask]
    y_clean = y[clean_mask]

    # 保存清洗后的数据
    np.save('training_data_X_rotation_clean_fixed.npy', X_clean)
    np.save('training_data_y_rotation_clean_fixed.npy', y_clean)

    print(f"\n旋转清洗完成！")
    print(f"清洗后数据形状: X={X_clean.shape}, y={y_clean.shape}")


    return X_clean, y_clean, clean_mask



if __name__ == "__main__":
    # 检查输入数据是否存在
    if not os.path.exists('training_data_X_clean.npy') or not os.path.exists('training_data_y_clean.npy'):
        print("错误: 未找到尺度和形状清洗后的数据")
        print("请先运行 clean_scale_shape_data.py")
        exit(1)

    # 运行旋转数据清洗
    X_clean, y_clean, clean_mask = clean_rotation_data_standalone()
