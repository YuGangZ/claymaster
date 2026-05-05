# clean_scale_shape_data.py
# 清洗规则 1.变化率 > 100% 2.变化量 > 3σ 3.尺度参数 ≤ 0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clean_scale_shape_data():
    # 加载原始数据
    X = np.load('training_data_X.npy')
    y = np.load('training_data_y.npy')

    print(f"原始数据: X形状={X.shape}, y形状={y.shape}")

    # 提取尺度和形状参数（假设前5维）
    X_scale_shape = X[:, :5]
    y_scale_shape = y[:, :5]

    # 计算变化量和变化率
    delta = y_scale_shape - X_scale_shape
    change_ratio = np.abs(delta) / (np.abs(X_scale_shape) + 1e-6)

    # 创建清洗掩码
    # 规则1：变化率超过100%的样本
    extreme_mask = np.any(change_ratio > 1.0, axis=1)
    print(f"规则1：变化超过100%的样本数: {np.sum(extreme_mask)}")

    # 规则2：变化量超过3个标准差的样本
    std_mask = np.zeros(len(delta), dtype=bool)
    # 逐参数计算标准差
    for i in range(5):
        param_std = np.std(delta[:, i])
        std_mask = std_mask | (np.abs(delta[:, i]) > 3 * param_std)
    print(f"规则2：变化超过3σ的样本数: {np.sum(std_mask)}")

    # 规则3：尺度参数物理不可能的情况（如尺度变为0或负数）
    physical_mask = np.any(y_scale_shape[:, :3] <= 0, axis=1) | np.any(X_scale_shape[:, :3] <= 0, axis=1)
    print(f"规则3：物理不可能（尺度<=0）的样本数: {np.sum(physical_mask)}")

    # 组合所有清洗规则
    clean_mask = ~(extreme_mask | std_mask | physical_mask)

    print(f"\n清洗统计:")
    print(f"  总样本数: {len(X)}")
    print(f"  保留样本数: {np.sum(clean_mask)}")
    print(f"  删除样本数: {len(X) - np.sum(clean_mask)}")
    print(f"  保留比例: {np.sum(clean_mask) / len(X) * 100:.2f}%")

    # 应用清洗
    X_clean = X[clean_mask]
    y_clean = y[clean_mask]

    # 保存清洗后的数据
    np.save('training_data_X_clean.npy', X_clean)
    np.save('training_data_y_clean.npy', y_clean)

    # 分析清洗效果
    print("\n=== 清洗前后对比 ===")
    for i, name in enumerate(['Scale X', 'Scale Y', 'Scale Z', 'Epsilon1', 'Epsilon2']):
        print(f"\n{name}:")
        print(f"  清洗前 - 变化量标准差: {np.std(delta[:, i]):.6f}")
        print(f"  清洗前 - 变化率均值: {np.mean(change_ratio[:, i]):.6f}")

        clean_delta = y_clean[:, i] - X_clean[:, i]
        clean_change_ratio = np.abs(clean_delta) / (np.abs(X_clean[:, i]) + 1e-6)
        print(f"  清洗后 - 变化量标准差: {np.std(clean_delta):.6f}")
        print(f"  清洗后 - 变化率均值: {np.mean(clean_change_ratio):.6f}")

    # 可视化清洗效果
    visualize_cleaning_effect(X_scale_shape, y_scale_shape, X_clean, y_clean, clean_mask)

    return X_clean, y_clean


def visualize_cleaning_effect(X_orig, y_orig, X_clean, y_clean, clean_mask):
    """可视化清洗效果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Data Cleaning Effect - Scale and Shape Parameters', fontsize=14)

    param_names = ['Scale X', 'Scale Y', 'Scale Z', 'Epsilon1', 'Epsilon2']

    # 绘制每个参数的变化量分布
    for i in range(5):
        row, col = divmod(i, 3)
        ax = axes[row, col]

        # 原始数据变化量
        orig_delta = y_orig[:, i] - X_orig[:, i]
        clean_delta = y_clean[:, i] - X_clean[:, i]

        # 直方图
        ax.hist(orig_delta, bins=50, alpha=0.5, label='Original', color='red')
        ax.hist(clean_delta, bins=50, alpha=0.5, label='Cleaned', color='blue')

        ax.set_title(f'{param_names[i]}')
        ax.set_xlabel('Delta (next - current)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 第6个子图：显示清洗的样本分布
    ax = axes[1, 2]
    ax.pie([np.sum(clean_mask), np.sum(~clean_mask)],
           labels=['Kept', 'Removed'],
           colors=['blue', 'red'],
           autopct='%1.1f%%')
    ax.set_title('Data Cleaning Distribution')

    plt.tight_layout()
    plt.savefig('data_cleaning_effect.png', dpi=150)
    plt.close()

    # 创建更详细的异常检测图
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    fig2.suptitle('Outlier Detection in Scale and Shape Parameters', fontsize=14)

    for i in range(5):
        row, col = divmod(i, 3)
        ax = axes2[row, col]

        # 当前值 vs 变化量
        scatter = ax.scatter(X_orig[:, i], y_orig[:, i] - X_orig[:, i],
                             c=clean_mask, cmap='coolwarm', alpha=0.6, s=20)

        ax.set_xlabel(f'Current {param_names[i]}')
        ax.set_ylabel(f'Delta {param_names[i]}')
        ax.set_title(f'{param_names[i]}: Blue=Kept, Red=Removed')
        ax.grid(True, alpha=0.3)

    # 移除第6个子图的刻度
    axes2[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('outlier_detection.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    X_clean, y_clean = clean_scale_shape_data()