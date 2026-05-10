# eval_model.py
# 模型测试脚本 - 测试模型并保存所有指标
import argparse
import numpy as np
import pandas as pd
import torch

import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_log_var.shape_predictor import ShapePredictor


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_trained_model(model_path, device, input_dim=17, output_dim=14):
    """加载训练好的模型 - 修改为14维状态"""
    model = ShapePredictor(input_dim=input_dim, output_dim=output_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ 加载模型: {model_path}")
    return model


def validate_data_dimensions(X_test, y_test, expected_state_dim=14):
    """验证数据维度 - 修改为14维状态"""
    expected_input_dim = expected_state_dim + 3  # 14维状态 + 3维控制
    assert X_test.shape[1] == expected_input_dim, f"X_test应为{expected_input_dim}维，实际为{X_test.shape[1]}维"
    assert y_test.shape[1] == expected_state_dim, f"y_test应为{expected_state_dim}维，实际为{y_test.shape[1]}维"
    print(f"✅ 数据维度验证通过: 输入={X_test.shape[1]}维, 输出={y_test.shape[1]}维")


def calculate_detailed_metrics_per_dimension(y_true, y_pred):
    """计算每个维度的详细指标"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    n_dims = y_true.shape[1]
    metrics_list = []

    for dim in range(n_dims):
        true_vals = y_true[:, dim]
        pred_vals = y_pred[:, dim]

        # 基础误差指标
        mse = mean_squared_error(true_vals, pred_vals)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_vals, pred_vals)

        # 相对误差
        abs_error = np.abs(pred_vals - true_vals)
        abs_true = np.abs(true_vals)

        if np.mean(abs_true) > 1e-10:
            relative_error = np.mean(abs_error) / np.mean(abs_true)
        else:
            relative_error = np.mean(abs_error)

        # 稳健的相对RMSE计算
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_errors = abs_error / (abs_true + 1e-8)
            # 过滤极端值（如 > 1000%）
            valid_mask = (relative_errors < 10) & (~np.isinf(relative_errors))
            if np.any(valid_mask):
                relative_rmse = np.sqrt(np.mean(relative_errors[valid_mask] ** 2))
            else:
                relative_rmse = 0.0

        # R²分数
        if np.var(true_vals) > 1e-10:
            r2 = r2_score(true_vals, pred_vals)
        else:
            r2 = 0.0

        # Pearson相关系数
        if len(true_vals) > 1 and np.std(true_vals) > 1e-10 and np.std(pred_vals) > 1e-10:
            pearson = np.corrcoef(true_vals, pred_vals)[0, 1]
        else:
            pearson = 0.0

        # 对称平均绝对百分比误差 (sMAPE)
        denominator = np.abs(true_vals) + np.abs(pred_vals) + 1e-8
        smape = 100 * np.mean(2 * abs_error / denominator)

        # 信噪比 (SNR) in dB
        signal_power = np.mean(true_vals ** 2)
        noise_power = np.mean((true_vals - pred_vals) ** 2)
        if noise_power > 1e-10:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 100.0  # 使用大值代替无穷大

        metrics_list.append({
            'dimension': dim,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'relative_error': relative_error,
            'relative_rmse': relative_rmse,
            'r2': r2,
            'pearson': pearson,
            'smape': smape,
            'snr_db': snr_db
        })

    return pd.DataFrame(metrics_list)


def test_and_save(model, save_dir, data_x_path, data_y_path, max_samples=5000, state_dim=14):
    """
    测试模型并保存所有结果 - 修改为适配14维状态

    参数:
        model: 训练好的模型
        save_dir: 结果保存目录
        max_samples: 最大样本数（用于加速测试）
        state_dim: 状态维度（默认为14）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    try:
        X = np.load(data_x_path)
        y = np.load(data_y_path)
        print(f"✅ 成功加载数据:")
        print(f"   X: {data_x_path}")
        print(f"   y: {data_y_path}")
    except FileNotFoundError as e:
        print(f"❌ 找不到数据文件: {e}")
        print("尝试加载默认文件名...")
        return None, None, None

    print(f"📊 原始数据维度:")
    print(f"   X形状: {X.shape}")
    print(f"   y形状: {y.shape}")
    print(f"   数据记录数: {len(X)}")

    # 使用查验后的数据作为测试集
    X_test, y_test = X, y

    # 验证数据维度
    validate_data_dimensions(X_test, y_test, expected_state_dim=state_dim)

    # 打印数据统计信息
    print(f"📊 测试数据统计:")
    print(f"   X形状: {X_test.shape} ({state_dim}维状态 + 3维控制)")
    print(f"   y形状: {y_test.shape} ({state_dim}维下一帧)")
    print(f"   输入范围: [{X_test.min():.2e}, {X_test.max():.2e}]")
    print(f"   输出范围: [{y_test.min():.2e}, {y_test.max():.2e}]")

    # 采样以减少计算时间
    if len(X_test) > max_samples:
        indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
        print(f"📝 采样 {max_samples} 个样本以加速测试")

    # 设备设置
    device = next(model.parameters()).device

    # 分离状态和控制
    state_test = X_test[:, :state_dim]  # 前14维：状态
    control_test = X_test[:, state_dim:]  # 后3维：控制

    state_tensor = torch.FloatTensor(state_test).to(device)
    control_tensor = torch.FloatTensor(control_test).to(device)

    # 模型预测
    print("🔮 进行模型预测...")
    with torch.no_grad():
        delta_state, uncertainty = model(state_tensor, control_tensor)
        pred_delta = delta_state.cpu().numpy()
        pred_uncertainty = uncertainty.cpu().numpy()
        # 预测下一状态 = 当前状态 + 预测增量
        y_pred = state_test + pred_delta

    # 计算详细指标
    print("📈 计算详细性能指标...")
    metrics_df = calculate_detailed_metrics_per_dimension(y_test, y_pred)

    # 14维状态名称（已移除flatness和convexity，保留smoothness）
    dim_names = [
        'scale_a1', 'scale_a2', 'scale_a3',  # 0-2: 尺度参数
        'shape_epsilon1', 'shape_epsilon2',  # 3-4: 形状参数
        'translation_x', 'translation_y', 'translation_z',  # 5-7: 位置
        'euler_rx', 'euler_ry', 'euler_rz',  # 8-10: 旋转
        'volume', 'elongation', 'smoothness'  # 11-13: 几何特征
    ]

    # 保存详细数据
    print("💾 保存详细数据...")
    all_data = {}
    for dim in range(state_dim):
        all_data[f'true_dim{dim}'] = y_test[:, dim]
        all_data[f'pred_dim{dim}'] = y_pred[:, dim]
        all_data[f'error_dim{dim}'] = y_pred[:, dim] - y_test[:, dim]
        all_data[f'uncertainty_dim{dim}'] = pred_uncertainty[:, dim]

    all_data_df = pd.DataFrame(all_data)
    all_data_df.to_csv(os.path.join(save_dir, 'all_dimensions_data.csv'), index=False)

    # 保存详细指标数据（包含所有指标）
    metrics_df['dimension_name'] = dim_names
    metrics_df.to_csv(os.path.join(save_dir, 'detailed_metrics_per_dimension.csv'), index=False)

    # 计算并保存整体指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    overall_metrics = {
        'overall_rmse': np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten())),
        'overall_mae': mean_absolute_error(y_test.flatten(), y_pred.flatten()),
        'overall_r2': r2_score(y_test.flatten(), y_pred.flatten()),
        'avg_dim_rmse': metrics_df['rmse'].mean(),
        'avg_dim_r2': metrics_df['r2'].mean(),
        'avg_dim_relative_error': metrics_df['relative_error'].mean(),
        'avg_dim_smape': metrics_df['smape'].mean(),
        'avg_dim_pearson': metrics_df['pearson'].mean(),
        'avg_dim_snr_db': metrics_df['snr_db'].mean(),
        'state_dimension': state_dim,
        'control_dimension': 3,
        'total_samples': len(X_test),
        'removed_features': 'flatness, convexity',
        'retained_features': 'scale_a1, scale_a2, scale_a3, shape_epsilon1, shape_epsilon2, translation_x, translation_y, translation_z, euler_rx, euler_ry, euler_rz, volume, elongation, smoothness'
    }

    overall_df = pd.DataFrame([overall_metrics])
    overall_df.to_csv(os.path.join(save_dir, 'overall_metrics.csv'), index=False)

    # 保存真值和预测值（用于绘图）
    np.savez(
        os.path.join(save_dir, 'test_results.npz'),
        y_true=y_test,
        y_pred=y_pred,
        dim_names=dim_names
    )

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("📋 测试结果摘要 (14维状态模型)")
    print("=" * 60)
    print(f"整体RMSE: {overall_metrics['overall_rmse']:.6e}")
    print(f"整体MAE: {overall_metrics['overall_mae']:.6e}")
    print(f"整体R²: {overall_metrics['overall_r2']:.4f}")
    print(f"平均维度R²: {overall_metrics['avg_dim_r2']:.4f}")
    print(f"平均维度sMAPE: {overall_metrics['avg_dim_smape']:.2f}%")
    print(f"平均维度Pearson: {overall_metrics['avg_dim_pearson']:.4f}")
    print(f"平均维度SNR: {overall_metrics['avg_dim_snr_db']:.2f} dB")
    print(f"状态维度: {state_dim}")
    print(f"测试样本数: {overall_metrics['total_samples']}")
    print(f"移除的特征: {overall_metrics['removed_features']}")
    print(f"保留的特征: {overall_metrics['retained_features']}")

    # 找出最佳和最差维度
    best_dim = metrics_df.loc[metrics_df['r2'].idxmax()]
    worst_dim = metrics_df.loc[metrics_df['r2'].idxmin()]
    print(
        f"\n最佳维度 (最高R²): 维度 {int(best_dim['dimension'])} ({dim_names[int(best_dim['dimension'])]}) - R²={best_dim['r2']:.4f}")
    print(
        f"最差维度 (最低R²): 维度 {int(worst_dim['dimension'])} ({dim_names[int(worst_dim['dimension'])]}) - R²={worst_dim['r2']:.4f}")

    # 按RMSE排序找出最准确和最不准确的维度
    best_rmse_dim = metrics_df.loc[metrics_df['rmse'].idxmin()]
    worst_rmse_dim = metrics_df.loc[metrics_df['rmse'].idxmax()]
    print(
        f"\n最准确维度 (最低RMSE): 维度 {int(best_rmse_dim['dimension'])} ({dim_names[int(best_rmse_dim['dimension'])]}) - RMSE={best_rmse_dim['rmse']:.6e}")
    print(
        f"最不准确维度 (最高RMSE): 维度 {int(worst_rmse_dim['dimension'])} ({dim_names[int(worst_rmse_dim['dimension'])]}) - RMSE={worst_rmse_dim['rmse']:.6e}")

    # 打印各维度R²排名
    print(f"\n📊 各维度R²排名:")
    sorted_metrics = metrics_df.sort_values('r2', ascending=False)
    for i, (_, row) in enumerate(sorted_metrics.iterrows()):
        dim_idx = int(row['dimension'])
        print(
            f"  {i + 1:2d}. 维度 {dim_idx:2d} ({dim_names[dim_idx]:15s}): R² = {row['r2']:.4f}, RMSE = {row['rmse']:.6e}")

    print(f"\n💾 数据已保存到: {save_dir}")
    print("生成的文件:")
    print(f"  • test_results.npz - 真值和预测值（用于绘图）")
    print(f"  • all_dimensions_data.csv - 所有维度详细数据")
    print(f"  • detailed_metrics_per_dimension.csv - 每个维度的详细指标")
    print(f"  • overall_metrics.csv - 整体指标")

    return y_test, y_pred, metrics_df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试形状预测模型并保存结果 (14维状态)')
    parser.add_argument('--model_path', type=str, default='../shape_predictor_best.pth',
                        help='训练模型路径')
    parser.add_argument('--save_dir', type=str, default='test_results_14dim',
                        help='结果保存目录')
    parser.add_argument('--data_x_path', type=str, default='../training_data_X_rotation_clean_fixed.npy',
                        help='X数据文件路径 (.npy文件)')
    parser.add_argument('--data_y_path', type=str, default='../training_data_y_rotation_clean_fixed.npy',
                        help='y数据文件路径 (.npy文件)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='最大样本数（用于加速测试）')
    parser.add_argument('--state_dim', type=int, default=14,
                        help='状态维度')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用设备: {device}")

    # 加载训练模型 - 使用14维状态配置
    input_dim = args.state_dim + 3  # 状态维度 + 控制维度
    model = load_trained_model(args.model_path, device,
                            input_dim=input_dim,
                            output_dim=args.state_dim)

    # 测试并保存结果
    y_test, y_pred, metrics_df = test_and_save(
        model,
        args.save_dir,
        args.data_x_path,
        args.data_y_path,
        args.max_samples,
        state_dim=args.state_dim
    )

    if y_test is not None:
        print(f"\n✅ 测试完成！")
        print(f"   运行: python plot_data.py --data_dir {args.save_dir}")
    else:
        print("❌ 测试失败")


if __name__ == '__main__':
    main()