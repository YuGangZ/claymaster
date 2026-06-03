# evaluation.py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.metrics import r2_score


def evaluate_uncertainty_quality(preds, uncertainties, targets):
    """评估不确定性预测质量"""
    errors = preds - targets
    abs_errors = np.abs(errors)
    variances = np.exp(uncertainties)  # 转换为标准差
    stds = np.sqrt(variances)

    results = {}

    # 1. 不确定性与绝对误差的相关性
    flat_uncertainties = uncertainties.flatten()
    flat_abs_errors = abs_errors.flatten()

    # 移除 NaN 和无穷大值
    mask = np.isfinite(flat_uncertainties) & np.isfinite(flat_abs_errors)
    flat_uncertainties = flat_uncertainties[mask]
    flat_abs_errors = flat_abs_errors[mask]

    if len(flat_uncertainties) > 10:  # 确保有足够的数据点
        try:
            corr_pearson, _ = stats.pearsonr(flat_uncertainties, flat_abs_errors)
            corr_spearman, _ = stats.spearmanr(flat_uncertainties, flat_abs_errors)
        except:
            corr_pearson = np.nan
            corr_spearman = np.nan
    else:
        corr_pearson = np.nan
        corr_spearman = np.nan

    results['pearson_correlation'] = corr_pearson
    results['spearman_correlation'] = corr_spearman

    # 2. 校准性评估
    conf_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    actual_coverages = []

    for conf in conf_levels:
        z = stats.norm.ppf((1 + conf) / 2)
        lower = preds - z * stds
        upper = preds + z * stds

        coverage = np.mean((targets >= lower) & (targets <= upper))
        actual_coverages.append(coverage)

    results['expected_confidence'] = conf_levels
    results['actual_coverage'] = actual_coverages
    results['calibration_error'] = np.mean(np.abs(np.array(conf_levels) - np.array(actual_coverages)))

    # 3. 方差与误差平方的匹配度
    expected_variance = np.mean(errors ** 2, axis=0)
    predicted_variance = np.mean(variances, axis=0)

    mask = (expected_variance > 1e-10) & (predicted_variance > 1e-10) & np.isfinite(expected_variance) & np.isfinite(
        predicted_variance)

    if np.sum(mask) > 1:
        expected_var_valid = expected_variance[mask]
        predicted_var_valid = predicted_variance[mask]
        correlation_matrix = np.corrcoef(expected_var_valid, predicted_var_valid)
        results['variance_match_correlation'] = correlation_matrix[0, 1]
        results['variance_ratio'] = np.mean(predicted_var_valid) / np.mean(expected_var_valid)
    else:
        results['variance_match_correlation'] = np.nan
        results['variance_ratio'] = np.nan

    return results


def plot_uncertainty_calibration(preds, uncertainties, targets, save_path='uncertainty_calibration.png'):
    """绘制不确定性校准图"""
    errors = preds - targets
    variances = np.exp(uncertainties)  # 转换为标准差
    stds = np.sqrt(variances)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Uncertainty Calibration Analysis', fontsize=16)

    # 1. 不确定性与绝对误差散点图
    ax1 = axes[0, 0]
    sample_size = min(5000, len(errors.flatten()))
    indices = np.random.choice(len(errors.flatten()), sample_size, replace=False)

    ax1.scatter(uncertainties.flatten()[indices],
                np.abs(errors).flatten()[indices],
                alpha=0.3, s=5)

    # 添加趋势线
    x = uncertainties.flatten()[indices]
    y = np.abs(errors).flatten()[indices]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(x)
    ax1.plot(x_sorted, p(x_sorted), "r-", linewidth=2,
             label=f'Corr: {np.corrcoef(x, y)[0, 1]:.3f}')

    ax1.set_xlabel('Uncertainty (log std)')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Uncertainty vs Absolute Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 校准曲线
    ax2 = axes[0, 1]
    conf_levels = np.linspace(0.05, 0.95, 19)
    actual_coverages = []

    for conf in conf_levels:
        z = stats.norm.ppf((1 + conf) / 2)
        lower = preds - z * stds
        upper = preds + z * stds
        coverage = np.mean((targets >= lower) & (targets <= upper))
        actual_coverages.append(coverage)

    ax2.plot(conf_levels, actual_coverages, 'bo-', linewidth=2, markersize=6)
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    ax2.set_xlabel('Expected Confidence Level')
    ax2.set_ylabel('Actual Coverage')
    ax2.set_title('Calibration Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', 'box')

    # 3. 标准化残差分布
    ax3 = axes[1, 0]
    standardized_errors = errors / (stds + 1e-8)
    ax3.hist(standardized_errors.flatten(), bins=50, density=True,
             alpha=0.6, color='g', edgecolor='black')

    # 绘制标准正态分布对比
    x_norm = np.linspace(-4, 4, 100)
    y_norm = stats.norm.pdf(x_norm)
    ax3.plot(x_norm, y_norm, 'r-', linewidth=2, label='Standard Normal')

    ax3.set_xlabel('Standardized Error (error / std)')
    ax3.set_ylabel('Density')
    ax3.set_title('Standardized Residual Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 分位数-分位数图
    ax4 = axes[1, 1]
    stats.probplot(standardized_errors.flatten(), dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot: Residuals vs Normal Distribution')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig


def plot_predictions_vs_truth(model, dataloader, save_dir='training_results', suffix='', num_samples=1000):
    """
    绘制预测的下一帧参数与真实下一帧的对比图
    支持 suffix 参数避免多个消融模式的文件覆盖
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 收集预测结果
    all_preds = []
    all_targets = []
    all_uncertainties = []
    all_current_states = []
    all_pred_deltas = []  # 收集预测的delta
    all_target_deltas = []  # 收集目标的delta

    with torch.no_grad():
        for batch_state, batch_control, batch_y in dataloader:
            total_samples = sum(p.shape[0] for p in all_preds)
            if total_samples >= num_samples:
                break

            # 直接使用原始数据输入模型
            pred_delta, pred_uncertainty = model(batch_state, batch_control)

            # 计算原始空间的下一状态
            pred_next = batch_state + pred_delta
            true_next = batch_y
            target_delta = batch_y - batch_state

            all_preds.append(pred_next.numpy())
            all_targets.append(true_next.numpy())  # 计算目标下一状态
            all_uncertainties.append(pred_uncertainty.numpy())
            all_current_states.append(batch_state.numpy())
            all_pred_deltas.append(pred_delta.numpy())
            all_target_deltas.append(target_delta.numpy())

    all_preds = np.vstack(all_preds)[:num_samples]
    all_targets = np.vstack(all_targets)[:num_samples]
    all_uncertainties = np.vstack(all_uncertainties)[:num_samples]
    all_current_states = np.vstack(all_current_states)[:num_samples]
    all_pred_deltas = np.vstack(all_pred_deltas)[:num_samples]
    all_target_deltas = np.vstack(all_target_deltas)[:num_samples]

    # 计算误差
    errors = all_preds - all_targets
    absolute_errors = np.abs(errors)
    relative_errors = np.abs(errors) / (np.abs(all_targets) + 1e-8)

    # 计算均方根误差 (RMSE)
    rmse_per_dim = np.sqrt(np.mean(errors ** 2, axis=0))
    mae_per_dim = np.mean(absolute_errors, axis=0)

    # 14维状态名称（已移除flatness和convexity，保留smoothness）
    dim_names_14 = [
        'scale_a1', 'scale_a2', 'scale_a3',  # 0-2: 尺度参数
        'shape_epsilon1', 'shape_epsilon2',  # 3-4: 形状参数
        'translation_x', 'translation_y', 'translation_z',  # 5-7: 位置
        'euler_rx', 'euler_ry', 'euler_rz',  # 8-10: 旋转
        'volume', 'elongation', 'smoothness'  # 11-13: 几何特征
    ]

    # 创建完整的可视化图表
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Predictions vs Ground Truth Analysis ({suffix})', fontsize=16, y=0.98)

    # 1. 关键维度预测vs真实值散点图
    key_dims = [0, 3, 5, 8, 11, 13]  # Scale X, Shape Epsilon1, Position X, Rotation X, Volume, Smoothness
    key_names = ['Scale X', 'Shape Epsilon1', 'Position X', 'Rotation X', 'Volume', 'Smoothness']

    # 创建2x3的子图
    for idx, (dim, name) in enumerate(zip(key_dims, key_names)):
        row = idx // 3
        col = idx % 3
        ax = plt.subplot(3, 3, idx + 1)

        # 散点图显示预测vs真实值
        scatter = ax.scatter(all_targets[:, dim], all_preds[:, dim],
                             alpha=0.6, s=15, c=all_uncertainties[:, dim],
                             cmap='viridis')

        # 添加对角线
        min_val = min(all_targets[:, dim].min(), all_preds[:, dim].min())
        max_val = max(all_targets[:, dim].max(), all_preds[:, dim].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name} ({dim_names_14[dim]})\nRMSE: {rmse_per_dim[dim]:.4f}')
        ax.grid(True, alpha=0.3)

        # 添加R²分数
        r2 = r2_score(all_targets[:, dim], all_preds[:, dim])
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 为第一个子图添加颜色条
        if idx == 0:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Uncertainty (log std)')

    # 2. 误差分布直方图
    ax_error = plt.subplot(3, 3, 7)
    flat_errors = errors.flatten()
    ax_error.hist(flat_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax_error.set_title('Error Distribution')
    ax_error.set_xlabel('Prediction Error')
    ax_error.set_ylabel('Frequency')
    ax_error.grid(True, alpha=0.3)

    # 添加误差统计
    mean_error = np.mean(flat_errors)
    std_error = np.std(flat_errors)
    ax_error.axvline(mean_error, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_error.text(0.05, 0.95, f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}',
                  transform=ax_error.transAxes, fontsize=10,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. 各维度RMSE比较
    ax_rmse = plt.subplot(3, 3, 8)
    dim_indices = range(len(rmse_per_dim))
    bars = ax_rmse.bar(dim_indices, rmse_per_dim, alpha=0.7)
    ax_rmse.set_title('RMSE per Dimension')
    ax_rmse.set_xlabel('Dimension Index')
    ax_rmse.set_ylabel('RMSE')
    ax_rmse.grid(True, alpha=0.3, axis='y')

    # 标记关键维度
    for dim in key_dims:
        bars[dim].set_color('red')

    # 4. 不确定性分析
    ax_uncertainty = plt.subplot(3, 3, 9)
    stds = np.sqrt(np.exp(all_uncertainties))  # 转换为标准差
    # 不确定性 vs 绝对误差
    flat_uncertainties = stds.flatten()
    flat_abs_errors = absolute_errors.flatten()

    # 采样以减少数据点
    if len(flat_uncertainties) > 5000:
        indices = np.random.choice(len(flat_uncertainties), 5000, replace=False)
        flat_uncertainties = flat_uncertainties[indices]
        flat_abs_errors = flat_abs_errors[indices]

    ax_uncertainty.scatter(flat_uncertainties, flat_abs_errors, alpha=0.5, s=10)
    ax_uncertainty.set_title('Uncertainty (std) vs Absolute Error')
    ax_uncertainty.set_xlabel('Uncertainty (std)')
    ax_uncertainty.set_ylabel('Absolute Error')
    ax_uncertainty.grid(True, alpha=0.3)

    # 添加相关系数
    corr = np.corrcoef(flat_uncertainties, flat_abs_errors)[0, 1]
    ax_uncertainty.text(0.05, 0.95, f'Correlation: {corr:.4f}',
                        transform=ax_uncertainty.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    # 保存时添加 suffix
    plt.savefig(os.path.join(save_dir, f'predictions_vs_truth_detailed_{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 额外：绘制delta预测质量
    fig_delta = plt.figure(figsize=(15, 10))
    fig_delta.suptitle(f'Delta Prediction Quality ({suffix})', fontsize=16, y=0.98)

    # 预测delta vs 目标delta
    ax1 = plt.subplot(2, 2, 1)
    # 选择几个关键维度
    for dim in [0, 5, 8]:  # scale_a1, translation_x, euler_rx
        ax1.scatter(all_target_deltas[:200, dim], all_pred_deltas[:200, dim],
                    alpha=0.6, s=15, label=dim_names_14[dim])

    # 添加对角线
    min_val = min(all_target_deltas.min(), all_pred_deltas.min())
    max_val = max(all_target_deltas.max(), all_pred_deltas.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

    ax1.set_xlabel('Target Delta')
    ax1.set_ylabel('Predicted Delta')
    ax1.set_title('Delta Prediction vs Target')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Delta误差分布
    ax2 = plt.subplot(2, 2, 2)
    delta_errors = all_pred_deltas - all_target_deltas
    ax2.hist(delta_errors.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Delta Error Distribution')
    ax2.set_xlabel('Delta Error')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)

    # 各维度delta误差统计
    ax3 = plt.subplot(2, 2, 3)
    delta_rmse = np.sqrt(np.mean(delta_errors ** 2, axis=0))
    bars_delta = ax3.bar(range(len(delta_rmse)), delta_rmse, alpha=0.7)
    ax3.set_title('Delta RMSE per Dimension')
    ax3.set_xlabel('Dimension Index')
    ax3.set_ylabel('Delta RMSE')
    ax3.grid(True, alpha=0.3, axis='y')

    # 标记关键维度
    for dim in key_dims:
        bars_delta[dim].set_color('red')

    # 预测delta的统计信息
    ax4 = plt.subplot(2, 2, 4)
    pred_delta_mean = np.mean(np.abs(all_pred_deltas), axis=0)
    target_delta_mean = np.mean(np.abs(all_target_deltas), axis=0)

    x = np.arange(len(pred_delta_mean))
    width = 0.35
    ax4.bar(x - width / 2, pred_delta_mean, width, label='Predicted', alpha=0.7)
    ax4.bar(x + width / 2, target_delta_mean, width, label='Target', alpha=0.7)
    ax4.set_title('Mean Absolute Delta Comparison')
    ax4.set_xlabel('Dimension Index')
    ax4.set_ylabel('Mean Absolute Delta')
    ax4.set_xticks(x)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'delta_prediction_quality_{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    eval_results = {}
    # 在预测可视化之后添加不确定性质量评估
    if len(all_preds) > 0:
        # 评估不确定性质量
        eval_results = evaluate_uncertainty_quality(
            all_preds, all_uncertainties, all_targets
        )

        # 打印评估结果
        print("\n" + "=" * 60)
        print(f"不确定性预测质量评估 ({suffix})")
        print("=" * 60)
        print(f"不确定性与绝对误差相关性: {eval_results['pearson_correlation']:.4f}")
        print(f"校准误差: {eval_results['calibration_error']:.4f}")
        print(f"方差匹配相关性: {eval_results['variance_match_correlation']:.4f}")
        print(f"方差比率: {eval_results['variance_ratio']:.4f}")

        # 打印delta统计
        print("\n" + "=" * 60)
        print("各维度Delta详细统计")
        print("=" * 60)

        # 定义阈值
        thresholds = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

        # 为每个维度计算统计
        for dim_idx in range(all_target_deltas.shape[1]):
            target_delta_dim = all_target_deltas[:, dim_idx]
            pred_delta_dim = all_pred_deltas[:, dim_idx]

            print(f"\n维度 {dim_idx:2d} ({dim_names_14[dim_idx]}):")
            print("-" * 40)

            # 目标delta统计
            target_abs = np.abs(target_delta_dim)

            # 统计目标delta不同数量级占比
            print("  目标delta绝对值分布:")
            for i in range(len(thresholds) - 1):
                lower = thresholds[i + 1]
                upper = thresholds[i]
                mask = (target_abs >= lower) & (target_abs < upper)
                count = np.sum(mask)
                percent = count / len(target_abs) * 100
                print(f"    [{lower:.0e}, {upper:.0e}): {count:6d} ({percent:6.2f}%)")

            # 统计小于最小阈值的情况
            mask = target_abs < thresholds[-1]
            count = np.sum(mask)
            percent = count / len(target_abs) * 100
            print(f"    <{thresholds[-1]:.0e}:        {count:6d} ({percent:6.2f}%)")

            # 预测delta统计
            pred_abs = np.abs(pred_delta_dim)

            print("  预测delta绝对值分布:")
            for i in range(len(thresholds) - 1):
                lower = thresholds[i + 1]
                upper = thresholds[i]
                mask = (pred_abs >= lower) & (pred_abs < upper)
                count = np.sum(mask)
                percent = count / len(pred_abs) * 100
                print(f"    [{lower:.0e}, {upper:.0e}): {count:6d} ({percent:6.2f}%)")

            # 统计小于最小阈值的情况
            mask = pred_abs < thresholds[-1]
            count = np.sum(mask)
            percent = count / len(pred_abs) * 100
            print(f"    <{thresholds[-1]:.0e}:        {count:6d} ({percent:6.2f}%)")

            # 计算分位数
            target_q25, target_q50, target_q75 = np.percentile(target_abs, [25, 50, 75])
            pred_q25, pred_q50, pred_q75 = np.percentile(pred_abs, [25, 50, 75])

            print(f"  目标delta分位数: Q25={target_q25:.2e}, Q50={target_q50:.2e}, Q75={target_q75:.2e}")
            print(f"  预测delta分位数: Q25={pred_q25:.2e}, Q50={pred_q50:.2e}, Q75={pred_q75:.2e}")

            # 计算匹配度：预测delta与目标delta在同一数量级的比例
            target_levels = np.floor(np.log10(target_abs + 1e-10))
            pred_levels = np.floor(np.log10(pred_abs + 1e-10))

            same_level = np.sum(target_levels == pred_levels)
            within_one_level = np.sum(np.abs(target_levels - pred_levels) <= 1)

            same_level_percent = same_level / len(target_abs) * 100
            within_one_level_percent = within_one_level / len(target_abs) * 100

            print(f"  数量级匹配度: 完全相同={same_level_percent:.1f}%, ±1级={within_one_level_percent:.1f}%")

        print("=" * 60)

        # 额外添加整体统计摘要
        print("\nDelta统计摘要:")
        print("-" * 40)

        # 计算每个维度目标delta的中位数
        target_medians = np.median(np.abs(all_target_deltas), axis=0)
        pred_medians = np.median(np.abs(all_pred_deltas), axis=0)

        # 找出delta最大的几个维度
        top_n = 5
        top_indices = np.argsort(target_medians)[-top_n:][::-1]

        print(f"目标delta绝对值中位数最大的 {top_n} 个维度:")
        for rank, dim_idx in enumerate(top_indices, 1):
            target_median = target_medians[dim_idx]
            pred_median = pred_medians[dim_idx]
            ratio = pred_median / target_median if target_median > 0 else np.nan

            print(f"  {rank}. 维度 {dim_idx:2d} ({dim_names_14[dim_idx]}):")
            print(f"     目标中位数: {target_median:.2e}, 预测中位数: {pred_median:.2e}, 比率: {ratio:.3f}")

        # 计算预测与目标的比例分布
        valid_mask = target_medians > 0
        if np.sum(valid_mask) > 0:
            median_ratios = pred_medians[valid_mask] / target_medians[valid_mask]
            print(f"\n预测/目标中位数比率统计:")
            print(f"  均值: {np.mean(median_ratios):.3f}, 中位数: {np.median(median_ratios):.3f}")
            print(f"  范围: [{np.min(median_ratios):.3f}, {np.max(median_ratios):.3f}]")

            # 统计比率在不同区间的分布
            ratio_bins = [0, 0.1, 0.5, 0.9, 1.1, 2.0, np.inf]
            bin_labels = ["<0.1", "0.1-0.5", "0.5-0.9", "0.9-1.1", "1.1-2.0", ">2.0"]

            print(f"  比率分布:")
            for i in range(len(bin_labels)):
                if i < len(ratio_bins) - 1:
                    mask = (median_ratios >= ratio_bins[i]) & (median_ratios < ratio_bins[i + 1])
                else:
                    mask = median_ratios >= ratio_bins[i]

                count = np.sum(mask)
                percent = count / len(median_ratios) * 100
                print(f"    {bin_labels[i]}: {count:2d}个维度 ({percent:5.1f}%)")

        print("=" * 60)

        # 绘制校准图
        plot_uncertainty_calibration(
            all_preds, all_uncertainties, all_targets,
            os.path.join(save_dir, f'uncertainty_calibration_{suffix}.png')
        )

    # 返回各项指标供汇总
    return {
        'preds': all_preds,
        'targets': all_targets,
        'uncertainties': all_uncertainties,
        'pred_deltas': all_pred_deltas,
        'target_deltas': all_target_deltas,
        'rmse_per_dim': rmse_per_dim,
        'mae_per_dim': mae_per_dim,
        'dim_names': dim_names_14,
        'mean_rmse': np.mean(rmse_per_dim),
        'corr_uncertainty': eval_results.get('pearson_correlation', np.nan),
        'calibration_error': eval_results.get('calibration_error', np.nan),
    }


def compare_ablation_models(results_base_dir, ablation_modes):
    """汇总对比不同消融模式的评估结果，自动加载数据和模型"""
    from torch.utils.data import DataLoader, TensorDataset
    from shape_predictor_ablation import ShapePredictor  # 导入模型类

    # 加载数据（与训练使用相同的数据集，也可改为独立测试集）
    X = np.load('training_data_X.npy')
    y = np.load('training_data_y.npy')
    state = torch.FloatTensor(X[:, :14])
    control = torch.FloatTensor(X[:, 14:])
    target = torch.FloatTensor(y)
    dataset = TensorDataset(state, control, target)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    metrics_list = []
    for mode in ablation_modes:
        model_path = os.path.join(results_base_dir, mode, f'shape_predictor_{mode}_best.pth')
        if not os.path.exists(model_path):
            print(f"警告: {model_path} 不存在，跳过 {mode}")
            continue
        model = ShapePredictor(input_dim=17, output_dim=14, ablation_mode=mode)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        temp_dir = os.path.join(results_base_dir, '_temp_eval')
        os.makedirs(temp_dir, exist_ok=True)
        metrics = plot_predictions_vs_truth(model, test_loader, save_dir=temp_dir, suffix=mode, num_samples=1000)
        metrics['ablation_mode'] = mode
        metrics_list.append(metrics)

    import shutil
    shutil.rmtree(os.path.join(results_base_dir, '_temp_eval'), ignore_errors=True)

    # 保存汇总指标到 CSV
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(os.path.join(results_base_dir, 'ablation_comparison.csv'), index=False)

    # 绘制各维度 RMSE 对比柱状图
    dim_names = metrics_list[0]['dim_names'] if metrics_list else []
    n_dims = len(dim_names)
    modes = [m['ablation_mode'] for m in metrics_list]
    rmse_data = np.array([m['rmse_per_dim'] for m in metrics_list])

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_dims)
    width = 0.8 / len(modes)
    for i, mode in enumerate(modes):
        offset = (i - len(modes) / 2 + 0.5) * width
        ax.bar(x + offset, rmse_data[i], width, label=mode, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names, rotation=45, ha='right')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE per Dimension for Different Ablation Modes')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_base_dir, 'ablation_rmse_comparison.png'), dpi=300)
    plt.close()

    # 绘制总体指标对比（平均 RMSE 和不确定性相关性）
    if 'mean_rmse' in metrics_list[0] and 'corr_uncertainty' in metrics_list[0]:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        mean_rmse = [m['mean_rmse'] for m in metrics_list]
        corr = [m['corr_uncertainty'] for m in metrics_list]
        ax2.plot(modes, mean_rmse, marker='o', label='Mean RMSE')
        ax2.plot(modes, corr, marker='s', label='Uncertainty Correlation')
        ax2.set_ylabel('Value')
        ax2.set_title('Overall Performance Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_base_dir, 'ablation_overall_comparison.png'), dpi=300)
        plt.close()

    print(f"对比结果已保存至: {results_base_dir}")
    return metrics_list