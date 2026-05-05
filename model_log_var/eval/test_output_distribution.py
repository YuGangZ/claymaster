# test_output_distribution.py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import os
from shape_predictor import ShapePredictor


def load_training_data():
    """加载训练数据"""
    print("加载训练数据...")

    X_train = np.load('training_data_X_rotation_clean_fixed.npy')
    y_train = np.load('training_data_y_rotation_clean_fixed.npy')

    print(f"原始数据维度:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # 分离原始状态和控制
    original_state_train = X_train[:, :16]  # 原始16维状态
    control_train = X_train[:, 16:]  # 3维控制

    # 从16维状态中移除flatness(索引13)和convexity(索引15)
    indices_to_remove = [13, 15]
    state_train = np.delete(original_state_train, indices_to_remove, axis=1)
    y_train = np.delete(y_train, indices_to_remove, axis=1)

    print(f"\n处理后数据维度:")
    print(f"  状态维度: {state_train.shape[1]} (期望14)")
    print(f"  控制维度: {control_train.shape[1]} (期望3)")
    print(f"  目标维度: {y_train.shape[1]} (期望14)")

    return state_train, control_train, y_train


def analyze_input_statistics(state_data, control_data):
    """分析输入数据的统计特性"""
    print("\n" + "=" * 80)
    print("输入数据统计信息")
    print("=" * 80)

    # 14维状态名称
    dim_names_14 = [
        'scale_a1', 'scale_a2', 'scale_a3',  # 0-2: 尺度参数
        'shape_epsilon1', 'shape_epsilon2',  # 3-4: 形状参数
        'translation_x', 'translation_y', 'translation_z',  # 5-7: 位置
        'euler_rx', 'euler_ry', 'euler_rz',  # 8-10: 旋转
        'volume', 'elongation', 'smoothness'  # 11-13: 几何特征
    ]

    print("\n状态数据各维度统计:")
    print(f"{'维度':<5} {'名称':<20} {'最小值':<15} {'最大值':<15} {'均值':<15} {'标准差':<15} {'量级':<10}")
    print("-" * 95)

    for i, name in enumerate(dim_names_14):
        dim_data = state_data[:, i]
        min_val = np.min(dim_data)
        max_val = np.max(dim_data)
        mean_val = np.mean(dim_data)
        std_val = np.std(dim_data)

        # 估计量级（以10为底的对数的绝对值）
        magnitude = np.floor(np.log10(np.abs(mean_val) + 1e-10))

        print(
            f"{i:<5} {name:<20} {min_val:<15.6e} {max_val:<15.6e} {mean_val:<15.6e} {std_val:<15.6e} 10^{magnitude:.0f}")

    print("\n控制数据统计:")
    control_names = ['control_1', 'control_2', 'control_3']
    for i, name in enumerate(control_names):
        dim_data = control_data[:, i]
        min_val = np.min(dim_data)
        max_val = np.max(dim_data)
        mean_val = np.mean(dim_data)
        std_val = np.std(dim_data)
        magnitude = np.floor(np.log10(np.abs(mean_val) + 1e-10))

        print(
            f"  {name}: 均值={mean_val:.6e}, 标准差={std_val:.6e}, 范围=[{min_val:.6e}, {max_val:.6e}], 量级≈10^{magnitude:.0f}")


def test_model_output_distribution(model, dataloader, num_samples=2000):
    """测试模型输出分布"""
    print("\n" + "=" * 80)
    print("模型输出分布分析")
    print("=" * 80)

    model.eval()

    # 收集预测结果
    all_pred_deltas = []
    all_pred_uncertainties = []
    all_target_deltas = []
    all_current_states = []
    all_pred_next_states = []

    with torch.no_grad():
        for batch_idx, (batch_state, batch_control, batch_y) in enumerate(dataloader):
            if len(all_pred_deltas) >= num_samples:
                break

            # 前向传播
            pred_delta, pred_uncertainty = model(batch_state, batch_control)

            # 计算目标变化量
            target_delta = batch_y - batch_state

            # 计算预测的下一状态
            pred_next = batch_state + pred_delta

            all_pred_deltas.append(pred_delta.numpy())
            all_pred_uncertainties.append(pred_uncertainty.numpy())
            all_target_deltas.append(target_delta.numpy())
            all_current_states.append(batch_state.numpy())
            all_pred_next_states.append(pred_next.numpy())

            if batch_idx % 10 == 0:
                print(f"  处理批次 {batch_idx}...")

    # 合并所有批次
    all_pred_deltas = np.vstack(all_pred_deltas)[:num_samples]
    all_pred_uncertainties = np.vstack(all_pred_uncertainties)[:num_samples]
    all_target_deltas = np.vstack(all_target_deltas)[:num_samples]
    all_current_states = np.vstack(all_current_states)[:num_samples]
    all_pred_next_states = np.vstack(all_pred_next_states)[:num_samples]

    print(f"\n收集了 {len(all_pred_deltas)} 个样本的预测结果")

    return {
        'pred_deltas': all_pred_deltas,
        'pred_uncertainties': all_pred_uncertainties,
        'target_deltas': all_target_deltas,
        'current_states': all_current_states,
        'pred_next_states': all_pred_next_states
    }


def analyze_delta_magnitude_distribution(deltas, name="Delta", dim_names=None, save_path=None):
    """
    分析delta值在各个数量级的分布

    参数:
        deltas: numpy数组, 形状为(n_samples, n_dims)
        name: 数据集的名称
        dim_names: 维度名称列表
        save_path: 保存统计结果的路径
    """
    print(f"\n{'=' * 80}")
    print(f"{name}值数量级分布分析")
    print('=' * 80)

    n_samples, n_dims = deltas.shape

    if dim_names is None:
        dim_names = [f"维度_{i}" for i in range(n_dims)]

    # 定义数量级范围
    magnitude_bins = np.arange(-15, 6, 1)  # 从10^-15到10^5
    magnitude_labels = [f"10^{int(m)}" for m in magnitude_bins]

    # 统计每个维度的数量级分布
    all_magnitude_counts = []
    all_magnitude_percentages = []

    print(f"\n{name}值各维度数量级分布:")
    print("-" * 100)

    for dim_idx, dim_name in enumerate(dim_names):
        dim_data = deltas[:, dim_idx]

        # 计算每个样本的数量级
        with np.errstate(divide='ignore', invalid='ignore'):
            magnitudes = np.floor(np.log10(np.abs(dim_data) + 1e-20))
            # 处理特殊情况
            magnitudes[np.isinf(magnitudes)] = -20  # 0值对应的数量级设为-20
            magnitudes[np.isnan(magnitudes)] = -20

        # 统计每个数量级的样本数
        magnitude_counts = np.zeros(len(magnitude_bins))
        for i, mag_bin in enumerate(magnitude_bins):
            if i == len(magnitude_bins) - 1:
                # 最后一个bin，包含所有大于该值的
                magnitude_counts[i] = np.sum(magnitudes >= mag_bin)
            else:
                magnitude_counts[i] = np.sum((magnitudes >= mag_bin) & (magnitudes < magnitude_bins[i + 1]))

        # 计算百分比
        magnitude_percentages = magnitude_counts / n_samples * 100

        # 找到主要数量级（样本数最多的）
        main_magnitude_idx = np.argmax(magnitude_counts)
        main_magnitude = magnitude_bins[main_magnitude_idx]
        main_percentage = magnitude_percentages[main_magnitude_idx]

        print(f"{dim_name:<20}: 主要数量级=10^{main_magnitude:.0f}, 占比={main_percentage:.1f}%")

        all_magnitude_counts.append(magnitude_counts)
        all_magnitude_percentages.append(magnitude_percentages)

    # 绘制数量级分布热图
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # 1. 样本数热图
    counts_matrix = np.array(all_magnitude_counts).T
    im1 = axes[0].imshow(np.log1p(counts_matrix), aspect='auto', cmap='YlOrRd')
    axes[0].set_xlabel('维度')
    axes[0].set_ylabel('数量级')
    axes[0].set_title(f'{name}值数量级分布热图（样本数对数）')
    axes[0].set_xticks(range(len(dim_names)))
    axes[0].set_xticklabels(dim_names, rotation=45, ha='right')
    axes[0].set_yticks(range(len(magnitude_labels)))
    axes[0].set_yticklabels(magnitude_labels)
    plt.colorbar(im1, ax=axes[0], label='log(样本数+1)')

    # 2. 百分比热图
    percentages_matrix = np.array(all_magnitude_percentages).T
    im2 = axes[1].imshow(percentages_matrix, aspect='auto', cmap='Blues', vmin=0, vmax=100)
    axes[1].set_xlabel('维度')
    axes[1].set_ylabel('数量级')
    axes[1].set_title(f'{name}值数量级分布热图（百分比）')
    axes[1].set_xticks(range(len(dim_names)))
    axes[1].set_xticklabels(dim_names, rotation=45, ha='right')
    axes[1].set_yticks(range(len(magnitude_labels)))
    axes[1].set_yticklabels(magnitude_labels)
    plt.colorbar(im2, ax=axes[1], label='百分比 (%)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n热图已保存到: {save_path}")

    plt.close()

    # 保存详细统计结果到文件
    if save_path:
        # 获取文件目录和基础名称
        dir_name = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        base_name = os.path.basename(save_path).replace('.png', '')
        stats_file = os.path.join(dir_name, f"{base_name}_stats.txt")

        with open(stats_file, 'w') as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"{name}值数量级分布详细统计\n")
            f.write(f"{'=' * 80}\n\n")

            f.write(f"总样本数: {n_samples}\n")
            f.write(f"维度数: {n_dims}\n\n")

            for dim_idx, dim_name in enumerate(dim_names):
                f.write(f"\n维度: {dim_name}\n")
                f.write(f"{'-' * 60}\n")
                f.write(f"{'数量级':<10} {'样本数':<15} {'百分比':<15}\n")
                f.write(f"{'-' * 60}\n")

                for i, mag_bin in enumerate(magnitude_bins):
                    count = all_magnitude_counts[dim_idx][i]
                    percentage = all_magnitude_percentages[dim_idx][i]

                    if count > 0:  # 只记录有样本的数量级
                        f.write(f"10^{mag_bin:<4.0f} {count:<15.0f} {percentage:<15.2f}\n")

            # 总结统计
            f.write(f"\n{'=' * 80}\n")
            f.write("总结统计\n")
            f.write(f"{'=' * 80}\n\n")

            f.write("各维度主要数量级:\n")
            for dim_idx, dim_name in enumerate(dim_names):
                main_magnitude_idx = np.argmax(all_magnitude_counts[dim_idx])
                main_magnitude = magnitude_bins[main_magnitude_idx]
                main_percentage = all_magnitude_percentages[dim_idx][main_magnitude_idx]
                f.write(f"  {dim_name:<20}: 10^{main_magnitude:.0f} ({main_percentage:.1f}%)\n")

        print(f"详细统计已保存到: {stats_file}")

    return {
        'magnitude_bins': magnitude_bins,
        'magnitude_labels': magnitude_labels,
        'magnitude_counts': all_magnitude_counts,
        'magnitude_percentages': all_magnitude_percentages,
        'dim_names': dim_names
    }


def compare_delta_magnitude_distributions(pred_deltas, target_deltas, dim_names, save_dir='magnitude_comparison'):
    """
    比较预测delta和目标delta的数量级分布
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("预测delta vs 目标delta数量级分布比较")
    print('=' * 80)

    # 分析预测delta的数量级分布
    pred_stats = analyze_delta_magnitude_distribution(
        pred_deltas,
        name="预测Delta",
        dim_names=dim_names,
        save_path=os.path.join(save_dir, 'pred_delta_magnitude_heatmap.png')
    )

    # 分析目标delta的数量级分布
    target_stats = analyze_delta_magnitude_distribution(
        target_deltas,
        name="目标Delta",
        dim_names=dim_names,
        save_path=os.path.join(save_dir, 'target_delta_magnitude_heatmap.png')
    )

    # 比较主要数量级
    print(f"\n{'=' * 80}")
    print("主要数量级比较")
    print('=' * 80)

    comparison_results = []

    for dim_idx, dim_name in enumerate(dim_names):
        # 预测的主要数量级
        pred_main_idx = np.argmax(pred_stats['magnitude_counts'][dim_idx])
        pred_main_mag = pred_stats['magnitude_bins'][pred_main_idx]
        pred_main_percentage = pred_stats['magnitude_percentages'][dim_idx][pred_main_idx]

        # 目标的主要数量级
        target_main_idx = np.argmax(target_stats['magnitude_counts'][dim_idx])
        target_main_mag = target_stats['magnitude_bins'][target_main_idx]
        target_main_percentage = target_stats['magnitude_percentages'][dim_idx][target_main_idx]

        # 数量级差异
        mag_difference = pred_main_mag - target_main_mag

        # 评估匹配程度
        if abs(mag_difference) <= 1:
            match_status = "✅ 匹配良好"
        elif abs(mag_difference) <= 2:
            match_status = "⚠️  基本匹配"
        else:
            match_status = "❌ 差异较大"

        comparison_results.append({
            'dim_name': dim_name,
            'pred_magnitude': pred_main_mag,
            'pred_percentage': pred_main_percentage,
            'target_magnitude': target_main_mag,
            'target_percentage': target_main_percentage,
            'difference': mag_difference,
            'match_status': match_status
        })

        print(f"{dim_name:<20}: 预测=10^{pred_main_mag:.0f}({pred_main_percentage:.1f}%) | "
              f"目标=10^{target_main_mag:.0f}({target_main_percentage:.1f}%)")

    # 绘制比较图
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. 主要数量级对比条形图
    pred_magnitudes = [r['pred_magnitude'] for r in comparison_results]
    target_magnitudes = [r['target_magnitude'] for r in comparison_results]

    x = np.arange(len(dim_names))
    width = 0.35

    axes[0, 0].bar(x - width / 2, pred_magnitudes, width, label='预测', color='skyblue', alpha=0.7)
    axes[0, 0].bar(x + width / 2, target_magnitudes, width, label='目标', color='lightcoral', alpha=0.7)
    axes[0, 0].set_xlabel('维度')
    axes[0, 0].set_ylabel('主要数量级 (10^x)')
    axes[0, 0].set_title('主要数量级对比')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(dim_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. 数量级差异热图
    differences = [r['difference'] for r in comparison_results]
    diff_matrix = np.array(differences).reshape(1, -1)
    im = axes[0, 1].imshow(diff_matrix, aspect='auto', cmap='RdYlBu_r',
                           vmin=-5, vmax=5)
    axes[0, 1].set_title('数量级差异 (预测-目标)')
    axes[0, 1].set_xlabel('维度')
    axes[0, 1].set_xticks(range(len(dim_names)))
    axes[0, 1].set_xticklabels(dim_names, rotation=45, ha='right')
    axes[0, 1].set_yticks([])
    plt.colorbar(im, ax=axes[0, 1], label='数量级差异')

    # 3. 匹配状态统计
    status_counts = {}
    for r in comparison_results:
        status = r['match_status']
        status_counts[status] = status_counts.get(status, 0) + 1

    colors = {'✅ 匹配良好': 'lightgreen', '⚠️  基本匹配': 'orange', '❌ 差异较大': 'lightcoral'}

    statuses = list(status_counts.keys())
    counts = list(status_counts.values())
    bar_colors = [colors.get(s, 'gray') for s in statuses]

    axes[1, 0].bar(statuses, counts, color=bar_colors, alpha=0.7)
    axes[1, 0].set_title('匹配状态统计')
    axes[1, 0].set_ylabel('维度数量')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 添加计数标签
    for i, count in enumerate(counts):
        axes[1, 0].text(i, count + 0.1, str(count), ha='center', va='bottom')

    # 4. 维度级别的详细比较
    cell_text = []
    for r in comparison_results:
        cell_text.append([
            f"{r['pred_magnitude']:.0f}",
            f"{r['pred_percentage']:.1f}%",
            f"{r['target_magnitude']:.0f}",
            f"{r['target_percentage']:.1f}%",
            f"{r['difference']:+.1f}",
            r['match_status']
        ])

    axes[1, 1].axis('off')
    table = axes[1, 1].table(
        cellText=cell_text,
        colLabels=['预测数量级', '预测占比', '目标数量级', '目标占比', '差异', '匹配状态'],
        colWidths=[0.12, 0.12, 0.12, 0.12, 0.1, 0.2],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    plt.suptitle('预测Delta vs 目标Delta数量级分布比较', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'magnitude_comparison_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存比较结果
    comparison_file = os.path.join(save_dir, 'magnitude_comparison_summary.txt')
    with open(comparison_file, 'w') as f:
        f.write(f"{'=' * 80}\n")
        f.write("预测Delta vs 目标Delta数量级分布比较报告\n")
        f.write(f"{'=' * 80}\n\n")

        f.write("总体统计:\n")
        f.write(f"  总维度数: {len(dim_names)}\n")
        f.write(f"  匹配良好(差异≤1): {status_counts.get('✅ 匹配良好', 0)}\n")
        f.write(f"  基本匹配(差异≤2): {status_counts.get('⚠️  基本匹配', 0)}\n")
        f.write(f"  差异较大(差异>2): {status_counts.get('❌ 差异较大', 0)}\n\n")

        f.write("各维度详细比较:\n")
        f.write(f"{'-' * 120}\n")
        f.write(f"{'维度':<20} {'预测(主要)':<15} {'目标(主要)':<15} {'差异':<10} {'状态':<15}\n")
        f.write(f"{'-' * 120}\n")

        for r in comparison_results:
            f.write(f"{r['dim_name']:<20} "
                    f"10^{r['pred_magnitude']:.0f}({r['pred_percentage']:.1f}%) "
                    f"10^{r['target_magnitude']:.0f}({r['target_percentage']:.1f}%) "
                    f"{r['difference']:+.1f} "
                    f"{r['match_status']}\n")

        f.write(f"\n{'=' * 80}\n")
        f.write("分析结论:\n")
        f.write(f"{'=' * 80}\n")

        good_match = status_counts.get('✅ 匹配良好', 0)
        if good_match / len(dim_names) > 0.7:
            f.write("✅ 模型在大部分维度上的预测数量级与目标数量级匹配良好\n")
        elif good_match / len(dim_names) > 0.5:
            f.write("⚠️  模型在一半以上的维度上预测数量级与目标数量级匹配良好\n")
        else:
            f.write("❌ 模型在多数维度上的预测数量级与目标数量级差异较大，可能需要调整\n")

    print(f"\n比较报告已保存到: {comparison_file}")

    return comparison_results


def analyze_output_statistics(output_data):
    """详细分析输出数据的统计特性"""
    pred_deltas = output_data['pred_deltas']
    pred_uncertainties = output_data['pred_uncertainties']
    target_deltas = output_data['target_deltas']

    # 14维状态名称
    dim_names_14 = [
        'scale_a1', 'scale_a2', 'scale_a3',
        'shape_epsilon1', 'shape_epsilon2',
        'translation_x', 'translation_y', 'translation_z',
        'euler_rx', 'euler_ry', 'euler_rz',
        'volume', 'elongation', 'smoothness'
    ]

    print("\n预测变化量(delta)统计:")
    print(f"{'维度':<5} {'名称':<20} {'最小值':<15} {'最大值':<15} {'均值':<15} {'标准差':<15} {'量级':<10}")
    print("-" * 95)

    pred_delta_stats = []
    for i, name in enumerate(dim_names_14):
        dim_data = pred_deltas[:, i]
        min_val = np.min(dim_data)
        max_val = np.max(dim_data)
        mean_val = np.mean(dim_data)
        std_val = np.std(dim_data)
        magnitude = np.floor(np.log10(np.abs(mean_val) + 1e-10))

        pred_delta_stats.append({
            'name': name,
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'magnitude': magnitude
        })

        print(
            f"{i:<5} {name:<20} {min_val:<15.6e} {max_val:<15.6e} {mean_val:<15.6e} {std_val:<15.6e} 10^{magnitude:.0f}")

    print("\n目标变化量(delta)统计:")
    print(f"{'维度':<5} {'名称':<20} {'最小值':<15} {'最大值':<15} {'均值':<15} {'标准差':<15} {'量级':<10}")
    print("-" * 95)

    target_delta_stats = []
    for i, name in enumerate(dim_names_14):
        dim_data = target_deltas[:, i]
        min_val = np.min(dim_data)
        max_val = np.max(dim_data)
        mean_val = np.mean(dim_data)
        std_val = np.std(dim_data)
        magnitude = np.floor(np.log10(np.abs(mean_val) + 1e-10))

        target_delta_stats.append({
            'name': name,
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'magnitude': magnitude
        })

        print(
            f"{i:<5} {name:<20} {min_val:<15.6e} {max_val:<15.6e} {mean_val:<15.6e} {std_val:<15.6e} 10^{magnitude:.0f}")

    print("\n不确定性(log variance)统计:")
    print(f"{'维度':<5} {'名称':<20} {'最小值':<15} {'最大值':<15} {'均值':<15} {'标准差':<15}")
    print("-" * 95)

    uncertainty_stats = []
    for i, name in enumerate(dim_names_14):
        dim_data = pred_uncertainties[:, i]
        min_val = np.min(dim_data)
        max_val = np.max(dim_data)
        mean_val = np.mean(dim_data)
        std_val = np.std(dim_data)

        uncertainty_stats.append({
            'name': name,
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val
        })

        print(f"{i:<5} {name:<20} {min_val:<15.6e} {max_val:<15.6e} {mean_val:<15.6e} {std_val:<15.6e}")

    # 计算不确定性对应的标准差(实际物理量级)
    print("\n不确定性对应的标准差(std = sqrt(exp(log_var)))统计:")
    print(f"{'维度':<5} {'名称':<20} {'最小值':<15} {'最大值':<15} {'均值':<15} {'量级':<10}")
    print("-" * 95)

    for i, name in enumerate(dim_names_14):
        log_var_data = pred_uncertainties[:, i]
        std_data = np.sqrt(np.exp(log_var_data))
        min_std = np.min(std_data)
        max_std = np.max(std_data)
        mean_std = np.mean(std_data)
        magnitude = np.floor(np.log10(mean_std + 1e-10))

        print(f"{i:<5} {name:<20} {min_std:<15.6e} {max_std:<15.6e} {mean_std:<15.6e} 10^{magnitude:.0f}")

    # 计算预测误差
    errors = pred_deltas - target_deltas
    abs_errors = np.abs(errors)

    print("\n预测误差统计:")
    print(f"{'维度':<5} {'名称':<20} {'MAE':<15} {'RMSE':<15} {'最大误差':<15} {'相对误差':<15}")
    print("-" * 95)

    for i, name in enumerate(dim_names_14):
        mae = np.mean(abs_errors[:, i])
        rmse = np.sqrt(np.mean(errors[:, i] ** 2))
        max_err = np.max(abs_errors[:, i])

        # 计算相对误差（只对非零目标）
        target_abs = np.abs(target_deltas[:, i])
        non_zero_mask = target_abs > 1e-10
        if np.sum(non_zero_mask) > 0:
            relative_error = np.mean(abs_errors[non_zero_mask, i] / target_abs[non_zero_mask])
        else:
            relative_error = 0.0

        print(f"{i:<5} {name:<20} {mae:<15.6e} {rmse:<15.6e} {max_err:<15.6e} {relative_error:<15.6e}")

    return {
        'pred_delta_stats': pred_delta_stats,
        'target_delta_stats': target_delta_stats,
        'uncertainty_stats': uncertainty_stats,
        'errors': errors,
        'abs_errors': abs_errors
    }


def plot_output_distributions(output_data, stats_results, save_dir='output_distribution_plots'):
    """绘制输出分布图"""
    os.makedirs(save_dir, exist_ok=True)

    pred_deltas = output_data['pred_deltas']
    target_deltas = output_data['target_deltas']
    pred_uncertainties = output_data['pred_uncertainties']
    errors = stats_results['errors']

    # 14维状态名称
    dim_names_14 = [
        'scale_a1', 'scale_a2', 'scale_a3',
        'shape_epsilon1', 'shape_epsilon2',
        'translation_x', 'translation_y', 'translation_z',
        'euler_rx', 'euler_ry', 'euler_rz',
        'volume', 'elongation', 'smoothness'
    ]

    # 1. 预测delta vs 目标delta散点图（关键维度）
    key_dims = [0, 3, 5, 8, 11, 13]  # 选择关键维度
    key_names = ['Scale X', 'Shape Epsilon1', 'Position X', 'Rotation X', 'Volume', 'Smoothness']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (dim, name) in enumerate(zip(key_dims, key_names)):
        ax = axes[idx]

        # 散点图
        scatter = ax.scatter(target_deltas[:1000, dim], pred_deltas[:1000, dim],
                             alpha=0.6, s=10, c=pred_uncertainties[:1000, dim],
                             cmap='viridis')

        # 添加对角线
        min_val = min(target_deltas[:, dim].min(), pred_deltas[:, dim].min())
        max_val = max(target_deltas[:, dim].max(), pred_deltas[:, dim].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

        ax.set_xlabel(f'Target Δ{name}')
        ax.set_ylabel(f'Predicted Δ{name}')
        ax.set_title(f'{name}\n{dim_names_14[dim]}')
        ax.grid(True, alpha=0.3)

        # 添加R²分数
        from sklearn.metrics import r2_score
        r2 = r2_score(target_deltas[:, dim], pred_deltas[:, dim])
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if idx == 0:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Uncertainty (log var)')

    plt.suptitle('Predicted vs Target Delta Values (Key Dimensions)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'delta_scatter_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 误差分布直方图
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    for i in range(min(14, len(axes))):
        ax = axes[i]
        dim_errors = errors[:, i]

        # 移除极端异常值
        q1, q3 = np.percentile(dim_errors, [1, 99])
        filtered_errors = dim_errors[(dim_errors >= q1) & (dim_errors <= q3)]

        ax.hist(filtered_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        mean_error = np.mean(dim_errors)
        std_error = np.std(dim_errors)

        ax.set_title(f'{dim_names_14[i]}\nμ={mean_error:.2e}, σ={std_error:.2e}')
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(14, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Prediction Error Distributions (Filtered 1st-99th Percentile)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 不确定性分布
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    for i in range(min(14, len(axes))):
        ax = axes[i]
        dim_uncertainty = pred_uncertainties[:, i]

        ax.hist(dim_uncertainty, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')

        mean_unc = np.mean(dim_uncertainty)
        median_unc = np.median(dim_uncertainty)

        ax.axvline(x=mean_unc, color='red', linestyle='-', linewidth=2, alpha=0.7, label=f'Mean={mean_unc:.2f}')
        ax.axvline(x=median_unc, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Median={median_unc:.2f}')

        ax.set_title(f'{dim_names_14[i]}\nUncertainty (log var)')
        ax.set_xlabel('log(Variance)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=8)

    for i in range(14, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Uncertainty (log variance) Distributions', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'uncertainty_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 量级分析图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 预测delta的量级分布
    pred_magnitudes = []
    for i in range(14):
        mean_val = np.mean(np.abs(pred_deltas[:, i]))
        if mean_val > 0:
            magnitude = np.floor(np.log10(mean_val))
            pred_magnitudes.append(magnitude)

    axes[0].bar(range(len(pred_magnitudes)), pred_magnitudes, alpha=0.7, color='skyblue')
    axes[0].set_xlabel('Dimension Index')
    axes[0].set_ylabel('Magnitude (log10)')
    axes[0].set_title('Predicted Delta Magnitudes (Mean Absolute Value)')
    axes[0].grid(True, alpha=0.3, axis='y')

    # 不确定性的量级分布（转换为标准差）
    uncertainty_std_magnitudes = []
    for i in range(14):
        mean_log_var = np.mean(pred_uncertainties[:, i])
        mean_std = np.sqrt(np.exp(mean_log_var))
        if mean_std > 0:
            magnitude = np.floor(np.log10(mean_std))
            uncertainty_std_magnitudes.append(magnitude)

    axes[1].bar(range(len(uncertainty_std_magnitudes)), uncertainty_std_magnitudes, alpha=0.7, color='lightgreen')
    axes[1].set_xlabel('Dimension Index')
    axes[1].set_ylabel('Magnitude (log10)')
    axes[1].set_title('Uncertainty (std) Magnitudes')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'magnitude_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 误差与不确定性的关系
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    for i in range(min(14, len(axes))):
        ax = axes[i]

        # 采样以减少数据点
        sample_size = min(1000, len(errors))
        indices = np.random.choice(len(errors), sample_size, replace=False)

        ax.scatter(np.abs(errors[indices, i]), pred_uncertainties[indices, i],
                   alpha=0.5, s=10, color='purple')

        # 添加趋势线
        if sample_size > 10:
            x = np.abs(errors[indices, i])
            y = pred_uncertainties[indices, i]
            valid_mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(valid_mask) > 10:
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                try:
                    z = np.polyfit(x_valid, y_valid, 1)
                    p = np.poly1d(z)
                    x_sorted = np.sort(x_valid)
                    ax.plot(x_sorted, p(x_sorted), "r-", linewidth=2)

                    # 计算相关系数
                    corr = np.corrcoef(x_valid, y_valid)[0, 1]
                    ax.text(0.05, 0.95, f'Corr={corr:.3f}', transform=ax.transAxes,
                            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except:
                    pass

        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Uncertainty (log var)')
        ax.set_title(f'{dim_names_14[i]}')
        ax.grid(True, alpha=0.3)

    for i in range(14, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Error vs Uncertainty Relationship', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_vs_uncertainty.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(stats_results, save_path='output_distribution_summary.txt'):
    """生成详细的统计报告"""
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("模型输出分布分析报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 关键发现总结\n")
        f.write("-" * 40 + "\n")

        # 分析预测delta的量级范围
        pred_means = [stat['mean'] for stat in stats_results['pred_delta_stats']]
        pred_stds = [stat['std'] for stat in stats_results['pred_delta_stats']]

        f.write(f"预测变化量(delta)的均值范围: {min(pred_means):.2e} 到 {max(pred_means):.2e}\n")
        f.write(f"预测变化量(delta)的标准差范围: {min(pred_stds):.2e} 到 {max(pred_stds):.2e}\n")

        # 分析不确定性的范围
        unc_means = [stat['mean'] for stat in stats_results['uncertainty_stats']]
        f.write(f"不确定性(log variance)的均值范围: {min(unc_means):.2f} 到 {max(unc_means):.2f}\n")

        # 计算不确定性对应的物理标准差范围
        stds_from_unc = [np.sqrt(np.exp(mean)) for mean in unc_means]
        f.write(f"不确定性对应的标准差范围: {min(stds_from_unc):.2e} 到 {max(stds_from_unc):.2e}\n")

        # 分析误差统计
        mae_values = np.mean(stats_results['abs_errors'], axis=0)
        rmse_values = np.sqrt(np.mean(stats_results['errors'] ** 2, axis=0))

        f.write(f"\n预测误差统计:\n")
        f.write(f"  平均绝对误差(MAE)范围: {min(mae_values):.2e} 到 {max(mae_values):.2e}\n")
        f.write(f"  均方根误差(RMSE)范围: {min(rmse_values):.2e} 到 {max(rmse_values):.2e}\n")

        # 分析量级
        pred_magnitudes = [stat['magnitude'] for stat in stats_results['pred_delta_stats']]
        f.write(f"\n预测变化量的量级(10的幂次)范围: {min(pred_magnitudes):.0f} 到 {max(pred_magnitudes):.0f}\n")

        f.write("\n2. 建议\n")
        f.write("-" * 40 + "\n")

        # 根据分析结果提供建议
        if max(pred_magnitudes) < -2:
            f.write("✅ 预测输出量级很小(<10^-2)，模型可能适合小变化预测\n")
        elif max(pred_magnitudes) < 0:
            f.write("✅ 预测输出量级适中(10^-2到1)，适合大多数物理系统\n")
        else:
            f.write("⚠️  预测输出量级较大(>1)，可能需要检查数据或调整模型\n")

        # 检查不确定性是否合理
        avg_unc = np.mean(unc_means)
        if avg_unc < -5:
            f.write("⚠️  不确定性值非常小(log_var < -5)，可能过于自信\n")
        elif avg_unc > 2:
            f.write("⚠️  不确定性值较大(log_var > 2)，预测可能不够准确\n")
        else:
            f.write("✅ 不确定性值在合理范围内\n")

        # 检查误差与不确定性的关系
        avg_mae = np.mean(mae_values)
        avg_std_from_unc = np.mean(stds_from_unc)
        ratio = avg_std_from_unc / avg_mae if avg_mae > 0 else 0

        f.write(f"\n误差与不确定性关系:\n")
        f.write(f"  平均绝对误差: {avg_mae:.2e}\n")
        f.write(f"  平均预测标准差: {avg_std_from_unc:.2e}\n")
        f.write(f"  标准差/MAE比率: {ratio:.2f}\n")

        if 0.5 < ratio < 2.0:
            f.write("✅ 不确定性与误差匹配良好\n")
        else:
            f.write("⚠️  不确定性与误差匹配可能不够理想\n")

        f.write("\n3. 各维度详细统计\n")
        f.write("-" * 40 + "\n")

        dim_names_14 = [
            'scale_a1', 'scale_a2', 'scale_a3',
            'shape_epsilon1', 'shape_epsilon2',
            'translation_x', 'translation_y', 'translation_z',
            'euler_rx', 'euler_ry', 'euler_rz',
            'volume', 'elongation', 'smoothness'
        ]

        for i, name in enumerate(dim_names_14):
            f.write(f"\n维度 {i}: {name}\n")
            f.write(f"  预测delta: 均值={stats_results['pred_delta_stats'][i]['mean']:.2e}, "
                    f"标准差={stats_results['pred_delta_stats'][i]['std']:.2e}\n")
            f.write(f"  目标delta: 均值={stats_results['target_delta_stats'][i]['mean']:.2e}, "
                    f"标准差={stats_results['target_delta_stats'][i]['std']:.2e}\n")
            f.write(f"  不确定性: 均值={stats_results['uncertainty_stats'][i]['mean']:.2f}, "
                    f"范围=[{stats_results['uncertainty_stats'][i]['min']:.2f}, "
                    f"{stats_results['uncertainty_stats'][i]['max']:.2f}]\n")
            f.write(f"  预测误差: MAE={mae_values[i]:.2e}, RMSE={rmse_values[i]:.2e}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")

    print(f"\n详细报告已保存到: {save_path}")


def main():
    """主函数"""
    print("开始分析模型输出分布...")

    # 1. 加载数据
    state_data, control_data, target_data = load_training_data()

    # 2. 分析输入数据统计
    analyze_input_statistics(state_data, control_data)

    # 3. 创建数据加载器（使用部分数据）
    num_test_samples = 2000
    test_indices = np.random.choice(len(state_data), min(num_test_samples, len(state_data)), replace=False)

    test_state = torch.FloatTensor(state_data[test_indices])
    test_control = torch.FloatTensor(control_data[test_indices])
    test_target = torch.FloatTensor(target_data[test_indices])

    test_dataset = TensorDataset(test_state, test_control, test_target)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. 加载模型
    print("\n" + "=" * 80)
    print("加载预训练模型...")

    model = ShapePredictor(input_dim=17, output_dim=14)

    try:
        # 尝试加载最佳模型
        model.load_state_dict(torch.load('shape_predictor_best.pth', map_location='cpu'))
        print("✅ 成功加载 shape_predictor_best.pth")
    except FileNotFoundError:
        try:
            # 尝试加载最终模型
            model.load_state_dict(torch.load('shape_predictor.pth', map_location='cpu'))
            print("✅ 成功加载 shape_predictor.pth")
        except FileNotFoundError:
            print("❌ 未找到预训练模型文件，使用随机初始化的模型")
            print("   请先运行 train.py 训练模型")

    # 5. 测试模型输出
    print("\n" + "=" * 80)
    print("运行模型进行预测...")
    output_data = test_model_output_distribution(model, test_loader, num_samples=2000)

    # 6. 分析输出统计
    stats_results = analyze_output_statistics(output_data)

    # 7. 绘制分布图
    print("\n" + "=" * 80)
    print("生成可视化图表...")
    plot_output_distributions(output_data, stats_results, save_dir='output_distribution_plots')

    # 8. 统计delta值数量级分布（新增功能）
    print("\n" + "=" * 80)
    print("分析Delta值数量级分布...")

    # 14维状态名称
    dim_names_14 = [
        'scale_a1', 'scale_a2', 'scale_a3',
        'shape_epsilon1', 'shape_epsilon2',
        'translation_x', 'translation_y', 'translation_z',
        'euler_rx', 'euler_ry', 'euler_rz',
        'volume', 'elongation', 'smoothness'
    ]

    # 统计预测delta和目标delta的数量级分布
    pred_deltas = output_data['pred_deltas']
    target_deltas = output_data['target_deltas']

    # 比较预测delta和目标delta的数量级分布
    comparison_results = compare_delta_magnitude_distributions(
        pred_deltas,
        target_deltas,
        dim_names_14,
        save_dir='magnitude_comparison'
    )

    # 9. 生成总结报告
    generate_summary_report(stats_results)

    print("\n" + "=" * 80)
    print("分析完成!")
    print(f"  1. 统计报告: output_distribution_summary.txt")
    print(f"  2. 可视化图表: output_distribution_plots/ 目录")
    print(f"  3. Delta数量级分析: magnitude_comparison/ 目录")
    print(f"  4. 测试样本数: {len(output_data['pred_deltas'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
