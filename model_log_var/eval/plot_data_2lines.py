import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Set font to avoid display issues
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def load_test_data(data_dir):
    """从保存的测试数据加载"""
    # 尝试从 .npz 文件加载
    npz_path = os.path.join(data_dir, 'test_results.npz')

    if not os.path.exists(npz_path):
        print(f"❌ 找不到数据文件: {npz_path}")
        return None, None, None

    # 加载数据
    data = np.load(npz_path, allow_pickle=True)
    y_true = data['y_true']
    y_pred = data['y_pred']
    dim_names = data['dim_names']

    print(f"✅ 加载数据: {data_dir}")
    print(f"   真值形状: {y_true.shape}")
    print(f"   预测值形状: {y_pred.shape}")

    return y_true, y_pred, dim_names


def load_uncertainty_data(data_dir, n_dims):
    """加载不确定性（标准差）数据"""
    uncertainty_path = os.path.join(data_dir, 'all_dimensions_data.csv')

    if os.path.exists(uncertainty_path):
        print(f"✅ 加载不确定性数据: {uncertainty_path}")
        uncertainty_df = pd.read_csv(uncertainty_path)

        # 提取所有维度的不确定性数据
        uncertainty_data = {}

        for dim in range(n_dims):
            col_name = f'uncertainty_dim{dim}'
            if col_name in uncertainty_df.columns:
                # log_std = uncertainty_df[col_name].values
                # std = np.exp(log_std)#*10
                # uncertainty_data[dim] = std
                uncertainty_data[dim] = uncertainty_df[col_name].values
        if uncertainty_data:
            print(f"   加载了 {len(uncertainty_data)} 个维度的不确定性数据")

            return uncertainty_data
        else:
            print("⚠️  未找到不确定性数据列")
            return None
    else:
        print(f"⚠️  找不到不确定性文件: {uncertainty_path}")
        return None


def load_metrics_data(data_dir):
    """从文件加载指标数据"""
    metrics_path = os.path.join(data_dir, 'detailed_metrics_per_dimension.csv')
    overall_path = os.path.join(data_dir, 'overall_metrics.csv')

    if os.path.exists(metrics_path):
        print(f"✅ 加载指标数据: {metrics_path}")
        metrics_df = pd.read_csv(metrics_path)
    else:
        print(f"⚠️  找不到指标文件: {metrics_path}")
        metrics_df = None

    if os.path.exists(overall_path):
        print(f"✅ 加载整体指标: {overall_path}")
        overall_df = pd.read_csv(overall_path)
        overall_metrics = overall_df.iloc[0].to_dict()
    else:
        print(f"⚠️  找不到整体指标文件: {overall_path}")
        overall_metrics = None

    return metrics_df, overall_metrics



def plot_all_dimensions_scatter(y_true, y_pred, metrics_df, dim_names, save_dir, uncertainty_data=None):
    """绘制所有维度的预测vs真实值散点图 - 使用不确定性着色"""
    os.makedirs(save_dir, exist_ok=True)
    n_dims = y_true.shape[1]

    # 只选择前14个维度
    selected_dim = list(range(min(14, n_dims)))

    # 创建3x5子图，调整图形尺寸为颜色条留出空间
    fig, axes = plt.subplots(2, 7, figsize=(64, 18),
                        subplot_kw={'aspect': 'equal'})
    axes = axes.ravel()

    # 设置颜色映射
    cmap = cm.viridis

    for i, dim in enumerate(selected_dim):
        ax = axes[i]
        true_vals = y_true[:, dim]
        pred_vals = y_pred[:, dim]

        # 根据是否有不确定性数据选择着色方式
        if uncertainty_data is not None and dim in uncertainty_data:
            # 使用绝对不确定性着色
            uncertainty = uncertainty_data[dim]

            # local_norm = Normalize(vmin=np.percentile(uncertainty, 5),
            #                     vmax=np.percentile(uncertainty, 95))
            local_norm = Normalize(vmin=uncertainty.min(), vmax=uncertainty.max())
            colors = cmap(local_norm(uncertainty))

            scatter = ax.scatter(true_vals, pred_vals, c=colors, alpha=0.6, s=10)

            # 计算不确定性与绝对误差的相关性
            abs_errors = np.abs(pred_vals - true_vals)

            # 安全地计算相关性
            if len(uncertainty) > 1:
                try:
                    # 检查是否有有效数据
                    valid_mask = np.isfinite(uncertainty) & np.isfinite(abs_errors)
                    if np.sum(valid_mask) > 1:
                        corr = np.corrcoef(np.exp(uncertainty[valid_mask]), abs_errors[valid_mask])[0, 1]
                    else:
                        corr = 0
                except:
                    corr = 0

                # if not np.isnan(corr):
                #     ax.text(0.05, 0.95, f'Corr={corr:.3f}', transform=ax.transAxes,
                #             fontsize=10, verticalalignment='top',
                #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # 计算平均不确定性
            mean_uncertainty = np.mean(uncertainty)
            uncertainty_info = f'\nAvg Unc: {mean_uncertainty:.3f}'

        else:
            # 没有不确定性数据，使用绝对误差着色
            abs_errors = np.abs(pred_vals - true_vals)

            # 创建子图独立的归一化
            # local_norm = Normalize(vmin=np.percentile(abs_errors, 5),
            #                     vmax=np.percentile(abs_errors, 95))
            local_norm = Normalize(vmin=abs_errors.min(), vmax=abs_errors.max())
            colors = cmap(local_norm(abs_errors))

            scatter = ax.scatter(true_vals, pred_vals, c=colors, alpha=0.6, s=10)

            # 计算平均绝对误差
            mean_abs_error = np.mean(abs_errors)
            uncertainty_info = f'\nAvg Error: {mean_abs_error:.3f}'

        # 绘制对角线（理想预测线）
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1.5)
        ax.tick_params(axis='both', labelsize=24)


        # 添加标题和指标
        if metrics_df is not None and not metrics_df.empty:
            metrics_df['dimension'] = metrics_df['dimension'].astype(int)
            dim_metrics = metrics_df[metrics_df['dimension'] == dim]
            if not dim_metrics.empty:
                r2 = dim_metrics['r2'].values[0]
                rmse = dim_metrics['rmse'].values[0]
                title = f'{dim_names[dim]}\nR²={r2:.3f}, RMSE={rmse:.2e}'
            else:
                title = f'{dim_names[dim]}'
        else:
            title = f'{dim_names[dim]}'

        ax.set_title(title, fontsize=26, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if i >= 7:  # 底部行
            ax.set_xlabel('True Value', fontsize=26, fontname='Times New Roman', fontweight='bold')
        if i % 7 == 0:  # 第一列
            ax.set_ylabel('Pred Value', fontsize=26, fontname='Times New Roman', fontweight='bold')

    # 隐藏未使用的子图
    for j in range(len(selected_dim), 14):
        axes[j].axis('off')

    # 添加全局颜色条参考
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.93, 0.55, 0.01, 0.4])

    # 创建归一化的颜色条（0-1范围）
    norm_ref = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_ref)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)

    if uncertainty_data is not None:
        cbar.set_label('Uncertainty', fontsize=28)
    else:
        cbar.set_label('Absolute Error', fontsize=14)

    # # 在颜色条上添加一些参考标记
    # cbar.set_ticks([0, 1])
    # cbar.set_ticklabels(['Low', 'High'])
    # cbar.ax.yaxis.tick_right()
    # cbar.ax.tick_params(labelsize=26)
    # 颜色条标签设置（正上方和正下方）
    cbar.set_ticks([])  # 隐藏默认刻度

    # # 在颜色条正下方添加"Low"
    # cbar.ax.text(0.5, -0.02, r'$\mathbf{Low}$',
    #              transform=cbar.ax.transAxes,
    #              ha='center', va='top', fontsize=26, fontname='Times New Roman',
    #              fontweight='bold')
    #
    # # 在颜色条正上方添加"High"
    # cbar.ax.text(0.5, 1.02, r'$\mathbf{High}$',
    #              transform=cbar.ax.transAxes,
    #              ha='center', va='bottom', fontsize=26, fontname='Times New Roman',
    #              fontweight='bold')
    cbar.ax.text(0.5, -0.02, 'Low', transform=cbar.ax.transAxes,
                 ha='center', va='top', fontsize=26, fontname='Times New Roman',
                 fontweight='bold')
    cbar.ax.text(0.5, 1.02, 'High', transform=cbar.ax.transAxes,
                 ha='center', va='bottom', fontsize=26, fontname='Times New Roman',
                 fontweight='bold')
    # 其他标签加粗示例
    cbar.set_label('Uncertainty', fontsize=28, fontname='Times New Roman', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.93, 1])  # 为颜色条留出右侧空间

    # 保存图像
    if uncertainty_data is not None:
        filename = 'all_dimensions_scatter_with_uncertainty.png'
        print("📊 Using absolute uncertainty coloring, each subplot independently normalized")
    else:
        filename = 'all_dimensions_scatter_with_error.png'
        print("📊 Using absolute error coloring, each subplot independently normalized")

    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Scatter plot saved: {os.path.join(save_dir, filename)}")

    return metrics_df



def print_summary(metrics_df, overall_metrics, dim_names):
    """打印结果摘要"""
    if metrics_df is None or overall_metrics is None:
        print("⚠️  没有指标数据，无法显示摘要")
        return

    print("\n" + "=" * 60)
    print("Overall Performance Metrics:")
    print("=" * 60)

    # 打印整体指标
    for key, value in overall_metrics.items():
        if 'rmse' in key or 'mae' in key:
            print(f"{key}: {value:.6e}")
        elif 'r2' in key or 'pearson' in key:
            print(f"{key}: {value:.4f}")
        elif 'smape' in key:
            print(f"{key}: {value:.2f}%")
        elif 'snr_db' in key:
            print(f"{key}: {value:.2f} dB")
        else:
            print(f"{key}: {value}")

    print("=" * 60)

    # 找出最佳和最差维度
    if not metrics_df.empty:
        best_dim = metrics_df.loc[metrics_df['r2'].idxmax()]
        worst_dim = metrics_df.loc[metrics_df['r2'].idxmin()]

        print(f"\nBest dimension (highest R²): Dimension {int(best_dim['dimension'])} "
              f"({dim_names[int(best_dim['dimension'])]}) - R²={best_dim['r2']:.4f}")
        print(f"Worst dimension (lowest R²): Dimension {int(worst_dim['dimension'])} "
              f"({dim_names[int(worst_dim['dimension'])]}) - R²={worst_dim['r2']:.4f}")

        # 按RMSE排序
        best_rmse_dim = metrics_df.loc[metrics_df['rmse'].idxmin()]
        worst_rmse_dim = metrics_df.loc[metrics_df['rmse'].idxmax()]

        print(f"\nMost accurate dimension (lowest RMSE): Dimension {int(best_rmse_dim['dimension'])} "
              f"- RMSE={best_rmse_dim['rmse']:.6e}")
        print(f"Least accurate dimension (highest RMSE): Dimension {int(worst_rmse_dim['dimension'])} "
              f"- RMSE={worst_rmse_dim['rmse']:.6e}")


def main():
    parser = argparse.ArgumentParser(description='Plot scatter plots from saved test data')
    parser.add_argument('--data_dir', type=str, default='test_results_14dim',
                        help='Directory containing saved test data')
    parser.add_argument('--save_dir', type=str, default='plots',
                        help='Directory to save plots')
    parser.add_argument('--use_uncertainty', action='store_false',
                        help='Use uncertainty data for coloring scatter points')
    args = parser.parse_args()

    print("🎨 Plotting scatter plots from saved test data...")

    if args.use_uncertainty:
        print("🎨 使用不确定性数据进行着色")
        print("📊 注意：现在使用相对不确定性（σ/|True|）进行着色")

    # 加载测试数据
    y_true, y_pred, dim_names = load_test_data(args.data_dir)

    if y_true is None or y_pred is None:
        print("❌ Failed to load test data")
        return

    # 保持你的dim_names不变
    dim_names = [
        r"$\mathbf{a_1}$", r"$\mathbf{a_2}$", r"$\mathbf{a_3}$",
        r"$\mathbf{\epsilon_1}$", r"$\mathbf{\epsilon_2}$",
        r"$\mathbf{T_x}$", r"$\mathbf{T_y}$", r"$\mathbf{T_z}$",
        r"$\mathbf{R_{ox}}$", r"$\mathbf{R_{oy}}$", r"$\mathbf{R_{oz}}$",
        r"$\mathbf{V_o}$", r"$\mathbf{l_o}$", r"$\mathbf{r_o}$", r"$\mathbf{r_o}$", r"$\mathbf{r_o}$"
    ]

    # 加载指标数据
    metrics_df, overall_metrics = load_metrics_data(args.data_dir)

    # 加载不确定性数据（如果需要）
    uncertainty_data = None
    if args.use_uncertainty:
        n_dims = y_true.shape[1]
        uncertainty_data = load_uncertainty_data(args.data_dir, n_dims)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 绘制散点图（带或不带不确定性着色）
    plot_all_dimensions_scatter(y_true, y_pred, metrics_df, dim_names, args.save_dir, uncertainty_data)

    # 打印结果摘要
    print_summary(metrics_df, overall_metrics, dim_names)

    print(f"\n✅ Plotting completed!")
    if args.use_uncertainty:
        print(f"   带相对不确定性着色的散点图已保存到: {args.save_dir}")
    else:
        print(f"   带相对误差着色的散点图已保存到: {args.save_dir}")


if __name__ == '__main__':
    main()