# [file name]: evaluate_both.py
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))


def evaluate_both_controllers():
    """同时评估MPC和RL控制器"""
    print("=" * 80)
    print("软体变形控制器综合评估")
    print("=" * 80)

    # 1. 评估MPC
    print("\n1. 评估MPC控制器...")
    from evaluate_mpc1 import MPCEvaluator

    mpc_evaluator = MPCEvaluator(
        data_dir="../mpc_control_data",
        output_dir="mpc_evaluation_results"
    )
    mpc_results = mpc_evaluator.generate_comprehensive_report()

    # 2. 评估RL
    print("\n2. 评估RL控制器...")
    from evaluate_rl import RLEvaluator

    rl_evaluator = RLEvaluator(
        model_path="sac_deformation_precise.zip",
        output_dir="rl_evaluation_results"
    )
    rl_results = rl_evaluator.run_complete_evaluation(
        num_episodes=10,
        mpc_evaluator=mpc_evaluator
    )

    # 3. 对比分析
    print("\n3. 创建对比分析...")
    create_comparison_report(mpc_results, rl_results)

    print("\n评估完成！")
    print("MPC结果保存在: mpc_evaluation_results/")
    print("RL结果保存在: rl_evaluation_results/")
    print("对比报告: comparison_report/")


def create_comparison_report(mpc_results, rl_results):
    """创建对比报告"""
    output_dir = Path("comparison_report")
    output_dir.mkdir(exist_ok=True)

    import matplotlib.pyplot as plt
    import numpy as np

    # 提取数据
    mpc_error = mpc_results.get('basic_metrics', {}).get('total_errors', {}).get('final', float('inf'))
    mpc_improvement = mpc_results.get('basic_metrics', {}).get('total_errors', {}).get('improvement', 0)

    rl_success = rl_results.get('evaluation_metrics', {}).get('success_rate', 0)
    rl_error = rl_results.get('evaluation_metrics', {}).get('average_final_error', float('inf'))

    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 性能对比柱状图
    categories = ['MPC', 'RL']
    error_values = [mpc_error, rl_error]
    axes[0].bar(categories, error_values, color=['blue', 'green'])
    axes[0].set_title('Final Error Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Error')
    axes[0].grid(True, alpha=0.3)

    # 添加数值标签
    for i, v in enumerate(error_values):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center')

    # RL成功率
    axes[1].bar(['Success', 'Failure'], [rl_success, 1 - rl_success], color=['green', 'red'])
    axes[1].set_title(f'RL Success Rate: {rl_success:.1%}', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Proportion')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 创建文本报告
    report = []
    report.append("=" * 80)
    report.append("MPC vs RL 控制器综合对比报告")
    report.append("=" * 80)

    report.append("\nMPC控制器性能:")
    if mpc_error != float('inf'):
        report.append(f"  最终误差: {mpc_error:.6f}")
        report.append(f"  误差改善率: {mpc_improvement:.2f}%")

    report.append("\nRL控制器性能:")
    report.append(f"  成功率: {rl_success:.1%}")
    if rl_error != float('inf'):
        report.append(f"  平均最终误差: {rl_error:.6f}")

    report.append("\n对比分析:")
    if rl_error < mpc_error:
        report.append("  ✓ RL策略在精度上优于MPC")
    else:
        report.append("  ○ MPC在精度上表现更好")

    if rl_success > 0.8:
        report.append("  ✓ RL策略具有高成功率")

    report.append("\n建议:")
    if rl_success > 0.8 and rl_error < mpc_error:
        report.append("  - RL策略在各方面均表现优异，推荐使用")
    elif mpc_improvement > 70:
        report.append("  - MPC收敛性好，适合确定性环境")
    else:
        report.append("  - 两种方法各有优劣，建议根据应用场景选择")

    report.append("\n" + "=" * 80)

    # 保存报告
    with open(output_dir / 'comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(report))

    print(f"对比报告已保存至: {output_dir}/")


if __name__ == "__main__":
    evaluate_both_controllers()