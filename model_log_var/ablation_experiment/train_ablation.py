# train_ablation.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from shape_predictor_ablation import ShapePredictor, Loss  # 使用消融版模型
import os
from datetime import datetime
from evaluation_ablation import plot_predictions_vs_truth, compare_ablation_models


def train_single(ablation_mode, results_base_dir='training_results'):
    """训练单个消融模式模型"""
    # 创建该模式的专属结果子目录
    mode_results_dir = os.path.join(results_base_dir, ablation_mode)
    os.makedirs(mode_results_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"开始训练消融模式: {ablation_mode}")
    print(f"结果保存至: {mode_results_dir}")
    print(f"{'=' * 60}")

    # 加载数据
    X_train = np.load('training_data_X.npy')
    y_train = np.load('training_data_y.npy')
    state_train = X_train[:, :14]
    control_train = X_train[:, 14:]

    # 转换为张量
    state_tensor = torch.FloatTensor(state_train)
    control_tensor = torch.FloatTensor(control_train)
    y_tensor = torch.FloatTensor(y_train)

    dataset = TensorDataset(state_tensor, control_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化模型（传入消融模式）
    model = ShapePredictor(input_dim=17, output_dim=14, ablation_mode=ablation_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    criterion = Loss(nll_weight=1.0, relative_weight=0.30)

    best_loss = float('inf')
    patience_counter = 0
    patience = 100
    max_epochs = 400

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch_state, batch_control, batch_y in train_loader:
            optimizer.zero_grad()
            pred_delta, pred_uncertainty = model(batch_state, batch_control)
            target_delta = batch_y - batch_state
            loss = criterion(pred_delta, pred_uncertainty, target_delta, batch_state)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        # 保存最佳模型（文件名带模式标识）
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_path = os.path.join(mode_results_dir, f'shape_predictor_{ablation_mode}_best.pth')
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d}/{max_epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}, Best: {best_loss:.6f}")

        if patience_counter >= patience:
            print(f"早停于 epoch {epoch}")
            break

    # 加载最佳模型并评估
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # 调用评估函数（保存详细图表到模式子目录）
    plot_predictions_vs_truth(model, train_loader, save_dir=mode_results_dir, suffix=ablation_mode, num_samples=1000)

    print(f"模式 {ablation_mode} 训练完成，最佳损失: {best_loss:.6f}\n")
    return best_loss


def main():
    # 定义四种消融模式
    ablation_modes = ['full', 'no_pose', 'no_geometry', 'only_shape_scale']
    results_base_dir = 'training_results_ablation'
    os.makedirs(results_base_dir, exist_ok=True)

    # 依次训练各模式
    for mode in ablation_modes:
        train_single(mode, results_base_dir)

    # 所有模式训练完成后，生成对比图表
    print("\n" + "=" * 60)
    print("所有消融实验完成，开始汇总对比评估...")
    print("=" * 60)

    # 调用评估模块中的对比函数（将在 evaluation.py 中实现）
    compare_ablation_models(results_base_dir, ablation_modes)

    print(f"\n全部完成！结果保存在 {results_base_dir}")


if __name__ == "__main__":
    main()