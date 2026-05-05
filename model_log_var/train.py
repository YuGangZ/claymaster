# train.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from shape_predictor import *
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
import seaborn as sns
from evaluation import *


def train():
    results_dir = 'training_results'
    os.makedirs(results_dir, exist_ok=True)

    start_time = datetime.now()

    X_train = np.load('training_data_X.npy')
    y_train = np.load('training_data_y.npy')

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    original_state_train = X_train[:, :16]
    control_train = X_train[:, 16:]

    dim_names_16 = [
        'scale_a1', 'scale_a2', 'scale_a3',
        'shape_epsilon1', 'shape_epsilon2',
        'translation_x', 'translation_y', 'translation_z',
        'euler_rx', 'euler_ry', 'euler_rz',
        'volume', 'elongation', 'flatness', 'smoothness', 'convexity'
    ]

    for i, name in enumerate(dim_names_16):
        print(f"  索引 {i:2d}: {name}")

    indices_to_remove = [13, 15]
    state_train = np.delete(original_state_train, indices_to_remove, axis=1)

    print(f"\n移除的维度: {[dim_names_16[i] for i in indices_to_remove]}")
    print(f"保留的维度: {[name for i, name in enumerate(dim_names_16) if i not in indices_to_remove]}")

    # 同样处理目标y（下一帧状态）
    y_train = np.delete(y_train, indices_to_remove, axis=1)

    # 验证维度
    print("\n处理后数据维度:")
    print(f"state_train shape: {state_train.shape} (期望: (n_samples, 14))")
    print(f"control_train shape: {control_train.shape} (期望: (n_samples, 3))")
    print(f"y_train shape: {y_train.shape} (期望: (n_samples, 14))")

    # 检查维度是否正确
    if state_train.shape[1] != 14:
        print(f"警告: 状态维度不是14，实际是 {state_train.shape[1]}")
    if y_train.shape[1] != 14:
        print(f"警告: 目标维度不是14，实际是 {y_train.shape[1]}")

    # 打印保留的14维状态名称
    dim_names_14 = []
    for i, name in enumerate(dim_names_16):
        if i not in indices_to_remove:
            dim_names_14.append(name)

    print(f"\n14维状态名称:")
    for i, name in enumerate(dim_names_14):
        print(f"  索引 {i:2d}: {name}")

    # 直接使用原始数据，不进行归一化
    state_tensor = torch.FloatTensor(state_train)  # 14维当前状态
    control_tensor = torch.FloatTensor(control_train)  # 3维控制
    y_tensor = torch.FloatTensor(y_train)  # 14维目标状态

    # 打印数据统计信息
    print("\n" + "=" * 60)
    print("DATA STATISTICS (14维状态，已移除flatness和convexity)")
    print("=" * 60)

    # 打印各维度统计
    print("\n状态数据各维度均值和标准差:")
    for i in range(min(5, state_train.shape[1])):  # 只显示前5维
        mean_val = state_train[:, i].mean()
        std_val = state_train[:, i].std()
        print(f"  维度 {i}: {dim_names_14[i]}: 均值 = {mean_val:.6f}, 标准差 = {std_val:.6f}")

    if state_train.shape[1] > 5:
        print(f"  其他维度: ...")

    control_mean = control_train.mean(axis=0)
    control_std = control_train.std(axis=0)
    print(f"\n控制数据统计:")
    print(f"  均值: [{control_mean[0]:.4f}, {control_mean[1]:.4f}, {control_mean[2]:.4f}]")
    print(f"  标准差: [{control_std[0]:.4f}, {control_std[1]:.4f}, {control_std[2]:.4f}]")

    print(f"\n训练样本数量: {len(state_train)}")
    print(f"状态维度: {state_train.shape[1]}")
    print(f"控制维度: {control_train.shape[1]}")
    print(f"目标维度: {y_train.shape[1]}")
    print("=" * 60)

    # 构建DataLoader
    dataset = TensorDataset(state_tensor, control_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化模型和优化器
    # 注意：输入维度 = 14维状态 + 3维控制 = 17维
    # 输出维度 = 14维状态变化量
    model = ShapePredictor(input_dim=17, output_dim=14)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print("=" * 60)

    # 调整学习率，因为数据量级可能较大
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)  # 更小的学习率

    # 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )

    # 调整损失函数参数，适应原始物理量级
    criterion = Loss(
        nll_weight=1.0,
        relative_weight=0.30,
    )
    # criterion = Loss(
    #     nll_weight=0.3,  # 降低NLL权重
    #     mse_weight=0.6,  # 提高MSE权重
    #     relative_weight=0.1  # 降低相对损失权重
    # )


    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    patience = max(100, int(os.environ.get('EPOCHS', '200')))

    # 记录训练历史
    loss_history = []
    component_losses = {}

    print("\n开始训练 (14维状态，无归一化)...")
    max_epochs = int(os.environ.get('EPOCHS', '600'))

    # 记录每个epoch的学习率
    learning_rates = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0

        for batch_state, batch_control, batch_y in train_loader:
            optimizer.zero_grad()

            # 前向传播
            pred_delta, pred_uncertainty = model(batch_state, batch_control)
            # 直接计算目标变化量（非归一化）
            target_delta = batch_y - batch_state  # batch_state是原始物理量

            # 计算损失
            loss = criterion(pred_delta, pred_uncertainty, target_delta, batch_state)

            # 反向传播
            loss.backward()

            # 梯度裁剪（使用更严格的裁剪，因为梯度可能较大）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        # 记录组件损失
        if hasattr(criterion, 'loss_components'):
            for key in component_losses.keys():
                if key in criterion.loss_components:
                    component_losses[key].append(criterion.loss_components[key])

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'shape_predictor_best.pth')
            # 同时保存优化器状态以便恢复训练
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'shape_predictor_checkpoint.pth')
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d}/{max_epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}, Best: {best_loss:.6f}")

            # 打印组件损失
            # if hasattr(criterion, 'loss_components'):
            #     nll_loss = criterion.loss_components.get('nll', 0)
            #     mse_loss = criterion.loss_components.get('mse', 0)
            #     relative_loss = criterion.loss_components.get('relative', 0)
            #     print(f"  Components - NLL: {nll_loss:.6f}, MSE: {mse_loss:.6f}, Relative: {relative_loss:.6f}")
            if hasattr(criterion, 'loss_components'):
                components_str = ', '.join([f'{key}: {value:.6f}' for key, value in criterion.loss_components.items()])
                print(f"  Components - {components_str}")


        # 早停检查
        if patience_counter >= patience:
            print(f"早停触发于第 {epoch} 轮")
            break

    # 计算训练时间
    training_time = datetime.now() - start_time

    # 加载最佳模型
    model.load_state_dict(torch.load('shape_predictor_best.pth'))
    torch.save(model.state_dict(), 'shape_predictor.pth')

    # 绘制最终结果
    plot_predictions_vs_truth(model, train_loader, results_dir)

    print("=" * 60)
    print("单步预测模型训练完成并保存")
    print(f"训练时间: {training_time}")
    print(f"最佳损失: {best_loss:.6f}")
    print(f"最终损失: {loss_history[-1]:.6f}")
    print(f"状态维度: 14维 (已移除flatness和convexity，保留smoothness)")
    print(f"训练结果保存在: {results_dir}")
    print("=" * 60)



if __name__ == "__main__":
    train()