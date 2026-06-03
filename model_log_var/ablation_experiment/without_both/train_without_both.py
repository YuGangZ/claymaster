import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from shape_predictor_ablation import *
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
import seaborn as sns
import argparse   # 用于接收消融模式参数

def train(ablation_mode='full'):
    # 根据消融模式创建结果子目录
    results_dir = f'training_results_{ablation_mode}'
    os.makedirs(results_dir, exist_ok=True)

    start_time = datetime.now()
    print(f"\n========== 开始消融实验: {ablation_mode} ==========")

    # 加载数据
    X_train = np.load(r'..\training_data_X.npy')
    y_train = np.load(r'..\training_data_y.npy')
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params:,} (消融模式: {ablation_mode})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    criterion = Loss(nll_weight=1.0, relative_weight=0.30)

    best_loss = float('inf')
    patience_counter = 0
    patience = 100
    max_epochs = 600

    loss_history = []
    # 保存最佳模型时使用带消融模式的文件名
    best_model_path = f'shape_predictor_{ablation_mode}_best.pth'
    checkpoint_path = f'shape_predictor_{ablation_mode}_checkpoint.pth'

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
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d}/{max_epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}, Best: {best_loss:.6f}")
            if hasattr(criterion, 'loss_components'):
                comp_str = ', '.join([f'{k}: {v:.6f}' for k, v in criterion.loss_components.items()])
                print(f"  Components - {comp_str}")

        if patience_counter >= patience:
            print(f"早停于 epoch {epoch}")
            break

    # 保存最终模型
    torch.save(model.state_dict(), f'shape_predictor_{ablation_mode}.pth')

    # 简单可视化（可选）
    plot_predictions_vs_truth(model, train_loader, results_dir)

    training_time = datetime.now() - start_time
    print(f"\n===== 消融模式 {ablation_mode} 完成 =====")
    print(f"训练时间: {training_time}")
    print(f"最佳损失: {best_loss:.6f}")
    print(f"模型保存为: {best_model_path}")
    print("="*60)

def plot_predictions_vs_truth(model, dataloader, save_dir):
    """与原代码类似的简单绘图，此处略作示意"""
    model.eval()
    with torch.no_grad():
        sample_state, sample_control, sample_y = next(iter(dataloader))
        pred_delta, _ = model(sample_state, sample_control)
        target_delta = sample_y - sample_state
        pred_next = sample_state + pred_delta
        target_next = sample_y

        # 画前4个维度的对比图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i, ax in enumerate(axes.flatten()):
            if i < 4:
                ax.plot(target_next[:, i].cpu(), label='True', marker='o', linestyle='-')
                ax.plot(pred_next[:, i].cpu(), label='Pred', marker='x', linestyle='--')
                ax.set_title(f'Dimension {i}')
                ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'predictions.png'))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='only_shape_scale',
                        choices=['full', 'no_pose', 'no_geometry', 'only_shape_scale'],
                        help='消融模式')
    args = parser.parse_args()
    train(ablation_mode=args.mode)