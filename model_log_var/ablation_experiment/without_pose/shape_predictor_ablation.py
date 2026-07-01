import numpy as np
import torch
import torch.nn as nn


class ShapePredictor(nn.Module):
    def __init__(self, input_dim=17, output_dim=14, ablation_mode='full'):
        """
        ablation_mode:
            'full'            -> 使用所有特征 (尺度、形状、位姿、几何)
            'no_pose'         -> 消融位姿 (仅保留尺度、形状、几何)
            'no_geometry'     -> 消融几何 (仅保留尺度、形状、位姿)
            'only_shape_scale'-> 仅保留尺度+形状 (消融位姿和几何)
        """
        super().__init__()
        self.ablation_mode = ablation_mode
        self.output_dim = output_dim

        # 1. 特征编码器（始终创建，但 forward 中可能被屏蔽）
        self.scale_encoder = self._create_scale_encoder()
        self.shape_encoder = self._create_shape_encoder()
        self.pose_encoder = self._create_pose_encoder()      # 可能被屏蔽
        self.geometry_encoder = self._create_geometry_encoder() # 可能被屏蔽
        self.control_encoder = self._create_control_encoder()

        # 2. 交叉注意力 (固定5个分支，特征维度不变)
        self.cross_attention = CrossAttentionModule(
            feature_dims=[32, 16, 24, 24, 16]   # scale, shape, pose, geometry, control
        )

        # 3. 融合网络
        self.fusion_network = FusionNetwork(
            input_dim=160,
            hidden_dims=[256, 128, 64],
            dropout_rates=[0.3, 0.2, 0.1]
        )

        # 4. 输出头
        self.delta_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim)
        )
        self.uncertainty_head = UncertaintyHead(input_dim=64, output_dim=output_dim)

        self._initialize_physics_aware()

    def _initialize_physics_aware(self):
        # 与原代码相同
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def _create_scale_encoder(self):
        return nn.Sequential(
            nn.Linear(3, 16), nn.LayerNorm(16), nn.LeakyReLU(0.01),
            nn.Linear(16, 32), nn.LayerNorm(32), nn.LeakyReLU(0.01)
        )

    def _create_shape_encoder(self):
        return nn.Sequential(
            nn.Linear(2, 8), nn.LayerNorm(8), nn.LeakyReLU(0.01),
            nn.Linear(8, 16), nn.LayerNorm(16), nn.LeakyReLU(0.01)
        )

    def _create_pose_encoder(self):
        class EulerToQuatLayer(nn.Module):
            def forward(self, x):
                # x: (batch, 6) -> position(3) + euler(3)
                position = x[:, :3]
                euler = x[:, 3:]
                half = euler / 2.0
                cx, sx = torch.cos(half[:, 0]), torch.sin(half[:, 0])
                cy, sy = torch.cos(half[:, 1]), torch.sin(half[:, 1])
                cz, sz = torch.cos(half[:, 2]), torch.sin(half[:, 2])
                qw = cx*cy*cz + sx*sy*sz
                qx = sx*cy*cz - cx*sy*sz
                qy = cx*sy*cz + sx*cy*sz
                qz = cx*cy*sz - sx*sy*cz
                norm = torch.sqrt(qw**2 + qx**2 + qy**2 + qz**2 + 1e-8)
                qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
                quat = torch.stack([qw, qx, qy, qz], dim=1)
                return torch.cat([position, quat], dim=1)
        return nn.Sequential(
            EulerToQuatLayer(),
            nn.Linear(7, 32), nn.LayerNorm(32), nn.LeakyReLU(0.01),
            nn.Linear(32, 24), nn.LayerNorm(24), nn.LeakyReLU(0.01)
        )

    def _create_geometry_encoder(self):
        return nn.Sequential(
            nn.Linear(3, 16), nn.LayerNorm(16), nn.LeakyReLU(0.01),
            nn.Linear(16, 24), nn.LayerNorm(24), nn.LeakyReLU(0.01)
        )

    def _create_control_encoder(self):
        return nn.Sequential(
            nn.Linear(3, 8), nn.LayerNorm(8), nn.LeakyReLU(0.01),
            nn.Linear(8, 16), nn.LayerNorm(16), nn.LeakyReLU(0.01)
        )

    def forward(self, current_state, control):
        batch_size = current_state.shape[0]
        device = current_state.device

        # 始终计算的尺度、形状、控制特征
        scale_feat = self.scale_encoder(current_state[:, :3])
        shape_feat = self.shape_encoder(current_state[:, 3:5])
        control_feat = self.control_encoder(control)

        # ----- 位姿特征（根据消融模式决定是否计算）-----
        if self.ablation_mode in ['full', 'no_geometry']:
            # 正常计算位姿
            pose_input = torch.cat([current_state[:, 5:8], current_state[:, 8:11]], dim=1)
            pose_feat = self.pose_encoder(pose_input)
        else:
            # 消融位姿：输出全零张量，维度与 pose_encoder 输出相同 (24)
            pose_feat = torch.zeros(batch_size, 24, device=device)

        # ----- 几何特征（根据消融模式决定是否计算）-----
        if self.ablation_mode in ['full', 'no_pose']:
            geometry_feat = self.geometry_encoder(current_state[:, 11:14])
        else:
            geometry_feat = torch.zeros(batch_size, 24, device=device)

        # 五个特征按顺序送入交叉注意力
        features = [scale_feat, shape_feat, pose_feat, geometry_feat, control_feat]
        attended_features = self.cross_attention(features)

        fused = self.fusion_network(attended_features)
        delta = self.delta_head(fused)
        uncertainty = self.uncertainty_head(fused)

        return delta, uncertainty


class CrossAttentionModule(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()
        self.feature_dims = feature_dims

        self.unified_dim = 32
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.unified_dim) for dim in feature_dims
        ])

        self.attention = nn.MultiheadAttention(
            embed_dim=self.unified_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, features):
        projected_features = []
        for i, feat in enumerate(features):
            projected = self.projections[i](feat)
            projected_features.append(projected.unsqueeze(1))

        all_features = torch.cat(projected_features, dim=1)

        attended_features = []

        for i in range(len(features)):
            query = projected_features[i]

            key = value = all_features

            attn_out, _ = self.attention(query, key, value)
            attended_features.append(attn_out.squeeze(1))

        return torch.cat(attended_features, dim=1)



class FusionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rates):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            residual_block = nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.01)
            )

            self.projection = nn.Linear(current_dim, hidden_dim) if current_dim != hidden_dim else nn.Identity()

            layers.append((residual_block, self.projection))
            current_dim = hidden_dim

        self.layers = nn.ModuleList([layer[0] for layer in layers])
        self.projections = nn.ModuleList([layer[1] for layer in layers])

    def forward(self, x):
        for block, projection in zip(self.layers, self.projections):
            residual = projection(x)
            x = block(x) + residual
            x = nn.LeakyReLU(0.01)(x)
        return x


class UncertaintyHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, output_dim),
            nn.Softplus()
        )
    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    if layer is self.network[-1]:
                        nn.init.constant_(layer.bias, -4.0)
                    else:
                        nn.init.constant_(layer.bias, 0.0)
    def forward(self, x):
        return self.network(x)


class Loss(nn.Module):
    def __init__(self, nll_weight=1.0, relative_weight=0.3):
        super().__init__()
        self.nll_weight = nll_weight
        self.relative_weight = relative_weight

    def forward(self, pred_delta, pred_uncertainty, target, current_state=None):
        nll_loss = self._nll_loss(pred_delta, pred_uncertainty, target)
        relative_loss = self._relative_loss(pred_delta, target)

        total_loss = self.nll_weight * nll_loss + self.relative_weight * relative_loss

        self.loss_components = {
            'nll': nll_loss.item(),
            'relative': relative_loss.item()

        }

        return total_loss

    def _nll_loss(self, pred, uncertainty, target):
        log_var = uncertainty
        precision = torch.exp(-log_var)
        squared_error = (pred - target) ** 2
        loss = 0.5 * (log_var + squared_error * precision)
        loss = loss + 0.5 * torch.log(torch.tensor(2 * torch.pi, device=pred.device))
        return torch.mean(loss)


    def _relative_loss(self, pred, target):
        target_abs = torch.abs(target)
        significant_mask = target_abs > torch.quantile(target_abs, 0.055)

        if significant_mask.sum() > 0:
            relative_error = torch.mean(
                (pred[significant_mask] - target[significant_mask]) ** 2 /
                (target[significant_mask] ** 2 + 1e-8)
            )
        else:
            relative_error = torch.tensor(0.0, device=pred.device)

        return relative_error