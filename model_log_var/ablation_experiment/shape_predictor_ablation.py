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

        # === 关键修改：根据消融模式计算特征数量 ===
        if ablation_mode == 'full':
            num_features = 5
        elif ablation_mode in ['no_pose', 'no_geometry']:
            num_features = 4
        elif ablation_mode == 'only_shape_scale':
            num_features = 3
        else:
            num_features = 5  # fallback
        fusion_input_dim = num_features * 32

        self.fusion_network = FusionNetwork(
            input_dim=fusion_input_dim,
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

        features = []
        active_indices = []  # 记录每个特征对应的原始分支索引

        # 尺度 (索引 0)
        features.append(scale_feat)
        active_indices.append(0)
        # 形状 (索引 1)
        features.append(shape_feat)
        active_indices.append(1)
        # 位姿 (索引 2)，根据模式决定是否加入
        if self.ablation_mode in ['full', 'no_geometry']:
            pose_input = torch.cat([current_state[:, 5:8], current_state[:, 8:11]], dim=1)
            pose_feat = self.pose_encoder(pose_input)
            features.append(pose_feat)
            active_indices.append(2)
        # 几何 (索引 3)，根据模式决定是否加入
        if self.ablation_mode in ['full', 'no_pose']:
            geometry_feat = self.geometry_encoder(current_state[:, 11:14])
            features.append(geometry_feat)
            active_indices.append(3)
        # 控制 (索引 4)
        features.append(control_feat)
        active_indices.append(4)

        # 传入 active_indices
        attended_features = self.cross_attention(features, active_indices=active_indices)

        fused = self.fusion_network(attended_features)
        delta = self.delta_head(fused)
        uncertainty = self.uncertainty_head(fused)

        return delta, uncertainty


class CrossAttentionModule(nn.Module):
    def __init__(self, feature_dims, num_branches=5):
        """
        feature_dims: 所有可能分支的输出维度列表，例如 [32,16,24,24,16]
        num_branches: 总分支数（固定为5，对应 scale, shape, pose, geometry, control）
        """
        super().__init__()
        self.num_branches = num_branches
        self.unified_dim = 32
        # 为每个可能的分支创建投影层
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.unified_dim) for dim in feature_dims
        ])
        self.attention = nn.MultiheadAttention(
            embed_dim=self.unified_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, features, active_indices=None):
        """
        features: 实际使用的特征列表（长度 <= num_branches）
        active_indices: 可选，指定每个特征对应原始分支的索引（默认按顺序 0,1,2,...）
        """
        if active_indices is None:
            # 假设传入的 features 顺序对应原始分支索引 0..len(features)-1
            active_indices = list(range(len(features)))
        # 投影实际使用的特征
        projected = []
        for idx, feat in zip(active_indices, features):
            proj = self.projections[idx](feat)
            projected.append(proj.unsqueeze(1))
        # 拼接所有投影特征作为 key/value
        all_proj = torch.cat(projected, dim=1)  # [B, M, D]
        # 对每个特征单独做交叉注意力
        attended = []
        for i, (idx, feat) in enumerate(zip(active_indices, features)):
            query = projected[i]  # [B,1,D]
            attn_out, _ = self.attention(query, all_proj, all_proj)
            attended.append(attn_out.squeeze(1))
        # 拼接所有注意力输出
        return torch.cat(attended, dim=1)



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