import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, List
from .config import ModelConfig
from .expert import ExpertGroup


class ConformerLayer(nn.Module):
    """1 Conformer Layer tiêu chuẩn (không MoE)"""
    model_dim: int = 1024
    num_heads: int = 16
    ff_multiplier: int = 4
    conv_kernel_size: int = 31
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # x: (batch, time, model_dim)
        # Conv module
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Conv(features=self.model_dim, kernel_size=(self.conv_kernel_size,), padding="SAME")(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = x + residual

        # Self-attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=deterministic
        )(x, x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = x + residual

        # FFN
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.model_dim * self.ff_multiplier)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(self.model_dim)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = x + residual

        return x


class MoEConformerLayer(nn.Module):
    """Conformer Layer có MoE-FFN (thay FFN bằng ExpertGroup)"""
    config: ModelConfig
    group_configs: List[ExpertGroupConfig]

    @nn.compact
    def __call__(self, x, group_ids, all_expert_groups, deterministic: bool = True):
        # x: (batch, time, model_dim)
        # group_ids: (batch, time) - nhóm được chọn cho mỗi frame
        # all_expert_groups: List[ExpertGroup] - 8 nhóm expert

        # Conv module
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Conv(features=self.config.model_dim, kernel_size=(self.config.conv_kernel_size,), padding="SAME")(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.1)(x, deterministic=deterministic)
        x = x + residual

        # Self-attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            dropout_rate=0.1,
            deterministic=deterministic
        )(x, x)
        x = nn.Dropout(rate=0.1)(x, deterministic=deterministic)
        x = x + residual

        # MoE-FFN: thay FFN bằng ExpertGroup tương ứng
        # Flatten batch và time để xử lý expert
        batch_size, seq_len, model_dim = x.shape
        x_flat = x.reshape(-1, model_dim)  # (batch*time, model_dim)
        group_ids_flat = group_ids.reshape(-1)  # (batch*time,)

        outputs = []
        for group_idx, expert_group in enumerate(all_expert_groups):
            # Lấy mask cho nhóm này
            mask = (group_ids_flat == group_idx)  # (batch*time,)
            if not jnp.any(mask):
                continue
            # Chỉ xử lý các frame thuộc nhóm này
            x_group = x_flat[mask]  # (num_frames_in_group, model_dim)
            # Chạy qua expert group: trả về (num_frames, 5, model_dim)
            group_outputs = expert_group(x_group, deterministic=deterministic)
            # Lấy trung bình hoặc chọn top-k trong nhóm (theo thiết kế: lấy cả 5 expert)
            # Ở đây lấy trung bình có trọng số (có thể cải thiện sau)
            group_output = jnp.mean(group_outputs, axis=1)  # (num_frames, model_dim)
            outputs.append((mask, group_output))

        # Ghép lại theo đúng thứ tự
        result = jnp.zeros_like(x_flat)
        for mask, out in outputs:
            result = result.at[mask].set(out)
        x = result.reshape(batch_size, seq_len, model_dim)

        x = nn.Dropout(rate=0.1)(x, deterministic=deterministic)
        x = x + residual

        return x


class MoEConformerEncoder(nn.Module):
    """MoE-Conformer Encoder tổng thể (24 layers)"""
    config: ModelConfig

    @nn.compact
    def __call__(self, x, group_ids, deterministic: bool = True):
        # x: (batch, time, model_dim)
        # group_ids: (batch, time) từ GatingNetwork

        # Khởi tạo 8 ExpertGroups
        expert_groups = [
            ExpertGroup(config=group_cfg)
            for group_cfg in self.config.group_configs
        ]

        # Layer 1-5: Dense Conformer (không MoE)
        for i in range(self.config.moe_start_layer):
            x = ConformerLayer(
                model_dim=self.config.model_dim,
                num_heads=self.config.num_heads,
                ff_multiplier=self.config.ff_multiplier,
                conv_kernel_size=self.config.conv_kernel_size,
            )(x, deterministic=deterministic)

        # Layer 6-18: MoE Conformer (có 8 ExpertGroups)
        for i in range(self.config.moe_start_layer, self.config.moe_end_layer):
            x = MoEConformerLayer(
                config=self.config,
                group_configs=self.config.group_configs
            )(x, group_ids, expert_groups, deterministic=deterministic)

        # Layer 19-24: Dense Conformer (không MoE)
        for i in range(self.config.moe_end_layer, self.config.num_layers):
            x = ConformerLayer(
                model_dim=self.config.model_dim,
                num_heads=self.config.num_heads,
                ff_multiplier=self.config.ff_multiplier,
                conv_kernel_size=self.config.conv_kernel_size,
            )(x, deterministic=deterministic)

        return x
