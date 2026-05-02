import torch
import torch.nn as nn
from typing import List
from .config import ModelConfig
from .expert import ExpertGroup


class ConformerLayer(nn.Module):
    def __init__(self, model_dim=1024, num_heads=16,
                 ff_multiplier=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.dropout = dropout
        self.conv_norm = nn.LayerNorm(model_dim)
        self.conv = nn.Conv1d(model_dim, model_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size//2)
        self.conv_gelu = nn.GELU()
        self.conv_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(model_dim)
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * ff_multiplier, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, deterministic=True):
        residual = x
        x = self.conv_norm(x)
        x_conv = x.permute(0, 2, 1)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.permute(0, 2, 1)
        x_conv = self.conv_gelu(x_conv)
        x_conv = self.conv_dropout(x_conv)
        x = x_conv + residual

        residual = x
        x = self.attn_norm(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.attn_dropout(attn_out)
        x = x + residual

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x


class MoEConformerLayer(nn.Module):
    def __init__(self, config, expert_groups):
        super().__init__()
        self.config = config
        self.expert_groups = nn.ModuleList(expert_groups)
        self.conv_norm = nn.LayerNorm(config.model_dim)
        self.conv = nn.Conv1d(config.model_dim, config.model_dim,
                               kernel_size=config.conv_kernel_size, padding=config.conv_kernel_size//2)
        self.conv_gelu = nn.GELU()
        self.conv_dropout = nn.Dropout(0.1)
        self.attn_norm = nn.LayerNorm(config.model_dim)
        self.self_attn = nn.MultiheadAttention(config.model_dim, config.num_heads, dropout=0.1, batch_first=True)
        self.attn_dropout = nn.Dropout(0.1)
        self.moe_dropout = nn.Dropout(0.1)

    def forward(self, x, group_ids, deterministic=True):
        residual = x
        x = self.conv_norm(x)
        x_conv = x.permute(0, 2, 1)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.permute(0, 2, 1)
        x_conv = self.conv_gelu(x_conv)
        x_conv = self.conv_dropout(x_conv)
        x = x_conv + residual

        residual = x
        x = self.attn_norm(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.attn_dropout(attn_out)
        x = x + residual

        batch_size, seq_len, model_dim = x.shape
        x_flat = x.reshape(-1, model_dim)
        group_ids_flat = group_ids.reshape(-1)

        outputs = []
        masks = []
        for group_idx, expert_group in enumerate(self.expert_groups):
            mask = (group_ids_flat == group_idx)
            if not mask.any():
                continue
            x_group = x_flat[mask]
            group_outputs = expert_group(x_group, deterministic=deterministic)
            group_output = group_outputs.mean(dim=1)
            outputs.append(group_output)
            masks.append(mask)

        result = torch.zeros_like(x_flat)
        for mask, out in zip(masks, outputs):
            result[mask] = out
        x = result.reshape(batch_size, seq_len, model_dim)

        x = self.moe_dropout(x)
        x = x + residual
        return x


class MoEConformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.expert_groups = nn.ModuleList([
            ExpertGroup(group_cfg) for group_cfg in config.group_configs
        ])
        self.pre_layers = nn.ModuleList([
            ConformerLayer(
                model_dim=config.model_dim,
                num_heads=config.num_heads,
                ff_multiplier=config.ff_multiplier,
                conv_kernel_size=config.conv_kernel_size,
            )
            for _ in range(config.moe_start_layer)
        ])
        self.moe_layers = nn.ModuleList([
            MoEConformerLayer(config=config, expert_groups=self.expert_groups)
            for _ in range(config.moe_start_layer, config.moe_end_layer)
        ])
        self.post_layers = nn.ModuleList([
            ConformerLayer(
                model_dim=config.model_dim,
                num_heads=config.num_heads,
                ff_multiplier=config.ff_multiplier,
                conv_kernel_size=config.conv_kernel_size,
            )
            for _ in range(config.moe_end_layer, config.num_layers)
        ])

    def forward(self, x, group_ids, deterministic=True):
        for layer in self.pre_layers:
            x = layer(x, deterministic=deterministic)
        for layer in self.moe_layers:
            x = layer(x, group_ids, deterministic=deterministic)
        for layer in self.post_layers:
            x = layer(x, deterministic=deterministic)
        return x
