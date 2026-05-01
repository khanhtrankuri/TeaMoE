import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class Expert(nn.Module):
    """Một Expert duy nhất (Feed-Forward Network)"""
    expert_dim: int = 1024
    ff_multiplier: int = 4
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # x: (batch*time, expert_dim)
        hidden = nn.Dense(self.expert_dim * self.ff_multiplier)(x)
        hidden = nn.gelu(hidden)
        hidden = nn.Dropout(rate=self.dropout)(hidden, deterministic=deterministic)
        output = nn.Dense(self.expert_dim)(hidden)
        return output


class ExpertGroup(nn.Module):
    """Nhóm 5 Expert chuyên biệt"""
    config: ExpertGroupConfig

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # x: (batch*time, expert_dim)
        experts_outputs = []
        for i in range(self.config.num_experts):
            expert_out = Expert(
                expert_dim=self.config.expert_dim,
                ff_multiplier=self.config.ff_multiplier,
                dropout=self.config.dropout
            )(x, deterministic=deterministic)
            experts_outputs.append(expert_out)
        # Stack: (batch*time, num_experts, expert_dim)
        return jnp.stack(experts_outputs, axis=1)

    def get_expert(self, idx: int):
        """Lấy expert thứ idx trong nhóm"""
        return Expert(
            expert_dim=self.config.expert_dim,
            ff_multiplier=self.config.ff_multiplier,
            dropout=self.config.dropout
        )
