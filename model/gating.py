import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class GatingNetwork(nn.Module):
    """Mạng chọn nhóm expert (8-way) cho mỗi frame"""
    num_groups: int = 8
    model_dim: int = 1024
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # x: (batch, time, model_dim)
        # Trả về: group_probs (batch, time, num_groups), group_ids (batch, time)
        hidden = nn.Dense(self.hidden_dim)(x)
        hidden = nn.relu(hidden)
        logits = nn.Dense(self.num_groups)(hidden)
        group_probs = nn.softmax(logits, axis=-1)
        group_ids = jnp.argmax(group_probs, axis=-1)
        return group_probs, group_ids
