import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


class Expert(nn.Module):
    def __init__(self, expert_dim=1024, ff_multiplier=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(expert_dim, expert_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim * ff_multiplier, expert_dim),
        )

    def _forward_impl(self, x, deterministic=True):
        return self.net(x)

    def forward(self, x, deterministic=True, use_checkpoint=False):
        if use_checkpoint and self.training and x.requires_grad:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, x, deterministic, use_reentrant=False)
        return self._forward_impl(x, deterministic)


class ExpertGroup(nn.Module):
    """Group of experts with configurable pooling strategy.

    Supports:
    - Mean pooling (default): Simple average of all expert outputs
    - Attention pooling: Learn to weight experts per token

    Args:
        config: Must contain:
            - num_experts: number of experts in group (typically 5)
            - expert_dim: dimension of expert input/output
        expert_pretrained_paths: Optional list of paths, one per expert
        use_attention_pooling: If True, use attention to combine expert outputs
        attn_heads: Number of attention heads (only if use_attention_pooling=True)
        attn_dropout: Dropout rate for attention weights
    """
    def __init__(self, config, expert_pretrained_paths: Optional[List[str]] = None,
                 use_attention_pooling: bool = False, attn_heads: int = 4,
                 attn_dropout: float = 0.1):
        super().__init__()
        self.config = config
        self.num_experts = config['num_experts']
        self.expert_dim = config['expert_dim']
        self.use_attention_pooling = use_attention_pooling

        # Create experts
        for i in range(self.num_experts):
            expert = Expert(
                expert_dim=config['expert_dim'],
                ff_multiplier=config['ff_multiplier'],
                dropout=config['dropout'],
            )
            if expert_pretrained_paths is not None and i < len(expert_pretrained_paths):
                pretrained_path = expert_pretrained_paths[i]
                if pretrained_path is not None:
                    state_dict = torch.load(pretrained_path, map_location='cpu')
                    expert.load_state_dict(state_dict)
            setattr(self, f"expert_{i}", expert)

        # Attention-based pooling (optional)
        if use_attention_pooling:
            self.attn_heads = attn_heads
            self.head_dim = self.expert_dim // attn_heads
            assert self.head_dim * attn_heads == self.expert_dim, \
                f"expert_dim ({self.expert_dim}) must be divisible by attn_heads ({attn_heads})"

            # Query: from input token representation
            self.query = nn.Linear(self.expert_dim, self.expert_dim)
            # Key & Value: from expert outputs
            self.key = nn.Linear(self.expert_dim, self.expert_dim)
            self.value = nn.Linear(self.expert_dim, self.expert_dim)

            # Output projection
            self.out_proj = nn.Linear(self.expert_dim, self.expert_dim)
            self.attn_dropout = nn.Dropout(attn_dropout)
            self.norm = nn.LayerNorm(self.expert_dim)

    def forward(self, x, deterministic=True, use_checkpoint=False):
        """Combine expert outputs.

        Args:
            x: [N, expert_dim] - input tokens assigned to this group
            deterministic: If True, use deterministic behavior

        Returns:
            [N, expert_dim] - combined output
        """
        # Get outputs from all experts
        outputs = []
        for i in range(self.num_experts):
            expert = getattr(self, f"expert_{i}")
            out = expert(x, deterministic=deterministic, use_checkpoint=use_checkpoint)
            outputs.append(out)

        stacked = torch.stack(outputs, dim=1)  # [N, E, D]

        if self.use_attention_pooling:
            # Attention-based combination
            N, E, D = stacked.shape

            # Compute query from input, key/value from expert outputs
            q = self.query(x).view(N, self.attn_heads, self.head_dim)  # [N, H, h]
            k = self.key(stacked).view(N, E, self.attn_heads, self.head_dim)
            v = self.value(stacked).view(N, E, self.attn_heads, self.head_dim)

            # Transpose for attention: [N, H, E, h]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Scaled dot-product attention
            # q: [N, H, h] -> [N, H, 1, h]
            # k: [N, H, E, h] -> [N, H, h, E]
            scores = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # scores: [N, H, 1, E]

            attn_weights = F.softmax(scores, dim=-1)  # [N, H, 1, E]
            attn_weights = self.attn_dropout(attn_weights)

            # Weighted sum of values
            # attn_weights: [N, H, 1, E] @ v: [N, H, E, h] -> [N, H, 1, h]
            context = torch.matmul(attn_weights, v)  # [N, H, 1, h]
            context = context.squeeze(2)  # [N, H, h]

            # Concatenate heads
            context = context.view(N, D)  # [N, D]

            # Output projection
            combined = self.out_proj(context)
            combined = self.norm(combined + x)  # Pre-norm style residual
        else:
            # Simple mean pooling (default)
            combined = stacked.mean(dim=1)

        return combined

    def get_expert(self, idx):
        return getattr(self, f"expert_{idx}")


class AttentionPoolingExpertGroup(ExpertGroup):
    """Convenience subclass with attention pooling enabled by default."""
    def __init__(self, config, expert_pretrained_paths=None, attn_heads=4, attn_dropout=0.1):
        super().__init__(
            config,
            expert_pretrained_paths=expert_pretrained_paths,
            use_attention_pooling=True,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout
        )
