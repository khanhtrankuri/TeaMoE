import torch
import torch.nn as nn
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
    def __init__(self, config, expert_pretrained_paths: Optional[List[str]] = None):
        super().__init__()
        self.config = config
        for i in range(config['num_experts']):
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

    def forward(self, x, deterministic=True, use_checkpoint=False):
        outputs = []
        for i in range(self.config['num_experts']):
            expert = getattr(self, f"expert_{i}")
            outputs.append(expert(x, deterministic=deterministic, use_checkpoint=use_checkpoint))
        return torch.stack(outputs, dim=1)

    def get_expert(self, idx):
        return getattr(self, f"expert_{idx}")
