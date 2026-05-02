import torch
import torch.nn as nn
from typing import Optional


class Expert(nn.Module):
    def __init__(self, expert_dim=1024, ff_multiplier=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(expert_dim, expert_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim * ff_multiplier, expert_dim),
        )

    def forward(self, x, deterministic=True):
        return self.net(x)


class ExpertGroup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        for i in range(config['num_experts']):
            setattr(self, f"expert_{i}", Expert(
                expert_dim=config['expert_dim'],
                ff_multiplier=config['ff_multiplier'],
                dropout=config['dropout'],
            ))

    def forward(self, x, deterministic=True):
        outputs = []
        for i in range(self.config['num_experts']):
            expert = getattr(self, f"expert_{i}")
            outputs.append(expert(x, deterministic=deterministic))
        return torch.stack(outputs, dim=1)

    def get_expert(self, idx):
        return getattr(self, f"expert_{idx}")
