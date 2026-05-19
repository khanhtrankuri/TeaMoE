import torch
import torch.nn as nn
from typing import Tuple


class GatingNetwork(nn.Module):
    def __init__(self, num_groups=8, model_dim=1024, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_groups),
        )

    def forward(self, x, deterministic=True, return_logits=False):
        logits = self.net(x)
        group_probs = torch.softmax(logits.float(), dim=-1)
        group_ids = torch.argmax(group_probs, dim=-1)
        if return_logits:
            return group_probs, group_ids, logits
        return group_probs, group_ids
