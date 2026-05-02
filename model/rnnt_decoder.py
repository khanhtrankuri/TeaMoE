import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PredictionNetwork(nn.Module):
    def __init__(self, hidden_dim=1024, num_layers=2, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                             dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, state=None, deterministic=True):
        embed = self.embedding(y)
        if state is not None:
            output, new_state = self.lstm(embed, state)
        else:
            output, new_state = self.lstm(embed)
        if not deterministic:
            output = self.dropout(output)
        return output, new_state


class JointNetwork(nn.Module):
    def __init__(self, joint_dim=1024, vocab_size=5000):
        super().__init__()
        self.encoder_proj = nn.Linear(joint_dim, joint_dim)
        self.pred_proj = nn.Linear(joint_dim, joint_dim)
        self.output = nn.Linear(joint_dim, vocab_size + 1)

    def forward(self, encoder_out, pred_out):
        encoder_proj = self.encoder_proj(encoder_out)
        pred_proj = self.pred_proj(pred_out)
        encoder_expanded = encoder_proj.unsqueeze(2)
        pred_expanded = pred_proj.unsqueeze(1)
        joint = encoder_expanded + pred_expanded
        joint = F.relu(joint)
        output = self.output(joint)
        return output


class RNNTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pred_net = PredictionNetwork(
            hidden_dim=config['decoder_hidden'],
            num_layers=config['decoder_layers'],
            vocab_size=config['vocab_size'],
            dropout=0.1,
        )
        self.joint_net = JointNetwork(
            joint_dim=config['decoder_hidden'],
            vocab_size=config['vocab_size'],
        )

    def forward(self, encoder_out, targets, deterministic=True):
        pred_out, _ = self.pred_net(targets, deterministic=deterministic)
        joint_out = self.joint_net(encoder_out, pred_out)
        return joint_out

    def compute_rnnt_loss(self, logits, targets, input_lengths, target_lengths):
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(
            logits[..., :-1].reshape(-1, vocab_size - 1),
            targets.reshape(-1),
            reduction='mean'
        )
        return loss
