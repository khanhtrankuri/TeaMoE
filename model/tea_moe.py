import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .config import ModelConfig
from .gating import GatingNetwork
from .moe_conformer import MoEConformerEncoder
from .rnnt_decoder import RNNTDecoder
from .losses import CombinedLoss


class TeaMoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.n_mels, config.model_dim)
        self.gating = GatingNetwork(
            num_groups=config.num_groups,
            model_dim=config.model_dim
        )
        self.encoder = MoEConformerEncoder(config=config)
        self.decoder = RNNTDecoder(config=config)
        num_phones = getattr(config, 'num_phones', 256)
        self.phone_head = nn.Linear(config.model_dim, num_phones)
        self.loss_fn = CombinedLoss(
            load_balance_weight=config.load_balance_weight,
            z_loss_weight=config.z_loss_weight,
            distillation_weight=config.distillation_weight,
            ctc_phone_weight=config.ctc_phone_weight,
            blank_id=config.blank_id
        )

    def forward(self, audio_features, targets, phone_targets=None, deterministic=True):
        x = self.input_proj(audio_features)
        group_probs, group_ids = self.gating(x, deterministic=deterministic)
        encoder_out = self.encoder(
            x,
            group_ids=group_ids,
            deterministic=deterministic
        )
        rnnt_logits = self.decoder(
            encoder_out,
            targets,
            deterministic=deterministic
        )
        phone_logits = self.phone_head(encoder_out)
        aux_outputs = {
            "group_probs": group_probs,
            "group_ids": group_ids,
            "encoder_out": encoder_out,
            "phone_logits": phone_logits,
        }
        return rnnt_logits, aux_outputs

    def compute_loss(self, rnnt_logits, targets, input_lengths, target_lengths,
                     group_probs, group_ids, group_logits, distillation_loss,
                     phone_logits=None, phone_targets=None, phone_lengths=None):
        return self.loss_fn.total_loss(
            rnnt_logits=rnnt_logits,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            group_probs=group_probs,
            group_ids=group_ids,
            group_logits=group_logits,
            distillation_loss=distillation_loss,
            phone_logits=phone_logits,
            phone_targets=phone_targets,
            phone_lengths=phone_lengths
        )
