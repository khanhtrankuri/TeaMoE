import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional

from .config import ModelConfig
from .gating import GatingNetwork
from .moe_conformer import MoEConformerEncoder
from .rnnt_decoder import RNNTDecoder
from .losses import CombinedLoss


class TeaMoEModel(nn.Module):
    """
    Tổng thể kiến trúc TeaMoE: Gating + MoE-Conformer + RNN-T Decoder
    Tích hợp cả cơ chế cạnh tranh và distillation (được gọi từ training loop)
    """
    config: ModelConfig

    def setup(self):
        self.gating = GatingNetwork(
            num_groups=self.config.num_groups,
            model_dim=self.config.model_dim
        )
        self.encoder = MoEConformerEncoder(config=self.config)
        self.decoder = RNNTDecoder(config=self.config)
        # Head phụ cho CTC phone recognition (tối ưu PER)
        self.phone_head = nn.Dense(256)  # 256 là số lượng phone classes (cần định nghĩa)
        self.loss_fn = CombinedLoss(
            load_balance_weight=self.config.load_balance_weight,
            z_loss_weight=self.config.z_loss_weight,
            distillation_weight=self.config.distillation_weight,
            ctc_phone_weight=self.config.ctc_phone_weight,
            blank_id=self.config.blank_id
        )

    def __call__(
        self,
        audio_features: jnp.ndarray,
        targets: jnp.ndarray,
        phone_targets: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Forward pass cho training/inference
        audio_features: (batch, time, n_mels)
        targets: (batch, target_len) - transcript tokens
        phone_targets: (batch, target_len) - phone tokens (cho CTC phone loss)
        Trả về: rnnt_logits, aux_outputs (chứa group_probs, group_ids, encoder_out, v.v.)
        """
        # 1. Gating: chọn nhóm expert cho mỗi frame
        group_probs, group_ids = self.gating(audio_features, deterministic=deterministic)
        # group_probs: (batch, time, num_groups)
        # group_ids: (batch, time)

        # 2. Encoder: MoE-Conformer
        encoder_out = self.encoder(
            audio_features,
            group_ids=group_ids,
            deterministic=deterministic
        )  # (batch, time, model_dim)

        # 3. RNN-T Decoder
        rnnt_logits = self.decoder(
            encoder_out,
            targets,
            deterministic=deterministic
        )  # (batch, time, target_len, vocab_size+1)

        # 4. Phone head (cho CTC phone loss)
        phone_logits = self.phone_head(encoder_out)  # (batch, time, num_phones)

        aux_outputs = {
            "group_probs": group_probs,
            "group_ids": group_ids,
            "encoder_out": encoder_out,
            "phone_logits": phone_logits,
        }

        return rnnt_logits, aux_outputs

    def compute_loss(
        self,
        rnnt_logits: jnp.ndarray,
        targets: jnp.ndarray,
        input_lengths: jnp.ndarray,
        target_lengths: jnp.ndarray,
        group_probs: jnp.ndarray,
        group_ids: jnp.ndarray,
        group_logits: jnp.ndarray,  # logits từ gating trước softmax
        distillation_loss: jnp.ndarray,
        phone_logits: Optional[jnp.ndarray] = None,
        phone_targets: Optional[jnp.ndarray] = None,
        phone_lengths: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, dict]:
        """Tính tổng loss sử dụng CombinedLoss"""
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
