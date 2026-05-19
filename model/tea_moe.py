import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .gating import GatingNetwork
from .moe_conformer import MoEConformerEncoder
from .rnnt_decoder import RNNTDecoder
from .losses import CombinedLoss


class TeaMoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_whisper = config.get('use_whisper_features', False)
        self.use_precomputed_whisper = config.get('use_precomputed_whisper', False)
        self.input_proj = nn.Linear(config['n_mels'], config['model_dim'])
        self.whisper_dim = config.get('whisper_dim', 512)
        self.time_reduction_factor = max(1, int(config.get('time_reduction_factor', 1)))

        # Initialize Whisper encoder only if NOT using precomputed features
        if self.use_whisper and not self.use_precomputed_whisper:
            from transformers import WhisperModel
            whisper_model_name = config.get('whisper_model_name', 'openai/whisper-base')
            self.whisper = WhisperModel.from_pretrained(whisper_model_name)
            self.whisper_dim = getattr(self.whisper.config, 'd_model', self.whisper_dim)
            self.whisper_freeze = config.get('whisper_freeze', True)

            # Freeze Whisper parameters if requested
            if self.whisper_freeze:
                for param in self.whisper.parameters():
                    param.requires_grad = False

            # Combined projection: mel + whisper features -> model_dim
            combined_input_dim = config['n_mels'] + self.whisper_dim
            proj_dropout = config.get('whisper_proj_dropout', 0.1)
            self.combined_proj = nn.Sequential(
                nn.Linear(combined_input_dim, config['model_dim']),
                nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()
            )
        elif self.use_precomputed_whisper:
            # Pre-computed features path: mel + precomputed whisper -> model_dim
            combined_input_dim = config['n_mels'] + self.whisper_dim
            proj_dropout = config.get('whisper_proj_dropout', 0.1)
            self.combined_proj = nn.Sequential(
                nn.Linear(combined_input_dim, config['model_dim']),
                nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()
            )

        self.gating = GatingNetwork(
            num_groups=config['num_groups'],
            model_dim=config['model_dim']
        )
        self.encoder = MoEConformerEncoder(config=config)
        self.decoder = RNNTDecoder(config=config)
        self.ctc_phone_weight = config.get('ctc_phone_weight', 0.0)
        num_phones = config.get('num_phones', 256)
        self.phone_head = nn.Linear(config['model_dim'], num_phones)
        self.loss_fn = CombinedLoss(
            load_balance_weight=config['load_balance_weight'],
            z_loss_weight=config['z_loss_weight'],
            distillation_weight=config['distillation_weight'],
            ctc_phone_weight=config['ctc_phone_weight'],
            blank_id=config['blank_id']
        )

    def reduce_lengths(self, lengths):
        if self.time_reduction_factor <= 1:
            return lengths
        return torch.div(
            lengths + self.time_reduction_factor - 1,
            self.time_reduction_factor,
            rounding_mode='floor',
        )

    def _reduce_time(self, x):
        if self.time_reduction_factor <= 1:
            return x
        return F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size=self.time_reduction_factor,
            stride=self.time_reduction_factor,
            ceil_mode=True,
        ).transpose(1, 2)

    def project_audio_features(self, audio_features, whisper_features=None):
        """
        Project mel-only or mel+Whisper features into the encoder dimension.

        Args:
            audio_features: [B, T, n_mels] - mel spectrogram (time-first format)
            whisper_features: [B, T_w, whisper_dim] - pre-computed Whisper features (optional)
        """
        if self.use_precomputed_whisper:
            # Use pre-computed Whisper features (loaded from disk)
            if whisper_features is None:
                # Fallback to mel-only if no features provided
                return self.input_proj(audio_features)

            if whisper_features.shape[-1] != self.whisper_dim:
                raise ValueError(
                    f"Expected Whisper feature dim {self.whisper_dim}, "
                    f"got {whisper_features.shape[-1]}"
                )

            # Interpolate Whisper features to match mel time dimension
            _, T, _ = audio_features.shape
            T_w = whisper_features.shape[1]
            if T_w != T:
                whisper_features = torch.nn.functional.interpolate(
                    whisper_features.transpose(1, 2),  # [B, whisper_dim, T_w]
                    size=T,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # [B, T, whisper_dim]

            # Concatenate mel and pre-computed whisper features
            combined = torch.cat([audio_features, whisper_features], dim=-1)
            return self.combined_proj(combined)  # [B, T, model_dim]

        elif self.use_whisper:
            # Compute Whisper features on-the-fly (original behavior)
            B, T, _ = audio_features.shape
            # Whisper expects [B, n_mels, T] format (channels first)
            audio_for_whisper = audio_features.transpose(1, 2)  # [B, 80, T]

            # Whisper encoder expects exactly 3000 mel frames and downsamples time.
            expected_seq_length = 3000  # Whisper's expected mel length
            if T <= expected_seq_length:
                pad_len = expected_seq_length - T
                audio_for_whisper = torch.nn.functional.pad(
                    audio_for_whisper, (0, pad_len), mode='constant', value=0
                )
            else:
                audio_for_whisper = audio_for_whisper[:, :, :expected_seq_length]

            # Extract Whisper features (with or without gradients)
            with torch.set_grad_enabled(not self.whisper_freeze):
                whisper_out = self.whisper.encoder(audio_for_whisper)
                whisper_features = whisper_out.last_hidden_state  # [B, T_whisper, whisper_dim]

            if whisper_features.shape[1] != T:
                whisper_features = torch.nn.functional.interpolate(
                    whisper_features.transpose(1, 2),  # [B, whisper_dim, T_whisper]
                    size=T,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # [B, T, whisper_dim]

            # Concatenate mel and whisper features
            combined = torch.cat([audio_features, whisper_features], dim=-1)
            return self.combined_proj(combined)  # [B, T, model_dim]

        # Standard path: mel only
        return self.input_proj(audio_features)  # [B, T, model_dim]

    def forward(self, audio_features, targets, phone_targets=None, deterministic=True,
                use_checkpoint=False, whisper_features=None):
        """
        Args:
            audio_features: [B, T, n_mels] - mel spectrogram (time-first format)
            targets: [B, U] - token ids
            phone_targets: [B, U] - phone ids (optional)
            deterministic: bool - routing mode
            use_checkpoint: bool - gradient checkpointing
            whisper_features: [B, T_w, whisper_dim] - pre-computed Whisper features (optional)

        Returns:
            rnnt_logits: [B, T, vocab_size + 1] by default
            aux_outputs: dict with auxiliary outputs
        """
        x = self.project_audio_features(audio_features, whisper_features)
        x = self._reduce_time(x)

        # Gating and encoder (unchanged)
        group_probs, group_ids, group_logits = self.gating(
            x,
            deterministic=deterministic,
            return_logits=True,
        )
        encoder_out = self.encoder(
            x,
            group_ids=group_ids,
            deterministic=deterministic,
            use_checkpoint=use_checkpoint
        )
        rnnt_logits = self.decoder(
            encoder_out,
            targets,
            deterministic=deterministic
        )
        phone_logits = (
            self.phone_head(encoder_out)
            if self.ctc_phone_weight > 0
            else None
        )
        aux_outputs = {
            "group_probs": group_probs,
            "group_ids": group_ids,
            "group_logits": group_logits,
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
