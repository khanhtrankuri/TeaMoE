import torch
import torch.nn as nn
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

        # Initialize Whisper encoder if enabled
        if self.use_whisper:
            from transformers import WhisperModel
            whisper_model_name = config.get('whisper_model_name', 'openai/whisper-base')
            self.whisper = WhisperModel.from_pretrained(whisper_model_name)
            self.whisper_dim = 512  # Whisper-base encoder output dimension
            self.whisper_freeze = config.get('whisper_freeze', True)

            # Freeze Whisper parameters if requested
            if self.whisper_freeze:
                for param in self.whisper.parameters():
                    param.requires_grad = False

            # Combined projection: mel + whisper features -> model_dim
            combined_input_dim = config['n_mels'] + self.whisper_dim  # 80 + 512 = 592
            proj_dropout = config.get('whisper_proj_dropout', 0.1)
            self.combined_proj = nn.Sequential(
                nn.Linear(combined_input_dim, config['model_dim']),
                nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()
            )
        else:
            # Standard input projection (mel only)
            self.input_proj = nn.Linear(config['n_mels'], config['model_dim'])

        self.gating = GatingNetwork(
            num_groups=config['num_groups'],
            model_dim=config['model_dim']
        )
        self.encoder = MoEConformerEncoder(config=config)
        self.decoder = RNNTDecoder(config=config)
        num_phones = config.get('num_phones', 256)
        self.phone_head = nn.Linear(config['model_dim'], num_phones)
        self.loss_fn = CombinedLoss(
            load_balance_weight=config['load_balance_weight'],
            z_loss_weight=config['z_loss_weight'],
            distillation_weight=config['distillation_weight'],
            ctc_phone_weight=config['ctc_phone_weight'],
            blank_id=config['blank_id']
        )

    def forward(self, audio_features, targets, phone_targets=None, deterministic=True, use_checkpoint=False):
        """
        Args:
            audio_features: [B, T, n_mels] - mel spectrogram (time-first format)
            targets: [B, U] - token ids
            phone_targets: [B, U] - phone ids (optional)
            deterministic: bool - routing mode
            use_checkpoint: bool - gradient checkpointing

        Returns:
            rnnt_logits: [B, U, T, vocab_size]
            aux_outputs: dict with auxiliary outputs
        """
        if self.use_whisper:
            B, T, _ = audio_features.shape
            # Whisper expects [B, n_mels, T] format (channels first)
            audio_for_whisper = audio_features.transpose(1, 2)  # [B, 80, T]

            # Whisper has a length constraint (multiple of 3000 for pretrained)
            # We handle this by padding to the next multiple of 3000, then cropping
            expected_seq_length = 3000  # Whisper's expected mel length
            if T <= expected_seq_length:
                # Pad to expected_seq_length
                pad_len = expected_seq_length - T
                audio_for_whisper = torch.nn.functional.pad(
                    audio_for_whisper, (0, pad_len), mode='constant', value=0
                )
                pad_needed = True
            else:
                # If longer, we'll process in chunks or just let it fail?
                # For simplicity, truncate to multiple of expected_seq_length
                # In production, you'd implement chunked processing
                new_T = (T // expected_seq_length) * expected_seq_length
                if new_T < T:
                    audio_for_whisper = audio_for_whisper[:, :, :new_T]
                    pad_needed = False
                else:
                    pad_needed = False

            # Extract Whisper features (with or without gradients)
            with torch.set_grad_enabled(not self.whisper_freeze):
                whisper_out = self.whisper.encoder(audio_for_whisper)
                whisper_features = whisper_out.last_hidden_state  # [B, T_whisper, 512]

            # Remove padding to match original T
            if pad_needed:
                whisper_features = whisper_features[:, :T, :]
            else:
                # If we truncated, we need to upsample back to original T
                if whisper_features.shape[1] != T:
                    whisper_features = torch.nn.functional.interpolate(
                        whisper_features.transpose(1, 2),  # [B, 512, T_whisper]
                        size=T,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)  # [B, T, 512]

            # Concatenate mel and whisper features
            combined = torch.cat([audio_features, whisper_features], dim=-1)  # [B, T, 80+512=592]
            x = self.combined_proj(combined)  # [B, T, model_dim]
        else:
            # Standard path: mel only
            x = self.input_proj(audio_features)  # [B, T, model_dim]

        # Gating and encoder (unchanged)
        group_probs, group_ids = self.gating(x, deterministic=deterministic)
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
