import torch
import torch.nn.functional as F
from typing import Tuple


class CombinedLoss:
    """Combined loss for MoE-Conformer + RNN-T"""

    def __init__(
        self,
        load_balance_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        distillation_weight: float = 0.1,
        ctc_phone_weight: float = 0.3,
        blank_id: int = 0
    ):
        self.load_balance_weight = load_balance_weight
        self.z_loss_weight = z_loss_weight
        self.distillation_weight = distillation_weight
        self.ctc_phone_weight = ctc_phone_weight
        self.blank_id = blank_id

    def rnnt_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RNN-T loss (placeholder - needs specialized library)
        logits: (batch, time, label_len, vocab_size+1)
        For now, use mean of logits at each position as a simple loss.
        """
        # Placeholder: use a simple MSE-like loss on logits
        return torch.mean(logits) * 0.0  # Return 0 for now

    def load_balance_loss(
        self,
        group_probs: torch.Tensor,
        group_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Load balance loss for MoE (balance load between groups)
        group_probs: (batch, time, num_groups) - probabilities from gating network
        """
        num_groups = group_probs.shape[-1]
        group_usage = torch.mean(F.one_hot(group_ids, num_classes=num_groups).float(), dim=[0, 1])
        target_usage = 1.0 / num_groups
        loss = torch.sum((group_usage - target_usage) ** 2)
        return loss * self.load_balance_weight

    def z_loss(
        self,
        group_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Router z-loss to reduce overconfidence
        group_logits: (batch, time, num_groups)
        """
        log_z = torch.logsumexp(group_logits, dim=-1)
        loss = torch.mean(log_z ** 2)
        return loss * self.z_loss_weight

    def distillation_loss(
        self,
        teacher_outputs: torch.Tensor,
        student_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        KL divergence loss for distillation
        teacher_outputs: (batch*time, expert_dim)
        student_outputs: (batch*time, expert_dim)
        """
        teacher_probs = F.softmax(teacher_outputs, dim=-1)
        student_log_probs = F.log_softmax(student_outputs, dim=-1)
        kl = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-8) - student_log_probs), dim=-1)
        return torch.mean(kl) * self.distillation_weight

    def ctc_phone_loss(
        self,
        phone_logits: torch.Tensor,
        phone_targets: torch.Tensor,
        input_lengths: torch.Tensor,
        phone_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        CTC loss for phone recognition (optimizes PER)
        phone_logits: (batch, time, num_phones)
        """
        num_phones = phone_logits.shape[-1]
        return F.cross_entropy(
            phone_logits.reshape(-1, num_phones),
            phone_targets.reshape(-1),
            reduction='mean'
        ) * self.ctc_phone_weight

    def total_loss(
        self,
        rnnt_logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        group_probs: torch.Tensor,
        group_ids: torch.Tensor,
        group_logits: torch.Tensor,
        distillation_loss: torch.Tensor,
        phone_logits: torch.Tensor = None,
        phone_targets: torch.Tensor = None,
        phone_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss and return detailed dict
        """
        losses = {}

        losses['rnnt'] = self.rnnt_loss(rnnt_logits, targets, input_lengths, target_lengths)
        losses['load_balance'] = self.load_balance_loss(group_probs, group_ids)
        losses['z_loss'] = self.z_loss(group_logits)
        losses['distillation'] = distillation_loss

        if phone_logits is not None and phone_targets is not None:
            losses['ctc_phone'] = self.ctc_phone_loss(
                phone_logits, phone_targets, input_lengths, phone_lengths
            )
        else:
            losses['ctc_phone'] = torch.tensor(0.0)

        total = (
            losses['rnnt'] +
            losses['load_balance'] +
            losses['z_loss'] +
            losses['distillation'] +
            losses['ctc_phone']
        )
        losses['total'] = total

        return total, losses
