import jax
import jax.numpy as jnp
from typing import Tuple


class CombinedLoss:
    """Tổng hợp tất cả loss cho MoE-Conformer + RNN-T"""

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
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        input_lengths: jnp.ndarray,
        target_lengths: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Tính RNN-T loss (placeholder - cần thư viện chuyên dụng)
        logits: (batch, time, label_len, vocab_size+1)
        """
        # Placeholder: dùng cross-entropy đơn giản
        # Trong thực tế cần warp-transducer hoặc jax-triton
        vocab_size = logits.shape[-1]
        loss = jax.nn.softmax_cross_entropy_with_logits(
            logits[..., :-1],  # bỏ blank
            jax.nn.one_hot(targets, vocab_size - 1)
        )
        return jnp.mean(loss)

    def load_balance_loss(
        self,
        group_probs: jnp.ndarray,
        group_ids: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Load balance loss cho MoE (cân bằng tải giữa các nhóm)
        group_probs: (batch, time, num_groups) - xác suất từ gating network
        """
        num_groups = group_probs.shape[-1]
        # Tính tỷ lệ sử dụng mỗi nhóm
        group_usage = jnp.mean(jax.nn.one_hot(group_ids, num_groups), axis=[0, 1])  # (num_groups,)
        # Mục tiêu: mỗi nhóm được dùng đều nhau (1/num_groups)
        target_usage = 1.0 / num_groups
        loss = jnp.sum((group_usage - target_usage) ** 2)
        return loss * self.load_balance_weight

    def z_loss(
        self,
        group_logits: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Router z-loss để giảm overconfidence
        group_logits: (batch, time, num_groups)
        """
        log_z = jnp.log(jnp.sum(jnp.exp(group_logits), axis=-1) + 1e-8)
        loss = jnp.mean(log_z ** 2)
        return loss * self.z_loss_weight

    def distillation_loss(
        self,
        teacher_outputs: jnp.ndarray,
        student_outputs: jnp.ndarray
    ) -> jnp.ndarray:
        """
        KL divergence loss cho distillation
        teacher_outputs: (batch*time, expert_dim)
        student_outputs: (batch*time, expert_dim)
        """
        teacher_probs = jax.nn.softmax(teacher_outputs, axis=-1)
        student_log_probs = jax.nn.log_softmax(student_outputs, axis=-1)
        kl = jnp.sum(teacher_probs * (jnp.log(teacher_probs + 1e-8) - student_log_probs), axis=-1)
        return jnp.mean(kl) * self.distillation_weight

    def ctc_phone_loss(
        self,
        phone_logits: jnp.ndarray,
        phone_targets: jnp.ndarray,
        input_lengths: jnp.ndarray,
        phone_lengths: jnp.ndarray
    ) -> jnp.ndarray:
        """
        CTC loss cho phone recognition (tối ưu PER)
        phone_logits: (batch, time, num_phones)
        """
        # Placeholder: dùng cross-entropy
        num_phones = phone_logits.shape[-1]
        loss = jax.nn.softmax_cross_entropy_with_logits(
            phone_logits,
            jax.nn.one_hot(phone_targets, num_phones)
        )
        return jnp.mean(loss) * self.ctc_phone_weight

    def total_loss(
        self,
        rnnt_logits: jnp.ndarray,
        targets: jnp.ndarray,
        input_lengths: jnp.ndarray,
        target_lengths: jnp.ndarray,
        group_probs: jnp.ndarray,
        group_ids: jnp.ndarray,
        group_logits: jnp.ndarray,
        distillation_loss: jnp.ndarray,
        phone_logits: jnp.ndarray = None,
        phone_targets: jnp.ndarray = None,
        phone_lengths: jnp.ndarray = None
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Tính tổng loss và trả về dict chi tiết
        """
        losses = {}

        # RNN-T loss
        losses['rnnt'] = self.rnnt_loss(rnnt_logits, targets, input_lengths, target_lengths)

        # Load balance loss
        losses['load_balance'] = self.load_balance_loss(group_probs, group_ids)

        # Z-loss
        losses['z_loss'] = self.z_loss(group_logits)

        # Distillation loss
        losses['distillation'] = distillation_loss

        # CTC phone loss (nếu có)
        if phone_logits is not None and phone_targets is not None:
            losses['ctc_phone'] = self.ctc_phone_loss(
                phone_logits, phone_targets, input_lengths, phone_lengths
            )
        else:
            losses['ctc_phone'] = 0.0

        # Tổng loss
        total = (
            losses['rnnt'] +
            losses['load_balance'] +
            losses['z_loss'] +
            losses['distillation'] +
            losses['ctc_phone']
        )
        losses['total'] = total

        return total, losses
