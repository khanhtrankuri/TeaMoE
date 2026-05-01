import jax
import jax.numpy as jnp
from typing import Tuple


class ExpertDistillation:
    """Cơ chế distillation nội bộ nhóm: Top 2 dạy Top 3-5"""

    def __init__(self, distillation_weight: float = 0.1):
        self.weight = distillation_weight

    def compute_kl_divergence(
        self,
        teacher_logits: jnp.ndarray,
        student_logits: jnp.ndarray,
        temperature: float = 1.0
    ) -> jnp.ndarray:
        """
        Tính KL divergence giữa teacher và student
        teacher_logits: (batch*time, expert_dim) - output của top 1/2 expert
        student_logits: (batch*time, expert_dim) - output của expert 3/4/5
        """
        teacher_probs = nn.softmax(teacher_logits / temperature, axis=-1)
        student_log_probs = nn.log_softmax(student_logits / temperature, axis=-1)
        kl = jnp.sum(teacher_probs * (jnp.log(teacher_probs + 1e-8) - student_log_probs), axis=-1)
        return jnp.mean(kl) * (temperature ** 2)

    def group_distillation_loss(
        self,
        group_outputs: jnp.ndarray,
        group_ids: jnp.ndarray,
        flat_group_ids: jnp.ndarray,
        rng_key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Tính distillation loss cho 1 nhóm expert
        group_outputs: (batch*time, num_experts_per_group, expert_dim) - output của 5 expert trong nhóm
        flat_group_ids: (batch*time,) - group_id cho mỗi frame
        Trả về: distillation loss trung bình
        """
        num_experts = group_outputs.shape[1]  # 5
        if num_experts < 3:
            return 0.0

        # Chọn top 2 expert (giả sử đã được xếp hạng từ trước, ở đây lấy 2 expert đầu)
        # Trong thực tế, top 2 được xác định bằng fitness score
        top_1_output = group_outputs[:, 0, :]  # (batch*time, expert_dim)
        top_2_output = group_outputs[:, 1, :]  # (batch*time, expert_dim)

        # Các expert cần học: 3, 4, 5
        student_outputs = group_outputs[:, 2:, :]  # (batch*time, 3, expert_dim)

        # Chỉ áp dụng cho 10% batch ngẫu nhiên
        rand_val = jax.random.uniform(rng_key, shape=())
        if rand_val > 0.1:
            return 0.0

        total_loss = 0.0
        for i in range(student_outputs.shape[1]):
            student_out = student_outputs[:, i, :]
            # KL divergence với top 1
            loss1 = self.compute_kl_divergence(top_1_output, student_out)
            # KL divergence với top 2
            loss2 = self.compute_kl_divergence(top_2_output, student_out)
            total_loss += (loss1 + loss2) / 2

        return total_loss * self.weight / 3  # Chia cho 3 student experts

    def compute_all_groups_distillation(
        self,
        all_group_outputs: jnp.ndarray,  # (batch*time, total_experts, expert_dim)
        group_assignments: jnp.ndarray,  # (total_experts,) group_id của từng expert
        flat_group_ids: jnp.ndarray,  # (batch*time,)
        rng_key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Tính distillation loss cho tất cả nhóm"""
        num_groups = jnp.max(group_assignments) + 1
        total_loss = 0.0

        for group_id in range(num_groups):
            # Lấy indices của expert thuộc nhóm này
            expert_indices = jnp.where(group_assignments == group_id)[0]
            if len(expert_indices) == 0:
                continue

            # Lấy output của các expert trong nhóm này
            group_outputs = all_group_outputs[:, expert_indices, :]  # (batch*time, experts_per_group, expert_dim)

            # Chỉ xử lý frame thuộc nhóm này
            mask = (flat_group_ids == group_id)
            if not jnp.any(mask):
                continue

            group_outputs_masked = group_outputs[mask]  # (num_frames, experts_per_group, expert_dim)
            flat_group_ids_masked = flat_group_ids[mask]

            # Tính loss cho nhóm này
            group_loss = self.group_distillation_loss(
                group_outputs_masked, group_id, flat_group_ids_masked, rng_key
            )
            total_loss += group_loss

        return total_loss
