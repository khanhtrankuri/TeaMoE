import torch
import torch.nn.functional as F
from typing import Tuple


class ExpertDistillation:
    """Intra-group distillation: Top 2 teach Top 3-5"""

    def __init__(self, distillation_weight: float = 0.1):
        self.weight = distillation_weight

    def compute_kl_divergence(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute KL divergence between teacher and student
        teacher_logits: (batch*time, expert_dim) - output of top 1/2 expert
        student_logits: (batch*time, expert_dim) - output of expert 3/4/5
        """
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        kl = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-8) - student_log_probs), dim=-1)
        return torch.mean(kl) * (temperature ** 2)

    def group_distillation_loss(
        self,
        group_outputs: torch.Tensor,
        group_ids: torch.Tensor,
        flat_group_ids: torch.Tensor,
        rng_key=None
    ) -> torch.Tensor:
        """
        Compute distillation loss for one expert group
        group_outputs: (batch*time, num_experts_per_group, expert_dim)
        flat_group_ids: (batch*time,) - group_id for each frame
        Returns: average distillation loss
        """
        num_experts = group_outputs.shape[1]  # 5
        if num_experts < 3:
            return torch.tensor(0.0)

        top_1_output = group_outputs[:, 0, :]  # (batch*time, expert_dim)
        top_2_output = group_outputs[:, 1, :]  # (batch*time, expert_dim)
        student_outputs = group_outputs[:, 2:, :]  # (batch*time, 3, expert_dim)

        # Only apply to 10% of batches randomly
        if rng_key is not None:
            rand_val = torch.rand(1).item()
            if rand_val > 0.1:
                return torch.tensor(0.0)

        total_loss = 0.0
        for i in range(student_outputs.shape[1]):
            student_out = student_outputs[:, i, :]
            loss1 = self.compute_kl_divergence(top_1_output, student_out)
            loss2 = self.compute_kl_divergence(top_2_output, student_out)
            total_loss += (loss1 + loss2) / 2

        return total_loss * self.weight / 3  # Divide by 3 student experts

    def compute_all_groups_distillation(
        self,
        all_group_outputs: torch.Tensor,
        group_assignments: torch.Tensor,
        flat_group_ids: torch.Tensor,
        rng_key=None
    ) -> torch.Tensor:
        """Compute distillation loss for all groups"""
        num_groups = int(torch.max(group_assignments)) + 1
        total_loss = torch.tensor(0.0)

        for group_id in range(num_groups):
            expert_indices = torch.where(group_assignments == group_id)[0]
            if len(expert_indices) == 0:
                continue

            group_outputs = all_group_outputs[:, expert_indices, :]
            mask = (flat_group_ids == group_id)
            if not mask.any():
                continue

            group_outputs_masked = group_outputs[mask]
            flat_group_ids_masked = flat_group_ids[mask]

            group_loss = self.group_distillation_loss(
                group_outputs_masked, group_id, flat_group_ids_masked, rng_key
            )
            total_loss = total_loss + group_loss

        return total_loss
