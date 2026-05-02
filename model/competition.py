import torch
import torch.nn.functional as F
from typing import Tuple, List


class NaturalNichesCompetition:
    """Natural Niches competition algorithm for Expert Groups"""

    def __init__(self, config):
        self.config = config
        self.alpha = config['alpha']
        self.use_matchmaker = config['use_matchmaker']
        self.num_experts = config['total_experts']
        self.num_groups = config['num_groups']
        self.experts_per_group = config['experts_per_group']

    def sample_parents(
        self,
        archive: List[dict],
        scores: torch.Tensor,
        rand_key,
        group_id: int
    ) -> Tuple[int, int]:
        """
        Select parents using Natural Niches algorithm
        archive: list of state_dicts (one per expert)
        scores: (num_experts, num_datapoints) - fitness of each expert on each datapoint
        Returns: (parent_1_idx, parent_2_idx) within group `group_id`
        """
        group_start = group_id * self.experts_per_group
        group_end = group_start + self.experts_per_group
        group_indices = list(range(group_start, group_end))
        group_scores = scores[group_indices]  # (experts_per_group, num_datapoints)

        # 1. Compute composite fitness with alpha normalization
        z = torch.sum(group_scores, dim=1) ** self.alpha  # (experts_per_group,)
        fitness_matrix = group_scores / (z[:, None] + 1e-8)  # (experts_per_group, num_datapoints)
        fitness = torch.sum(fitness_matrix, dim=1)  # (experts_per_group,)

        # 2. Select parent 1 by fitness distribution
        probs = F.softmax(fitness, dim=0)
        parent_1_local_idx = torch.multinomial(probs, 1).item()
        parent_1_idx = group_indices[parent_1_local_idx]

        # 3. MATCHMAKER - special competition mechanism
        if self.use_matchmaker:
            parent_1_fitness = fitness_matrix[parent_1_local_idx, :]  # (num_datapoints,)
            match_score = torch.sum(torch.clamp(fitness_matrix - parent_1_fitness[None, :], min=0), dim=1)
            probs2 = F.softmax(match_score, dim=0)
            parent_2_local_idx = torch.multinomial(probs2, 1).item()
            parent_2_idx = group_indices[parent_2_local_idx]
        else:
            sorted_indices = torch.argsort(fitness, descending=True)
            parent_2_local_idx = sorted_indices[1].item() if len(sorted_indices) > 1 else sorted_indices[0].item()
            parent_2_idx = group_indices[parent_2_local_idx]

        return parent_1_idx, parent_2_idx

    def update_archive(
        self,
        archive: List[dict],
        scores: torch.Tensor,
        new_expert_weights: dict,
        group_id: int
    ) -> List[dict]:
        """
        Update archive: add new expert, remove worst expert
        archive: list of state_dicts (num_experts,)
        new_expert_weights: state_dict of new expert
        Returns: updated archive
        """
        group_start = group_id * self.experts_per_group
        group_end = group_start + self.experts_per_group
        group_scores = scores[group_start:group_end]  # (experts_per_group, num_datapoints)

        z = torch.sum(group_scores, dim=1) ** self.alpha
        fitness_matrix = group_scores / (z[:, None] + 1e-8)
        fitness = torch.sum(fitness_matrix, dim=1)

        worst_local_idx = torch.argmin(fitness).item()
        worst_idx = group_start + worst_local_idx

        archive[worst_idx] = new_expert_weights
        return archive

    def run_competition_step(
        self,
        archive: List[dict],
        scores: torch.Tensor,
        model: torch.nn.Module
    ) -> List[dict]:
        """
        Run one competition step for ALL groups
        Returns: updated archive with evolved expert weights
        """
        new_archive = [w.copy() if isinstance(w, dict) else {k: v.clone() for k, v in w.items()} for w in archive]
        for group_id in range(self.num_groups):
            parent_1_idx, parent_2_idx = self.sample_parents(
                new_archive, scores, None, group_id
            )
            parent_1_weights = new_archive[parent_1_idx]
            parent_2_weights = new_archive[parent_2_idx]

            # Slerp interpolation
            new_weights = {}
            for key in parent_1_weights:
                w1 = parent_1_weights[key].flatten()
                w2 = parent_2_weights[key].flatten()
                dot = torch.sum(w1 * w2)
                norm = torch.norm(w1) * torch.norm(w2) + 1e-8
                omega = torch.acos(torch.clamp(dot / norm, -1.0, 1.0))
                t = 0.5
                slerp = (torch.sin((1 - t) * omega) * w1 + torch.sin(t * omega) * w2) / (torch.sin(omega + 1e-8) + 1e-8)
                new_weights[key] = slerp.reshape(parent_1_weights[key].shape)

            new_archive = self.update_archive(new_archive, scores, new_weights, group_id)
        return new_archive
