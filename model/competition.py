import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, List
from .config import ModelConfig


class NaturalNichesCompetition:
    """Thuật toán cạnh tranh Natural Niches cho Expert Groups"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.alpha = config.alpha
        self.use_matchmaker = config.use_matchmaker
        self.num_experts = config.total_experts  # 40
        self.num_groups = config.num_groups  # 8
        self.experts_per_group = config.experts_per_group  # 5

    def sample_parents(
        self,
        archive: jnp.ndarray,
        scores: jnp.ndarray,
        rand_key: jax.random.PRNGKey,
        group_id: int
    ) -> Tuple[int, int]:
        """
        Chọn parent theo thuật toán Natural Niches
        archive: (num_experts, ...) - trọng số expert
        scores: (num_experts, num_datapoints) - fitness của từng expert trên từng datapoint
        Trả về: (parent_1_idx, parent_2_idx) trong nhóm `group_id`
        """
        # Lấy indices của expert thuộc nhóm này
        group_start = group_id * self.experts_per_group
        group_end = group_start + self.experts_per_group
        group_indices = jnp.arange(group_start, group_end)
        group_scores = scores[group_indices]  # (experts_per_group, num_datapoints)

        # 1. Tính fitness tổng hợp với alpha normalization
        z = jnp.sum(group_scores, axis=0) ** self.alpha  # (num_datapoints,)
        fitness_matrix = group_scores / z[None, :]  # (experts_per_group, num_datapoints)
        fitness = jnp.sum(fitness_matrix, axis=1)  # (experts_per_group,)

        # 2. Chọn parent 1 theo phân phối fitness
        probs = nn.softmax(fitness)
        k1, k2 = jax.random.split(rand_key)
        parent_1_local_idx = jax.random.choice(k1, self.experts_per_group, shape=(1,), p=probs)[0]
        parent_1_idx = group_indices[parent_1_local_idx]

        # 3. MATCHMAKER - cơ chế cạnh tranh đặc biệt
        if self.use_matchmaker:
            # Tính match_score: độ khác biệt giữa parent_1 và các cá thể khác
            parent_1_fitness = fitness_matrix[parent_1_local_idx, :]  # (num_datapoints,)
            match_score = jnp.maximum(0, fitness_matrix - parent_1_fitness[None, :]).sum(axis=1)
            # Chọn parent 2 theo match_score (cạnh tranh để tìm cá thể khác biệt)
            probs2 = nn.softmax(match_score)
            parent_2_local_idx = jax.random.choice(k2, self.experts_per_group, shape=(1,), p=probs2)[0]
            parent_2_idx = group_indices[parent_2_local_idx]
        else:
            # Không dùng matchmaker: chọn parent 2 theo fitness cao thứ 2
            sorted_indices = jnp.argsort(fitness)[::-1]
            parent_2_local_idx = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
            parent_2_idx = group_indices[parent_2_local_idx]

        return int(parent_1_idx), int(parent_2_idx)

    def update_archive(
        self,
        archive: jnp.ndarray,
        scores: jnp.ndarray,
        new_expert_weights: jnp.ndarray,
        group_id: int
    ) -> jnp.ndarray:
        """
        Cập nhật archive: thêm expert mới, loại bỏ expert kém nhất
        archive: (num_experts, ...) trọng số hiện tại
        new_expert_weights: (experts_per_group, ...) trọng số con mới
        Trả về: archive mới
        """
        group_start = group_id * self.experts_per_group
        group_end = group_start + self.experts_per_group
        group_scores = scores[group_start:group_end]  # (experts_per_group, num_datapoints)

        # Tính fitness tổng hợp cho nhóm
        z = jnp.sum(group_scores, axis=0) ** self.alpha
        fitness_matrix = group_scores / z[None, :]
        fitness = jnp.sum(fitness_matrix, axis=1)  # (experts_per_group,)

        # Tìm expert có fitness thấp nhất
        worst_local_idx = jnp.argmin(fitness)
        worst_idx = group_start + worst_local_idx

        # Thay thế expert yếu nhất bằng expert mới (lấy expert mới đầu tiên)
        new_weights = new_expert_weights[0]  # (...)
        updated_archive = archive.at[worst_idx].set(new_weights)

        return updated_archive

    def run_competition_step(
        self,
        archive: jnp.ndarray,
        scores: jnp.ndarray,
        rand_key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Chạy 1 bước cạnh tranh cho TẤT CẢ các nhóm
        Trả về: archive mới sau khi cập nhật
        """
        new_archive = archive
        for group_id in range(self.num_groups):
            # Chọn parents
            parent_1_idx, parent_2_idx = self.sample_parents(
                archive, scores, rand_key, group_id
            )
            # Tạo offspring bằng slerp (spherical linear interpolation)
            parent_1_weights = archive[parent_1_idx]
            parent_2_weights = archive[parent_2_idx]
            # Slerp interpolation
            omega = jnp.arccos(jnp.sum(parent_1_weights * parent_2_weights) /
                                (jnp.linalg.norm(parent_1_weights) * jnp.linalg.norm(parent_2_weights) + 1e-8))
            t = 0.5  # Trung bình
            slerp = (jnp.sin((1 - t) * omega) * parent_1_weights + jnp.sin(t * omega) * parent_2_weights) / jnp.sin(omega + 1e-8)
            new_expert_weights = slerp[None, ...]  # (1, ...)

            # Cập nhật archive
            new_archive = self.update_archive(
                new_archive, scores, new_expert_weights, group_id
            )
        return new_archive
