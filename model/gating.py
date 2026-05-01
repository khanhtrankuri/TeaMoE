import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class GatingNetwork(nn.Module):
    """Mạng chọn nhóm expert (8-way) cho mỗi frame"""
    num_groups: int = 8
    model_dim: int = 1024
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # x: (batch, time, model_dim)
        # Trả về: group_probs (batch, time, num_groups), group_ids (batch, time)
        hidden = nn.Dense(self.hidden_dim)(x)
        hidden = nn.relu(hidden)
        logits = nn.Dense(self.num_groups)(hidden)
        group_probs = nn.softmax(logits, axis=-1)
        group_ids = jnp.argmax(group_probs, axis=-1)
        return group_probs, group_ids


def select_experts_by_group(
    group_ids: jnp.ndarray,
    all_expert_outputs: jnp.ndarray,
    group_assignments: jnp.ndarray,
    top_k: int = 5
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Chọn expert dựa trên group_ids.
    all_expert_outputs: (batch*time, total_experts, expert_dim)
    group_assignments: (total_experts,) chứa group_id của từng expert
    Trả về: selected_experts (batch*time, top_k, expert_dim), selection_weights (batch*time, top_k)
    """
    # Lấy mask cho từng frame: expert nào thuộc group được chọn thì mask=1
    # group_ids: (batch, time) -> mở rộng thành (batch*time,)
    flat_group_ids = group_ids.reshape(-1)
    # group_assignments: (total_experts,) -> (1, total_experts)
    group_mask = (group_assignments[None, :] == flat_group_ids[:, None])  # (batch*time, total_experts)
    # Lấy top_k expert trong nhóm (giả sử mỗi nhóm có đúng top_k expert, ở đây là 5)
    # Thực tế: mỗi nhóm có 5 expert, ta lấy tất cả 5 expert đó
    # Nếu muốn linh hoạt hơn, có thể dùng top_k trong nhóm (nhưng thiết kế là lấy cả nhóm)
    # Ở đây ta lấy tất cả expert thuộc nhóm được chọn
    # Tuy nhiên, all_expert_outputs đã chứa output của tất cả expert (40 expert), ta cần lọc
    # Cách đơn giản: với mỗi frame, lấy các expert có group_mask=1
    # Nhưng số lượng expert mỗi nhóm cố định là 5, nên ta có thể precompute indices
    # Để đơn giản, ta sẽ trả về tất cả expert trong nhóm (5 expert)
    # Và dùng trọng số bằng nhau (hoặc có thể dùng gating weight riêng cho từng expert)
    # Theo thiết kế: chọn nhóm -> kích hoạt toàn bộ 5 expert trong nhóm
    # Vậy selection_weights có thể là đều nhau 1/5 cho 5 expert đó
    # Ta sẽ xử lý ở mức higher-level: sau khi có group_id, ta lấy output của 5 expert thuộc nhóm đó
    # Ở đây, ta trả về mask và để hàm gọi xử lý
    return group_mask  # Trả về mask để sử dụng sau
