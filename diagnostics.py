"""
Diagnostic tools for tracking expert specialization in shared-pretrained TeaMoE.

This module provides functions to:
- Track weight divergence between same-expert across different groups
- Monitor routing patterns per group
- Measure cross-group similarity matrices
- Track load balancing metrics per group
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def compute_expert_weight_distance(
    model,
    expert_idx: int,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Compute pairwise distances between expert_idx across all groups.

    Args:
        model: TeaMoEModel instance
        expert_idx: Which expert index to compare (0-4)
        metric: "cosine" or "l2"

    Returns:
        Distance matrix of shape [num_groups, num_groups]
    """
    num_groups = len(model.encoder.expert_groups)
    vectors = []

    for group_idx in range(num_groups):
        expert = model.encoder.expert_groups[group_idx].get_expert(expert_idx)
        params = torch.cat([p.detach().cpu().flatten() for p in expert.parameters()])
        vectors.append(params)

    vectors = torch.stack(vectors)  # [G, D]

    if metric == "cosine":
        normed = F.normalize(vectors, dim=1)
        sim_matrix = normed @ normed.T  # [G, G] similarity
        dist_matrix = 1 - sim_matrix.numpy()
    elif metric == "l2":
        dist_matrix = torch.cdist(vectors, vectors, p=2).numpy()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return dist_matrix


def compute_group_specialization_scores(
    model,
    group_pretrained_weights: Optional[List[Dict[str, torch.Tensor]]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute specialization scores for each group's experts.

    If pretrained_weights provided, measures divergence from initial state.
    Otherwise, measures intra-group diversity (expert_0 vs expert_1, etc. within same group).

    Returns:
        dict with keys:
        - "group_expert_diversity": [num_groups] - Average pairwise distance within group
        - "cross_group_expert_similarity": [num_groups, num_experts] - How similar expert i is to same expert in other groups
        - "specialization_index": [num_groups] - Higher = more specialized (diverge from shared init)
    """
    num_groups = len(model.encoder.expert_groups)
    experts_per_group = model.encoder.expert_groups[0].config["num_experts"]

    results = {
        "group_expert_diversity": np.zeros(num_groups),
        "cross_group_expert_similarity": np.zeros((num_groups, experts_per_group)),
        "specialization_index": np.zeros(num_groups),
    }

    # Compute intra-group diversity
    for g in range(num_groups):
        group = model.encoder.expert_groups[g]
        expert_vectors = []
        for i in range(experts_per_group):
            expert = group.get_expert(i)
            params = torch.cat([p.detach().cpu().flatten() for p in expert.parameters()])
            expert_vectors.append(F.normalize(params, dim=0))

        # Average pairwise cosine distance within group
        vecs = torch.stack(expert_vectors)
        sim = vecs @ vecs.T
        mask = torch.triu(torch.ones_like(sim), diagonal=1).bool()
        avg_sim = sim[mask].mean().item()
        results["group_expert_diversity"][g] = 1 - avg_sim

    # Compute cross-group similarity for each expert index
    for e_idx in range(experts_per_group):
        vectors = []
        for g in range(num_groups):
            expert = model.encoder.expert_groups[g].get_expert(e_idx)
            params = torch.cat([p.detach().cpu().flatten() for p in expert.parameters()])
            vectors.append(F.normalize(params, dim=0))

        vecs = torch.stack(vectors)  # [G, D]
        sim = vecs @ vecs.T  # [G, G]
        # Average similarity of this expert across all pairs of groups
        mask = ~torch.eye(num_groups, dtype=torch.bool)
        avg_cross_sim = sim[mask].mean().item()
        results["cross_group_expert_similarity"][:, e_idx] = 1 - avg_cross_sim

    # Specialization index: how much each group's experts differ from their initial shared state
    if group_pretrained_weights is not None:
        for g in range(num_groups):
            group_dists = []
            for e_idx in range(experts_per_group):
                current_expert = model.encoder.expert_groups[g].get_expert(e_idx)
                current_params = torch.cat([p.detach().cpu().flatten() for p in current_expert.parameters()])
                init_params = torch.cat([
                    group_pretrained_weights[g][f"expert_{e_idx}.{k}"].flatten()
                    for k in list(next(iter(group_pretrained_weights[g].values())).keys())
                ])

                # Normalize and compute cosine distance
                current_norm = F.normalize(current_params, dim=0)
                init_norm = F.normalize(init_params, dim=0)
                dist = 1 - (current_norm @ init_norm).item()
                group_dists.append(dist)

            results["specialization_index"][g] = np.mean(group_dists)

    return results


def analyze_routing_patterns(
    model,
    dataloader,
    device,
    num_batches: int = 10
) -> Dict[str, np.ndarray]:
    """
    Analyze routing patterns: which groups route to which experts?

    Returns:
        dict with:
        - "group_routing_matrix": [num_groups, num_groups] - How much each gating group routes to each expert group
        - "expert_usage_per_group": [num_groups, experts_per_group] - Usage of each expert within each group
        - "group_selection_entropy": [num_groups] - Entropy of group selection per gating decision
    """
    model.eval()
    num_groups = model.config.get("num_groups", 8)
    experts_per_group = model.config.get("experts_per_group", 5)

    group_routing = np.zeros((num_groups, num_groups))
    expert_usage = np.zeros((num_groups, experts_per_group))
    group_counts = np.zeros(num_groups)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            audio_features, targets, phone_targets, input_lengths, target_lengths, phone_lengths = batch
            audio_features = audio_features.to(device)

            x = model.input_proj(audio_features)
            group_probs, group_ids = model.gating(x, deterministic=True)

            # Count group selections
            batch_size, seq_len = group_ids.shape
            flat_ids = group_ids.cpu().numpy().flatten()

            for selected_gid in flat_ids:
                if 0 <= selected_gid < num_groups:
                    group_counts[selected_gid] += 1

            # For each token, track which expert group it went to
            # Note: In current implementation, group_ids directly maps to expert group
            for src_gid in range(num_groups):
                mask = (flat_ids == src_gid)
                if mask.any():
                    # All tokens from gating group src_gid went to expert group src_gid
                    group_routing[src_gid, src_gid] += mask.sum()

                    # Also track which expert within that group was used
                    # This requires accessing the expert group's output
                    # For now, we assume equal distribution or track via separate hook

            # Could also track actual expert usage by modifying forward pass

    # Normalize
    if group_counts.sum() > 0:
        group_routing = group_routing / group_counts.sum()

    results = {
        "group_routing_matrix": group_routing,
        "expert_usage_per_group": expert_usage,
        "group_selection_counts": group_counts,
        "group_selection_entropy": -np.sum(
            (group_counts / (group_counts.sum() + 1e-8)) *
            np.log(group_counts / (group_counts.sum() + 1e-8) + 1e-8)
        ),
    }

    return results


def print_specialization_report(model, routing_stats: Optional[Dict] = None):
    """
    Print a formatted report on expert specialization.
    """
    print("\n" + "="*70)
    print("EXPERT SPECIALIZATION REPORT")
    print("="*70)

    num_groups = len(model.encoder.expert_groups)
    experts_per_group = model.encoder.expert_groups[0].config["num_experts"]

    print(f"\nArchitecture: {num_groups} groups × {experts_per_group} experts")
    print(f"Total distinct expert instances: {num_groups * experts_per_group}")

    # Group names
    group_names = [gc.get("group_name", f"Group_{i}")
                   for i, gc in enumerate(model.config.get("group_configs", []))]
    if not group_names:
        group_names = [f"Group_{i}" for i in range(num_groups)]

    # Compute intra-group diversity
    print("\n1. Intra-Group Expert Diversity (cosine distance, higher = more diverse)")
    print("-" * 70)
    for g in range(num_groups):
        group = model.encoder.expert_groups[g]
        expert_vectors = []
        for i in range(experts_per_group):
            expert = group.get_expert(i)
            params = torch.cat([p.detach().cpu().flatten() for p in expert.parameters()])
            expert_vectors.append(F.normalize(params, dim=0))

        vecs = torch.stack(expert_vectors)
        sim = vecs @ vecs.T
        mask = torch.triu(torch.ones_like(sim), diagonal=1).bool()
        avg_sim = sim[mask].mean().item()
        diversity = 1 - avg_sim
        print(f"  {group_names[g]:15s}: {diversity:.4f}")

    # Compute cross-group similarity for expert_0
    print("\n2. Cross-Group Similarity for expert_0 (cosine distance)")
    print("-" * 70)
    print("  (higher = more specialized per group)")
    expert_idx = 0
    vectors = []
    for g in range(num_groups):
        expert = model.encoder.expert_groups[g].get_expert(expert_idx)
        params = torch.cat([p.detach().cpu().flatten() for p in expert.parameters()])
        vectors.append(F.normalize(params, dim=0))

    vecs = torch.stack(vectors)
    sim = vecs @ vecs.T
    dist = 1 - sim.numpy()

    print("  " + " ".join(f"{name[:8]:8s}" for name in group_names[:4]))
    for i in range(min(num_groups, 4)):
        row = "  " + " ".join(f"{dist[i, j]:.3f}    " for j in range(min(num_groups, 4)))
        print(row)

    # Routing stats if provided
    if routing_stats is not None:
        print("\n3. Routing Distribution")
        print("-" * 70)
        counts = routing_stats["group_selection_counts"]
        total = counts.sum()
        for g in range(num_groups):
            if total > 0:
                pct = counts[g] / total * 100
                print(f"  {group_names[g]:15s}: {counts[g]:8.0f} tokens ({pct:5.1f}%)")

        entropy = routing_stats["group_selection_entropy"]
        print(f"\n  Gating entropy: {entropy:.4f} (max = {np.log(num_groups):.4f})")

    print("\n" + "="*70)


def track_specialization_during_training(
    model,
    pretrained_checkpoint_dir: str,
    output_file: str = "specialization_tracking.jsonl"
):
    """
    Track specialization progress during training.

    Call this periodically (e.g., every eval step) to log:
    - Weight distance from initial pretrained state per expert
    - Cross-group similarity matrices
    """
    import json

    num_groups = len(model.encoder.expert_groups)
    experts_per_group = model.encoder.expert_groups[0].config["num_experts"]

    # Load pretrained checkpoints
    pretrained_weights = []
    for g in range(num_groups):
        group_weights = {}
        for e in range(experts_per_group):
            ckpt_path = Path(pretrained_checkpoint_dir) / f"expert_M{e+1}.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu")
                group_weights[f"expert_{e}"] = ckpt["expert_state_dict"]
        pretrained_weights.append(group_weights)

    # Compute distances
    tracking = {
        "global_step": model.global_step if hasattr(model, "global_step") else 0,
        "expert_distances": {},
        "avg_specialization_per_group": [],
    }

    for e_idx in range(experts_per_group):
        tracking["expert_distances"][f"expert_{e_idx}"] = compute_expert_weight_distance(
            model, e_idx, metric="cosine"
        ).tolist()

    # Compute average specialization per group
    spec_scores = compute_group_specialization_scores(model, pretrained_weights)
    tracking["avg_specialization_per_group"] = spec_scores["specialization_index"].tolist()

    # Append to file
    with open(output_file, "a") as f:
        f.write(json.dumps(tracking) + "\n")

    return tracking


if __name__ == "__main__":
    # Quick test
    print("Diagnostics module loaded successfully.")
    print("Functions available:")
    print("  - compute_expert_weight_distance()")
    print("  - compute_group_specialization_scores()")
    print("  - analyze_routing_patterns()")
    print("  - print_specialization_report()")
    print("  - track_specialization_during_training()")
