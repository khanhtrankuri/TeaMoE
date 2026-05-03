# TeaMoE: Shared Pretrained Experts Architecture

## Overview

TeaMoE supports **shared pretrained expert models** across acoustic groups. Instead of 8 separate checkpoints (one per group), all groups share the same 5 pretrained experts — **87.5% storage reduction** with cross-group knowledge transfer.

### Key Innovation

```
PRETRAINED PHASE:                    FINE-TUNING PHASE:

    M₁ ──────────────┐                  M₁ᵛ ∈ Group 0 (vowels)
    M₂ ──────────────┤                  M₂ᵛ ∈ Group 0
    M₃ ──────────────┤                  M₃ᵛ ∈ Group 0
    M₄ ──────────────┤      shared      M₄ᵛ ∈ Group 0
    M₅ ──────────────┘   initialization  M₅ᵛ ∈ Group 0
                                          ↓ fine-tune
    M₁ ──────────────┐                  M₁ᵖ ∈ Group 1 (plosives)
    M₂ ──────────────┤                  M₂ᵖ ∈ Group 1
    M₃ ──────────────┤                  M₃ᵖ ∈ Group 1
    M₄ ──────────────┤                  M₄ᵖ ∈ Group 1
    M₅ ──────────────┘                  M₅ᵖ ∈ Group 1
                                          ↓
    ... same M₁-M₅ for all 8 groups      ... specialized per group
```

Each expert M1-M5 trains on diverse mixed acoustic data. During fine-tuning, the same M1 in Group 0 (vowels) diverges from M1 in Group 1 (plosives) through gradient updates.

## Quick Start

### 1. Train Pretrained Experts (M1-M5)

```bash
python train_pretrained_experts.py \
    --config config/default.yaml \
    --output-dir checkpoints/pretrained \
    --epochs 10 \
    --batch-size 64 \
    --lr 1e-3
```

Creates:
- `checkpoints/pretrained/expert_M1.pt`
- `checkpoints/pretrained/expert_M2.pt`
- ...
- `checkpoints/pretrained/expert_M5.pt`

### 2. Generate Shared Config

```bash
python generate_shared_config.py \
    --base-config config/default.yaml \
    --pretrained-dir checkpoints/pretrained \
    --output config/shared_pretrained.yaml
```

This config makes all 8 groups reference the same 5 pretrained models.

### 3. (Optional) Enable Attention Pooling

Edit `config/shared_pretrained.yaml`:

```yaml
model:
  use_attention_pooling: true    # NEW: Attention instead of mean
  attn_heads: 4                  # Multi-head attention
  attn_dropout: 0.1
```

**Benefits**: Token-specific expert weighting (+1M params/group, ~8M total).

### 4. Fine-tune TeaMoE

```bash
python train.py \
    --config config/shared_pretrained.yaml \
    --output-dir checkpoints/finetuned_shared \
    --num-epochs 20 \
    --learning-rate 1e-4
```

The model loads shared pretrained experts and fine-tunes them per group.

### 5. Monitor Specialization

During evaluation, additional metrics appear:
- `SpecializationIndex`: Cosine distance from pretrained weights (0→0.6)
- `GroupSpecialization`: Per-group scores

### 6. Analyze Results

```bash
python run_shared_pretrained_workflow.py --phase analyze \
    --output-dir checkpoints/finetuned_shared \
    --pretrained-dir checkpoints/pretrained
```

Generates report with intra-group diversity, cross-group similarity, specialization progress.

## Pooling Strategies

### Mean Pooling (Default)

```python
combined = stacked_outputs.mean(dim=1)  # [N, 5, D] → [N, D]
```

- **Pros**: Simple, no extra parameters, stable
- **Cons**: Equal weight to all experts regardless of token
- **Use for**: Baselines, memory-constrained training

### Attention Pooling (Recommended)

```yaml
use_attention_pooling: true
attn_heads: 4
attn_dropout: 0.1
```

**Architecture**:
```
x (input to group): [N, D]
   │
   ├─→ Query: q = W_q(x)      → [N, H, head_dim]
   │
   └─→ Expert outputs stacked: [N, 5, D]
           │
           ├─→ Keys: k = W_k(stacked)  → [N, 5, H, head_dim]
           └─→ Values: v = W_v(stacked)→ [N, 5, H, head_dim]
                   │
                   └─→ Attention:
                       scores = softmax(q·k^T/√d)  → [N, H, 1, 5]
                       context = Σ(scores * v)     → [N, H, head_dim]
                       output = LayerNorm(W_out(context) + x)
```

**Pros**:
- Learns token-specific expert weights
- Vowel "a" may weight M3 highly, plosive "p" may weight M1
- Better utilization of diverse experts

**Cons**:
- +1M parameters per group (D=1024, H=4)
- Slightly slower

**Use for**: Final production models, maximizing accuracy

### Parameter Overhead

| Model Dim | Experts | Mean | Attention | Overhead |
|-----------|---------|------|-----------|----------|
| 512       | 5       | 3.3M | 4.1M      | +0.8M    |
| 1024      | 5       | 13M  | 14M       | +1.1M    |

For TeaMoE (1024D, 8 groups): **+8-10M total** (negligible vs 200M base model)

## Configuration

### Shared Pretrained Paths

```yaml
model:
  num_groups: 8
  experts_per_group: 5
  total_experts: 40

  # Each group gets SAME list of 5 paths
  group_expert_pretrained_paths:
    - ["M1.pt", "M2.pt", "M3.pt", "M4.pt", "M5.pt"]  # Group 0 (vowels)
    - ["M1.pt", "M2.pt", "M3.pt", "M4.pt", "M5.pt"]  # Group 1 (plosives)
    - [...]  # Groups 2-7: all identical
```

### Mixed Sharing (Advanced)

```yaml
group_expert_pretrained_paths:
  - ["M1.pt", "M2.pt", null, null, null]   # Group 0: 2 shared + 3 scratch
  - [null, null, null, null, null]         # Group 1: all scratch
```

### Legacy Format (Deprecated)

```yaml
# OLD: one checkpoint per group (8 total)
group_pretrained_paths:
  - "group0.pt"
  - "group1.pt"
  - ...
```

## How It Works

### ExpertGroup Initialization

```python
class ExpertGroup(nn.Module):
    def __init__(self, config, expert_pretrained_paths=None):
        for i in range(5):  # num_experts
            expert = Expert(...)
            if expert_pretrained_paths and expert_pretrained_paths[i]:
                expert.load_state_dict(torch.load(expert_pretrained_paths[i]))
            setattr(self, f"expert_{i}", expert)
```

All groups load M1-M5 identically initially.

### Fine-tuning Divergence

```
Step 0:    M1_0 == M1_1 == M1_2 (identical weights)
Step 1k:   M1_0 ≈ M1_1 ≈ M1_2 (minor gradient differences)
Step 10k:  M1_0 ≠ M1_1 ≠ M1_2 (specialization)
Step 50k+: M1_0 ⊥ M1_1 ⊥ M1_2 (orthogonal-ish, highly specialized)
```

Gradient flow:
- Token "a" (vowel) → Group 0 → updates M1_0, M2_0, M3_0, M4_0, M5_0
- Token "p" (plosive) → Group 1 → updates M1_1, M2_1, M3_1, M4_1, M5_1

Same M1 initialization, different gradients → different specialists.

## Workflow Comparison

| Approach | Pretrained Models | Storage | Cross-Group Learning |
|----------|------------------|---------|---------------------|
| Baseline | 8 (1 per group) | 8 files | ❌ |
| Per-Expert | 40 (1 per expert) | 40 files | ❌ |
| **Shared** | **5 (shared)** | **5 files** | **✅** |

## Benefits

1. **Storage Efficiency**: 87.5% reduction vs baseline
2. **Cross-Domain Transfer**: Each expert learns patterns from all acoustic groups
3. **Specialization via Fine-tuning**: Same building block → different specialists
4. **Flexible**: Mix shared and scratch experts per group

## Expected Specialization Trajectory

```
Step       Specialization Index
0          0.000 (identical to pretrained)
1,000      0.05-0.10 (beginning divergence)
10,000     0.15-0.25 (noticeable)
50,000+    0.30-0.60 (converged)
```

SpecializationIndex = mean(1 - cosine_similarity(current_weights, pretrained_weights))

## Diagnostics

### Built-in Metrics (WandB)

During evaluation if `group_expert_pretrained_paths` is set:
- `WER`, `PER`: Task performance
- `Gini`: Load balance (0=balanced, 1=unbalanced)
- `ExpertCosineDistance`: Diversity among all experts
- `SpecializationIndex`: Divergence from pretrained (new!)
- `GroupSpecialization`: Per-group array

### Command-line Diagnostics

```python
from diagnostics import print_specialization_report

print_specialization_report(model)
```

Output:
```
======================================================================
EXPERT SPECIALIZATION REPORT
======================================================================

Architecture: 8 groups × 5 experts
Total distinct expert instances: 40

1. Intra-Group Expert Diversity (cosine distance)
--------------------------------------------------
  vowels       : 0.3652
  plosives     : 0.3521
  fricatives   : 0.3710

2. Cross-Group Similarity for expert_0 (cosine distance)
--------------------------------------------------------
  vowels plos fric nas ...
  vowels   0.000  0.18  0.16  0.15 ...
  plosives 0.18   0.000 0.17  0.16 ...

3. Routing Distribution
-----------------------
  vowels       :  12,345 tokens (15.3%)
  plosives     :   8,901 tokens (11.0%)
  Gating entropy: 1.8424 (max = 2.0794)
```

## FAQ

**Q: Do M1-M5 stay identical across groups before training?**
A: Yes, after loading checkpoints all groups' expert_0 have identical weights. After first gradient update, they diverge.

**Q: How do I know if specialization is working?**
A: Monitor `SpecializationIndex`. Increases from 0 → 0.3+ means experts diverging as expected. If stuck near 0, LR may be too low.

**Q: Will shared experts hurt accuracy?**
A: Hypothesis: Cross-group transfer **improves** accuracy by letting M1 learn both vowel and plosive patterns. Requires empirical validation.

**Q: Can I freeze pretrained experts?**
A: Yes:
```python
for group in model.encoder.expert_groups:
    for i in range(5):
        for p in group.get_expert(i).parameters():
            p.requires_grad = False
```

**Q: Can I mix shared and non-shared experts?**
A: Yes! Set some paths to `null` in `group_expert_pretrained_paths`.

## Implementation Checklist

- [x] `ExpertGroup` accepts per-expert pretrained paths
- [x] Config format: `group_expert_pretrained_paths: List[List[str]]`
- [x] All groups can load same M1-M5
- [x] Expert instances are independent (`id()` unique)
- [x] Attention pooling option
- [x] Specialization tracking in `evaluate()`
- [x] WandB logging for `SpecializationIndex`
- [x] Workflow script: pretrain → finetune → analyze
- [x] Diagnostic tools: `diagnostics.py`
- [x] Documentation
- [x] Unit tests

## Future Work

1. Adaptive sharing: Learn which groups should share which experts
2. Hierarchical experts: M1-M5 → group-specific → task-specific
3. Cross-group attention: Let experts peek at other groups' activations
4. Progressive sharing: Start unique, merge similar experts during training
5. Theoretical analysis of specialization convergence
