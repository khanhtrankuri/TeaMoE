# TeaMoE: Shared Pretrained Experts Architecture

## Overview

TeaMoE now supports **shared pretrained expert models** across acoustic groups. Instead of each of the 8 groups having its own unique pretrained model (8 files), all groups share the same set of 5 pretrained experts (5 files) — a **87.5% storage reduction**.

### Key Innovation

```
Pretrained Phase:                   Fine-tuning Phase:
                                                    
M1 (generalist)    M1 (generalist)   M1_vowels   M1_plosives   M1_fricatives
      │                  │               ↓            ↓             ↓
M2 (generalist)    M2 (generalist)   M2_vowels   M2_plosives   M2_fricatives
      │                  │               ↓            ↓             ↓
M3 (generalist)    M3 (generalist)   M3_vowels   M3_plosives   M3_fricatives
      │                  │               ↓            ↓             ↓
M4 (generalist)    M4 (generalist)   M4_vowels   M4_plosives   M4_fricatives
      │                  │               ↓            ↓             ↓
M5 (generalist)    M5 (generalist)   M5_vowels   M5_plosives   M5_fricatives
```

Each expert M1-M5 is trained on **diverse mixed acoustic data** initially. During fine-tuning, the same M1 instance in Group 0 (vowels) learns vowel-specific representations, while M1 in Group 1 (plosives) learns plosive-specific representations — all through gradient updates.

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

This creates:
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

This creates a config where all 8 groups reference the same 5 pretrained models.

### 3. Fine-tune TeaMoE with Shared Experts

```bash
python train.py \
    --config config/shared_pretrained.yaml \
    --output-dir checkpoints/finetuned_shared \
    --num-epochs 20 \
    --learning-rate 1e-4
```

The model will load the shared pretrained experts and fine-tune them per group.

### 4. Monitor Specialization

During training, you'll see additional metrics:
- `SpecializationIndex`: Average cosine distance between current expert weights and their pretrained initialization. Higher = more specialized.
- `GroupSpecialization`: Per-group specialization scores.

### 5. Analyze Results

```bash
python run_shared_pretrained_workflow.py --phase analyze \
    --output-dir checkpoints/finetuned_shared \
    --pretrained-dir checkpoints/pretrained
```

Generates a detailed report on:
- Intra-group expert diversity
- Cross-group similarity for each expert index
- Specialization progress

## Configuration

### Config Structure

```yaml
model:
  # ... standard config ...

  # NEW: List of lists, one per group, each containing paths to 5 experts
  group_expert_pretrained_paths:
    - ["checkpoints/pretrained/expert_M1.pt",
       "checkpoints/pretrained/expert_M2.pt",
       "checkpoints/pretrained/expert_M3.pt",
       "checkpoints/pretrained/expert_M4.pt",
       "checkpoints/pretrained/expert_M5.pt"]  # Group 0 (vowels)
    - ["checkpoints/pretrained/expert_M1.pt", ...]  # Group 1 (plosives)
    # ... repeat for all 8 groups
```

### Backward Compatibility

The old format `group_pretrained_paths` (one path per group) is deprecated but still supported.

## How It Works

### 1. ExpertGroup Initialization

```python
class ExpertGroup(nn.Module):
    def __init__(self, config, expert_pretrained_paths: Optional[List[str]] = None):
        for i in range(config['num_experts']):
            expert = Expert(...)
            if expert_pretrained_paths and expert_pretrained_paths[i]:
                state_dict = torch.load(expert_pretrained_paths[i])
                expert.load_state_dict(state_dict)
            setattr(self, f"expert_{i}", expert)
```

Each expert in the group loads from a **different** pretrained checkpoint. Across groups, `expert_0` loads from the same M1 checkpoint, creating shared initialization.

### 2. Independent Fine-tuning

During training, gradients flow to the specific expert instance:
- `Group 0 → expert_0` receives gradients for vowel tokens
- `Group 1 → expert_0` receives gradients for plosive tokens
- Same M1 initialization → different fine-tuned variants

### 3. Tracking Specialization

Specialization index measures divergence from initial pretrained state:

```python
specialization = 1 - cosine_similarity(current_weights, pretrained_weights)
```

Values range from 0 (identical to pretrained) to 1 (completely different).

## File Structure

```
TeaMoE/
├── config/
│   ├── default.yaml                    # Standard config
│   └── shared_pretrained.yaml         # Generated shared config
├── checkpoints/
│   ├── pretrained/
│   │   ├── expert_M1.pt               # Pretrained expert models
│   │   ├── expert_M2.pt
│   │   ├── expert_M3.pt
│   │   ├── expert_M4.pt
│   │   └── expert_M5.pt
│   └── finetuned_shared/
│       ├── best_model.pt              # Fine-tuned TeaMoE
│       └── config.yaml
├── train_pretrained_experts.py        # Phase 1: Train M1-M5
├── generate_shared_config.py          # Generate shared config
├── train.py                           # Phase 2: Fine-tune TeaMoE
├── run_shared_pretrained_workflow.py  # End-to-end workflow
├── diagnostics.py                     # Specialization analysis tools
├── test_pretrained_groups.py          # Unit tests
└── ARCHITECTURE.md                    # Detailed architecture
```

## Workflow Comparison

| Approach | Pretrained Models | Storage | Cross-Group Learning |
|----------|------------------|---------|---------------------|
| **Baseline** | 8 (1 per group) | 8 files | ❌ No |
| **Per-expert** | 40 (1 per expert) | 40 files | ❌ No |
| **Shared (NEW)** | 5 (shared across all) | **5 files** | ✅ Yes |

## Benefits

1. **Storage Efficiency**: 87.5% reduction vs baseline (5 vs 40 files)
2. **Cross-Domain Transfer**: Each expert learns patterns from all acoustic groups
3. **Specialization via Fine-tuning**: Same building block → different specialists
4. **Flexibility**: Can mix shared and non-shared experts in same config

## Expected Behavior

### Early Training (Step 0)
- All groups' `expert_0` have identical weights (from M1)
- Routing may be unbalanced (all groups might prefer same experts)
- Specialization index near 0

### Mid Training (Step 10k)
- Gradual divergence: M1 in Group 0 differs from M1 in Group 1
- Routing starts specializing: vowels → certain experts, plosives → others
- Specialization index rising (0.1-0.3)

### Late Training (Step 50k+)
- Strong specialization: each expert-group combo is unique
- Balanced routing across experts
- Specialization index stable (0.3-0.6 typical)

## Diagnostics

### Print Specialization Report

```python
from diagnostics import print_specialization_report
from train import TeaMoEModel
import torch

model = TeaMoEModel(config)
checkpoint = torch.load('checkpoints/finetuned_shared/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

print_specialization_report(model)
```

### Track During Training

Specialization metrics are automatically logged to WandB during evaluation if `group_expert_pretrained_paths` is set.

## Research Questions

### Q: Will all groups' expert_0 diverge in different directions?

**Yes** — gradients for token "a" (vowel) flow to Group 0's expert_0, while gradients for token "p" (plosive) flow to Group 1's expert_0. Over time, they specialize.

### Q: What if one expert (M1) is much better than others?

Initially, all groups might route heavily to expert_0. The load balancing loss penalizes this, encouraging exploration. The gating network learns to route vowel tokens to the best vowel expert, which might be M3 in Group 0 but M1 in Group 1.

### Q: Can I freeze pretrained experts?

Yes! In config, set `requires_grad=False` on expert parameters after loading:

```python
for group in model.encoder.expert_groups:
    for i in range(group.config['num_experts']):
        expert = group.get_expert(i)
        for p in expert.parameters():
            p.requires_grad = False
```

Then only the gating network and other components train.

## Troubleshooting

### Issue: "Size of tensor a (6) must match tensor b (5)"

**Cause**: Even convolution kernel size. The padding formula `kernel_size // 2` only preserves length for odd kernels.

**Fix**: Use odd `conv_kernel_size` (default is 31, which is odd).

### Issue: Experts don't seem to specialize

**Check**:
- Is training long enough? Specialization takes thousands of steps
- Is the learning rate too low for expert fine-tuning?
- Try increasing `distillation_weight` to preserve some general knowledge

## Citation

If you use this architecture, please cite:

```bibtex
@inproceedings{teamoeshared2025,
  title={TeaMoE: Shared Pretrained Experts for Efficient Acoustic Modeling},
  author={Your Name},
  year={2025}
}
```
