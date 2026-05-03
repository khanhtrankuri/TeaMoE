# TeaMoE Architecture: Shared Pretrained Experts

## 1. Core Innovation: Parameter-Efficient MoE

### The Problem
Traditional MoE: Each acoustic group needs its own pretrained model → 8 separate checkpoints → no cross-group knowledge transfer.

### The Solution
**5 pretrained experts (M1-M5) shared across all 8 groups**:
- Each expert trained on diverse mixed acoustic data (no group split)
- During fine-tuning, same M1 in Group 0 (vowels) diverges from M1 in Group 1 (plosives)
- **Storage: 5 files instead of 40 (87.5% reduction!)**

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

## 2. Complete Architecture Flow

```
Input: [Batch, Seq_len, n_mels=80]
         │
         ▼
┌─────────────────────────────────────────┐
│  Input Projection                       │
│  Linear(80 → model_dim=1024)            │
│  Output: [B, S, 1024]                   │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  Gating Network                         │
│  MLP: 1024 → 256 → 8                    │
│  Output:                                │
│    • group_probs: [B, S, 8]             │
│    • group_ids: [B, S] (0-7)            │
│  Routes each token to an acoustic group│
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  Encoder Stack (24 layers)              │
├─────────────────────────────────────────┤
│                                         │
│  Layers 0-5: Pre-MoE                    │
│  ┌─────────────────────────────┐       │
│  │  ConformerLayer             │       │
│  │  • Conv1D (depthwise)       │       │
│  │  • Multi-Head Attention     │       │
│  │  • FeedForward              │       │
│  └─────────────────────────────┘       │
│           (shared across all)          │
│                                         │
│  Layers 6-17: MoE Layers                │
│  ┌─────────────────────────────┐       │
│  │  MoEConformerLayer          │       │
│  │  • Conv1D + Attention       │       │
│  │  • Expert Group Routing    │─────┐ │
│  └─────────────────────────────┘     │ │
│        (12 identical layers)         │ │
│                │                      │ │
│                ├─────────────────────┼─┘
│                │                     │
│                ▼                     │
│    ┌──────────────────────┐        │
│    │  Expert Groups (8)   │        │
│    │  ┌─────────────────┐ │        │
│    │  │   Group 0       │ │        │
│    │  │  (vowels)       │ │        │
│    │  │  ┌──────────┐   │ │        │
│    │  │  │exp₀: M₁  │   │ │        │
│    │  │  │exp₁: M₂  │   │ │        │
│    │  │  │exp₂: M₃  │   │ │        │
│    │  │  │exp₃: M₄  │   │ │        │
│    │  │  │exp₄: M₅  │   │ │        │
│    │  │  └──────────┘   │ │        │
│    │  └─────────────────┘ │        │
│    │  ┌─────────────────┐ │        │
│    │  │   Group 1       │ │        │
│    │  │  (plosives)     │ │        │
│    │  │  ┌──────────┐   │ │        │
│    │  │  │exp₀: M₁  │   │ │        │
│    │  │  │exp₁: M₂  │   │ │        │
│    │  │  │exp₂: M₃  │   │ │        │
│    │  │  │exp₃: M₄  │   │ │        │
│    │  │  │exp₄: M₅  │   │ │        │
│    │  │  └──────────┘   │ │        │
│    │  └─────────────────┘ │        │
│    │         ...           │        │
│    │  ┌─────────────────┐ │        │
│    │  │   Group 7       │ │        │
│    │  │  (other)        │ │        │
│    │  │  ┌──────────┐   │ │        │
│    │  │  │exp₀: M₁  │   │ │        │
│    │  │  │exp₁: M₂  │   │ │        │
│    │  │  │exp₂: M₃  │   │ │        │
│    │  │  │exp₃: M₄  │   │ │        │
│    │  │  │exp₄: M₅  │   │ │        │
│    │  │  └──────────┘   │ │        │
│    │  └─────────────────┘ │        │
│    └──────────────────────┘        │
│                │                    │
│                │ mean pooling       │
│                ▼                    │
│    [active_tokens, 1024]           │
│                │                    │
│                └────────────────────┼─┐
│                                     │
│  Layers 18-23: Post-MoE             │
│  ┌─────────────────────────────┐   │
│  │  ConformerLayer             │   │
│  │  (shared, same as pre-MoE)  │   │
│  └─────────────────────────────┘   │
│           (6 layers)                │
│                                     │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  Decoder (RNNT)                         │
│  • Prediction Network (2-layer LSTM)   │
│  • Joint Network (Linear)              │
│  Output: [B, T, vocab_size=5000]       │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  Phone CTC Head (optional)              │
│  Linear(1024 → num_phones=256)          │
│  Output: [B, S, 256]                    │
└─────────────────────────────────────────┘
```

## 3. ExpertGroup Detailed Structure

### 3.1 Initialization with Shared Pretrained Models

```python
class ExpertGroup(nn.Module):
    def __init__(self, config, expert_pretrained_paths: Optional[List[str]] = None):
        """
        Args:
            expert_pretrained_paths: List of 5 paths [M1.pt, M2.pt, M3.pt, M4.pt, M5.pt]
                                     All groups receive the SAME list
        """
        super().__init__()
        self.config = config

        # Create 5 distinct expert instances
        for i in range(config['num_experts']):  # num_experts = 5
            expert = Expert(
                expert_dim=config['expert_dim'],
                ff_multiplier=config['ff_multiplier'],
                dropout=config['dropout'],
            )

            # Load pretrained weights if provided
            if expert_pretrained_paths is not None and i < len(expert_pretrained_paths):
                pretrained_path = expert_pretrained_paths[i]
                if pretrained_path is not None:
                    state_dict = torch.load(pretrained_path, map_location='cpu')
                    expert.load_state_dict(state_dict)
                    print(f"  Loaded {pretrained_path} into expert_{i}")

            setattr(self, f"expert_{i}", expert)

    def forward(self, x, deterministic=True, use_checkpoint=False):
        """Process input through all 5 experts in parallel.

        Args:
            x: [N, model_dim] - tokens assigned to this group

        Returns:
            [N, model_dim] - mean pooled output of all 5 experts
        """
        outputs = []
        for i in range(self.config['num_experts']):
            expert = getattr(self, f"expert_{i}")
            out = expert(x, deterministic=deterministic, use_checkpoint=use_checkpoint)
            outputs.append(out)

        # Stack: [N, 5, model_dim] → mean: [N, model_dim]
        stacked = torch.stack(outputs, dim=1)
        return stacked.mean(dim=1)
```

### 3.2 Shared vs Independent Instances

**Key Point**: Although all groups load from the same 5 checkpoints, each group creates **independent expert instances**:

```
Memory after initialization:

Group 0 (vowels)                    Group 1 (plosives)
┌─────────────┐                    ┌─────────────┐
│ expert_0    │  ← M1 checkpoint   │ expert_0    │  ← M1 checkpoint
│ (params: M1) │───────────────────→│ (params: M1) │
└─────────────┘                    └─────────────┘
┌─────────────┐                    ┌─────────────┐
│ expert_1    │  ← M2 checkpoint   │ expert_1    │  ← M2 checkpoint
│ (params: M2) │───────────────────→│ (params: M2) │
└─────────────┘                    └─────────────┘
    ...                                    ...

After first gradient update:

Group 0 (vowels)                    Group 1 (plosives)
┌─────────────┐                    ┌─────────────┐
│ expert_0    │  M1 + Δᵛ           │ expert_0    │  M1 + Δᵖ
│ (specialized)│                   │ (different)  │
└─────────────┘                    └─────────────┘

where Δᵛ ≠ Δᵖ because:
  - Group 0 receives gradients from vowel tokens
  - Group 1 receives gradients from plosive tokens
```

## 4. MoEConformerLayer Forward Pass (Step-by-Step)

### 4.1 Input and Gating

```
Input x: [B, S, 1024]
group_ids: [B, S]  (from gating network)

Example:
  token "a" (vowel)    → group_ids[0,5] = 0 (Group 0)
  token "p" (plosive)  → group_ids[0,6] = 1 (Group 1)
```

### 4.2 Shared Conformer Branches

```python
def forward(self, x, group_ids, deterministic=True):
    residual = x

    # 1. Convolution Branch (shared)
    x = self.conv_norm(x)
    x_conv = self.conv(x.permute(0,2,1)).permute(0,2,1)
    x_conv = self.conv_gelu(x_conv)
    x_conv = self.conv_dropout(x_conv)
    x = x_conv + residual

    # 2. Attention Branch (shared)
    residual = x
    x = self.attn_norm(x)
    attn_out, _ = self.self_attn(x, x, x)
    x = self.attn_dropout(attn_out)
    x = x + residual

    # x now contains shared representation [B, S, 1024]
```

### 4.3 Expert Routing

```python
    batch_size, seq_len, model_dim = x.shape
    x_flat = x.reshape(-1, model_dim)      # [B×S, 1024]
    group_ids_flat = group_ids.reshape(-1) # [B×S]

    outputs = []
    masks = []

    # Process each expert group
    for group_idx, expert_group in enumerate(self.expert_groups):
        # Find tokens assigned to this group
        mask = (group_ids_flat == group_idx)

        if not mask.any():
            continue

        x_group = x_flat[mask]  # [N_group, 1024]

        # Forward through the group's 5 experts
        group_out = expert_group(x_group, deterministic, use_checkpoint)
        # expert_group returns: mean([N, 5, 1024]) = [N, 1024]

        outputs.append(group_out)
        masks.append(mask)

    # Scatter back to original positions
    result = torch.zeros_like(x_flat)
    for mask, out in zip(masks, outputs):
        result[mask] = out

    x_moe = result.reshape(batch_size, seq_len, model_dim)
```

### 4.4 Final Residual

```python
    x = self.moe_dropout(x_moe)
    x = x + residual  # residual from shared branches
    return x
```

## 5. Data Shapes Throughout Model

| Stage | Shape | Description |
|-------|-------|-------------|
| Input | `[B, T, 80]` | Raw mel-spectrogram |
| After input_proj | `[B, T, 1024]` | Projected features |
| After gating | `group_probs: [B, T, 8]` | Group assignment probs |
| | `group_ids: [B, T]` | Hard assignment |
| Pre-MoE output | `[B, T, 1024]` | Shared encoding |
| MoE x_flat | `[B×T, 1024]` | Flattened for routing |
| Group routing mask | `[B×T]` boolean | Which tokens per group |
| Expert group input | `[N_g, 1024]` | Tokens for group g |
| Expert outputs stacked | `[N_g, 5, 1024]` | 5 experts per token |
| After mean pooling | `[N_g, 1024]` | Combined group output |
| After scatter | `[B×T, 1024]` | Reconstructed full sequence |
| MoE output | `[B, T, 1024]` | After residual |
| Post-MoE output | `[B, T, 1024]` | Final encoder state |
| RNNT logits | `[B, U, V]` | Decoder predictions |
| Phone logits | `[B, T, 256]` | CTC phone predictions |

Where:
- B = batch size
- T = input sequence length (time frames)
- U = output sequence length (target tokens)
- V = vocabulary size (~5000)
- 8 = num_groups
- 5 = experts_per_group
- 1024 = model_dim

## 6. Configuration

### 6.1 Shared Pretrained Paths Format

```yaml
model:
  num_groups: 8
  experts_per_group: 5
  total_experts: 40

  # Each group gets the SAME list of 5 paths
  group_expert_pretrained_paths:
    - ["checkpoints/pretrained/expert_M1.pt",
       "checkpoints/pretrained/expert_M2.pt",
       "checkpoints/pretrained/expert_M3.pt",
       "checkpoints/pretrained/expert_M4.pt",
       "checkpoints/pretrained/expert_M5.pt"]  # ← Group 0 (vowels)
    - ["checkpoints/pretrained/expert_M1.pt",
       "checkpoints/pretrained/expert_M2.pt",
       ...]  # ← Group 1 (plosives) - SAME paths!
    - ...  # Groups 2-7: all identical lists
```

### 6.2 Legacy Format (Deprecated)

```yaml
# OLD: one path per group (8 paths total)
group_pretrained_paths:
  - "checkpoints/group0.pt"
  - "checkpoints/group1.pt"
  - ...
```

### 6.3 Mixed Configuration

You can mix shared and None:

```yaml
group_expert_pretrained_paths:
  - ["M1.pt", "M2.pt", None, None, None]  # Group 0: only 2 pretrained
  - [None, None, None, None, None]        # Group 1: train all from scratch
```

## 7. Specialization Dynamics

### 7.1 What Gets Tracked

```python
SpecializationIndex = mean(
    1 - cosine_similarity(
        current_expert_weights,
        pretrained_expert_weights
    )
) across all (group, expert) pairs
```

### 7.2 Expected Trajectory

```
Step       Specialization Index
0          0.000 (all experts identical to M1-M5)
1,000      0.05-0.10 (beginning divergence)
10,000     0.15-0.25 (noticeable specialization)
50,000     0.30-0.45 (strong specialization)
100,000+   0.40-0.60 (converged)
```

### 7.3 Cross-Group Similarity Matrix

For expert_0 across all groups (M1 derivatives):

```
           Group0  Group1  Group2  Group3  ...
Group0      1.00    0.85    0.82    0.80
Group1      0.85    1.00    0.84    0.81
Group2      0.82    0.84    1.00    0.83
Group3      0.80    0.81    0.83    1.00
...
```

Diagonal = 1.0 (self). Off-diagonal < 1.0 shows divergence.

### 7.4 Intra-Group Diversity

Within Group 0, how different are its 5 experts?

```
Group 0 diversity score:
  expert_0 (M1ᵛ) vs expert_1 (M2ᵛ): cosine_dist = 0.35
  expert_0 vs expert_2 (M3ᵛ):          cosine_dist = 0.38
  expert_0 vs expert_3 (M4ᵛ):          cosine_dist = 0.36
  expert_0 vs expert_4 (M5ᵛ):          cosine_dist = 0.37
  → Average: 0.365

Higher intra-group diversity = experts are truly different
```

## 8. Gradient Flow

### 8.1 Forward Pass Token Journey

```
Token "a" at position t:
  1. Gating assigns: group_id = 0 (vowels)
  2. Shared Conv+Attn: x[t] = f_shared(x[t])
  3. MoE routing: mask[t] = True for group 0
  4. Group 0 receives x[t], passes to ALL 5 experts:
     - expert_0: h₀ = M1ᵛ(x[t])
     - expert_1: h₁ = M2ᵛ(x[t])
     - expert_2: h₂ = M3ᵛ(x[t])
     - expert_3: h₃ = M4ᵛ(x[t])
     - expert_4: h₄ = M5ᵛ(x[t])
  5. Mean: h = mean(h₀, h₁, h₂, h₃, h₄)
  6. Output: x_out[t] = h + residual
  7. Decoder predicts token

Token "p" at position t:
  Same flow, but:
    - step 1: group_id = 1 (plosives)
    - step 4: Group 1's experts (M1ᵖ, M2ᵖ, M3ᵖ, M4ᵖ, M5ᵖ) process it
```

### 8.2 Backward Pass

```
Loss L computed from decoder output.

Gradients flow back:
  ∂L/∂x_out[t] → residual connection
  ∂L/∂h = ∂L/∂x_out[t] (copy to all 5 experts)

For token "a" (vowel) at position t:
  ∂L/∂M1ᵛ receives: ∂L/∂h₀ (via mean → split equally)
  ∂L/∂M2ᵛ receives: ∂L/∂h₁
  ...
  These gradients update M1ᵛ, M2ᵛ, ... in Group 0 ONLY

For token "p" (plosive):
  ∂L/∂M1ᵖ receives gradients → updates M1ᵖ in Group 1 ONLY

Result: Same initialization (M1), different updates → M1ᵛ ≠ M1ᵖ
```

## 9. Load Balancing with Shared Experts

### 9.1 The Challenge

With shared M1-M5 initialization, all groups might initially favor `expert_0` (M1 derivative), causing:

- M1 to become over-specialized in all groups
- M2-M5 underutilized

### 9.2 Auxiliary Load Balancing Loss

```python
# In CombinedLoss / MoEConformerLayer
group_probs: [B, S, 8]  # gating output per token
expert_usage: [8, 5]     # tokens assigned to (group, expert)

# Importance loss (encourage uniform expert usage within group)
importance = group_probs.mean(dim=1)  # [B, 8, 5]
cv = std(importance) / mean(importance)  # coefficient of variation
load_balance_loss = cv²

# Total loss = main_loss + λ * load_balance_loss
```

### 9.3 Per-Group Monitoring

```python
expert_usage[group_idx, expert_idx] = count(
    tokens where group_ids == group_idx
    AND that expert was selected within that group
)
```

For shared experts, monitor separately per group:
- `usage[0, 0]`: How many vowel tokens used M1ᵛ
- `usage[1, 0]`: How many plosive tokens used M1ᵖ

## 10. Training Phases

### 10.1 Phase 1: Pretrain M1-M5

**Objective**: Train 5 diverse generalist experts on mixed acoustic data.

```bash
python train_pretrained_experts.py \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-3
```

**Data**: All acoustic classes mixed together (vowels + plosives + fricatives + ...)

**Loss**: Autoencoder reconstruction (MSE) or contrastive learning.

**Output**: 5 checkpoints with different random seeds → diverse initializations.

### 10.2 Phase 2: Fine-tune TeaMoE with Shared Experts

**Objective**: Train full model with shared expert initialization.

```bash
python train.py \
  --config config/shared_pretrained.yaml \
  --num-epochs 20 \
  --learning-rate 1e-4
```

**What trains**:
- Gating network: ✅ trainable
- Expert parameters: ✅ trainable (fine-tune from pretrained)
- Input projection: ✅ trainable
- Decoder: ✅ trainable

**Expected**: Specialization index increases from 0 → 0.4+ over training.

### 10.3 Phase 3: Analyze

```bash
python run_shared_pretrained_workflow.py --phase analyze
```

Generates report with:
- Specialization per group
- Cross-group similarity matrices
- Intra-group diversity
- Routing patterns

## 11. Comparison Table

| Aspect | Baseline (8 group-specific) | Per-Expert (40 unique) | **Shared (5 across 8)** |
|--------|---------------------------|------------------------|-------------------------|
| Pretrained files | 8 | 40 | **5** ✅ |
| Storage reduction | 1× | 0.2× | **5×** ✅ |
| Cross-group learning | ❌ | ❌ | **✅** |
| M1 learns vowel + plosive patterns | ❌ | ❌ | **✅** |
| Expert diversity (intra-group) | medium | high | medium |
| Specialization per group | high | very high | **medium-high** ⚖️ |
| Initialization diversity | low (1 per group) | high (40 unique) | **medium** (5 unique) |
| Parameter count (total) | same | same | **same** ⚠️ |

**Key insight**: Shared approach achieves **87.5% storage reduction** while maintaining cross-group knowledge transfer. The trade-off: slightly less initial diversity (5 vs 40) but gains from seeing all groups during pretraining.

## 12. Advanced Features

### 12.1 Partial Sharing

Mix shared and non-shared in same model:

```yaml
group_expert_pretrained_paths:
  - ["M1.pt", "M2.pt", None, None, None]   # Group 0: 2 shared + 3 scratch
  - [None, None, None, None, None]         # Group 1: all scratch
  - ["M1.pt", "M2.pt", "M3.pt", None, None]  # Group 2: 3 shared
```

### 12.2 Freezing Pretrained Experts

To train only gating + decoder (experts frozen):

```python
# After model creation
for group in model.encoder.expert_groups:
    for i in range(group.config['num_experts']):
        expert = group.get_expert(i)
        for p in expert.parameters():
            p.requires_grad = False

# Now optimizer only sees non-expert parameters
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

### 12.3 Distillation from Pretrained Experts

Add distillation loss to preserve general knowledge:

```python
# In training loop
for group in model.encoder.expert_groups:
    for i in range(5):
        current = group.get_expert(i)
        pretrained = load_pretrained_weights(i)  # M1, M2, etc.
        distill_loss += F.mse_loss(current(params), pretrained(params))

total_loss = main_loss + 0.1 * distill_loss
```

## 13. Diagnostics & Monitoring

### 13.1 Built-in Metrics (logged to WandB)

During `evaluate()`, if `group_expert_pretrained_paths` is set:

```python
metrics = {
    "WER": ...,
    "PER": ...,
    "Gini": ...,  # load balancing
    "ExpertCosineDistance": ...,  # diversity
    "SpecializationIndex": ...,  # 🆕 divergence from pretrained
    "GroupSpecialization": [...],  # 🆕 per-group scores
}
```

### 13.2 Command-Line Diagnostics

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
  ...

2. Cross-Group Similarity for expert_0 (cosine distance)
--------------------------------------------------------
  vowels plos fric nas ...
  vowels   0.000  0.18  0.16  0.15 ...
  plosives 0.18   0.000 0.17  0.16 ...

3. Routing Distribution
-----------------------
  vowels       :  12,345 tokens (15.3%)
  plosives     :   8,901 tokens (11.0%)
  ...
  Gating entropy: 1.8424 (max = 2.0794)
```

## 14. FAQ

**Q: Do M1-M5 stay identical across groups before training?**

A: Yes, after `load_state_dict()`, all groups' `expert_0` have identical weights (from M1.pt). After first gradient update, they diverge.

**Q: Can I use different pretrained models per group?**

A: Yes! Set `group_expert_pretrained_paths` with different paths per group. Current implementation supports arbitrary combinations.

**Q: How do I know if specialization is working?**

A: Monitor `SpecializationIndex`:
- Increases from 0.0 → 0.3+ means experts are diverging as expected
- If it stays near 0, either LR is too low or loss isn't propagating

**Q: Will shared experts hurt accuracy?**

A: Research hypothesis: Cross-group transfer should **improve** accuracy by allowing M1 to learn both vowel and plosive patterns, making it a better general feature extractor. Needs empirical validation.

**Q: Can I still use Natural Niches Competition?**

A: Yes! Competition operates on all 40 expert instances independently. M1 derivatives in different groups compete separately.

## 15. Implementation Checklist

- [x] `ExpertGroup.__init__` accepts `expert_pretrained_paths: List[str]`
- [x] Config format: `group_expert_pretrained_paths: List[List[str]]`
- [x] All groups can load same M1-M5
- [x] Expert instances remain independent (`id(expert)` unique)
- [x] Specialization tracking in `evaluate()`
- [x] WandB logging for `SpecializationIndex`
- [x] Workflow script: pretrain → finetune → analyze
- [x] Diagnostic tools: `diagnostics.py`
- [x] Documentation: `SHARED_PRETRAINED.md`
- [x] Unit tests: `test_pretrained_groups.py`

## 16. Future Work

1. **Adaptive sharing**: Learn which groups should share which experts
2. **Hierarchical experts**: M1-M5 → group-specific → task-specific
3. **Cross-group attention**: Allow experts to peek at other groups' activations
4. **Progressive sharing**: Start with unique experts, gradually merge similar ones
5. **Theoretical analysis**: Prove specialization convergence bounds
