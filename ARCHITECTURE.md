# TeaMoE Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [Core Innovation: Shared Pretrained Experts](#core-innovation)
3. [Complete Architecture](#complete-architecture)
4. [Component Deep Dive](#components)
5. [Data Flow](#data-flow)
6. [Configuration](#configuration)
7. [Training Pipeline](#training-pipeline)
8. [Specialization Dynamics](#specialization)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

---

## Overview <a name="overview"></a>

TeaMoE (**T**oken-aware **E**xpert **a**ttention **Mo****E**) is a speech recognition model combining:
- **MoE-Conformer Encoder**: 24-layer Conformer with Mixture-of-Experts in layers 6-17
- **RNN-T Decoder**: Transducer decoder for sequence-to-sequence ASR
- **Natural Niches Competition**: Evolutionary algorithm for expert specialization
- **Shared Pretrained Experts**: Storage-efficient expert initialization

### Quick Stats
- **Parameters**: ~958M (model_dim=1024)
- **Experts**: 8 groups × 5 experts = 40 total
- **Pretrained storage**: 5 files instead of 40 (87.5% reduction)
- **Target tasks**: WER (Word Error Rate), PER (Phone Error Rate)

---

## Core Innovation: Shared Pretrained Experts <a name="core-innovation"></a>

### The Problem
Traditional MoE: Each acoustic group needs its own pretrained model → 8 separate checkpoints → no cross-group knowledge transfer.

### The Solution
**5 pretrained experts (M1-M5) shared across all 8 groups**:
- Each expert trained on diverse mixed acoustic data
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

**Key Insight**: Same initialization, different gradients → specialized experts.

---

## Complete Architecture <a name="complete-architecture"></a>

```
INPUT: Audio Waveform
         │
         ▼
┌──────────────────────────────┐
│  Mel Spectrogram              │
│  [B, T, 80]                   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Input Projection             │
│  Linear(80 → model_dim)       │
│  [B, T, model_dim]            │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Gating Network               │
│  MLP: model_dim → 256 → G     │
│  Outputs:                     │
│    • group_probs: [B, T, G]   │
│    • group_ids: [B, T]        │
│  Routes tokens to acoustic   │
│  groups (G=8)                 │
└──────────┬───────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  ENCODER STACK (24 layers)                                 │
├─────────────────────────────────────────────────────────────┤
│                                                            │
│  Layers 0-5: PRE-MOE (Shared)                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  ConformerLayer                                     │  │
│  │    • LayerNorm                                      │  │
│  │    • Conv1D (depthwise, k=31)                       │  │
│  │    • Multi-Head Attention (H=16)                    │  │
│  │    • FeedForward (FFN ×4)                           │  │
│  └─────────────────────────────────────────────────────┘  │
│           (Processes all tokens identically)              │
│                                                            │
│  Layers 6-17: MOE LAYERS (12 layers)                      │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  MoEConformerLayer                                 │  │
│  │    • Conv1D + Attention (shared pre/post MoE)      │  │
│  │    • ┌─────────────────────────────────────────┐   │  │
│  │    • │  Expert Groups (8 groups × 5 experts)  │   │  │
│  │    • │  ┌─────────────┐   ┌─────────────┐    │   │  │
│  │    • │  │ Group 0     │   │ Group 1     │    │   │  │
│  │    • │  │ (vowels)    │   │ (plosives)  │    │   │  │
│  │    • │  │ ┌─────────┐ │   │ ┌─────────┐ │    │   │  │
│  │    • │  │ │exp0: M1│ │   │ │exp0: M1│ │    │   │  │
│  │    • │  │ │exp1: M2│ │   │ │exp1: M2│ │    │   │  │
│  │    • │  │ │exp2: M3│ │   │ │exp2: M3│ │    │   │  │
│  │    • │  │ │exp3: M4│ │   │ │exp3: M4│ │    │   │  │
│  │    • │  │ │exp4: M5│ │   │ │exp4: M5│ │    │   │  │
│  │    • │  │ └─────────┘ │   │ └─────────┘ │    │   │  │
│  │    • │  └─────────────┘   └─────────────┘    │   │  │
│  │    • │         ... all 8 groups ...           │   │  │
│  │    • └─────────────────────────────────────────┘   │  │
│  │    •    Each expert: FFN(model_dim → model_dim×4) │  │
│  │    •    Pooling: Mean or Attention (configurable) │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  Layers 18-23: POST-MOE (Shared)                          │
│  (Same as pre-MOE ConformerLayers)                        │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Encoder Output               │
│  [B, T, model_dim]            │
└──────────┬───────────────────┘
           │
           ├─────────────┬─────────────┐
           ▼             ▼             ▼
    ┌────────────┐ ┌────────────┐ ┌────────────┐
    │  RNN-T     │ │  Phone     │ │  Aux       │
    │  Decoder   │ │  CTC Head  │ │  Outputs   │
    └────────────┘ └────────────┘ └────────────┘
```

---

## Components Deep Dive <a name="components"></a>

### 1. Expert (`model/expert.py`)

**Simple feedforward network**:
```python
class Expert(nn.Module):
    def __init__(self, expert_dim=1024, ff_multiplier=4, dropout=0.1):
        self.net = nn.Sequential(
            nn.Linear(expert_dim, expert_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim * ff_multiplier, expert_dim),
        )
```

**Parameters**: ~4.2M per expert (for model_dim=1024)

### 2. ExpertGroup (`model/expert.py`)

**Container for 5 experts with pooling**:

```python
class ExpertGroup(nn.Module):
    def __init__(self, config, expert_pretrained_paths=None,
                 use_attention_pooling=False, attn_heads=4, attn_dropout=0.1):
        # Create 5 expert instances
        for i in range(5):
            expert = Expert(...)
            if expert_pretrained_paths and expert_pretrained_paths[i]:
                expert.load_state_dict(torch.load(expert_pretrained_paths[i]))
            setattr(self, f"expert_{i}", expert)

        # Optional attention pooling
        if use_attention_pooling:
            self.query = nn.Linear(expert_dim, expert_dim)
            self.key = nn.Linear(expert_dim, expert_dim)
            self.value = nn.Linear(expert_dim, expert_dim)
            self.out_proj = nn.Linear(expert_dim, expert_dim)
```

**Forward pass**:
```python
def forward(self, x):
    # x: [N, model_dim]
    outputs = [expert_i(x) for i in range(5)]  # Each: [N, D]
    stacked = torch.stack(outputs, dim=1)      # [N, 5, D]

    if self.use_attention_pooling:
        # Token-aware weighting
        combined = attention_pooling(x, stacked)
    else:
        # Simple average
        combined = stacked.mean(dim=1)         # [N, D]

    return combined
```

**Key**: All groups receive same pretrained paths, but create **independent instances**. After first gradient update, experts in different groups diverge.

### 3. GatingNetwork (`model/gating.py`)

**Routes tokens to acoustic groups**:
```python
class GatingNetwork(nn.Module):
    def __init__(self, num_groups=8, model_dim=1024, hidden_dim=256):
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_groups),
        )

    def forward(self, x):
        logits = self.net(x)                    # [B, T, G]
        probs = torch.softmax(logits, dim=-1)  # [B, T, G]
        ids = torch.argmax(probs, dim=-1)      # [B, T]
        return probs, ids
```

### 4. ConformerLayer (`model/moe_conformer.py`)

**Standard Conformer block** (used in pre/post MoE layers):
```
x → LayerNorm → Conv1D (depthwise) → +x → LayerNorm → MultiHeadAttn → +x →
LayerNorm → FeedForward → +x
```

### 5. MoEConformerLayer (`model/moe_conformer.py`)

**MoE-augmented Conformer**:
```python
class MoEConformerLayer(nn.Module):
    def forward(self, x, group_ids):
        # Shared branches
        x = conv_branch(x) + x
        x = attn_branch(x) + x

        # MoE routing
        batch_size, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)
        group_ids_flat = group_ids.reshape(-1)

        outputs = []
        masks = []
        for group_idx, expert_group in enumerate(self.expert_groups):
            mask = (group_ids_flat == group_idx)
            if mask.any():
                x_group = x_flat[mask]
                group_out = expert_group(x_group)  # [N_group, dim]
                outputs.append(group_out)
                masks.append(mask)

        # Scatter back
        result = torch.zeros_like(x_flat)
        for mask, out in zip(masks, outputs):
            result[mask] = out
        x = result.reshape(batch_size, seq_len, dim)

        x = moe_dropout(x) + residual
        return x
```

### 6. MoEConformerEncoder (`model/moe_conformer.py`)

**Complete encoder stack**:
```python
class MoEConformerEncoder(nn.Module):
    def __init__(self, config):
        # 8 ExpertGroups (shared across all MoE layers)
        self.expert_groups = nn.ModuleList([
            ExpertGroup(group_cfg, expert_pretrained_paths[i], ...)
            for i, group_cfg in enumerate(config['group_configs'])
        ])

        # Layer stacks
        self.pre_layers = [ConformerLayer for _ in range(moe_start_layer)]
        self.moe_layers = [MoEConformerLayer for _ in range(moe_start_layer, moe_end_layer)]
        self.post_layers = [ConformerLayer for _ in range(moe_end_layer, num_layers)]
```

### 7. TeaMoEModel (`model/tea_moe.py`)

**Full model wrapper**:
```python
class TeaMoEModel(nn.Module):
    def __init__(self, config):
        self.input_proj = nn.Linear(n_mels, model_dim)
        self.gating = GatingNetwork(num_groups, model_dim)
        self.encoder = MoEConformerEncoder(config)
        self.decoder = RNNTDecoder(config)
        self.phone_head = nn.Linear(model_dim, num_phones)
        self.loss_fn = CombinedLoss(...)

    def forward(self, audio, targets, phone_targets, deterministic=True):
        x = self.input_proj(audio)
        group_probs, group_ids = self.gating(x)
        encoder_out = self.encoder(x, group_ids)
        rnnt_logits = self.decoder(encoder_out, targets)
        phone_logits = self.phone_head(encoder_out)
        return rnnt_logits, {
            "group_probs": group_probs,
            "group_ids": group_ids,
            "encoder_out": encoder_out,
            "phone_logits": phone_logits,
        }
```

### 8. RNNTDecoder (`model/rnnt_decoder.py`)

**Transducer decoder** (simplified placeholder):
- Prediction Network: LSTM
- Joint Network: Linear projection
- Implementation: See file for details

### 9. CombinedLoss (`model/losses.py`)

**Multi-task loss**:
```python
class CombinedLoss:
    def total_loss(self, rnnt_logits, targets, group_probs, group_ids, ...):
        losses = {
            'rnnt': self.rnnt_loss(...),           # Main ASR loss
            'load_balance': self.load_balance_loss(...),  # Balance expert usage
            'z_loss': self.z_loss(...),            # Router regularization
            'distillation': distillation_loss,      # From teacher (if any)
            'ctc_phone': self.ctc_phone_loss(...) # Auxiliary phone task
        }
        total = sum(losses.values())
        return total, losses
```

### 10. NaturalNichesCompetition (`model/competition.py`)

**Evolutionary expert selection**:
- Computes fitness scores based on group routing frequency
- Selects parents using fitness-proportional sampling
- Creates child via slerp interpolation
- Replaces worst expert in each group

---

## Data Flow <a name="data-flow"></a>

| Stage | Shape | Description |
|-------|-------|-------------|
| **Input** | `[B, T, 80]` | Raw mel-spectrogram |
| After `input_proj` | `[B, T, D]` | Projected to model_dim |
| **Gating** | | |
| - `group_probs` | `[B, T, G]` | Softmax probabilities |
| - `group_ids` | `[B, T]` | Hard assignment (0-7) |
| **Encoder** | | |
| Pre-MoE output | `[B, T, D]` | Shared encoding |
| MoE input (flattened) | `[B×T, D]` | Reshape for routing |
| Group mask | `[B×T]` bool | Which tokens per group |
| Expert input (group g) | `[N_g, D]` | Tokens routed to group g |
| Expert outputs | `[N_g, 5, D]` | 5 experts in parallel |
| After pooling | `[N_g, D]` | Mean or attention |
| After scatter | `[B×T, D]` | Reconstruct sequence |
| MoE output | `[B, T, D]` | With residual |
| **Post-MoE output** | `[B, T, D]` | Final encoder state |
| **RNNT logits** | `[B, U, V]` | Decoder predictions |
| **Phone logits** | `[B, T, P]` | CTC phone predictions |

Where:
- **B** = batch size
- **T** = input time frames
- **U** = output sequence length
- **D** = model_dim (typically 1024)
- **G** = num_groups (8)
- **V** = vocab_size (5000)
- **P** = num_phones (256)

---

## Configuration <a name="configuration"></a>

### Full Config Structure (`config/default.yaml`)

```yaml
model:
  # Encoder backbone
  num_layers: 24
  moe_start_layer: 6
  moe_end_layer: 18
  model_dim: 1024
  num_heads: 16
  conv_kernel_size: 31
  ff_multiplier: 4

  # MoE configuration
  num_groups: 8
  experts_per_group: 5
  total_experts: 40

  # Shared pretrained experts
  group_expert_pretrained_paths: null  # or List[List[str]]

  # Pooling strategy
  use_attention_pooling: false
  attn_heads: 4
  attn_dropout: 0.1

  # Group configurations (8 groups)
  group_configs:
    - {group_id: 0, group_name: "vowels", ...}
    - {group_id: 1, group_name: "plosives", ...}
    ...

  # Decoder
  vocab_size: 5000
  decoder_hidden: 1024
  decoder_layers: 2
  blank_id: 0
  num_phones: 256

  # Loss weights
  competition_freq_steps: 1000
  distillation_weight: 0.1
  load_balance_weight: 0.01
  z_loss_weight: 0.001
  ctc_phone_weight: 0.3

  # Audio
  sample_rate: 16000
  n_mels: 80
  hop_length: 256
  win_length: 1024

  # Other
  alpha: 0.5
  use_matchmaker: true

data:
  manifests_dir: "datasets/processed_data_librispeech/manifests"
  train_manifest: "train.jsonl"
  valid_manifest: "validation.jsonl"
  max_duration: 30.0
  num_workers: 4

training:
  num_epochs: 100
  batch_size: 16
  learning_rate: 0.001
  warmup_steps: 50000
  total_steps: 500000
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  weight_decay: 0.01
  use_amp: true
  use_checkpoint: true
  eval_every_n_steps: 1000
  output_dir: "checkpoints"
  wandb_project: "TeaMoE"
```

---

## Training Pipeline <a name="training-pipeline"></a>

### Phase 1: Pretrain Experts with HuggingFace Distillation

**Goal**: Create 5 diverse expert checkpoints (M1-M5) by distilling from large pretrained models.

```bash
# Install dependencies
pip install transformers torchaudio librosa speechbrain

# Distill from HuggingFace models
python distill_hf_to_experts.py \
  --output-dir checkpoints/pretrained \
  --model-names \
    "facebook/mms-300m" \
    "speechbrain/asr-conformersmall-transformerlm-librispeech" \
    "facebook/hubert-large-ls960-ft" \
    "facebook/wav2vec2-large-xlsr-53" \
    "openai/whisper-large-v3" \
  --epochs 3 \
  --batch-size 16 \
  --expert-dim 1024
```

**Output**: `checkpoints/pretrained/expert_M1.pt` through `expert_M5.pt`

**What happens**:
1. Load each HuggingFace model
2. Extract encoder features from audio
3. Compute mel spectrogram (target)
4. Train student network (input_proj → Expert → output_proj) to reconstruct mel from HF features
5. Save only the Expert weights in checkpoint

### Phase 2: Train from Scratch (Alternative)

If you don't have HuggingFace models:

```bash
python train_pretrained_experts.py \
  --config config/default.yaml \
  --output-dir checkpoints/pretrained \
  --epochs 5 \
  --batch-size 32 \
  --experts 5
```

This trains 5 experts using autoencoder reconstruction on mel-spectrograms.

### Phase 3: Generate Shared Config

```bash
python generate_shared_config.py \
  --base-config config/default.yaml \
  --pretrained-dir checkpoints/pretrained \
  --output config/shared_pretrained.yaml \
  --num-groups 8 \
  --experts-per-group 5
```

This creates a config where all 8 groups share the same 5 pretrained paths.

### Phase 4: Fine-tune TeaMoE

```bash
python train.py \
  --config config/shared_pretrained.yaml \
  --output-dir checkpoints/finetuned_shared \
  --num-epochs 20 \
  --learning-rate 1e-4
```

**What happens**:
- Loads pretrained M1-M5 into every group's expert_0 through expert_4
- Each expert instance is independent (different gradients)
- Over time, M1 in Group 0 diverges from M1 in Group 1
- Specialization emerges through acoustic group routing

---

## Specialization Dynamics <a name="specialization"></a>

### Tracking Specialization

During evaluation, compute cosine distance between current expert and its pretrained initialization:

```python
def compute_specialization(model, pretrained_weights):
    scores = []
    for group_idx, group in enumerate(model.encoder.expert_groups):
        for expert_idx in range(5):
            current = group.get_expert(expert_idx).state_dict()
            pretrained = pretrained_weights[group_idx][f"expert_{expert_idx}"]

            # Flatten and normalize
            cur_vec = flatten_and_normalize(current)
            pre_vec = flatten_and_normalize(pretrained)

            similarity = torch.dot(cur_vec, pre_vec).item()
            distance = 1.0 - similarity  # Cosine distance
            scores.append(distance)

    return np.mean(scores)  # Specialization Index
```

### Expected Trajectory

| Step | Specialization Index |
|------|---------------------|
| 0 | 0.000 (identical to pretrained) |
| 1,000 | 0.05-0.10 |
| 10,000 | 0.15-0.25 |
| 50,000 | 0.30-0.45 |
| 100,000+ | 0.40-0.60 |

Higher = more specialized (diverged from shared initialization).

### Cross-Group Similarity Matrix

For expert_0 across all groups:

```
           G0     G1     G2     G3     ...
G0        1.00   0.85   0.82   0.80
G1        0.85   1.00   0.84   0.81
G2        0.82   0.84   1.00   0.83
G3        0.80   0.81   0.83   1.00
```

Diagonal = 1.0 (self). Off-diagonal shows divergence between groups.

---

## API Reference <a name="api-reference"></a>

### Core Classes

#### `Expert(config)`
Simple FFN expert.

**Args**:
- `expert_dim`: Hidden dimension
- `ff_multiplier`: Expansion factor (default 4)
- `dropout`: Dropout rate

**Input**: `[N, expert_dim]`
**Output**: `[N, expert_dim]`

---

#### `ExpertGroup(config, expert_pretrained_paths=None, use_attention_pooling=False, ...)`
Container for 5 experts with pooling.

**Args**:
- `config`: Dict with `num_experts`, `expert_dim`, `ff_multiplier`, `dropout`
- `expert_pretrained_paths`: List of 5 checkpoint paths (or None)
- `use_attention_pooling`: If True, use attention; else mean pool
- `attn_heads`: Attention heads (only if pooling=attention)
- `attn_dropout`: Attention dropout

**Input**: `[N, expert_dim]`
**Output**: `[N, expert_dim]`

**Method**: `get_expert(idx)` → returns Expert instance at index

---

#### `GatingNetwork(num_groups=8, model_dim=1024, hidden_dim=256)`
Routes tokens to groups.

**Input**: `[B, T, model_dim]`
**Output**: `(group_probs, group_ids)` where:
- `group_probs`: `[B, T, num_groups]`
- `group_ids`: `[B, T]` (int64)

---

#### `MoEConformerEncoder(config)`
Full MoE encoder stack.

**Args**: `config` dict with all model parameters

**Input**: `x: [B, T, D]`, `group_ids: [B, T]`
**Output**: `[B, T, D]`

**Attributes**:
- `expert_groups`: ModuleList of 8 ExpertGroup

---

#### `TeaMoEModel(config)`
Complete model.

**Input**:
- `audio_features`: `[B, T, 80]`
- `targets`: `[B, U]` (token IDs)
- `phone_targets`: `[B, U]` (phone IDs)
- `deterministic`: bool (routing mode)
- `use_checkpoint`: bool (gradient checkpointing)

**Output**:
- `rnnt_logits`: `[B, U, T, vocab_size]`
- `aux_outputs`: dict with `group_probs`, `group_ids`, `encoder_out`, `phone_logits`

**Method**: `compute_loss(...)` → returns total loss, loss_dict

---

### Utility Functions

#### `compute_expert_cosine_distances(model)`
Compute average cosine distance between all expert pairs.

#### `collect_expert_usage(aux_outputs, num_experts, device)`
Extract usage counts from routing.

#### `greedy_decode_rnnt(logits, blank_id=0)`
Simple greedy decoder for RNN-T.

---

## Troubleshooting <a name="troubleshooting"></a>

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'transformers'`

**Solution**:
```bash
pip install transformers torchaudio
```

For SpeechBrain models:
```bash
pip install speechbrain
```

---

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `model_dim` in config (1024 → 512 or 256)
2. Reduce `batch_size` (16 → 4 or 2)
3. Enable gradient checkpointing: `use_checkpoint: true`
4. Use gradient accumulation: `gradient_accumulation_steps: 4`
5. Enable mixed precision: `use_amp: true`

---

### Experts Not Specializing

**Symptom**: `SpecializationIndex` stays near 0.0 during training.

**Possible causes**:
1. Learning rate too low → increase LR (1e-4 → 5e-4)
2. Load balancing too strong → reduce `load_balance_weight`
3. Routing entropy too low → check `group_probs` distribution
4. Insufficient training steps → train longer

**Debug**:
```python
# Log group distribution
group_probs = aux_outputs['group_probs']
print("Group distribution:", group_probs.mean(dim=[0,1]))
```

---

### Checkpoint Loading Fails

**Error**: `RuntimeError: size mismatch for expert_0.net.0.weight`

**Cause**: Expert architecture mismatch between checkpoint and current config.

**Solution**:
1. Check `expert_dim` matches between pretrained and current config
2. Verify `ff_multiplier` is consistent
3. Regenerate checkpoints with correct config

---

### Distillation Script Issues

**Error**: `AttributeError: 'XXXProcessor' object has no attribute 'feature_extractor'`

**Cause**: Some HuggingFace models use different processor API.

**Solution**: The script handles both standard and SpeechBrain models. For custom models, add wrapper in `HFModelWrapper.extract_features()`.

---

### Windows Unicode Errors

**Symptom**: `UnicodeEncodeError: 'charmap' codec can't encode character`

**Cause**: Windows console uses cp1252 encoding, cannot print emojis.

**Solution**: All scripts now use `[OK]` and `[FAIL]` instead of emojis.

---

## File Structure

```
TeaMoE/
├── config/
│   ├── default.yaml           # Full config (24GB+ VRAM)
│   ├── 8gb.yaml              # Reduced config (8GB VRAM)
│   └── shared_pretrained.yaml # Generated shared config
├── model/
│   ├── __init__.py
│   ├── expert.py              # Expert, ExpertGroup
│   ├── gating.py              # GatingNetwork
│   ├── moe_conformer.py       # ConformerLayer, MoEConformerLayer, MoEConformerEncoder
│   ├── tea_moe.py             # TeaMoEModel
│   ├── rnnt_decoder.py        # RNNTDecoder
│   ├── losses.py              # CombinedLoss
│   └── competition.py         # NaturalNichesCompetition
├── checkpoints/
│   ├── pretrained/           # M1-M5 checkpoints
│   └── finetuned_shared/     # Fine-tuned model
├── datasets/
│   └── processed_data_librispeech/
│       ├── manifests/
│       │   ├── train.jsonl
│       │   ├── validation.jsonl
│       │   └── test.jsonl
│       └── audio/
├── distill_hf_to_experts.py   # NEW: Distill HF models
├── train_pretrained_experts.py # Train M1-M5 from scratch
├── generate_shared_config.py  # Generate shared config
├── train.py                   # Main training script
├── test_all.py                # Comprehensive test suite
└── ARCHITECTURE.md            # This file
```

---

## FAQ

**Q: Can I use different pretrained models per group?**

A: Yes. Edit `group_expert_pretrained_paths` to have different lists per group. Current implementation supports arbitrary combinations.

**Q: How do I freeze pretrained experts?**

A:
```python
for group in model.encoder.expert_groups:
    for i in range(5):
        for p in group.get_expert(i).parameters():
            p.requires_grad = False
# Optimizer will only update non-frozen parameters
```

**Q: What's the difference between mean pooling and attention pooling?**

A:
- **Mean**: Simple average, 0 extra params. All experts equally weighted per token.
- **Attention**: Token-aware weighting. Each token learns to weight experts differently. ~1M extra params per group. Recommended for shared pretrained experts.

**Q: Can I run without pretrained experts?**

A: Yes. Set `group_expert_pretrained_paths: null` or omit the key. Experts will train from random initialization.

**Q: How do I monitor specialization during training?**

A: The `evaluate()` function automatically computes `SpecializationIndex` if `group_expert_pretrained_paths` is set. It logs to WandB and prints during evaluation.

---

## References

- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
- [RNN-T: Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
- [Mixture of Experts (MoE)](https://arxiv.org/abs/1701.06538)
- [Outrageously Large Neural Networks: The Sparse MoE](https://arxiv.org/abs/1701.06538)

---

**Last Updated**: 2026-05-03
**Version**: 1.0
