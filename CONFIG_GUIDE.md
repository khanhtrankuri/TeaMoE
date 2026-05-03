# Configuration Guide for TeaMoE

## Quick Reference

| Config File | Use Case | Model Size | GPU Memory |
|-------------|----------|------------|------------|
| `config/small.yaml` | Development, testing, prototyping | ~17M params | 4-8GB |
| `config/default.yaml` | Full training, production (A100 40GB+) | ~200M params | 32GB+ |
| `config/shared_pretrained.yaml` | Shared pretrained experts + attention | ~200M params | 32GB+ |

## Config Structure

All configs follow this structure:

```yaml
model:      # Model architecture settings
data:       # Dataset and dataloader settings
training:   # Training hyperparameters and logging
```

## Model Options

### Encoder Backbone

| Parameter | Default (full) | Small | Description |
|-----------|----------------|-------|-------------|
| `num_layers` | 24 | 6 | Total Conformer layers |
| `moe_start_layer` | 6 | 2 | First MoE layer (0-indexed) |
| `moe_end_layer` | 18 | 5 | Last MoE layer (exclusive) |
| `model_dim` | 1024 | 256 | Hidden dimension |
| `num_heads` | 16 | 4 | Attention heads |
| `conv_kernel_size` | 31 | 31 | Depthwise conv kernel (must be odd) |
| `ff_multiplier` | 4 | 4 | FFN expansion factor |

**Memory scaling**: Roughly quadratic in `model_dim`, linear in `num_layers`.
- `model_dim=256`: ~17M params
- `model_dim=512`: ~65M params
- `model_dim=1024`: ~200M params

### MoE Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_groups` | 8 | Number of acoustic groups (vowels, plosives, etc.) |
| `experts_per_group` | 5 | Experts per group |
| `total_experts` | 40 | Auto: `num_groups × experts_per_group` |
| `top_k_groups` | 1 | Route to top-k groups during training |
| `top_k_inference` | 2 | Route to top-k groups during eval |

### Shared Pretrained Experts

```yaml
group_expert_pretrained_paths: null  # Train from scratch (default)

# OR with shared pretrained:
group_expert_pretrained_paths:
  - ["M1.pt", "M2.pt", "M3.pt", "M4.pt", "M5.pt"]  # Group 0
  - ["M1.pt", "M2.pt", "M3.pt", "M4.pt", "M5.pt"]  # Group 1
  # ... repeat for all groups
```

**Storage**: With shared pretrained, only 5 checkpoint files needed (M1-M5) instead of 40.

### Expert Pooling

How to combine outputs from multiple experts within a group:

```yaml
# Mean pooling (default): simple average
use_attention_pooling: false

# Attention pooling: token-aware weighting (+1M params/group)
use_attention_pooling: true
attn_heads: 4          # Number of attention heads (default 4)
attn_dropout: 0.1      # Dropout on attention weights
```

**Parameter overhead** (for model_dim=1024, 5 experts):
- Mean: 13M/group
- Attention: 14M/group (+1M)

**Recommendation**: Use attention pooling with shared pretrained experts for best results.

### Decoder (RNNT)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 5000 | Output token vocabulary |
| `decoder_hidden` | 1024 | Prediction network hidden dim |
| `decoder_layers` | 2 | LSTM layers in prediction network |
| `blank_id` | 0 | Blank token for RNNT |

### Loss Weights

```yaml
competition_freq_steps: 1000    # Natural niches competition (0=disabled)
distillation_weight: 0.1        # Teacher-student distillation
load_balance_weight: 0.01      # Balance expert usage (important!)
z_loss_weight: 0.001           # Logits stability
ctc_phone_weight: 0.3          # Auxiliary phone CTC task
alpha: 0.5                     # Matchmaker loss weight
use_matchmaker: true           # Enable matchmaker network
```

**Important**: `load_balance_weight` prevents one expert from dominating. Increase if gating becomes too unbalanced.

## Data Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `manifests_dir` | `datasets/processed_data_librispeech/manifests` | Path to manifest directory |
| `train_manifest` | `train.jsonl` | Training manifest filename |
| `valid_manifest` | `validation.jsonl` | Validation manifest filename |
| `test_manifest` | `test.jsonl` | Test manifest filename |
| `max_duration` | 30.0 | Max audio length in seconds |
| `num_workers` | 4 (full) / 2 (small) | DataLoader workers |

**Manifest format** (JSONL):
```json
{
  "audio_filepath": "/path/to/audio.wav",
  "text": "transcript text",
  "duration_seconds": 5.23
}
```

## Training Options

| Parameter | Default (full) | Small | Description |
|-----------|----------------|-------|-------------|
| `num_epochs` | 100 | 10 | Total epochs |
| `batch_size` | 16 | 8 | Per-GPU batch size |
| `learning_rate` | 0.001 | 0.001 | Peak learning rate |
| `warmup_steps` | 50000 | 1000 | Linear warmup steps |
| `total_steps` | 500000 | 10000 | For cosine scheduler |
| `gradient_accumulation_steps` | 1 | 2 | Accumulate N batches before step |
| `max_grad_norm` | 1.0 | 1.0 | Gradient clipping |
| `use_amp` | true | true | Mixed precision (FP16) |
| `use_checkpoint` | true | true | Gradient checkpointing (saves memory) |
| `use_compile` | false | false | torch.compile (PyTorch 2.0+) |

### Effective Batch Size

```
effective_batch = batch_size × gradient_accumulation_steps × num_gpus
```

Example: `batch_size=8, accum=2, 4 GPUs` → effective batch = 64

### Scheduling

Cosine decay with warmup:
```
LR = lr × min(1, step/warmup) × 0.5 × (1 + cos(π × progress))
```
where `progress = (step - warmup) / (total - warmup)`

## Logging & Checkpoints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_dir` | `checkpoints` | Save directory |
| `eval_every_n_steps` | 1000 | Run validation frequency |
| `log_every_n_steps` | 10 | Log metrics frequency |
| `save_every_n_steps` | 10000 | Save checkpoint frequency |
| `keep_last_n_checkpoints` | 5 | Number of recent checkpoints to keep |
| `resume` | false | Resume from `output_dir/checkpoint.pt` |

### Weights & Biases

```yaml
wandb_project: "TeaMoE"
wandb_entity: null              # Your W&B username/team (required to log)
wandb_tags: []                  # e.g., ["shared-pretrained", "attention"]
wandb_notes: ""                 # Description of this run
```

**Disable WandB**: Set `wandb_entity: null` or `WANDB_MODE=offline` env var.

## Hardware

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | null (auto) | "cuda", "cpu", or null for auto-detect |
| `seed` | 42 | Random seed for reproducibility |

## Creating Custom Configs

### 1. Start from base

```bash
cp config/default.yaml config/my_experiment.yaml
```

### 2. Modify key parameters

```yaml
training:
  batch_size: 32              # Increase if you have more VRAM
  learning_rate: 5e-5         # Lower LR for fine-tuning
  num_epochs: 50

model:
  use_attention_pooling: true  # Enable attention
```

### 3. Run training

```bash
python train.py --config config/my_experiment.yaml --output-dir checkpoints/my_exp
```

## Memory Optimization Tips

1. **Reduce `model_dim`**: 1024 → 512 → 256 (quadratic memory reduction)
2. **Reduce `num_layers`**: 24 → 12 → 6 (linear reduction)
3. **Use gradient checkpointing**: `use_checkpoint: true` (saves ~60% activation memory)
4. **Reduce `batch_size`**: With gradient accumulation to maintain effective batch
5. **Disable attention pooling**: Saves ~8-10M parameters
6. **Use `torch.compile`**: `use_compile: true` (PyTorch 2.0+, can speed up 20-30%)

## Troubleshooting

### CUDA Out of Memory

1. Reduce `batch_size`
2. Set `use_checkpoint: true`
3. Reduce `model_dim` or `num_layers`
4. Enable `use_attention_pooling: false`

### Slow Training

1. Increase `batch_size` (with gradient accumulation if needed)
2. Enable `use_compile: true`
3. Check data loading: increase `num_workers`
4. Disable WandB logging temporarily

### Poor Convergence

1. Adjust learning rate: try `1e-4` to `5e-4`
2. Increase `warmup_steps`: 50k → 100k
3. Check `load_balance_weight`: increase if one expert dominates
4. Verify data quality and manifest format

## Migration Guide

### From Baseline to Shared Pretrained

1. Train M1-M5: `python train_pretrained_experts.py`
2. Generate config: `python generate_shared_config.py --pretrained-dir checkpoints/pretrained`
3. Enable attention: Edit config, set `use_attention_pooling: true`
4. Lower LR: Set `learning_rate: 1e-4` (fine-tuning)
5. Train: `python train.py --config config/shared_pretrained.yaml`

### From Old Config Format

Old configs use `group_pretrained_paths` (list of 8 paths). New format uses `group_expert_pretrained_paths` (list of 8 lists, each with 5 paths). Use `generate_shared_config.py` to migrate.
