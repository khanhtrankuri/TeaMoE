# Hướng dẫn Tối ưu Thời gian Training TeaMoE

## Tổng quan

Đây là hướng dẫn để tối ưu thời gian training cho TeaMoE model bằng cách sử dụng **pre-computed Whisper features**.

## Vấn đề

Khi training với Whisper features, model tính toán features **mỗi lần** forward pass:
- Mỗi audio file: ~8-10 giây để extract Whisper features
- Với 100,000 audio files: ~278 giờ chỉ để extract features!
- Features bị tính lại **lặp đi lặp lại** mỗi epoch

## Giải pháp

**Pre-compute Whisper features 1 lần duy nhất** trước khi training:
- Extract features 1 lần: ~278 giờ (chạy 1 lần)
- Training: Load features từ disk (~2-3 giây/audio)
- **Tổng thời gian giảm ~3-4x**

## Các bước thực hiện

### Bước 1: Pre-compute Whisper Features

```bash
# Pre-compute cho training set
python precompute_whisper_features.py \
    --manifest datasets/processed_data_librispeech/manifests/train.jsonl \
    --output-manifest datasets/processed_data_librispeech/manifests/train_with_whisper.jsonl \
    --output-dir datasets/whisper_features \
    --whisper-model openai/whisper-base \
    --device cuda

# Pre-compute cho validation set
python precompute_whisper_features.py \
    --manifest datasets/processed_data_librispeech/manifests/validation.jsonl \
    --output-manifest datasets/processed_data_librispeech/manifests/validation_with_whisper.jsonl \
    --output-dir datasets/whisper_features \
    --whisper-model openai/whisper-base \
    --device cuda
```

**Kết quả:**
- Files `.npy` chứa Whisper features được lưu trong `datasets/whisper_features/`
- Manifest mới (`.jsonl`) có thêm field `whisper_feature_path`

### Bước 2: Training với Pre-computed Features

```bash
python train.py \
    --config config/fast_training.yaml \
    --train-manifest datasets/processed_data_librispeech/manifests/train_with_whisper.jsonl \
    --valid-manifest datasets/processed_data_librispeech/manifests/validation_with_whisper.jsonl
```

## Cấu hình

### config/fast_training.yaml

```yaml
model:
  # BẬT pre-computed Whisper features
  use_precomputed_whisper: true
  use_whisper_features: false  # Tắt on-the-fly computation

  # Giảm model size để training nhanh hơn
  model_dim: 512              # 1024 → 512
  num_layers: 12              # 24 → 12
  num_groups: 4               # 8 → 4

training:
  use_compile: true           # Bật torch.compile (+10-20%)
  use_checkpoint: false       # Tắt để dùng torch.compile
  batch_size: 64              # Tăng batch size
  eval_every_n_steps: 2000    # Giảm tần suất eval
```

## Các optimizations khác

### 1. Gradient Checkpointing vs torch.compile

```yaml
# Option A: Tiết kiệm VRAM
use_checkpoint: true
use_compile: false

# Option B: Tốc độ cao (recommended khi có đủ VRAM)
use_checkpoint: false
use_compile: true
```

### 2. Giảm Evaluation Frequency

```yaml
eval_every_n_steps: 2000      # Thay vì 500
save_every_n_steps: 10000     # Thay vì 5000
```

### 3. Tắt Các Metrics Không Cần Thiết

```yaml
logging:
  compute_per: false          # Tắt PER
  compute_gini: false         # Tắt Gini
  compute_expert_diversity: false  # Tắt cosine distance
```

## Tốc độ tăng dự kiến

| Optimization | Speedup |
|--------------|---------|
| Pre-computed Whisper | ~2-3x |
| torch.compile | +10-20% |
| Reduced model size | ~50% |
| Larger batch size | Fewer iterations |
| Reduced eval frequency | ~20% |
| **TOTAL** | **~3-4x** |

## Lưu ý

1. **Pre-compute 1 lần duy nhất**: Features được lưu vĩnh viễn
2. **Manifest mới**: Phải dùng manifest có `whisper_feature_path`
3. **Disk space**: Dự kiến ~50GB cho LibriSpeech (80 dim × 1500 frames × 4 bytes × 100k files)

## Troubleshooting

### Vấn đề: "Whisper features not found"

**Giải pháp:** Kiểm tra đường dẫn trong manifest:
```json
{
  "audio_filepath": "...",
  "whisper_feature_path": "datasets/whisper_features/utterance_id.npy"
}
```

### Vấn đề: "CUDA out of memory"

**Giải pháp:** Giảm batch size hoặc model_dim:
```yaml
training:
  batch_size: 32  # Giảm từ 64

model:
  model_dim: 512  # Giảm từ 1024
```

## File Structure

```
datasets/
├── whisper_features/           # Pre-computed features
│   ├── utterance_001.npy
│   ├── utterance_002.npy
│   └── ...
└── processed_data_librispeech/
    └── manifests/
        ├── train.jsonl              # Original (không có whisper)
        ├── train_with_whisper.jsonl # New (có whisper_feature_path)
        ├── validation.jsonl
        └── validation_with_whisper.jsonl
```