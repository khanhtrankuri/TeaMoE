# TeaMoE: MoE-Conformer + RNN-T with Natural Niches Competition

## Gioi thieu

TeaMoE la mot mo hinh Automatic Speech Recognition (ASR) ket hop kien truc **MoE-Conformer** va **RNN-T (Recurrent Neural Network Transducer)**, tich hop co che **Natural Niches Competition** de huan luyen cac expert chuyen biet.

### Dac diem chinh

- **MoE-Conformer Encoder**: 24 lop Conformer, trong do cac lop 6-18 su dung Mixture of Experts (MoE) voi 8 nhom expert, moi nhom co 5 expert chuyen biet
- **RNN-T Decoder**: Su dung Prediction Network (LSTM) va Joint Network de decode truc tiep tu audio sang van ban
- **Natural Niches Competition**: Co che canh tranh tien hoa giua cac expert trong cung nhom, su dung thuat toan selection dua tren fitness score
- **Expert Distillation**: Chuyen giao kien thuc tu top experts (top 1, 2) sang cac experts yeu hon (3, 4, 5) trong cung nhom
- **Multi-task Learning**: Ket hop RNN-T loss, Load Balance loss, Z-loss, Distillation loss va CTC Phone loss de toi uu ca WER va PER
- **Specialized Experts**: 8 nhom expert chuyen biet cho cac loai am thanh khac nhau (vowels, plosives, fricatives, nasals, male/female speakers, clean/other audio)

### Metrics

- **WER** (Word Error Rate): Danh gia do chinh xac nhan dang tu
- **PER** (Phone Error Rate): Danh gia do chinh xac nhan dang am vi
- **Gini Coefficient**: Do luong do can bang tai giua cac experts
- **Cosine Distance**: Do luong su da dang giua cac experts

## Cai dat moi truong

### Yeu cau he thong

- Python 3.8+
- PyTorch 2.0+ (voi CUDA support de train tren GPU)
- torchaudio
- librosa
- PyYAML
- NumPy
- tqdm
- wandb (tuy chon, de theo doi train)

### Cac buoc cai dat

```bash
# Tao virtual environment (khuyen dung)
python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Cai dat cac thu vien can thiet
pip install torch torchaudio librosa pyyaml numpy tqdm wandb

# Clone repository (neu chua co)
git clone <repository-url>
cd TeaMoE
```

### Cau truc thu muc

```
TeaMoE/
├── config/
│   ├── default.yaml        # Cau hinh mac dinh (24GB+ VRAM)
│   └── 8gb.yaml           # Cau hinh cho 8GB VRAM
├── model/
│   ├── __init__.py
│   ├── tea_moe.py          # Kien truc tong the
│   ├── moe_conformer.py    # MoE-Conformer Encoder
│   ├── rnnt_decoder.py     # RNN-T Decoder
│   ├── gating.py           # Gating Network
│   ├── expert.py           # Expert va ExpertGroup
│   ├── competition.py      # Natural Niches Competition
│   ├── distillation.py     # Expert Distillation
│   └── losses.py          # Combined Loss
├── datasets/
│   └── processed_data_librispeech/  # Du lieu da xu ly
│       ├── manifests/       # File manifest (train.jsonl, validation.jsonl, test.jsonl)
│       └── audio/          # File audio WAV
├── load_dataset/
│   ├── process_libri.py    # Script tai va xu ly LibriSpeech
│   └── text_utils.py
├── train.py                # Script huan luyen chinh
└── README.md
```

## Huong dan huan luyen

### 1. Chuan bi du lieu

Tai va tien xu ly du lieu LibriSpeech:

```bash
# Tai du lieu va chuyen doi FLAC -> WAV, tao manifest
python load_dataset/process_libri.py

# Du lieu se duoc luu tai:
#   datasets/processed_data_librispeech/manifests/  (train.jsonl, validation.jsonl, test.jsonl)
#   datasets/processed_data_librispeech/audio/       (file .wav)
```

File manifest dinh dang JSONL, moi dong la mot JSON object chua:
- `audio_filepath`: Duong dan den file WAV
- `text`: Ban ghi van ban tuong ung
- `speaker_id`: ID nguoi noi
- `gender`: Gioi tinh (M/F)
- `duration_seconds`: Do dai audio

### 2. Cau hinh mo hinh

Tuy chinh cau hinh thong qua file YAML trong thu muc `config/`:

**`config/default.yaml`** — Cho GPU 24GB+ VRAM:
```yaml
model:
  num_layers: 24
  model_dim: 1024
  num_heads: 16
  num_groups: 8
  experts_per_group: 5
  total_experts: 40
  # ...

data:
  manifests_dir: "datasets/processed_data_librispeech/manifests"
  train_manifest: "train.jsonl"
  valid_manifest: "validation.jsonl"
  test_manifest: "test.jsonl"

training:
  num_epochs: 100
  batch_size: 16
  learning_rate: 0.001
```

**`config/8gb.yaml`** — Cho GPU 8GB VRAM (tham so nho hon):
```yaml
model:
  num_layers: 12
  model_dim: 512
  num_heads: 8
  num_groups: 4
  experts_per_group: 4
  total_experts: 16
  # ...

data:
  manifests_dir: "datasets/processed_data_librispeech/manifests"

training:
  batch_size: 4
```

### 3. Chay huan luyen

```bash
# Huan luyen voi cau hinh mac dinh (24GB+ VRAM)
python train.py --config config/default.yaml

# Huan luyen voi cau hinh 8GB VRAM
python train.py --config config/8gb.yaml --batch-size 4

# Chi dinh cac tham so tuy chinh (ghi de config)
python train.py \
    --config config/default_optimized.yaml \
    --num-epochs 50 \
    --batch-size 4 \
    --learning-rate 1e-3 \
    --output-dir checkpoints

# Tiep tuc huan luyen tu checkpoint
python train.py --config config/default.yaml --resume --output-dir checkpoints
```

### 4. Theo doi qua trinh huan luyen

- Loss trung binh moi epoch hien thi qua tqdm progress bar
- WER duoc danh gia tren validation set moi epoch
- Neu dung wandb, metrics duoc log tu dong
- Checkpoint duoc luu sau moi 10 epoch tai thu muc output

### 5. Giai thich tham so MoE quan trong

| Tham so | Y nghia |
|---------|--------|
| `num_groups` | So nhom expert (moi nhom chuyen biet mot loai du lieu) |
| `experts_per_group` | So expert trong moi nhom |
| `total_experts` | Tong so expert = num_groups x experts_per_group |
| `top_k_groups` | So nhom duoc kich hoat moi lan suy luan (thuong la 1) |
| `top_k_inference` | So expert duoc chon trong nhom da kich hoat (thuong la 2) |

Luong du lieu: `Audio -> Gating Network -> Chon top_k_groups nhom -> Trong nhom chon top_k_inference expert -> Ket hop dau ra`

### 6. Cac buoc tiep theo (TODO)

- [x] Tich hop load du lieu LibriSpeech thuc te (da hoan thanh)
- [x] Chuyen tu JAX sang PyTorch (da hoan thanh)
- [x] Su dung YAML config thay vi JSON/dataclass (da hoan thanh)
- [ ] Tich hop RNN-T loss thuc te (hien dang la placeholder)
- [ ] Hoan thien co che Competition de update trong so expert thuc su
- [ ] Them evaluation tren test set voi WER/PER day du
- [ ] Toi uu toc do huan luyen (multi-GPU, mixed precision)

## Tham khao

- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
- [RNN-T: Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
- [Mixture of Experts (MoE)](https://arxiv.org/abs/1701.06538)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [torchaudio Documentation](https://pytorch.org/audio/)
