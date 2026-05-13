# Whisper Features Integration - Hướng dẫn sử dụng

## 📝 Tổng quan

TeaMoE hiện tại hỗ trợ tích hợp **Whisper-base encoder features** để cải thiện chất lượng features input cho model.

**Cách hoạt động:**
- Mel spectrogram (80 bins) + Whisper features (512 dim) được concatenate
- Sau đó chiếu qua `combined_proj`Linear layer để có model_dim
- Whisper encoder có thể được **frozen** (không train) hoặc **finetune**

---

## 🚀 Cách sử dụng

### **1. Cài đặt thêm dependencies**

```bash
pip install transformers>=4.30.0
```

### **2. Cấu hình config**

Thêm các tham số vào config YAML (dưới section `model:`):

```yaml
model:
  # ... các tham số khác ...

  # Whisper Feature Integration
  use_whisper_features: true           # Enable Whisper features
  whisper_model_name: "openai/whisper-base"  # Model từ HuggingFace
  whisper_freeze: true                 # Freeze Whisper encoder (recommended)
  whisper_proj_dropout: 0.1            # Dropout on projection layer (optional)
```

**Các giá trị có thể:**
- `use_whisper_features`: `true` hoặc `false` (default: `false`)
- `whisper_model_name`: bất kỳ Whisper variant nào:
  - `"openai/whisper-base"` (default, 72M params, 512 dim)
  - `"openai/whisper-small"` (244M params, 768 dim) → cần sửa code chút
  - `"openai/whisper-large"` (1.55B params, 1280 dim) → cần sửa code chút
- `whisper_freeze`: `true` (recommended) để giữ Whisper nguyên, `false` để train toàn bộ
- `whisper_proj_dropout`: dropout rate (0.0 - 0.5)

### **3. Chạy training**

```bash
# Với Whisper features
python train.py \
  --config config/simple.yaml \
  --output-dir checkpoints/with_whisper

# Hoặc với complete config
python train.py \
  --config config/complete.yaml \
  --output-dir checkpoints/with_whisper_full
```

**Lưu ý:** Trong file config, bạn cần set `use_whisper_features: true` ở phần model.

---

## 🧪 Test integration

Chạy test nhanh để xác nhận everything hoạt động:

```bash
# Test với Whisper enabled
python example_whisper_features.py --use-whisper --test-training

# Test backward compatibility (không dùng Whisper)
python example_whisper_features.py --test-training
```

**Expected output:**
```
[OK] Forward pass successful!
[OK] Training step successful!
[OK] All tests passed! Ready for training.
```

---

## 📊 Thông số kỹ thuật

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| **Mel spectrogram** | 80 bins | Acoustic features từ audio |
| **Whisper output dim** | 512 | Base variant encoder dimension |
| **Combined input dim** | 592 | 80 + 512 |
| **Whisper params** | ~72M | Khi frozen, không tính vào trainable params |
| **Additional params** | ~1.5M | Combined projection layer (592 → D) |
| **VRAM overhead** | ~500MB | Chỉ weights, không tính activations |
| **Speed impact** | +20-30% | Thêm Whisper forward pass mỗi step |

---

## ⚙️ Kiến trúc chi tiết

```
Input: audio → mel spectrogram [B, T, 80]
         │
         ├──────────────┐
         │              │
         ▼              ▼
    Mel direct      Whisper Encoder
    [B, T, 80]      [B, 80, T] → [B, T, 512]
         │              │
         └──────┬───────┘
                ▼
         Concatenate: torch.cat([mel, whisper], dim=-1)
                ↓
            [B, T, 592]
                ↓
         combined_proj: Linear(592 → D)
                ↓
            [B, T, D]
                ↓
         Gating → Encoder → Decoder (như cũ)
```

**Xử lý length constraint của Whisper:**
- Whisper yêu cầu mel input có độ dài chia hết cho 3000 (do pretraining)
- Code tự động:
  - Pad nếu `T < 3000` (thêm zeros)
  - Crop nếu `T` không phải bội của 3000 (lấy phần đầu)
  - Sau khi extract features, loại bỏ padding để về đúng `T` ban đầu

---

## 🎯 Best Practices

### **Khuyến nghị cho首次实验:**

1. **Bắt đầu với frozen Whisper:**
   ```yaml
   use_whisper_features: true
   whisper_freeze: true
   ```
   - Nhanh, ít memory
   - Chỉ train projection layer + TeaMoE
   - Dễ debug

2. **Nếu thấy improvement, thử finetune Whisper:**
   ```yaml
   whisper_freeze: false
   ```
   - Thêm `--lr 1e-5` hoặc separate LR cho Whisper
   - Cần gradient checkpointing để tiết kiệm memory

3. **Tuning learning rate:**
   ```python
   # Trong train.py, có thể set param groups:
   params = [
       {'params': model.whisper.parameters(), 'lr': 1e-5},  # nếu finetune
       {'params': model.combined_proj.parameters(), 'lr': 1e-3},
       {'params': model.encoder.parameters(), 'lr': 1e-3},
       {'params': model.decoder.parameters(), 'lr': 1e-3},
   ]
   ```

---

## 🔄 Chuyển đổi giữa Mel-only và Whisper mode

**Không cần sửa code**, chỉ cần thay đổi config:

```yaml
# Mel-only (cũ)
model:
  use_whisper_features: false

# Whisper-enhanced (mới)
model:
  use_whisper_features: true
  whisper_freeze: true  # hoặc false
```

Model tự động chọn input projection phù hợp.

---

## 🐛 Troubleshooting

### **Lỗi: `IndexError: index out of range`**
- **Nguyên nhân:** `targets` chứa token ID >= `vocab_size`
- **Fix:** Kiểm tra config `vocab_size` và đảm bảo targets trong range `[0, vocab_size-1]`

### **Lỗi: `CUDA out of memory`**
- **Nguyên nhân:** Whisper + TeaMoE quá nặng
- **Fix:**
  1. Reduce `batch_size`
  2. Frozen Whisper (`whisper_freeze: true`)
  3. Reduce `model_dim`
  4. Dùng config `simple.yaml` thay `complete.yaml`

### **Lỗi: `Whisper expects the mel input features to be of length X`**
- **Nguyên nhân:** Audio length không chia hết cho 3000
- **Fix:** Code đã tự động pad/crop. Nếu vẫn lỗi, kiểm tra shape truyền vào model.

### **Lỗi: ModuleNotFoundError: No module named 'transformers'**
- **Fix:** `pip install transformers>=4.30.0`

---

## 📈 Kiểm tra Specialization

Khi dùng Whisper features, bạn có thể theo dõi các metric:

- **WER** (Word Error Rate): Có giảm không?
- **Expert diversity** (cosine distance): Experts có specialization tốt hơn?
- **Group distribution:** Gating network có route đúng acoustic groups không?

So sánh với baseline (Mel-only) để đánh giá impact.

---

## 🎓 Cải tiến tiềm năng (Future Work)

1. **Different Whisper variants:**
   - Whisper-small (768 dim) cho balance performance/quality
   - Whisper-large-v3 cho SOTA

2. **Learnable fusion:**
   - Thay vì concat fixed, dùng attention để học weighting
   - Gating có thể chọn dynamically giữa mel và whisper features

3. **Multi-scale Whisper:**
   - Extract từ intermediate layers (không chỉ last_hidden_state)
   - Mỗi layer chứa thông tin khác nhau (low-level vs high-level)

4. **Distillation from Whisper decoder:**
   - Dùng Whisper decoder outputs làm teacher signal
   - Cải thiện RNN-T decoder quality

---

## 📁 Files đã thay đổi

| File | Thay đổi |
|------|----------|
| `model/tea_moe.py` | Thêm Whisper encoder + combined_proj logic |
| `config/complete.yaml` | Thêm `use_whisper_features`, `whisper_model_name`, `whisper_freeze`, `whisper_proj_dropout` |
| `config/simple.yaml` | Thêm các tham số Whisper tương tự |
| `config/shared_pretrained_template.yaml` | Thêm các tham số Whisper |
| `requirements.txt` | Đã có `transformers`, không cần sửa |
| `example_whisper_features.py` | **MỚI** - Test script và ví dụ |

---

## 💡 FAQ

**Q: Có dùng được Whisper-large không?**
A: Có, nhưng cần sửa `whisper_dim` từ 512 → 1280 và update `combined_proj` input dim. Code hiện tại hard-coded cho whisper-base.

**Q: Whisper có được train trên tiếng Việt không?**
A: Whisper được train trên 99 ngôn ngữ, bao gồm nhiều ngôn ngữ. Tuy nhiên, quality trên tiếng Việt có thể không tốt như tiếng Anh do data imbalance.

**Q: Có cần preprocess audio khác không?**
A: Không. Input mel spectrogram format yêu cầu giống Whisper training: 80 mel bins, 16kHz sample rate, hop_length=256, win_length=1024. Các tham số này đã phù hợp.

**Q: Inference speed bị ảnh hưởng thế nào?**
A: Khoảng +20-30% do thêm Whisper forward pass. Có thể tối ưu bằng:
- Cache Whisper features nếu audio cố định
- Distill Whisper features vào smaller model

**Q: Làm sao biết Whisper có đang đóng góp không?**
A: So sánh WER/PER với và không có Whisper. Có thể visualize:
- Group distribution (gating network outputs)
- Expert usage patterns

---

**Last Updated:** 2026-05-13
**Version:** 1.0
