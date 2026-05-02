# CÁC THUẬT TOÁN TRONG MoE-CONFORMER ENCODER

## 1. GATING NETWORK (Mạng chọn nhóm Expert)

**Mục đích:** Chọn 1 trong 8 nhóm expert cho mỗi frame audio

**Thuật toán:**
```
Input: x (batch, time, model_dim)
1. hidden = Dense(256)(x)           // Project xuống hidden dimension
2. hidden = ReLU(hidden)            // Activation
3. logits = Dense(8)(hidden)       // 8 groups -> 8 logits
4. group_probs = Softmax(logits)    // Xác suất cho mỗi nhóm
5. group_ids = ArgMax(group_probs)  // Chọn nhóm có xác suất cao nhất
Output: group_probs (batch, time, 8), group_ids (batch, time)
```

**Chi tiết:** Mỗi frame được gán cho đúng 1 nhóm (top-1 selection)

---

## 2. CONFORMER LAYER (Tiêu chuẩn - không MoE)

**Mục đích:** Xử lý sequence với Conv + Self-Attention + FFN

**Cấu trúc (Pre-Norm):**
```
Input: x (batch, time, model_dim)
1. Residual Connection + Conv Module:
   - LayerNorm(x)
   - Conv1D(kernel=31, padding=SAME)
   - GELU activation
   - Dropout(0.1)
   - Add residual: x = x + conv_out

2. Residual Connection + Self-Attention:
   - LayerNorm(x)
   - Multi-Head Attention (16 heads)
   - Dropout(0.1)
   - Add residual: x = x + attn_out

3. Residual Connection + FFN:
   - LayerNorm(x)
   - Dense(model_dim * 4) → GELU → Dropout
   - Dense(model_dim) → Dropout
   - Add residual: x = x + ffn_out
Output: x (batch, time, model_dim)
```

---

## 3. MoE CONFORMER LAYER (Có MoE-FFN)

**Mục đích:** Thay FFN bằng ExpertGroup (5 experts) dựa trên group_ids

**Thuật toán:**
```
Input: x (batch, time, model_dim), group_ids (batch, time)
1. Conv Module (giống Conformer Layer)
2. Self-Attention (giống Conformer Layer)

3. MoE-FFN (thay vì FFN thông thường):
   a. Flatten: x_flat (batch*time, model_dim)
                group_ids_flat (batch*time,)

   b. Với mỗi ExpertGroup (0 đến 7):
      - Tạo mask: frames nào thuộc group này?
      - Nếu không có frame nào → skip
      - Lấy x_group = x_flat[mask] (num_frames_in_group, model_dim)
      - Chạy qua ExpertGroup: group_outputs (num_frames, 5, model_dim)
      - Tính trung bình: group_output = Mean(group_outputs, axis=1)
      - Lưu (mask, group_output)

   c. Ghép lại: result[mask] = group_output cho từng group
   d. Reshape về (batch, time, model_dim)

4. Residual Connection: x = x + result
Output: x (batch, time, model_dim)
```

**Điểm đặc biệt:** Mỗi frame chỉ đi qua 1 nhóm (5 experts), không phải tất cả 40 experts.

---

## 4. MoE-CONFORMER ENCODER (24 Layers tổng thể)

**Kiến trúc phân lớp:**
```
Input: x (batch, time, model_dim), group_ids (batch, time)

┌─────────────────────────────────────────────┐
│  Pre-layers (1-5): 5 Dense Conformer      │
│  - Không dùng MoE                          │
│  - Mỗi layer: Conv → Attn → FFN            │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  MoE-layers (6-18): 13 MoE Conformer      │
│  - Dùng MoE-FFN thay vì FFN                │
│  - Mỗi layer: Conv → Attn → MoE-FFN        │
│  - MoE-FFN: Chọn 1/8 nhóm (5 experts)     │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Post-layers (19-24): 6 Dense Conformer    │
│  - Không dùng MoE                          │
│  - Mỗi layer: Conv → Attn → FFN            │
└─────────────────────────────────────────────┘
Output: encoder_out (batch, time, model_dim)
```

---

## 5. EXPERT GROUP (5 Experts trong 1 nhóm)

**Mục đích:** Chứa 5 experts chuyên biệt, xử lý cùng lúc

**Thuật toán:**
```
Input: x (num_frames, model_dim) - các frame thuộc nhóm này
1. Với mỗi expert i từ 0 đến 4:
   - expert_out = Expert_i(x)
     • Dense(model_dim * 4) → GELU → Dropout
     • Dense(model_dim)
   - Lưu expert_out vào list

2. Stack tất cả outputs
Output: (num_frames, 5, model_dim)
```

**Specialization:** Mỗi nhóm có chuyên môn riêng:
- Group 0: Vowels
- Group 1: Plosives
- Group 2: Fricatives
- Group 3: Nasals
- Group 4: Male speakers
- Group 5: Female speakers
- Group 6: Clean audio
- Group 7: Other audio

---

## 6. NATURAL NICHES COMPETITION (Cạnh tranh giữa Experts)

**Mục đích:** Tiến hóa các experts thông qua chọn lọc tự nhiên

**Thuật toán chọn Parents (Sample Parents):**
```
Input: archive (40, ...), scores (40, num_datapoints), group_id
1. Lấy indices của 5 experts trong nhóm:
   group_indices = [group_id*5, group_id*5+1, ..., group_id*5+4]

2. Tính fitness với Alpha Normalization:
   - group_scores = scores[group_indices]  (5, num_datapoints)
   - z = sum(group_scores, axis=0) ^ alpha   (num_datapoints,)
   - fitness_matrix = group_scores / z        (5, num_datapoints)
   - fitness = sum(fitness_matrix, axis=1)    (5,)

3. Chọn Parent 1:
   - probs = Softmax(fitness)
   - parent_1_idx = Random choice theo phân phối probs

4. Chọn Parent 2 (MATCHMAKER):
   Nếu use_matchmaker = True:
     - parent_1_fitness = fitness_matrix[parent_1_local_idx]
     - match_score = max(0, fitness_matrix - parent_1_fitness)
     - probs2 = Softmax(sum(match_score, axis=1))
     - parent_2_idx = Random choice theo probs2
   Ngược lại:
     - Chọn expert có fitness cao thứ 2

Output: (parent_1_idx, parent_2_idx)
```

**Thuật toán tạo Offspring (Slerp Interpolation):**
```
Input: parent_1_weights, parent_2_weights
1. Tính góc giữa 2 vectors (trên sphere):
   omega = arccos(dot(p1, p2) / (||p1|| * ||p2||))

2. Slerp (Spherical Linear Interpolation) với t = 0.5:
   slerp = [sin((1-t)*omega)*p1 + sin(t*omega)*p2] / sin(omega)

Output: new_expert_weights (trung bình spherical giữa 2 parents)
```

**Thuật toán cập nhật Archive:**
```
Input: archive, scores, new_expert_weights, group_id
1. Tính fitness cho nhóm (giống bước chọn parents)
2. Tìm expert có fitness THẤP NHẤT (worst_idx)
3. Thay thế: archive[worst_idx] = new_expert_weights
Output: archive mới
```

**Chạy cho tất cả 8 nhóm mỗi 1000 bước huấn luyện.**

---

## 7. EXPERT DISTILLATION (Knowledge Distillation nội bộ)

**Mục đích:** Top 1-2 experts (giỏi nhất) dạy Top 3-5 (yếu hơn)

**Thuật toán tính KL Divergence:**
```
Input: teacher_logits (batch*time, expert_dim), student_logits, temperature=1.0
1. teacher_probs = Softmax(teacher_logits / temperature)
2. student_log_probs = LogSoftmax(student_logits / temperature)
3. KL = Σ teacher_probs * (log(teacher_probs) - student_log_probs)
4. loss = Mean(KL) * temperature^2
Output: KL divergence loss
```

**Thuật toán Distillation cho 1 nhóm:**
```
Input: group_outputs (batch*time, 5, expert_dim), flat_group_ids, rng_key
1. Chọn teachers: top_1_output = group_outputs[:, 0, :]
                    top_2_output = group_outputs[:, 1, :]

2. Chọn students: student_outputs = group_outputs[:, 2:, :]  (3 experts)

3. Chỉ áp dụng cho 10% batch ngẫu nhiên:
   - rand_val = Random uniform()
   - Nếu rand_val > 0.1 → return 0.0

4. Với mỗi student i (0 đến 2):
   - loss1 = KL_Divergence(top_1_output, student_i)
   - loss2 = KL_Divergence(top_2_output, student_i)
   - total_loss += (loss1 + loss2) / 2

5. Chia cho 3 students: total_loss = total_loss * weight / 3
Output: Distillation loss cho nhóm này
```

**Tính cho tất cả 8 nhóm và cộng dồn.**

---

## 8. COMBINED LOSS (Tổng hợp tất cả Losses)

**Các thành phần loss:**

### 8.1 RNN-T Loss (Transcription Loss)
```
Input: logits (batch, time, label_len, vocab_size+1), targets
- Placeholder: Cross-entropy (thực tế dùng warp-transducer)
```

### 8.2 Load Balance Loss (Cân bằng tải giữa các nhóm)
```
Input: group_probs (batch, time, 8), group_ids (batch, time)
1. Tính tỷ lệ sử dụng: group_usage = Mean(OneHot(group_ids), axis=[0,1])
2. Mục tiêu: target_usage = 1/8 (đều nhau)
3. loss = Σ(group_usage - target_usage)^2
4. loss = loss * load_balance_weight (0.01)
```

### 8.3 Z-Loss (Chống overconfidence cho Gating)
```
Input: group_logits (batch, time, 8)
1. log_z = log(Σ exp(group_logits) + 1e-8)
2. loss = Mean(log_z^2)
3. loss = loss * z_loss_weight (0.001)
```

### 8.4 Distillation Loss
```
- Lấy từ ExpertDistillation.group_distillation_loss()
- Nhân với distillation_weight (0.1)
```

### 8.5 CTC Phone Loss (Tối ưu PER)
```
Input: phone_logits (batch, time, num_phones), phone_targets
- Placeholder: Cross-entropy (thực tế dùng CTC)
- Nhân với ctc_phone_weight (0.3)
```

**Tổng loss:**
```
total_loss = rnnt_loss + load_balance_loss + z_loss + distillation_loss + ctc_phone_loss
```

---

## 9. RNN-T DECODER (Giải mã với RNN-T)

**Prediction Network (LSTM):**
```
Input: y (batch, seq_len) - target tokens
1. Embedding(y) → (batch, seq_len, hidden_dim)
2. Với mỗi layer 1 đến 2:
   - LSTM(layer_input) → hidden_state
   - Dropout(hidden_state)
3. Output: pred_out (batch, seq_len, 1024)
```

**Joint Network:**
```
Input: encoder_out (batch, time, 1024), pred_out (batch, seq_len, 1024)
1. encoder_proj = Dense(1024)(encoder_out)  → (batch, time, 1024)
2. pred_proj = Dense(1024)(pred_out)         → (batch, seq_len, 1024)
3. Broadcast và cộng:
   - encoder_expanded = encoder_proj[:, :, None, :]  → (batch, time, 1, 1024)
   - pred_expanded = pred_proj[:, None, :, :]        → (batch, 1, seq_len, 1024)
   - joint = encoder_expanded + pred_expanded          → (batch, time, seq_len, 1024)
4. joint = ReLU(joint)
5. output = Dense(vocab_size + 1)(joint)
Output: (batch, time, seq_len, vocab_size+1)
```

---

## TỔNG KẾT LUỒNG DỮ LIỆU

```
Audio Features (batch, time, 80)
        ↓
Gating Network → group_ids (batch, time) [Chọn 1/8 nhóm]
        ↓
MoE-Conformer Encoder (24 layers):
  - Layers 1-5: Dense Conformer (Conv → Attn → FFN)
  - Layers 6-18: MoE Conformer (Conv → Attn → MoE-FFN)
      • Mỗi frame đi qua 1 nhóm (5 experts)
      • Expert outputs được trung bình lại
  - Layers 19-24: Dense Conformer
        ↓
Encoder Output (batch, time, 1024)
        ↓
RNN-T Decoder:
  - Prediction Network (2-layer LSTM)
  - Joint Network (Encoder + Pred → Add → Dense)
        ↓
RNN-T Logits (batch, time, seq_len, vocab+1)
        ↓
Combined Loss (RNN-T + Load Balance + Z-Loss + Distillation + CTC Phone)
```

**Training loop:**
- Mỗi 1000 bước: Chạy Natural Niches Competition để tiến hóa experts
- Mỗi batch: Tính Distillation Loss (10% xác suất) để top experts dạy lower experts
