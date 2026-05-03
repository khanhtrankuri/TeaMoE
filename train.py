"""
Train script for TeaMoE: MoE-Conformer + RNN-T with Natural Niches Competition
Metrics: WER, PER, Gini coefficient, Cosine distance between experts
"""
import os
import math
import json
import argparse
import contextlib
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import yaml
import librosa

from model.tea_moe import TeaMoEModel
from model.competition import NaturalNichesCompetition
from model.losses import CombinedLoss

# Giảm phân mảnh bộ nhớ CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BLANK_ID = 0


# ==================== Dataset ====================

class LibriSpeechDataset(Dataset):
    def __init__(self, manifest_path: str, config: dict, is_train: bool = True):
        self.config = config
        self.is_train = is_train
        self.sample_rate = config.get("sample_rate", 16000)
        self.n_mels = config.get("n_mels", 80)
        self.hop_length = config.get("hop_length", 256)
        self.win_length = config.get("win_length", 1024)
        self.max_duration = config.get("max_duration", 30.0)

        self.records = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("duration_seconds", 0) > self.max_duration:
                    continue
                # Sửa đường dẫn audio nếu cần
                if "audio_filepath" in record:
                    record["audio_filepath"] = record["audio_filepath"].replace(
                        "load_dataset", "datasets"
                    )
                self.records.append(record)

        print(f"Loaded {len(self.records)} records from {manifest_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        audio_path = record["audio_filepath"]
        text = record["text"]

        # Tải audio
        try:
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"[WARN] Error loading {audio_path}: {e}")
            waveform = np.zeros(self.sample_rate, dtype=np.float32)

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel = torch.from_numpy(mel).T.float()  # (T, n_mels)

        # Tokenize ký tự
        tokens = [ord(c) for c in text if ord(c) < 5000]
        tokens = tokens if tokens else [1]
        tokens = torch.tensor(tokens, dtype=torch.long)

        return {
            "audio_features": mel,
            "targets": tokens,
            "audio_path": audio_path,
        }


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, ...]:
    """Ghép batch: padding audio và targets."""
    max_time = max(item["audio_features"].shape[0] for item in batch)
    n_mels = batch[0]["audio_features"].shape[1]

    padded_features, feature_lengths = [], []
    for item in batch:
        feat = item["audio_features"]
        t = feat.shape[0]
        feature_lengths.append(t)
        if t < max_time:
            feat = torch.cat([feat, torch.zeros(max_time - t, n_mels)], dim=0)
        padded_features.append(feat)

    audio_features = torch.stack(padded_features)  # (B, T, n_mels)

    max_target = max(item["targets"].shape[0] for item in batch)
    padded_targets, target_lengths = [], []
    for item in batch:
        tgt = item["targets"]
        tl = tgt.shape[0]
        target_lengths.append(tl)
        if tl < max_target:
            pad = torch.full((max_target - tl,), BLANK_ID, dtype=torch.long)
            tgt = torch.cat([tgt, pad])
        padded_targets.append(tgt)

    targets = torch.stack(padded_targets)  # (B, L)
    input_lengths = torch.tensor(feature_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    # Phone targets: dùng chung targets (placeholder)
    phone_targets = targets.clone()
    phone_lengths = target_lengths.clone()

    return (
        audio_features,
        targets,
        phone_targets,
        input_lengths,
        target_lengths,
        phone_lengths,
    )


# ==================== Metrics ====================

def compute_edit_distance(pred: List[int], target: List[int]) -> int:
    """Levenshtein distance dùng cho WER/PER."""
    m, n = len(pred), len(target)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if pred[i - 1] == target[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def compute_wer(predictions: List[List[int]], targets: List[List[int]]) -> float:
    """Word Error Rate (%). Trả về giá trị 0–100."""
    total_errors = sum(
        compute_edit_distance(p, t) for p, t in zip(predictions, targets)
    )
    total_words = sum(len(t) for t in targets)
    return (total_errors / total_words * 100.0) if total_words > 0 else 0.0


def compute_per(
    phone_preds: List[List[int]], phone_targets: List[List[int]]
) -> float:
    """Phone Error Rate (%). Dùng cùng edit distance với WER."""
    return compute_wer(phone_preds, phone_targets)


def compute_gini_coefficient(expert_usage_counts: np.ndarray) -> float:
    """Gini coefficient để đo độ cân bằng tải giữa các expert.
    
    = 0 → cân bằng hoàn hảo, = 1 → tập trung hoàn toàn.
    """
    counts = np.sort(np.abs(expert_usage_counts))
    n = len(counts)
    total = np.sum(counts)
    if n == 0 or total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.dot(index, counts)) / (n * total) - (n + 1) / n)


def compute_expert_cosine_distances(model: nn.Module) -> float:
    """Tính khoảng cách cosine trung bình giữa các cặp expert.
    
    Dùng để đo tính đa dạng của expert (càng cao càng chuyên biệt).
    """
    expert_vectors = []
    try:
        for group in model.encoder.expert_groups:
            num_experts = group.config["num_experts"]
            for i in range(num_experts):
                expert = group.get_expert(i)
                # Nối toàn bộ tham số thành 1 vector
                params = torch.cat(
                    [p.detach().cpu().flatten() for p in expert.parameters()]
                )
                expert_vectors.append(F.normalize(params, dim=0))
    except AttributeError:
        return 0.0

    if len(expert_vectors) < 2:
        return 0.0

    vecs = torch.stack(expert_vectors)  # (N, D)
    # Ma trận cosine similarity
    sim_matrix = vecs @ vecs.T  # (N, N)
    n = sim_matrix.shape[0]
    # Lấy phần tam giác trên (không tính đường chéo)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    avg_similarity = sim_matrix[mask].mean().item()
    return 1.0 - avg_similarity  # cosine distance


# ==================== Decoding ====================

def greedy_decode_rnnt(logits: torch.Tensor, blank_id: int = BLANK_ID) -> List[int]:
    """Greedy decoding đơn giản cho RNN-T output.
    
    Args:
        logits: Tensor shape (T, U, V) hoặc (T, V)
        blank_id: ID của blank token
    
    Returns:
        Danh sách token IDs đã decode (bỏ blank và lặp liên tiếp)
    """
    if logits.dim() == 3:
        # (T, U, V): lấy token có prob cao nhất theo chiều U trước
        # Đơn giản: dùng bước cuối của U
        logits = logits[:, -1, :]  # (T, V)

    decoded, prev = [], -1
    for t in range(logits.shape[0]):
        token = int(torch.argmax(logits[t]))
        if token != blank_id and token != prev:
            decoded.append(token)
        prev = token if token != blank_id else prev
    return decoded


# ==================== Expert Usage Tracking ====================

def collect_expert_usage(aux_outputs: Dict, num_experts: int, device) -> np.ndarray:
    """Trích xuất thống kê sử dụng expert từ aux_outputs."""
    usage = np.zeros(num_experts)
    group_ids = aux_outputs.get("group_ids")
    if group_ids is not None:
        ids = group_ids.detach().cpu().numpy().flatten()
        for gid in ids:
            if 0 <= gid < num_experts:
                usage[int(gid)] += 1
    return usage


# ==================== Training Step ====================

def train_step(
    model: nn.Module,
    batch: Tuple,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_checkpoint: bool = False,
    gradient_accumulation_steps: int = 1,
) -> Tuple[torch.Tensor, Dict, Dict]:
    """Một bước forward + loss. Không bao gồm optimizer.step().

    Returns:
        total_loss: loss đã chia cho gradient_accumulation_steps (để backward)
        loss_dict: dict các thành phần loss (giá trị gốc, để log)
        aux_outputs: dict từ model (group_probs, group_ids, ...)
    """
    (
        audio_features,
        targets,
        phone_targets,
        input_lengths,
        target_lengths,
        phone_lengths,
    ) = batch

    audio_features = audio_features.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    phone_targets = phone_targets.to(device, non_blocking=True)
    input_lengths = input_lengths.to(device, non_blocking=True)
    target_lengths = target_lengths.to(device, non_blocking=True)
    phone_lengths = phone_lengths.to(device, non_blocking=True)

    amp_ctx = (
        torch.amp.autocast("cuda")
        if scaler is not None
        else contextlib.nullcontext()
    )

    with amp_ctx:
        rnnt_logits, aux_outputs = model(
            audio_features,
            targets,
            phone_targets,
            deterministic=False,
            use_checkpoint=use_checkpoint,
        )

        group_probs = aux_outputs.get("group_probs")
        group_ids = aux_outputs.get("group_ids")
        phone_logits = aux_outputs.get("phone_logits")
        group_logits = (
            torch.log(group_probs + 1e-8) if group_probs is not None else None
        )
        distillation_loss = torch.tensor(0.0, device=device)

        total_loss, loss_dict = model.compute_loss(
            rnnt_logits=rnnt_logits,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            group_probs=group_probs,
            group_ids=group_ids,
            group_logits=group_logits,
            distillation_loss=distillation_loss,
            phone_logits=phone_logits,
            phone_targets=phone_targets,
            phone_lengths=phone_lengths,
        )

    # Chia loss cho gradient accumulation (backward theo loss đã chia)
    scaled_loss = total_loss / gradient_accumulation_steps

    if scaler is not None:
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    return total_loss, loss_dict, aux_outputs


# ==================== Evaluation ====================

def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    use_checkpoint: bool = False,
    pretrained_checkpoint_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Đánh giá mô hình trên tập validation.

    Returns:
        dict gồm: WER, PER, Gini, ExpertCosineDistance, SpecializationIndex (if pretrained provided)
    """
    all_preds: List[List[int]] = []
    all_targets: List[List[int]] = []
    all_phone_preds: List[List[int]] = []
    all_phone_targets: List[List[int]] = []
    expert_usage_total: Optional[np.ndarray] = None

    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            (
                audio_features,
                targets,
                phone_targets,
                input_lengths,
                target_lengths,
                phone_lengths,
            ) = batch

            audio_features = audio_features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            phone_targets = phone_targets.to(device, non_blocking=True)

            rnnt_logits, aux_outputs = model(
                audio_features,
                targets,
                phone_targets,
                deterministic=True,
                use_checkpoint=use_checkpoint,
            )

            phone_logits = aux_outputs.get("phone_logits")

            B = audio_features.shape[0]
            for i in range(B):
                pred = greedy_decode_rnnt(rnnt_logits[i], BLANK_ID)
                all_preds.append(pred)
                all_targets.append(targets[i].cpu().tolist())

                if phone_logits is not None:
                    phone_pred = greedy_decode_rnnt(phone_logits[i], BLANK_ID)
                    all_phone_preds.append(phone_pred)
                else:
                    all_phone_preds.append(pred)
                all_phone_targets.append(phone_targets[i].cpu().tolist())

            try:
                num_experts = (
                    model.config["num_groups"] * model.config["experts_per_group"]
                )
            except (AttributeError, KeyError):
                num_experts = 8

            usage = collect_expert_usage(aux_outputs, num_experts, device)
            if expert_usage_total is None:
                expert_usage_total = usage
            else:
                expert_usage_total += usage

    model.train()

    wer = compute_wer(all_preds, all_targets)
    per = compute_per(all_phone_preds, all_phone_targets)
    gini = (
        compute_gini_coefficient(expert_usage_total)
        if expert_usage_total is not None
        else 0.0
    )
    cosine_dist = compute_expert_cosine_distances(model)

    metrics = {
        "WER": wer,
        "PER": per,
        "Gini": gini,
        "ExpertCosineDistance": cosine_dist,
    }

    # Compute specialization if pretrained weights available
    if pretrained_checkpoint_dir is not None:
        try:
            from diagnostics import compute_group_specialization_scores
            pretrained_dir = Path(pretrained_checkpoint_dir)
            num_groups = model.config.get("num_groups", 8)
            experts_per_group = model.config.get("experts_per_group", 5)

            pretrained_weights = []
            for g in range(num_groups):
                group_weights = {}
                for e in range(experts_per_group):
                    ckpt_path = pretrained_dir / f"expert_M{e+1}.pt"
                    if ckpt_path.exists():
                        pretrained_ckpt = torch.load(ckpt_path, map_location="cpu")
                        group_weights[f"expert_{e}"] = pretrained_ckpt["expert_state_dict"]
                pretrained_weights.append(group_weights)

            scores = compute_group_specialization_scores(model, pretrained_weights)
            metrics["SpecializationIndex"] = float(np.mean(scores["specialization_index"]))
            metrics["GroupSpecialization"] = scores["specialization_index"].tolist()
        except Exception as e:
            print(f"[WARN] Specialization tracking failed: {e}")

    return metrics


# ==================== Natural Niches Competition ====================

def build_expert_archive(model: nn.Module) -> List[Dict]:
    """Lưu trữ state_dict của tất cả expert."""
    archive = []
    try:
        for group in model.encoder.expert_groups:
            num_experts = group.config["num_experts"]
            for i in range(num_experts):
                expert = group.get_expert(i)
                archive.append(
                    {k: v.clone() for k, v in expert.state_dict().items()}
                )
    except AttributeError:
        pass
    return archive


def apply_archive_to_model(model: nn.Module, archive: List[Dict]) -> None:
    """Ghi lại state_dict của expert từ archive vào model."""
    idx = 0
    try:
        for group in model.encoder.expert_groups:
            num_experts = group.config["num_experts"]
            for i in range(num_experts):
                if idx >= len(archive):
                    break
                group.get_expert(i).load_state_dict(archive[idx])
                idx += 1
    except AttributeError:
        pass


def compute_expert_scores(
    model: nn.Module,
    batch: Tuple,
    device: torch.device,
    num_samples: int = 32,
) -> torch.Tensor:
    """Tính điểm fitness cho mỗi expert dựa trên tần suất được chọn.

    Returns:
        scores: Tensor shape (num_experts,)
    """
    model.eval()
    with torch.no_grad():
        audio_features = batch[0][:num_samples].to(device)

        try:
            x = model.input_proj(audio_features)
            _, group_ids = model.gating(x, deterministic=True)

            num_groups = model.config["num_groups"]
            experts_per_group = model.config["experts_per_group"]
            num_experts = num_groups * experts_per_group

            group_ids_flat = group_ids.reshape(-1).long()
            group_counts = torch.zeros(num_groups, device=device)
            for g in range(num_groups):
                group_counts[g] = (group_ids_flat == g).float().sum()

            # Phân bổ điểm cho từng expert trong nhóm
            scores = torch.zeros(num_experts, device=device)
            for g in range(num_groups):
                start = g * experts_per_group
                noise = torch.randn(experts_per_group, device=device) * 0.01
                scores[start : start + experts_per_group] = group_counts[g] + noise

        except Exception as e:
            print(f"[WARN] compute_expert_scores failed: {e}")
            scores = torch.zeros(1, device=device)

    model.train()
    return scores


def run_competition(
    model: nn.Module,
    competition: "NaturalNichesCompetition",
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 2,
) -> nn.Module:
    """Chạy một vòng natural niches competition để tiến hóa expert."""
    archive = build_expert_archive(model)
    if not archive:
        return model

    all_scores = []
    data_iter = iter(dataloader)
    for _ in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        scores = compute_expert_scores(model, batch, device)
        all_scores.append(scores)

    if not all_scores:
        return model

    # Trung bình điểm qua các batch, shape (num_experts,)
    avg_scores = torch.stack(all_scores, dim=0).mean(dim=0)

    # Competition API nhận (num_experts, num_datapoints) → expand
    scores_2d = avg_scores.unsqueeze(1)  # (num_experts, 1)

    try:
        new_archive = competition.run_competition_step(archive, scores_2d, model)
        apply_archive_to_model(model, new_archive)
    except Exception as e:
        print(f"[WARN] Competition step failed: {e}")

    return model


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Train TeaMoE Model")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Config YAML"
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint"
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config, "r") as f:
        full_config = yaml.safe_load(f)

    model_cfg = full_config["model"]
    train_cfg = full_config.get("training", {})
    data_cfg = full_config.get("data", {})

    # CLI args ghi đè config
    output_dir = Path(args.output_dir or train_cfg.get("output_dir", "checkpoints"))
    num_epochs = args.num_epochs or train_cfg.get("num_epochs", 100)
    batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    learning_rate = args.learning_rate or train_cfg.get("learning_rate", 1e-3)
    warmup_steps = train_cfg.get("warmup_steps", 50_000)
    total_steps = train_cfg.get("total_steps", 500_000)
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    use_amp = train_cfg.get("use_amp", True)
    use_checkpoint = train_cfg.get("use_checkpoint", True)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    eval_every = train_cfg.get("eval_every_n_steps", 1000)
    competition_every = train_cfg.get("competition_every_n_steps", 0)
    wandb_project = args.wandb_project or train_cfg.get("wandb_project", "TeaMoE")
    wandb_entity = args.wandb_entity or train_cfg.get("wandb_entity", None)

    device_str = args.device or train_cfg.get("device", None)
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    manifests_dir = data_cfg.get(
        "manifests_dir",
        "datasets/processed_data_librispeech/manifests",
    )
    train_manifest = os.path.join(
        manifests_dir, data_cfg.get("train_manifest", "train.jsonl")
    )
    valid_manifest = os.path.join(
        manifests_dir, data_cfg.get("valid_manifest", "validation.jsonl")
    )
    num_workers = data_cfg.get("num_workers", 0)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Lưu config để reproducibility
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(full_config, f)

    # --- WandB ---
    use_wandb = bool(wandb_project and wandb_entity)
    if use_wandb:
        try:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config={
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    **{k: model_cfg.get(k) for k in (
                        "model_dim", "num_layers", "num_groups",
                        "total_experts", "distillation_weight", "load_balance_weight"
                    ) if k in model_cfg},
                },
            )
        except Exception as e:
            print(f"[WARN] WandB init failed: {e}")
            use_wandb = False

    # --- Dataset & DataLoader ---
    pin_memory = device_str == "cuda"
    persistent_workers = num_workers > 0

    train_dataset = LibriSpeechDataset(train_manifest, data_cfg, is_train=True)
    valid_dataset = LibriSpeechDataset(valid_manifest, data_cfg, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # --- Model ---
    model = TeaMoEModel(config=model_cfg).to(device)

    # torch.compile (chỉ dùng khi không dùng gradient checkpointing)
    use_compile = train_cfg.get("use_compile", False)
    if use_compile and hasattr(torch, "compile") and not use_checkpoint:
        print("Applying torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    elif use_compile and use_checkpoint:
        print("[WARN] torch.compile disabled (incompatible với gradient checkpointing)")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if progress >= 1.0:
            return 1e-5 / learning_rate
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = (
        torch.amp.GradScaler("cuda")
        if use_amp and device_str == "cuda"
        else None
    )

    # Competition
    competition = NaturalNichesCompetition(model_cfg)

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    best_wer = float("inf")
    ckpt_path = output_dir / "checkpoint.pt"

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_wer = ckpt.get("best_wer", float("inf"))
        print(f"Resumed from epoch {start_epoch}, step {global_step}, best WER {best_wer:.2f}%")

    # --- Info ---
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Device        : {device}")
    print(f"Model params  : {num_params:,}")
    print(f"Train samples : {len(train_dataset):,}")
    print(f"Valid samples : {len(valid_dataset):,}")
    print(f"Epochs        : {num_epochs}")
    print(f"Batch size    : {batch_size}  (accum × {grad_accum_steps})")
    print(f"AMP           : {use_amp}  | Checkpoint : {use_checkpoint}")
    print(f"{'='*60}\n")

    # ==================== Training Loop ====================

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        accum_counter = 0

        optimizer.zero_grad()  # Reset gradient ở đầu epoch

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            total_loss, loss_dict, aux_outputs = train_step(
                model=model,
                batch=batch,
                device=device,
                scaler=scaler,
                use_checkpoint=use_checkpoint,
                gradient_accumulation_steps=grad_accum_steps,
            )

            accum_counter += 1

            # Optimizer step sau khi tích lũy đủ gradient
            if accum_counter >= grad_accum_steps:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                accum_counter = 0
                global_step += 1

                # --- Ghi nhận loss ---
                current_loss = (
                    loss_dict["total"].item()
                    if isinstance(loss_dict, dict) and "total" in loss_dict
                    else total_loss.item()
                )
                epoch_loss += current_loss
                epoch_steps += 1
                current_lr = scheduler.get_last_lr()[0]

                pbar.set_postfix(
                    loss=f"{current_loss:.4f}", lr=f"{current_lr:.2e}"
                )

                # --- Log WandB (mỗi 10 step) ---
                if use_wandb and global_step % 10 == 0:
                    log_dict = {
                        "train/loss": current_loss,
                        "train/lr": current_lr,
                        "step": global_step,
                    }
                    if isinstance(loss_dict, dict):
                        for k, v in loss_dict.items():
                            if k != "total" and isinstance(v, torch.Tensor):
                                log_dict[f"train/loss_{k}"] = v.item()
                    wandb.log(log_dict, step=global_step)

                # --- Evaluation ---
                if global_step % eval_every == 0:
                    # Get pretrained checkpoint dir for specialization tracking
                    pretrained_dir = None
                    if model_cfg.get("group_expert_pretrained_paths") is not None:
                        # Extract directory from first path
                        first_path = model_cfg["group_expert_pretrained_paths"][0][0]
                        if first_path is not None:
                            pretrained_dir = Path(first_path).parent

                    eval_metrics = evaluate(
                        model, valid_loader, device, use_checkpoint, pretrained_dir
                    )
                    wer = eval_metrics["WER"]
                    per = eval_metrics["PER"]
                    gini = eval_metrics["Gini"]
                    cosine_d = eval_metrics["ExpertCosineDistance"]
                    spec_idx = eval_metrics.get("SpecializationIndex", None)

                    pbar.write(
                        f"[Step {global_step}] "
                        f"WER={wer:.2f}%  PER={per:.2f}%  "
                        f"Gini={gini:.4f}  CosDist={cosine_d:.4f}"
                        + (f"  Spec={spec_idx:.4f}" if spec_idx is not None else "")
                    )

                    if use_wandb:
                        wandb_log = {
                            "eval/wer": wer,
                            "eval/per": per,
                            "eval/gini": gini,
                            "eval/expert_cosine_distance": cosine_d,
                        }
                        if "SpecializationIndex" in eval_metrics:
                            wandb_log["eval/specialization_index"] = eval_metrics["SpecializationIndex"]
                        wandb.log(wandb_log, step=global_step)

                    # Lưu model tốt nhất
                    if wer < best_wer:
                        best_wer = wer
                        torch.save(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "best_wer": best_wer,
                                "config": model_cfg,
                            },
                            output_dir / "best_model.pt",
                        )
                        pbar.write(
                            f"  → New best model saved (WER: {best_wer:.2f}%)"
                        )

                # --- Expert Competition ---
                if (
                    competition_every > 0
                    and global_step % competition_every == 0
                ):
                    pbar.write("Running expert competition...")
                    model = run_competition(
                        model, competition, train_loader, device, num_batches=2
                    )
                    pbar.write("Competition completed.")

        # --- Cuối epoch ---
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        print(
            f"Epoch {epoch+1}/{num_epochs} done | "
            f"avg_loss={avg_loss:.4f} | best_WER={best_wer:.2f}%"
        )

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_wer": best_wer,
                "config": model_cfg,
            },
            ckpt_path,
        )

    print(f"\nTraining completed! Best WER: {best_wer:.2f}%")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()