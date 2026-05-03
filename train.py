"""
Train script for TeaMoE: MoE-Conformer + RNN-T with Natural Niches Competition
Metrics: WER, PER, Gini coefficient, Cosine distance between experts
"""
import os
import math
import json
import argparse
from typing import List, Dict, Tuple
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

# Set CUDA allocation config to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# ==================== Dataset ====================

class LibriSpeechDataset(Dataset):
    def __init__(self, manifest_path: str, config: dict, is_train: bool = True):
        self.config = config
        self.is_train = is_train
        self.sample_rate = config.get('sample_rate', 16000)
        self.n_mels = config.get('n_mels', 80)
        self.hop_length = config.get('hop_length', 256)
        self.win_length = config.get('win_length', 1024)
        self.max_duration = config.get('max_duration', 30.0)
        self.audio_dir = config.get('audio_dir', None)

        self.records = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                duration = record.get('duration_seconds', 0)
                if duration > self.max_duration:
                    continue
                # Fix audio path: replace 'load_dataset' with 'datasets'
                if 'audio_filepath' in record:
                    record['audio_filepath'] = record['audio_filepath'].replace(
                        'load_dataset', 'datasets'
                    )
                self.records.append(record)

        print(f"Loaded {len(self.records)} records from {manifest_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        audio_path = record['audio_filepath']
        text = record['text']

        # Load audio with librosa
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            waveform = np.zeros(self.sample_rate)

        # Compute mel spectrogram with librosa
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, time)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel = torch.from_numpy(mel).T.float()  # (time, n_mels)

        # Tokenize text into character-level tokens
        tokens = [ord(c) for c in text if ord(c) < 5000]
        if len(tokens) == 0:
            tokens = [1]
        tokens = torch.tensor(tokens, dtype=torch.long)

        return {
            'audio_features': mel,
            'targets': tokens,
            'audio_path': audio_path,
        }


BLANK_ID = 0

def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, ...]:
    """Collate function for LibriSpeech dataset batches."""
    # Pad audio features to max length in batch
    max_time = max((item['audio_features'].shape[0] for item in batch))
    n_mels = batch[0]['audio_features'].shape[1]

    padded_features = []
    feature_lengths = []
    for item in batch:
        feat = item['audio_features']
        t = feat.shape[0]
        feature_lengths.append(t)
        if t < max_time:
            pad = torch.zeros(max_time - t, n_mels)
            feat = torch.cat([feat, pad], dim=0)
        padded_features.append(feat)

    audio_features = torch.stack(padded_features)  # (B, T, n_mels)

    # Pad targets to max length
    max_target = max((item['targets'].shape[0] for item in batch))
    padded_targets = []
    target_lengths = []
    for item in batch:
        tgt = item['targets']
        tl = tgt.shape[0]
        target_lengths.append(tl)
        if tl < max_target:
            pad = torch.full((max_target - tl,), BLANK_ID, dtype=torch.long)
            tgt = torch.cat([tgt, pad])
        padded_targets.append(tgt)

    targets = torch.stack(padded_targets)  # (B, L)

    input_lengths = torch.tensor(feature_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    # Phone targets placeholder (same as targets for now)
    phone_targets = targets.clone()
    phone_lengths = target_lengths.clone()

    return audio_features, targets, phone_targets, input_lengths, target_lengths, phone_lengths


# ==================== Metrics Computation ====================

def compute_edit_distance(pred_tokens: List[int], target_tokens: List[int]) -> int:
    """Edit distance (Levenshtein) for WER/PER"""
    m, n = len(pred_tokens), len(target_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == target_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]


def compute_wer(predictions: List[List[int]], targets: List[List[int]]) -> float:
    """Word Error Rate"""
    total_errors, total_words = 0, 0
    for pred, target in zip(predictions, targets):
        errors = compute_edit_distance(pred, target)
        total_errors += errors
        total_words += len(target)
    return (total_errors / total_words * 100) if total_words > 0 else 0.0


def compute_per(phone_predictions: List[List[int]], phone_targets: List[List[int]]) -> float:
    """Phone Error Rate"""
    return compute_wer(phone_predictions, phone_targets)


def compute_gini_coefficient(expert_usage_counts: np.ndarray) -> float:
    """Gini coefficient for expert load balance"""
    counts = np.sort(expert_usage_counts)
    n = len(counts)
    if n == 0 or np.sum(counts) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n


# ==================== Decoding ====================

def greedy_decode_rnnt(logits: torch.Tensor, blank_id: int = 0) -> List[int]:
    """Greedy decoding for RNN-T output"""
    time_steps, label_len = logits.shape[0], logits.shape[1]
    decoded = []
    prev = -1
    for t in range(time_steps):
        for l in range(label_len):
            token = int(torch.argmax(logits[t, l]))
            if token != blank_id and token != prev:
                decoded.append(token)
                prev = token
    return decoded


# ==================== Training Step ====================

def train_step(model, batch, device, use_checkpoint=False):
    """Single training step (without optimizer step for gradient accumulation)

    Returns: total_loss, loss_dict
    """
    audio_features, targets, phone_targets, input_lengths, target_lengths, phone_lengths = batch
    audio_features = audio_features.to(device)
    targets = targets.to(device)
    phone_targets = phone_targets.to(device)
    input_lengths = input_lengths.to(device)
    target_lengths = target_lengths.to(device)
    phone_lengths = phone_lengths.to(device)

    rnnt_logits, aux_outputs = model(
        audio_features, targets, phone_targets,
        deterministic=False, use_checkpoint=use_checkpoint
    )

    group_probs = aux_outputs.get('group_probs')
    group_ids = aux_outputs.get('group_ids')
    phone_logits = aux_outputs.get('phone_logits')
    group_logits = torch.log(group_probs + 1e-8) if group_probs is not None else None

    # Distillation loss placeholder
    distillation_loss = torch.tensor(0.0, device=audio_features.device)

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
        phone_lengths=phone_lengths
    )

    return total_loss, loss_dict


# ==================== Competition Step ====================

def build_expert_archive(model):
    """Build archive of all expert weights from the model"""
    archive = []
    for group in model.encoder.expert_groups:
        for i in range(group.config['num_experts']):
            expert = group.get_expert(i)
            archive.append({k: v.clone() for k, v in expert.state_dict().items()})
    return archive


def apply_archive_to_model(model, archive):
    """Apply updated expert weights from archive back to the model"""
    idx = 0
    for group in model.encoder.expert_groups:
        for i in range(group.config['num_experts']):
            expert = group.get_expert(i)
            expert.load_state_dict(archive[idx])
            idx += 1


def compute_expert_scores(model, batch, device, num_samples=100):
    """Compute fitness scores for each expert on a batch of data"""
    model.eval()
    with torch.no_grad():
        audio_features, targets, phone_targets, input_lengths, target_lengths, phone_lengths = batch
        audio_features = audio_features[:num_samples].to(device)
        input_lengths = input_lengths[:num_samples].to(device)

        # Get encoder outputs and group assignments
        x = model.input_proj(audio_features)
        group_probs, group_ids = model.gating(x, deterministic=True)
        encoder_out = model.encoder(x, group_ids=group_ids, deterministic=True)

        # Compute scores based on group assignments
        batch_size, seq_len, model_dim = encoder_out.shape
        group_ids_flat = group_ids.reshape(-1)

        # Base score: how much each group was used
        base_scores = torch.zeros(
            model.config['num_groups'],
            device=device
        )
        for g in range(model.config['num_groups']):
            mask = (group_ids_flat == g)
            base_scores[g] = mask.float().sum()

        # Distribute to experts with small random variations
        num_experts = model.config['num_groups'] * model.config['experts_per_group']
        scores = torch.zeros(num_experts, device=device)
        for group_idx in range(model.config['num_groups']):
            base_score = base_scores[group_idx]
            start_idx = group_idx * model.config['experts_per_group']
            # Add small noise to differentiate experts
            noise = torch.randn(model.config['experts_per_group'], device=device) * 0.01
            scores[start_idx:start_idx + model.config['experts_per_group']] = base_score + noise

    model.train()
    return scores.unsqueeze(0).expand(num_samples, -1).t()  # (num_experts, num_samples)


def run_competition(model, competition, dataloader, device, num_batches=1):
    """Run natural niches competition to evolve experts"""
    # Build initial archive
    archive = build_expert_archive(model)

    # Collect scores from a few batches
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

    # Average scores across batches
    scores = torch.stack(all_scores, dim=0).mean(dim=0)  # (num_experts, num_datapoints)

    # Run competition step
    new_archive = competition.run_competition_step(archive, scores, model)

    # Apply updated weights
    apply_archive_to_model(model, new_archive)

    return model


# ==================== Evaluation ====================

def evaluate(model, eval_loader, device: str = "cpu", use_checkpoint: bool = False):
    """Evaluate model on validation set"""
    all_preds = []
    all_targets = []
    model.eval()
    blank_id = model.loss_fn.blank_id

    with torch.no_grad():
        for batch in eval_loader:
            audio_features, targets, phone_targets, input_lengths, target_lengths, phone_lengths = batch
            audio_features = audio_features.to(device)
            targets = targets.to(device)

            rnnt_logits, aux_outputs = model(
                audio_features, targets, phone_targets,
                deterministic=True, use_checkpoint=use_checkpoint
            )

            for i in range(len(audio_features)):
                pred = greedy_decode_rnnt(rnnt_logits[i], blank_id)
                all_preds.append(pred)
                all_targets.append(list(targets[i].cpu().numpy()))

    model.train()

    wer = compute_wer(all_preds, all_targets)
    return {'WER': wer}


# ==================== Main Training Loop ====================

def main():
    parser = argparse.ArgumentParser(description="Train TeaMoE Model")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name (overrides config)")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity name (overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Device (overrides config)")
    args = parser.parse_args()

    # Load YAML config
    config_path = args.config
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    model_cfg = full_config['model']
    train_cfg = full_config.get('training', {})
    data_cfg = full_config.get('data', {})

    # CLI args override config values
    output_dir = args.output_dir or train_cfg.get('output_dir', 'checkpoints')
    num_epochs = args.num_epochs or train_cfg.get('num_epochs', 100)
    batch_size = args.batch_size or train_cfg.get('batch_size', 16)
    learning_rate = args.learning_rate or train_cfg.get('learning_rate', 1e-3)
    warmup_steps = train_cfg.get('warmup_steps', 50000)
    total_steps = train_cfg.get('total_steps', 500000)
    gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 1)
    use_amp = train_cfg.get('use_amp', True)
    use_checkpoint = train_cfg.get('use_checkpoint', True)
    checkpoint_every_n_layers = train_cfg.get('checkpoint_every_n_layers', 1)
    max_grad_norm = train_cfg.get('max_grad_norm', 1.0)
    wandb_project = args.wandb_project or train_cfg.get('wandb_project', 'TeaMoE')
    wandb_entity = args.wandb_entity or train_cfg.get('wandb_entity', None)
    device_str = args.device or train_cfg.get('device', None)
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    manifests_dir = data_cfg.get('manifests_dir', 'datasets/processed_data_librispeech/manifests')
    train_manifest = os.path.join(manifests_dir, data_cfg.get('train_manifest', 'train.jsonl'))
    valid_manifest = os.path.join(manifests_dir, data_cfg.get('valid_manifest', 'validation.jsonl'))
    num_workers = data_cfg.get('num_workers', 0)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WandB (optional)
    use_wandb = wandb_project and wandb_entity is not None
    if use_wandb:
        try:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config={
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "model_dim": model_cfg['model_dim'],
                    "num_layers": model_cfg['num_layers'],
                    "num_groups": model_cfg['num_groups'],
                    "total_experts": model_cfg['total_experts'],
                    "distillation_weight": model_cfg['distillation_weight'],
                    "load_balance_weight": model_cfg['load_balance_weight'],
                }
            )
        except Exception as e:
            print(f"WandB init failed: {e}")
            use_wandb = False

    device = torch.device(device_str)

    # Create datasets and loaders
    train_dataset = LibriSpeechDataset(train_manifest, data_cfg, is_train=True)
    valid_dataset = LibriSpeechDataset(valid_manifest, data_cfg, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True if device_str == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True if device_str == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
    )

    model = TeaMoEModel(config=model_cfg).to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Use torch.compile if available (PyTorch 2.0+) - but disable if using gradient checkpointing
    # as they are incompatible due to data-dependent branching in checkpoint
    use_compile = train_cfg.get('use_compile', False)
    use_checkpoint = train_cfg.get('use_checkpoint', True)
    if use_compile and hasattr(torch, 'compile') and not use_checkpoint:
        print("Applying torch.compile() for memory optimization...")
        model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
    elif use_compile and use_checkpoint:
        print("Warning: torch.compile disabled when using gradient checkpointing (incompatible)")

    competition = NaturalNichesCompetition(model_cfg)

    # Initialize mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp and device_str == 'cuda' else None

    # Learning rate scheduler with warmup + cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if progress >= 1.0:
            return 1e-5 / learning_rate
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    if args.resume and (output_dir / "checkpoint.pt").exists():
        checkpoint = torch.load(output_dir / "checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    print(f"Starting training for {num_epochs} epochs on {device}...")
    print(f"Model config: {model_cfg}")
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Valid dataset: {len(valid_dataset)} samples")

    # Training tracking
    global_step = 0
    best_wer = float('inf')
    accumulation_counter = 0

    # Main training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            audio_features, targets, phone_targets, input_lengths, target_lengths, phone_lengths = batch
            audio_features = audio_features.to(device)
            targets = targets.to(device)
            phone_targets = phone_targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            phone_lengths = phone_lengths.to(device)

            # Training step
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    rnnt_logits, aux_outputs = model(
                        audio_features, targets, phone_targets,
                        deterministic=False, use_checkpoint=use_checkpoint
                    )

                    group_probs = aux_outputs.get('group_probs')
                    group_ids = aux_outputs.get('group_ids')
                    phone_logits = aux_outputs.get('phone_logits')
                    group_logits = torch.log(group_probs + 1e-8) if group_probs is not None else None

                    # Placeholder for distillation loss
                    distillation_loss = torch.tensor(0.0, device=audio_features.device)

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
                        phone_lengths=phone_lengths
                    )

                    if gradient_accumulation_steps > 1:
                        total_loss = total_loss / gradient_accumulation_steps

                scaler.scale(total_loss).backward()
            else:
                rnnt_logits, aux_outputs = model(
                    audio_features, targets, phone_targets,
                    deterministic=False, use_checkpoint=use_checkpoint
                )

                group_probs = aux_outputs.get('group_probs')
                group_ids = aux_outputs.get('group_ids')
                phone_logits = aux_outputs.get('phone_logits')
                group_logits = torch.log(group_probs + 1e-8) if group_probs is not None else None

                distillation_loss = torch.tensor(0.0, device=audio_features.device)

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
                    phone_lengths=phone_lengths
                )

                if gradient_accumulation_steps > 1:
                    total_loss = total_loss / gradient_accumulation_steps

                total_loss.backward()

            accumulation_counter += 1

            # Gradient accumulation step
            if accumulation_counter >= gradient_accumulation_steps:
                # Gradient clipping
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                accumulation_counter = 0
                global_step += 1

                # Update metrics
                current_loss = loss_dict['total'].item() if isinstance(loss_dict, dict) else loss_dict.item()
                epoch_loss += current_loss
                epoch_steps += 1
                pbar.set_postfix({'loss': current_loss, 'lr': scheduler.get_last_lr()[0]})

                # Log to WandB
                if use_wandb and global_step % 10 == 0:
                    wandb.log({
                        'train/loss': current_loss,
                        'train/lr': scheduler.get_last_lr()[0],
                        'step': global_step
                    }, step=global_step)

            # Periodic evaluation
            if global_step > 0 and global_step % train_cfg.get('eval_every_n_steps', 1000) == 0:
                eval_metrics = evaluate(model, valid_loader, device=device, use_checkpoint=use_checkpoint)
                eval_wer = eval_metrics['WER']
                pbar.write(f"Step {global_step}: WER = {eval_wer:.2%}")

                if use_wandb:
                    wandb.log({'eval/wer': eval_wer}, step=global_step)

                # Save best model
                if eval_wer < best_wer:
                    best_wer = eval_wer
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_wer': best_wer,
                        'config': model_cfg,
                    }, output_dir / "best_model.pt")
                    pbar.write(f"New best model saved (WER: {best_wer:.2%})")

                # Periodic competition
                if train_cfg.get('competition_every_n_steps', 0) > 0 and global_step % train_cfg.get('competition_every_n_steps', 5000) == 0:
                    pbar.write("Running expert competition...")
                    model = run_competition(model, competition, train_loader, device, num_batches=2)
                    pbar.write("Competition completed")

        # Epoch end
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Save checkpoint at end of epoch
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_wer': best_wer,
            'config': model_cfg,
        }, output_dir / "checkpoint.pt")

    print(f"Training completed! Best WER: {best_wer:.2%}")

    if use_wandb:
        wandb.finish()
