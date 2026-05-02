"""
Train script for TeaMoE: MoE-Conformer + RNN-T with Natural Niches Competition
Metrics: WER, PER, Gini coefficient, Cosine distance between experts
"""
import os
import json
import argparse
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from model.config import ModelConfig
from model.tea_moe import TeaMoEModel
from model.competition import NaturalNichesCompetition
from model.distillation import ExpertDistillation
from model.losses import CombinedLoss


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


def compute_cosine_distance_experts(expert_weights: List[torch.Tensor]) -> float:
    """Average cosine distance between experts for diversity measurement"""
    n = len(expert_weights)
    if n < 2:
        return 0.0
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            w1, w2 = expert_weights[i], expert_weights[j]
            dot = torch.sum(w1 * w2)
            norm = torch.norm(w1) * torch.norm(w2) + 1e-8
            cos_sim = dot / norm
            total_dist += 1.0 - cos_sim
            count += 1
    return float(total_dist / count) if count > 0 else 0.0


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


def greedy_decode_phone(phone_logits: torch.Tensor) -> List[int]:
    """Decode phone predictions"""
    preds = torch.argmax(phone_logits, dim=-1)
    return [int(p) for p in preds]


# ==================== Data Loading Placeholder ====================

def load_data_batch(batch_size: int = 16, split: str = "train", device: str = "cpu"):
    """Placeholder for data loading from processed_data_librispeech/"""
    audio_features = torch.randn(batch_size, 100, 80, device=device)
    targets = torch.randint(1, 5000, (batch_size, 20), device=device)
    phone_targets = torch.randint(1, 256, (batch_size, 100), device=device)
    input_lengths = torch.full((batch_size,), 100, device=device)
    target_lengths = torch.full((batch_size,), 20, device=device)
    phone_lengths = torch.full((batch_size,), 100, device=device)
    return audio_features, targets, phone_targets, input_lengths, target_lengths, phone_lengths


# ==================== Training Step ====================

def train_step(model, optimizer, batch, rng_key=None):
    """Single training step"""
    audio_features, targets, phone_targets, input_lengths, target_lengths, phone_lengths = batch

    optimizer.zero_grad()

    rnnt_logits, aux_outputs = model(
        audio_features, targets, phone_targets, deterministic=False
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

    total_loss.backward()
    optimizer.step()

    # Update expert usage statistics
    expert_usage = np.zeros(model.config.total_experts)
    if group_ids is not None:
        group_ids_flat = group_ids.reshape(-1).cpu()
        usage = np.bincount(group_ids_flat.numpy(), minlength=model.config.num_groups)
        expert_usage = usage

    loss_val = loss_dict['total'].item() if isinstance(loss_dict, dict) and 'total' in loss_dict else loss_dict.item()
    return model, {'loss': loss_val, 'loss_dict': loss_dict}, expert_usage


# ==================== Competition Step ====================

def run_competition(model, competition, rng_key=None):
    """Run natural niches competition to evolve experts"""
    # Simplified: generate random expert weights for demonstration
    expert_weights = [p.clone() for p in model.parameters()]

    # Generate random fitness scores for demonstration
    num_datapoints = 100
    scores = torch.randn(competition.num_experts, num_datapoints)

    # Run competition for each group
    for group_id in range(competition.num_groups):
        parent_1_idx, parent_2_idx = competition.sample_parents(
            expert_weights, scores, rng_key, group_id
        )
        # In practice, would update actual params using competition.run_competition_step

    return model


# ==================== Evaluation ====================

def evaluate(model, eval_batches, device: str = "cpu"):
    """Evaluate model on validation set"""
    all_preds = []
    all_targets = []
    all_phone_preds = []
    all_phone_targets = []

    model.eval()
    with torch.no_grad():
        for batch in eval_batches:
            audio_features, targets, phone_targets, input_lengths, target_lengths, phone_lengths = batch

            rnnt_logits, aux_outputs = model(
                audio_features, targets, phone_targets, deterministic=True
            )
            phone_logits = aux_outputs.get('phone_logits')

            for i in range(len(audio_features)):
                pred = greedy_decode_rnnt(rnnt_logits[i], model.loss_fn.blank_id)
                all_preds.append(pred)
                all_targets.append(list(targets[i].cpu().numpy()))

            if phone_logits is not None:
                for i in range(len(audio_features)):
                    pred_phones = greedy_decode_phone(phone_logits[i])
                    all_phone_preds.append(pred_phones)
                    all_phone_targets.append(list(phone_targets[i].cpu().numpy()))
    model.train()

    wer = compute_wer(all_preds, all_targets)
    per = compute_per(all_phone_preds, all_phone_targets) if all_phone_preds else 0.0
    gini = compute_gini_coefficient(np.zeros(model.config.total_experts))  # Placeholder

    return {
        'WER': wer,
        'PER': per,
        'Gini': gini,
    }


# ==================== Main Training Loop ====================

def main():
    parser = argparse.ArgumentParser(description="Train TeaMoE Model")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--wandb-project", type=str, default="TeaMoE", help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)
    else:
        config = ModelConfig()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "model_dim": config.model_dim,
            "num_layers": config.num_layers,
            "num_groups": config.num_groups,
            "total_experts": config.total_experts,
            "distillation_weight": config.distillation_weight,
            "load_balance_weight": config.load_balance_weight,
        }
    )

    device = torch.device(args.device)
    model = TeaMoEModel(config=config).to(device)
    competition = NaturalNichesCompetition(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Learning rate scheduler with warmup
    warmup_steps = 50000
    total_steps = 500000
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(step / warmup_steps, 1.0) * 0.5 * (1 + torch.cos(torch.pi * min(step, total_steps) / total_steps)) + 1e-5 / args.learning_rate
        if step < total_steps else 1e-5 / args.learning_rate
    )

    start_epoch = 0
    if args.resume and (output_dir / "checkpoint.pt").exists():
        checkpoint = torch.load(output_dir / "checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    print(f"Starting training for {args.num_epochs} epochs on {device}...")
    print(f"Model config: {config}")

    step_count = 0
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.num_epochs} ===")

        num_train_batches = 100
        pbar = tqdm(range(num_train_batches), desc="Training")
        epoch_loss = 0.0
        epoch_metrics = {}
        expert_usage = np.zeros(config.total_experts)

        for batch_idx in pbar:
            batch = load_data_batch(args.batch_size, split="train", device=device)
            model, metrics, batch_usage = train_step(model, optimizer, batch)
            scheduler.step()
            step_count += 1

            epoch_loss += metrics['loss']
            expert_usage += batch_usage

            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)

            if step_count % config.competition_freq_steps == 0 and step_count > 0:
                print(f"\nRunning competition at step {step_count}...")
                model = run_competition(model, competition)

            pbar.set_postfix({'loss': metrics['loss']})

        avg_loss = epoch_loss / num_train_batches

        # Log training metrics to WandB
        wandb_log = {
            "epoch": epoch + 1,
            "train/avg_loss": avg_loss,
        }

        if isinstance(metrics.get('loss_dict'), dict):
            for loss_name, loss_val in metrics['loss_dict'].items():
                if loss_name != 'total':
                    wandb_log[f"train/{loss_name}_loss"] = float(loss_val)

        print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        print("Evaluating...")
        eval_batches = [load_data_batch(args.batch_size, split="validation", device=device) for _ in range(10)]
        eval_metrics = evaluate(model, eval_batches, device)
        print("Evaluation results:")
        for key, val in eval_metrics.items():
            print(f"  {key}: {val:.4f}")
            wandb_log[f"eval/{key.lower()}"] = val

        wandb.log(wandb_log, step=epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, output_dir / "checkpoint.pt")
            print(f"Checkpoint saved to {output_dir}")

    wandb.finish()
    print("\nTraining completed!")
    print(f"Final checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
