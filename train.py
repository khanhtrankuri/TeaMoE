"""
Train script for TeaMoE: MoE-Conformer + RNN-T with Natural Niches Competition
Metrics: WER, PER, Gini coefficient, Cosine distance between experts
"""
import os
import json
import argparse
from typing import List, Dict, Tuple
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.training import train_state, checkpoints
import numpy as np
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


def compute_cosine_distance_experts(expert_weights: List[jnp.ndarray]) -> float:
    """Average cosine distance between experts for diversity measurement"""
    n = len(expert_weights)
    if n < 2:
        return 0.0
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            w1, w2 = expert_weights[i], expert_weights[j]
            dot = jnp.sum(w1 * w2)
            norm = jnp.linalg.norm(w1) * jnp.linalg.norm(w2) + 1e-8
            cos_sim = dot / norm
            total_dist += 1.0 - cos_sim
            count += 1
    return float(total_dist / count) if count > 0 else 0.0


# ==================== Decoding ====================

def greedy_decode_rnnt(logits: jnp.ndarray, blank_id: int = 0) -> List[int]:
    """Greedy decoding for RNN-T output"""
    time_steps, label_len = logits.shape[0], logits.shape[1]
    decoded = []
    prev = -1
    for t in range(time_steps):
        for l in range(label_len):
            token = int(jnp.argmax(logits[t, l]))
            if token != blank_id and token != prev:
                decoded.append(token)
                prev = token
    return decoded


def greedy_decode_phone(phone_logits: jnp.ndarray) -> List[int]:
    """Decode phone predictions"""
    preds = jnp.argmax(phone_logits, axis=-1)
    return [int(p) for p in preds]


# ==================== Data Loading Placeholder ====================

def load_data_batch(batch_size: int = 16, split: str = "train"):
    """Placeholder for data loading from processed_data_librispeech/"""
    key = random.PRNGKey(42)
    audio_features = random.normal(key, (batch_size, 100, 80))
    targets = random.randint(key, (batch_size, 20), minval=1, maxval=5000)
    phone_targets = random.randint(key, (batch_size, 100), minval=1, maxval=256)
    input_lengths = jnp.array([100] * batch_size)
    target_lengths = jnp.array([20] * batch_size)
    phone_lengths = jnp.array([100] * batch_size)
    return audio_features, targets, phone_targets, input_lengths, target_lengths, phone_lengths


# ==================== Train State ====================

class TrainState(train_state.TrainState):
    """Extended train state for TeaMoE"""
    competition: NaturalNichesCompetition = None
    distillation: ExpertDistillation = None
    loss_fn: CombinedLoss = None
    rng: jnp.ndarray = None
    expert_usage: np.ndarray = None
    step_count: int = 0


def create_train_state(
    rng: jnp.ndarray,
    config: ModelConfig,
    learning_rate: float = 1e-3
) -> TrainState:
    """Initialize train state"""
    model = TeaMoEModel(config=config)
    competition = NaturalNichesCompetition(config)
    distillation = ExpertDistillation(config.distillation_weight)
    loss_fn = CombinedLoss(
        load_balance_weight=config.load_balance_weight,
        z_loss_weight=config.z_loss_weight,
        distillation_weight=config.distillation_weight,
        ctc_phone_weight=config.ctc_phone_weight,
        blank_id=config.blank_id
    )

    rng, init_rng = random.split(rng)
    dummy_audio = jnp.ones((1, 100, config.n_mels))
    dummy_targets = jnp.ones((1, 10), dtype=jnp.int32)
    dummy_phone_targets = jnp.ones((1, 100), dtype=jnp.int32)
    variables = model.init(init_rng, dummy_audio, dummy_targets, dummy_phone_targets, deterministic=False)
    params = variables['params']

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=50000,
        decay_steps=500000,
        end_value=1e-5
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=0.01)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        competition=competition,
        distillation=distillation,
        loss_fn=loss_fn,
        rng=rng,
        expert_usage=np.zeros(config.total_experts),
        step_count=0
    )


# ==================== Training Step ====================

@jax.jit
def train_step(state: TrainState, batch, rng_key):
    """Single training step"""
    audio_features, targets, phone_targets, input_lengths, target_lengths, phone_lengths = batch

    rng_key, dropout_rng = random.split(rng_key)

    def loss_fn(params):
        rng1, rng2 = random.split(dropout_rng)
        outputs = state.apply_fn(
            {'params': params},
            audio_features,
            targets,
            phone_targets,
            deterministic=False,
            rngs={'dropout': rng1}
        )
        # outputs is a tuple: (model_outputs, aux_outputs)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            model_out, aux_outputs = outputs
        else:
            model_out = outputs
            aux_outputs = {}

        rnnt_logits = model_out[0] if isinstance(model_out, tuple) else model_out
        group_probs = aux_outputs.get('group_probs')
        group_ids = aux_outputs.get('group_ids')
        phone_logits = aux_outputs.get('phone_logits')

        group_logits = jnp.log(group_probs + 1e-8) if group_probs is not None else None

        # Distillation loss placeholder
        distillation_loss = 0.0

        total_loss, loss_dict = state.loss_fn.total_loss(
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
        return total_loss, (loss_dict, group_ids)

    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, (loss_dict, group_ids) = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    # Update expert usage statistics
    if group_ids is not None:
        group_ids_flat = group_ids.reshape(-1)
        usage = np.bincount(np.array(group_ids_flat), minlength=state.competition.num_groups)
        state = state.replace(
            expert_usage=state.expert_usage + usage
        )

    state = state.replace(
        step_count=state.step_count + 1
    )

    loss_val = loss_dict['total'] if isinstance(loss_dict, dict) and 'total' in loss_dict else loss_dict
    return state, {'loss': float(loss_val), 'loss_dict': loss_dict}, rng_key


# ==================== Competition Step ====================

def run_competition(state: TrainState, rng_key):
    """Run natural niches competition to evolve experts"""
    rng_key, subkey = random.split(rng_key)

    # Simplified: generate random expert weights for demonstration
    expert_weights = jnp.ones((state.competition.num_groups, state.competition.experts_per_group, 1024))

    # Generate random fitness scores for demonstration
    num_datapoints = 100
    scores = jax.random.normal(subkey, (state.competition.num_experts, num_datapoints))

    # Run competition for each group
    for group_id in range(state.competition.num_groups):
        parent_1_idx, parent_2_idx = state.competition.sample_parents(
            expert_weights.reshape(-1, 1024), scores, subkey, group_id
        )
        # In practice, would update actual params using competition.run_competition_step

    return state, rng_key


# ==================== Evaluation ====================

def evaluate(state: TrainState, eval_batches):
    """Evaluate model on validation set"""
    all_preds = []
    all_targets = []
    all_phone_preds = []
    all_phone_targets = []

    for batch in eval_batches:
        audio_features, targets, phone_targets, input_lengths, target_lengths, phone_lengths = batch

        rng = random.PRNGKey(0)
        outputs = state.apply_fn(
            {'params': state.params},
            audio_features,
            targets,
            phone_targets,
            deterministic=True,
            rngs={'dropout': rng}
        )

        if isinstance(outputs, tuple) and len(outputs) == 2:
            model_out, aux_outputs = outputs
        else:
            model_out = outputs
            aux_outputs = {}

        rnnt_logits = model_out[0] if isinstance(model_out, tuple) else model_out
        phone_logits = aux_outputs.get('phone_logits')

        for i in range(len(audio_features)):
            pred = greedy_decode_rnnt(rnnt_logits[i], state.loss_fn.blank_id)
            all_preds.append(pred)
            all_targets.append(list(targets[i]))

        if phone_logits is not None:
            for i in range(len(audio_features)):
                pred_phones = greedy_decode_phone(phone_logits[i])
                all_phone_preds.append(pred_phones)
                all_phone_targets.append(list(phone_targets[i]))

    wer = compute_wer(all_preds, all_targets)
    per = compute_per(all_phone_preds, all_phone_targets) if all_phone_preds else 0.0
    gini = compute_gini_coefficient(state.expert_usage)

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

    rng = random.PRNGKey(0)
    state = create_train_state(rng, config, args.learning_rate)

    if args.resume:
        state = checkpoints.restore_checkpoint(str(output_dir), state)

    print(f"Starting training for {args.num_epochs} epochs...")
    print(f"Model config: {config}")

    for epoch in range(args.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.num_epochs} ===")

        num_train_batches = 100
        pbar = tqdm(range(num_train_batches), desc="Training")
        epoch_loss = 0.0
        epoch_metrics = {}

        for batch_idx in pbar:
            batch = load_data_batch(args.batch_size, split="train")
            state, metrics, rng = train_step(state, batch, rng)
            epoch_loss += metrics['loss']

            # Accumulate metrics
            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)

            if state.step_count % config.competition_freq_steps == 0 and state.step_count > 0:
                print(f"\nRunning competition at step {state.step_count}...")
                state, rng = run_competition(state, rng)

            pbar.set_postfix({'loss': metrics['loss']})

        avg_loss = epoch_loss / num_train_batches

        # Log training metrics to WandB
        wandb_log = {
            "epoch": epoch + 1,
            "train/avg_loss": avg_loss,
        }

        # Log individual loss components if available
        if isinstance(metrics.get('loss_dict'), dict):
            for loss_name, loss_val in metrics['loss_dict'].items():
                if loss_name != 'total':
                    wandb_log[f"train/{loss_name}_loss"] = float(loss_val)

        print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        print("Evaluating...")
        eval_batches = [load_data_batch(args.batch_size, split="validation") for _ in range(10)]
        eval_metrics = evaluate(state, eval_batches)
        print("Evaluation results:")
        for key, val in eval_metrics.items():
            print(f"  {key}: {val:.4f}")
            wandb_log[f"eval/{key.lower()}"] = val

        # Log expert usage statistics
        if state.expert_usage is not None:
            usage_normalized = state.expert_usage / (np.sum(state.expert_usage) + 1e-8)
            wandb_log["eval/expert_usage_gini"] = compute_gini_coefficient(state.expert_usage)

        wandb.log(wandb_log, step=epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1:
            checkpoints.save_checkpoint(str(output_dir), state, step=state.step_count, overwrite=True)
            print(f"Checkpoint saved to {output_dir}")

        state = state.replace(expert_usage=np.zeros(config.total_experts))

    wandb.finish()
    print("\nTraining completed!")
    print(f"Final checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
