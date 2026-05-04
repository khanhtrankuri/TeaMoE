"""
Train Pretrained Expert Models (M1-M5)

This script trains 5 diverse expert models on mixed acoustic data.
These experts will later be shared across all 8 groups in TeaMoE.

Strategy:
- Train 5 separate Expert models (not MoE yet)
- Use diverse data (all acoustic classes mixed)
- Different random seeds for diversity
- Each becomes a "generalist" feature extractor
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import librosa
import json

from model.expert import Expert


# ==================== Dataset (Same as train.py) ====================

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: str, config: dict):
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)
        self.n_mels = config.get("n_mels", 80)
        self.hop_length = config.get("hop_length", 256)
        self.win_length = config.get("win_length", 1024)
        self.max_duration = config.get("max_duration", 30.0)
        self.model_dim = config.get("model_dim", 1024)

        self.records = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("duration_seconds", 0) > self.max_duration:
                    continue
                if "audio_filepath" in record:
                    # Normalize path: relative to project root
                    audio_path = record["audio_filepath"]
                    if not os.path.isabs(audio_path):
                        # If path starts with 'load_dataset', replace with 'datasets'
                        if audio_path.startswith("load_dataset"):
                            audio_path = audio_path.replace("load_dataset", "datasets", 1)
                        # If path is relative, make it absolute relative to manifest dir
                        if not os.path.exists(audio_path):
                            manifest_dir = os.path.dirname(manifest_path)
                            audio_path = os.path.normpath(os.path.join(manifest_dir, audio_path))
                    record["audio_filepath"] = audio_path
                self.records.append(record)

        print(f"Loaded {len(self.records)} records from {manifest_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        audio_path = record["audio_filepath"]
        text = record.get("text", "")

        try:
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"[WARN] Error loading {audio_path}: {e}")
            # Return zero waveform of reasonable length
            waveform = np.zeros(self.sample_rate, dtype=np.float32)

        # Mel spectrogram [n_mels, time] -> [time, n_mels]
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel = torch.from_numpy(mel).T.float()  # [T, n_mels]

        # Text tokens (not used in pretraining but kept for compatibility)
        tokens = [ord(c) for c in text if ord(c) < 5000]
        tokens = tokens if tokens else [1]
        tokens = torch.tensor(tokens, dtype=torch.long)

        return {
            "audio_features": mel,
            "targets": tokens,
            "audio_path": audio_path,
        }


def collate_fn_pretrain(batch):
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

    audio_features = torch.stack(padded_features)
    targets = torch.stack([
        F.pad(item["targets"], (0, max(len(item["targets"]) for item in batch) - len(item["targets"])))
        for item in batch
    ])
    input_lengths = torch.tensor(feature_lengths, dtype=torch.long)
    target_lengths = torch.tensor([len(item["targets"]) for item in batch], dtype=torch.long)

    return audio_features, targets, input_lengths, target_lengths


# ==================== Simple Pretrain Model ====================

class PretrainModel(nn.Module):
    """Autoencoder-style pretraining for expert.

    Architecture:
    mel -> InputProj -> Expert -> OutputProj -> mel_recon

    This mimics the actual usage in TeaMoE where:
    mel -> InputProj (in main model) -> Expert (our pretrained) -> Output
    """
    def __init__(self, n_mels=80, expert_dim=1024, ff_multiplier=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, expert_dim)
        self.expert = Expert(expert_dim, ff_multiplier, dropout)
        self.output_proj = nn.Linear(expert_dim, n_mels)

    def forward(self, mel):
        # mel: [B, T, n_mels]
        x = self.input_proj(mel)  # [B, T, expert_dim]
        h = self.expert(x)        # [B, T, expert_dim]
        recon = self.output_proj(h)  # [B, T, n_mels]
        return recon, h


# ==================== Training ====================

def train_pretrain_expert(
    expert_id: int,
    config: dict,
    train_manifest: str,
    valid_manifest: str,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cuda",
):
    """Train a single expert model (M1, M2, ...)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Dataset
    train_dataset = PretrainDataset(train_manifest, config)
    valid_dataset = PretrainDataset(valid_manifest, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_pretrain,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_pretrain,
        pin_memory=True,
    )

    # Model - includes input_proj and output_proj for reconstruction
    n_mels = config.get("n_mels", 80)
    model_dim = config.get("model_dim", 1024)
    ff_multiplier = config.get("ff_multiplier", 4)
    dropout = config.get("dropout", 0.1)

    model = PretrainModel(
        n_mels=n_mels,
        expert_dim=model_dim,
        ff_multiplier=ff_multiplier,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"\n{'='*60}")
    print(f"Training Expert M{expert_id+1}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Model: input_proj({n_mels}->{model_dim}) + Expert + output_proj({model_dim}->{n_mels})")
    print(f"{'='*60}\n")

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            audio_features, targets, input_lengths, target_lengths = batch
            audio_features = audio_features.to(device, non_blocking=True)

            # Pretraining objective: reconstruct mel-spectrogram
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                recon, _ = model(audio_features)
                # MSE reconstruction loss, masked by valid lengths
                mask = (audio_features != 0).float()  # Simple mask for padding
                loss = F.mse_loss(recon * mask, audio_features * mask, reduction="sum") / mask.sum()

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_steps = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation", leave=False):
                audio_features, _, _, _ = batch
                audio_features = audio_features.to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                    recon, _ = model(audio_features)
                    mask = (audio_features != 0).float()
                    loss = F.mse_loss(recon * mask, audio_features * mask, reduction="sum") / mask.sum()

                valid_loss += loss.item()
                valid_steps += 1

        avg_train = train_loss / train_steps if train_steps > 0 else 0
        avg_valid = valid_loss / valid_steps if valid_steps > 0 else 0

        print(f"Epoch {epoch+1}: train_loss={avg_train:.4f}, valid_loss={avg_valid:.4f}")

        if avg_valid < best_loss:
            best_loss = avg_valid
            checkpoint_path = output_dir / f"expert_M{expert_id+1}.pt"
            torch.save({
                'expert_state_dict': model.expert.state_dict(),
                'input_proj_state_dict': model.input_proj.state_dict(),
                'output_proj_state_dict': model.output_proj.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'config': {
                    'n_mels': n_mels,
                    'expert_dim': model_dim,
                    'ff_multiplier': ff_multiplier,
                    'dropout': dropout,
                }
            }, checkpoint_path)
            print(f"  → Saved checkpoint: {checkpoint_path}")

    print(f"\nTraining M{expert_id+1} complete! Best validation loss: {best_loss:.4f}")
    return best_loss


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Train Pretrained Expert Models M1-M5")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--output-dir", type=str, default="checkpoints/pretrained")
    parser.add_argument("--train-manifest", type=str,
                        default="datasets/processed_data_librispeech/manifests/train.jsonl")
    parser.add_argument("--valid-manifest", type=str,
                        default="datasets/processed_data_librispeech/manifests/validation.jsonl")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--experts", type=int, default=5,
                        help="Number of expert models to train (M1-M5)")
    parser.add_argument("--base-seed", type=int, default=1000,
                        help="Base seed; experts use seed, seed+1, ...")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        full_config = yaml.safe_load(f)

    model_cfg = full_config["model"]
    data_cfg = full_config.get("data", {})

    # Override paths from data config if not provided
    if args.train_manifest == "datasets/processed_data_librispeech/manifests/train.jsonl":
        args.train_manifest = os.path.join(
            data_cfg.get("manifests_dir", "datasets/processed_data_librispeech/manifests"),
            data_cfg.get("train_manifest", "train.jsonl")
        )
    if args.valid_manifest == "datasets/processed_data_librispeech/manifests/validation.jsonl":
        args.valid_manifest = os.path.join(
            data_cfg.get("manifests_dir", "datasets/processed_data_librispeech/manifests"),
            data_cfg.get("valid_manifest", "validation.jsonl")
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_dir / "pretrain_config.yaml", "w") as f:
        yaml.dump({
            'model': model_cfg,
            'training': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'base_seed': args.base_seed,
            }
        }, f)

    print(f"Training {args.experts} pretrained expert models")
    print(f"Output directory: {output_dir}")
    print(f"Train manifest: {args.train_manifest}")
    print(f"Valid manifest: {args.valid_manifest}")

    # Train each expert with different seed
    losses = []
    for i in range(args.experts):
        seed = args.base_seed + i
        print(f"\n{'='*60}")
        print(f"Training Expert M{i+1} with seed {seed}")
        print(f"{'='*60}")

        loss = train_pretrain_expert(
            expert_id=i,
            config=model_cfg,
            train_manifest=args.train_manifest,
            valid_manifest=args.valid_manifest,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=seed,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        losses.append(loss)

    print(f"\n{'='*60}")
    print("PRETRAINING COMPLETE")
    print(f"{'='*60}")
    for i, loss in enumerate(losses):
        print(f"  M{i+1}: best validation loss = {loss:.4f}")
    print(f"\nPretrained models saved in: {output_dir}")


if __name__ == "__main__":
    import json
    main()
