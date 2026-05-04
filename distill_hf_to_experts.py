"""
Distill HuggingFace Pretrained Speech Models into TeaMoE Expert Checkpoints

This script takes 5 large pretrained speech models and distills their encoder
representations into simple FFN expert checkpoints compatible with TeaMoE.

Models to distill:
1. facebook/mms-300m (Multilingual Multimodal)
2. speechbrain/asr-conformersmall-transformerlm-librispeech
3. facebook/hubert-large-ls960-ft
4. facebook/wav2vec2-large-xlsr-53
5. openai/whisper-large-v3

Output: expert_M1.pt through expert_M5.pt

Usage:
  python distill_hf_to_experts.py \
    --output-dir checkpoints/pretrained \
    --model-names "facebook/mms-300m" "speechbrain/asr-conformersmall-transformerlm-librispeech" \
    --batch-size 32 \
    --epochs 5
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import librosa

# HuggingFace imports will be checked/imported lazily in HFModelWrapper.__init__
# This avoids ImportError at module load time


# ==================== Configuration ====================

MODEL_HANDLES = {
    "mms-300m": {
        "hf_name": "facebook/mms-300m",
        "target_dim": 1024,  # MMS hidden size
        "processor": None,
    },
    "speechbrain-conformer": {
        "hf_name": "speechbrain/asr-conformersmall-transformerlm-librispeech",
        "target_dim": 256,  # SpeechBrain conformer dim
        "processor": None,
    },
    "hubert-large": {
        "hf_name": "facebook/hubert-large-ls960-ft",
        "target_dim": 1024,
        "processor": None,
    },
    "wav2vec2-large-xlsr": {
        "hf_name": "facebook/wav2vec2-large-xlsr-53",
        "target_dim": 1024,
        "processor": None,
    },
    "whisper-large-v3": {
        "hf_name": "openai/whisper-large-v3",
        "target_dim": 1280,  # Whisper encoder dim
        "processor": None,
    },
}


# ==================== Dataset ====================

class DistillationDataset(Dataset):
    """Dataset for distillation using LibriSpeech or any audio files."""
    def __init__(self, manifest_path: str, sample_rate: int = 16000, max_duration: float = 30.0):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.records = []
        self.manifest_dir = os.path.dirname(manifest_path)

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping invalid JSON line: {e}")
                    continue

                if record.get("duration_seconds", 0) > self.max_duration:
                    continue

                # Normalize audio path
                audio_path = record.get("audio_filepath", "")
                if audio_path:
                    # Fix common path issues
                    # 1. Replace load_dataset with datasets (if data was moved)
                    if "load_dataset" in audio_path:
                        audio_path = audio_path.replace("load_dataset", "datasets", 1)

                    # 2. Make relative to manifest dir if not absolute
                    if not os.path.isabs(audio_path):
                        # Try relative to manifest directory
                        audio_path = os.path.normpath(os.path.join(self.manifest_dir, "..", audio_path))
                        # Also try direct join if audio_path is already relative to manifests dir
                        if not os.path.exists(audio_path):
                            audio_path = os.path.normpath(os.path.join(self.manifest_dir, "..", "audio", audio_path))

                    # 3. Verify file exists
                    if os.path.exists(audio_path):
                        record["audio_filepath"] = audio_path
                        self.records.append(record)
                    else:
                        # Only print warning for first few missing files
                        if len(self.records) < 5:
                            print(f"[WARN] Audio file not found: {audio_path}")

        print(f"Loaded {len(self.records)} records from {manifest_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        audio_path = record["audio_filepath"]

        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"[WARN] Error loading {audio_path}: {e}")
            waveform = np.zeros(self.sample_rate, dtype=np.float32)

        # Normalize audio
        if np.abs(waveform).max() > 0:
            waveform = waveform / np.abs(waveform).max() * 0.9

        return {
            "waveform": torch.from_numpy(waveform).float(),
            "audio_path": audio_path,
        }


def collate_fn_distill(batch):
    """Collate for distillation: pad waveforms to same length."""
    waveforms = [item["waveform"] for item in batch]
    lengths = [w.shape[0] for w in waveforms]
    max_len = max(lengths)

    padded = torch.zeros(len(batch), max_len)
    for i, w in enumerate(waveforms):
        padded[i, :lengths[i]] = w

    lengths = torch.tensor(lengths, dtype=torch.long)
    audio_paths = [item["audio_path"] for item in batch]

    return padded, lengths, audio_paths


# ==================== Expert Model ====================

class Expert(nn.Module):
    """Simple FFN expert - same as in model/expert.py"""
    def __init__(self, expert_dim=1024, ff_multiplier=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(expert_dim, expert_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim * ff_multiplier, expert_dim),
        )

    def forward(self, x, deterministic=True):
        return self.net(x)


class DistillationStudent(nn.Module):
    """Student network: input_proj -> Expert -> output_proj.

    Maps from HF model's representation space to mel and back.
    """
    def __init__(self, hf_dim: int, expert_dim: int, n_mels: int = 80, ff_multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hf_dim = hf_dim
        self.expert_dim = expert_dim
        self.n_mels = n_mels

        self.input_proj = nn.Linear(hf_dim, expert_dim)
        self.expert = Expert(expert_dim, ff_multiplier, dropout)
        self.output_proj = nn.Linear(expert_dim, n_mels)

    def forward(self, hf_features):
        # hf_features: [B, T, hf_dim]
        x = self.input_proj(hf_features)
        h = self.expert(x)
        recon = self.output_proj(h)
        return recon, h


# ==================== HuggingFace Model Wrappers ====================

class HFModelWrapper(nn.Module):
    """Wrapper to extract encoder features from various HF speech models."""
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Loading {model_name}...")

        # Import transformers here to avoid errors if not installed
        try:
            from transformers import (
                Wav2Vec2Model,
                Wav2Vec2Processor,
                HubertModel,
                WhisperModel,
                WhisperProcessor,
                AutoModel,
                AutoProcessor,
            )
        except ImportError as e:
            raise ImportError(f"[ERROR] transformers not installed. Run: pip install transformers\n  {e}")

        # Determine model type and load accordingly
        if "mms" in model_name.lower():
            # MMS uses Wav2Vec2 architecture but without tokenizer
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            # MMS doesn't need processor for feature extraction
            self.processor = None
            self.target_key = "last_hidden_state"
            self.use_processor = False
        elif "hubert" in model_name.lower():
            self.model = HubertModel.from_pretrained(model_name)
            # Hubert can use Wav2Vec2Processor as fallback
            try:
                self.processor = AutoProcessor.from_pretrained(model_name)
            except:
                try:
                    self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                except:
                    self.processor = None
            self.target_key = "last_hidden_state"
            self.use_processor = self.processor is not None
        elif "wav2vec2" in model_name.lower():
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            try:
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            except:
                try:
                    self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                except:
                    self.processor = None
            self.target_key = "last_hidden_state"
            self.use_processor = self.processor is not None
        elif "whisper" in model_name.lower():
            self.model = WhisperModel.from_pretrained(model_name)
            try:
                self.processor = WhisperProcessor.from_pretrained(model_name)
            except:
                self.processor = None
            self.target_key = "last_hidden_state"
            self.use_processor = False  # Whisper uses different input handling
        elif "speechbrain" in model_name.lower():
            # SpeechBrain uses a different API
            from speechbrain.pretrained import EncoderDecoderASR
            self.model = EncoderDecoderASR.from_hparams(
                source=model_name,
                savedir="pretrained_models/speechbrain_temp"
            )
            self.processor = None
            self.target_key = "encoder"
            self.use_processor = False
        else:
            # Generic AutoModel - try with processor, may fail
            self.model = AutoModel.from_pretrained(model_name)
            try:
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.use_processor = True
            except:
                print(f"[WARN] Could not load processor for {model_name}, will use raw waveform")
                self.processor = None
                self.use_processor = False
            self.target_key = "last_hidden_state"

        self.model = self.model.to(self.device)
        self.model.eval()
        self.target_dim = self.model.config.hidden_size
        print(f"  Loaded {model_name} with hidden_size={self.target_dim}")
        print(f"  Using processor: {self.processor is not None}")

    def extract_features(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Extract encoder features from audio waveforms.

        Args:
            waveforms: [B, max_len] audio in [-1, 1]
            lengths: [B] actual lengths

        Returns:
            features: [B, T', target_dim] extracted features
        """
        with torch.no_grad():
            if "speechbrain" in self.model_name.lower():
                # SpeechBrain API
                outputs = self.model.encode_batch(waveforms.to(self.device))
                # SpeechBrain returns [B, 1, T', D], squeeze
                features = outputs.squeeze(1)
                return features

            # For models without processor (MMS, some Whisper)
            if not self.use_processor or self.processor is None:
                # Direct waveform input (normalize to expected range)
                inputs = waveforms.to(self.device)
                # Some models expect specific sampling rate, but we assume 16kHz already
            else:
                # Standard processor (Wav2Vec2, Hubert)
                try:
                    inputs = self.processor(
                        waveforms.cpu().numpy(),
                        sampling_rate=self.processor.feature_extractor.sampling_rate,
                        return_tensors="pt",
                        padding=True,
                    ).input_values.to(self.device)
                except:
                    # Fallback: raw waveform
                    inputs = waveforms.to(self.device)

            outputs = self.model(inputs)
            features = getattr(outputs, self.target_key)

            return features


# ==================== Mel Spectrogram Extraction ====================

def compute_mel_spectrogram(
    waveforms: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    hop_length: int = 256,
    win_length: int = 1024,
) -> torch.Tensor:
    """Compute mel spectrogram from waveforms."""
    batch_size = waveforms.shape[0]
    mels = []

    for i in range(batch_size):
        wav = waveforms[i].cpu().numpy()
        # Trim padding
        non_zero = np.where(np.abs(wav) > 1e-6)[0]
        if len(non_zero) > 0:
            wav = wav[:non_zero[-1] + 1]

        if len(wav) == 0:
            mel = np.zeros((1, n_mels), dtype=np.float32)
        else:
            mel = librosa.feature.melspectrogram(
                y=wav,
                sr=sample_rate,
                n_mels=n_mels,
                hop_length=hop_length,
                win_length=win_length,
            )
            mel = librosa.power_to_db(mel, ref=np.max)
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        mel = torch.from_numpy(mel).T.float()  # [T_mel, n_mels]
        mels.append(mel)

    # Pad to same length
    max_len = max(m.shape[0] for m in mels)
    padded = torch.zeros(batch_size, max_len, n_mels)
    for i, mel in enumerate(mels):
        padded[i, :mel.shape[0]] = mel

    return padded  # [B, T_mel, n_mels]


# ==================== Training ====================

def distill_one_expert(
    hf_wrapper: HFModelWrapper,
    student: DistillationStudent,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device = torch.device("cuda"),
    output_dir: Optional[Path] = None,
):
    """Distill one expert from HF model."""
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_loss = float("inf")

    for epoch in range(epochs):
        student.train()
        train_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for waveforms, lengths, _ in pbar:
            waveforms = waveforms.to(device, non_blocking=True)

            # Extract HF features (teacher)
            with torch.no_grad():
                teacher_features = hf_wrapper.extract_features(waveforms, lengths)
                # Teacher features: [B, T_teacher, hf_dim]

            # Compute mel spectrogram (target)
            with torch.no_grad():
                target_mel = compute_mel_spectrogram(waveforms)  # [B, T_mel, n_mels]
                target_mel = target_mel.to(device)

            # Align lengths: teacher_features and target_mel may have different time dims
            # We'll match by interpolating teacher features to mel time dimension
            B, T_mel, _ = target_mel.shape
            T_teacher = teacher_features.shape[1]

            if T_teacher != T_mel:
                # Interpolate teacher features to mel time dimension
                teacher_features = F.interpolate(
                    teacher_features.transpose(1, 2),  # [B, hf_dim, T_teacher]
                    size=T_mel,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # [B, T_mel, hf_dim]

            # Student forward
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                recon, _ = student(teacher_features)
                loss = F.mse_loss(recon, target_mel)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation
        student.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for waveforms, lengths, _ in tqdm(val_loader, desc="Validation", leave=False):
                waveforms = waveforms.to(device, non_blocking=True)
                teacher_features = hf_wrapper.extract_features(waveforms, lengths)
                target_mel = compute_mel_spectrogram(waveforms).to(device)

                B, T_mel, _ = target_mel.shape
                T_teacher = teacher_features.shape[1]
                if T_teacher != T_mel:
                    teacher_features = F.interpolate(
                        teacher_features.transpose(1, 2),
                        size=T_mel,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)

                recon, _ = student(teacher_features)
                loss = F.mse_loss(recon, target_mel)
                val_loss += loss.item()
                val_steps += 1

        avg_train = train_loss / train_steps if train_steps > 0 else 0
        avg_val = val_loss / val_steps if val_steps > 0 else 0

        print(f"Epoch {epoch+1}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

        if avg_val < best_loss and output_dir:
            best_loss = avg_val
            checkpoint_path = output_dir / f"expert_M{hf_wrapper.model_name.split('-')[-1] if '-' in hf_wrapper.model_name else '1'}.pt"
            torch.save({
                'expert_state_dict': student.expert.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'config': {
                    'expert_dim': student.expert_dim,
                    'ff_multiplier': student.expert.net[0].out_features // student.expert_dim,
                    'dropout': student.expert.net[2].p if hasattr(student.expert.net[2], 'p') else 0.1,
                    'hf_model': hf_wrapper.model_name,
                }
            }, checkpoint_path)
            print(f"  → Saved checkpoint: {checkpoint_path}")

    student.train()
    return best_loss


def main():
    parser = argparse.ArgumentParser(description="Distill HuggingFace models to TeaMoE experts")
    parser.add_argument("--output-dir", type=str, default="checkpoints/pretrained",
                        help="Output directory for expert checkpoints")
    parser.add_argument("--model-names", type=str, nargs="+",
                        default=["facebook/mms-300m", "speechbrain/asr-conformersmall-transformerlm-librispeech",
                                 "facebook/hubert-large-ls960-ft", "facebook/wav2vec2-large-xlsr-53",
                                 "openai/whisper-large-v3"],
                        help="List of HF model names to distill")
    parser.add_argument("--train-manifest", type=str,
                        default="datasets/processed_data_librispeech/manifests/train.jsonl",
                        help="Training manifest")
    parser.add_argument("--valid-manifest", type=str,
                        default="datasets/processed_data_librispeech/manifests/validation.jsonl",
                        help="Validation manifest")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Epochs per expert")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--expert-dim", type=int, default=1024,
                        help="Expert hidden dimension")
    parser.add_argument("--n-mels", type=int, default=80,
                        help="Number of mel bins")
    args = parser.parse_args()

    # Check manifest files exist
    train_manifest_path = Path(args.train_manifest)
    valid_manifest_path = Path(args.valid_manifest)

    if not train_manifest_path.exists() or not valid_manifest_path.exists():
        print("\n[ERROR] Manifest files not found!")
        print(f"  Train: {train_manifest_path} {'[exists]' if train_manifest_path.exists() else '[MISSING]'}")
        print(f"  Valid: {valid_manifest_path} {'[exists]' if valid_manifest_path.exists() else '[MISSING]'}")
        print("\nPlease prepare your data first:")
        print("  1. Download LibriSpeech dataset")
        print("  2. Run: python load_dataset/process_libri.py")
        print("     (This will create manifests and audio files)")
        print("\nOr specify custom manifests:")
        print(f"  python distill_hf_to_experts.py --train-manifest PATH/TO/train.jsonl --valid-manifest PATH/TO/val.jsonl ...")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Save distillation config
    with open(output_dir / "distill_config.yaml", "w") as f:
        import yaml
        yaml.dump(vars(args), f, default_flow_style=False)

    # Datasets
    print("\nLoading datasets...")
    try:
        train_dataset = DistillationDataset(args.train_manifest)
        valid_dataset = DistillationDataset(args.valid_manifest)
    except Exception as e:
        print(f"[ERROR] Failed to load datasets: {e}")
        return

    if len(train_dataset) == 0:
        print("[ERROR] Training dataset is empty! Check your manifest file.")
        print(f"  Manifest: {args.train_manifest}")
        print("  Expected format: JSONL with 'audio_filepath' and 'text' fields")
        return

    if len(valid_dataset) == 0:
        print("[WARNING] Validation dataset is empty! Using training set for validation.")
        valid_dataset = train_dataset

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Valid samples: {len(valid_dataset)}")

    if len(train_dataset) < 10:
        print("[WARNING] Very small training set! Results may be poor.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_distill,
        pin_memory=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_distill,
        pin_memory=True,
    )

    # Distill each model
    print(f"\nStarting distillation of {len(args.model_names)} models:")
    for i, model_name in enumerate(args.model_names):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(args.model_names)}] Distilling: {model_name}")
        print(f"{'='*60}")

        try:
            # Load HF wrapper
            hf_wrapper = HFModelWrapper(model_name, device=device.type)

            # Create student with matching dimensions
            student = DistillationStudent(
                hf_dim=hf_wrapper.target_dim,
                expert_dim=args.expert_dim,
                n_mels=args.n_mels,
            ).to(device)

            print(f"Student: input_proj({hf_wrapper.target_dim}->{args.expert_dim}) + Expert + output_proj({args.expert_dim}->{args.n_mels})")
            print(f"Total params: {sum(p.numel() for p in student.parameters()):,}")

            # Distill
            best_loss = distill_one_expert(
                hf_wrapper=hf_wrapper,
                student=student,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                output_dir=output_dir,
            )

            print(f"[OK] Distillation complete. Best val loss: {best_loss:.4f}")

        except ImportError as e:
            print(f"[ERROR] Missing dependencies for {model_name}: {e}")
            print("  Install required packages: pip install transformers torchaudio")
            continue
        except Exception as e:
            print(f"[ERROR] Failed to distill {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("DISTILLATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nCheckpoints saved in: {output_dir}")
    print("\nNext steps:")
    print("  1. Generate shared config:")
    print(f"     python generate_shared_config.py --pretrained-dir {output_dir} --output config/shared_pretrained.yaml")
    print("  2. Train TeaMoE with shared experts:")
    print(f"     python train.py --config config/shared_pretrained.yaml --output-dir checkpoints/finetuned")


if __name__ == "__main__":
    main()
