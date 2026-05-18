#!/usr/bin/env python3
"""
Pre-compute Whisper features cho toàn bộ dataset.

Features được lưu dưới dạng .npy và đường dẫn được thêm vào manifest.
Chỉ chạy 1 lần, sau đó training sẽ load features sẵn thay vì compute lại.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
from transformers import WhisperModel


def extract_whisper_features(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Extract Whisper encoder features cho 1 audio file.

    Args:
        audio_path: Path to audio file
        sample_rate: Sample rate (must be 16kHz for Whisper)

    Returns:
        Whisper features: [T, whisper_dim] where whisper_dim=512 for whisper-base
    """
    # Load audio
    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Whisper expects log-Mel spectrogram with specific parameters
    # Transformers Whisper expects:
    # - 30-second audio at 16kHz
    # - 80-channel log-Mel spectrogram

    # Pad or trim to 30 seconds
    target_length = sample_rate * 30  # 480000 samples
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)))
    else:
        waveform = waveform[:target_length]

    # Normalize (similar to Whisper processor)
    waveform = waveform / np.max(np.abs(waveform) + 1e-8)

    return waveform


def process_manifest_with_whisper(
    manifest_path: str,
    output_manifest_path: str,
    output_dir: str,
    whisper_model_name: str = "openai/whisper-base",
    device: str = "cuda"
):
    """Process manifest và extract Whisper features cho tất cả audio.

    Args:
        manifest_path: Path to input JSONL manifest
        output_manifest_path: Path to output JSONL manifest (with whisper_feature_path)
        output_dir: Directory to save Whisper features as .npy files
        whisper_model_name: Whisper model name
        device: Device to run Whisper on
    """
    # Load Whisper model
    print(f"Loading Whisper model: {whisper_model_name}")
    model = WhisperModel.from_pretrained(whisper_model_name)
    model = model.to(device)
    model.eval()

    whisper_dim = model.config.d_model  # 512 for whisper-base
    print(f"Whisper feature dimension: {whisper_dim}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read manifest
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Processing {len(records)} audio files...")

    # Process each record
    updated_records = []
    for record in tqdm(records, desc="Extracting Whisper features"):
        audio_path = record["audio_filepath"]

        if not os.path.exists(audio_path):
            print(f"[WARN] Audio not found: {audio_path}")
            record["whisper_feature_path"] = None
            updated_records.append(record)
            continue

        # Generate unique ID for this audio
        audio_id = Path(audio_path).stem
        whisper_feat_path = output_dir / f"{audio_id}.npy"

        if whisper_feat_path.exists():
            # Already computed
            record["whisper_feature_path"] = str(whisper_feat_path)
            updated_records.append(record)
            continue

        # Extract audio
        try:
            waveform = extract_whisper_features(audio_path)
            waveform = torch.from_numpy(waveform).to(device)
            waveform = waveform.unsqueeze(0)  # [1, T]

            # Extract Whisper encoder features
            with torch.no_grad():
                encoder_output = model.encoder(waveform)
                features = encoder_output.last_hidden_state  # [1, T_features, whisper_dim]
                # T_features = 1500 for 30s audio (downsampled by 2)

            # Save features
            features_np = features.cpu().numpy()[0]  # [T_features, whisper_dim]
            np.save(whisper_feat_path, features_np)

            record["whisper_feature_path"] = str(whisper_feat_path)
            record["whisper_feature_shape"] = list(features_np.shape)

        except Exception as e:
            print(f"[ERROR] Failed to extract features for {audio_path}: {e}")
            record["whisper_feature_path"] = None

        updated_records.append(record)

    # Write updated manifest
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        for record in updated_records:
            f.write(json.dumps(record) + "\n")

    print(f"Whisper features saved to: {output_dir}")
    print(f"Updated manifest saved to: {output_manifest_path}")

    # Summary
    success_count = sum(1 for r in updated_records if r.get("whisper_feature_path") is not None)
    print(f"Successfully extracted features: {success_count}/{len(updated_records)}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Whisper features for dataset")
    parser.add_argument("--manifest", type=str, required=True, help="Input manifest path")
    parser.add_argument("--output-manifest", type=str, required=True, help="Output manifest path")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save features")
    parser.add_argument("--whisper-model", type=str, default="openai/whisper-base",
                        help="Whisper model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    process_manifest_with_whisper(
        manifest_path=args.manifest,
        output_manifest_path=args.output_manifest,
        output_dir=args.output_dir,
        whisper_model_name=args.whisper_model,
        device=args.device
    )


if __name__ == "__main__":
    main()