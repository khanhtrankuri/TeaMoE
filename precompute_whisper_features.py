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
from typing import Optional
from tqdm import tqdm
import argparse
import librosa
from transformers import WhisperFeatureExtractor, WhisperModel


def resolve_audio_path(record: dict, manifest_path: str) -> Optional[str]:
    """Resolve stale manifest paths after moving load_dataset -> datasets."""
    candidates = []
    audio_path = record.get("audio_filepath")
    if audio_path:
        candidates.append(audio_path)
        candidates.append(audio_path.replace("load_dataset", "datasets", 1))

    audio_relpath = record.get("audio_relpath")
    if audio_relpath:
        manifest_dir = Path(manifest_path).resolve().parent
        dataset_root = manifest_dir.parent
        candidates.append(str(dataset_root / audio_relpath))
        candidates.append(str(Path("datasets/processed_data_librispeech") / audio_relpath))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return audio_path


def load_audio_for_whisper(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load and normalize one audio file for Whisper feature extraction.

    Args:
        audio_path: Path to audio file
        sample_rate: Sample rate (must be 16kHz for Whisper)

    Returns:
        Waveform array at the requested sample rate
    """
    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform


def process_manifest_with_whisper(
    manifest_path: str,
    output_manifest_path: str,
    output_dir: str,
    whisper_model_name: str = "openai/whisper-base",
    device: str = "cuda",
    sample_rate: int = 16000,
):
    """Process manifest và extract Whisper features cho tất cả audio.

    Args:
        manifest_path: Path to input JSONL manifest
        output_manifest_path: Path to output JSONL manifest (with whisper_feature_path)
        output_dir: Directory to save Whisper features as .npy files
        whisper_model_name: Whisper model name
        device: Device to run Whisper on
        sample_rate: Audio sample rate for loading and feature extraction
    """
    # Load Whisper model
    print(f"Loading Whisper model: {whisper_model_name}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
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
        audio_path = resolve_audio_path(record, manifest_path)

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
            waveform = load_audio_for_whisper(audio_path, sample_rate=sample_rate)
            inputs = feature_extractor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(device)  # [1, 80, 3000]

            # Extract Whisper encoder features
            with torch.no_grad():
                encoder_output = model.encoder(input_features)
                features = encoder_output.last_hidden_state  # [1, T_features, whisper_dim]
                # T_features = 1500 for 30s audio (downsampled by 2)

            # Save features
            features_np = features.cpu().numpy()[0]  # [T_features, whisper_dim]
            np.save(whisper_feat_path, features_np)

            record["audio_filepath"] = audio_path
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
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate for audio loading")

    args = parser.parse_args()

    process_manifest_with_whisper(
        manifest_path=args.manifest,
        output_manifest_path=args.output_manifest,
        output_dir=args.output_dir,
        whisper_model_name=args.whisper_model,
        device=args.device,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
