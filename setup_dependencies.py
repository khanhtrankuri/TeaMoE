#!/usr/bin/env python
"""
Setup script for TeaMoE dependencies
"""
import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    print(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"[OK] {package} installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install {package}: {e}")
        return False

def main():
    print("=" * 60)
    print("TeaMoE Dependency Setup")
    print("=" * 60)

    # Core dependencies
    core_packages = [
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "librosa",
        "numpy",
        "tqdm",
        "pyyaml",
        "wandb",
    ]

    # HuggingFace dependencies (for distillation)
    hf_packages = [
        "transformers>=4.30.0",
        "sentencepiece",  # Required by some HF models
        "protobuf",       # Required by some HF models
    ]

    # SpeechBrain (optional)
    speechbrain_packages = [
        "speechbrain",
    ]

    print("\n1. Installing core packages...")
    for pkg in core_packages:
        install_package(pkg)

    print("\n2. Installing HuggingFace packages (for distillation)...")
    for pkg in hf_packages:
        install_package(pkg)

    response = input("\n3. Install SpeechBrain? (y/N): ").strip().lower()
    if response == 'y':
        for pkg in speechbrain_packages:
            install_package(pkg)

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Prepare LibriSpeech data:")
    print("     python load_dataset/process_libri.py")
    print("\n  2. Train from scratch:")
    print("     python train_pretrained_experts.py")
    print("\n  3. OR distill from HuggingFace:")
    print("     python distill_hf_to_experts.py")
    print("\n  4. Train TeaMoE:")
    print("     python train.py --config config/simple.yaml")
    print("=" * 60)

if __name__ == "__main__":
    main()
