"""
Complete workflow: Train shared pretrained experts then fine-tune TeaMoE.

Usage:
  python run_shared_pretrained_workflow.py --phase pretrain   # Train M1-M5
  python run_shared_pretrained_workflow.py --phase finetune  # Train TeaMoE with shared experts
  python run_shared_pretrained_workflow.py --phase analyze   # Analyze specialization
"""
import argparse
import yaml
import torch
from pathlib import Path
import subprocess
import sys
import json


def phase_pretrain(args):
    """Phase 1: Train 5 pretrained expert models."""
    print("="*70)
    print("PHASE 1: Training Pretrained Experts M1-M5")
    print("="*70)

    cmd = [
        sys.executable, "train_pretrained_experts.py",
        "--config", args.config,
        "--output-dir", args.pretrained_dir,
        "--epochs", str(args.pretrain_epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.pretrain_lr),
        "--experts", "5",
        "--base-seed", "1000",
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Verify all 5 checkpoints exist
    pretrained_dir = Path(args.pretrained_dir)
    for i in range(5):
        ckpt = pretrained_dir / f"expert_M{i+1}.pt"
        if not ckpt.exists():
            print(f"ERROR: Missing checkpoint {ckpt}")
            return False

    print(f"\n✓ All 5 pretrained models saved in {pretrained_dir}")
    return True


def phase_finetune(args):
    """Phase 2: Fine-tune TeaMoE with shared pretrained experts."""
    print("\n" + "="*70)
    print("PHASE 2: Fine-tuning TeaMoE with Shared Pretrained Experts")
    print("="*70)

    # Load and modify config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set shared pretrained paths
    pretrained_dir = Path(args.pretrained_dir)
    shared_paths = [
        str(pretrained_dir / f"expert_M{i+1}.pt")
        for i in range(5)
    ]

    # Each group gets the same 5 models
    config["model"]["group_expert_pretrained_paths"] = [
        shared_paths for _ in range(config["model"]["num_groups"])
    ]

    # Adjust training settings for fine-tuning
    train_config = config.get("training", {})
    train_config.update({
        "num_epochs": args.finetune_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.finetune_lr,
        "eval_every_n_steps": args.eval_every,
    })
    config["training"] = train_config

    # Save modified config
    finetune_config_path = Path(args.output_dir) / "finetune_config.yaml"
    finetune_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(finetune_config_path, "w") as f:
        yaml.dump(config, f)

    print(f"Fine-tune config saved to: {finetune_config_path}")
    print(f"Shared pretrained paths: {shared_paths}")

    # Run training
    cmd = [
        sys.executable, "train.py",
        "--config", str(finetune_config_path),
        "--output-dir", args.output_dir,
        "--num-epochs", str(args.finetune_epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.finetune_lr),
    ]

    if args.resume:
        cmd.append("--resume")

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Fine-tuning complete! Best model saved to: {args.output_dir}/best_model.pt")
        return True
    else:
        print(f"\n✗ Fine-tuning failed with exit code {result.returncode}")
        return False


def phase_analyze(args):
    """Phase 3: Analyze specialization after training."""
    print("\n" + "="*70)
    print("PHASE 3: Analyzing Expert Specialization")
    print("="*70)

    import json
    import numpy as np
    from train import TeaMoEModel
    import yaml

    # Load final checkpoint
    checkpoint_path = Path(args.output_dir) / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return False

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Load config
    config_path = Path(args.output_dir) / "finetune_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    model = TeaMoEModel(config=model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"Model loaded successfully")
    print(f"  - {len(model.encoder.expert_groups)} expert groups")
    print(f"  - {model.encoder.expert_groups[0].config['num_experts']} experts per group")

    # Load pretrained weights for comparison
    pretrained_dir = Path(args.pretrained_dir)
    pretrained_weights = []
    for g in range(model_cfg["num_groups"]):
        group_weights = {}
        for e in range(model_cfg["experts_per_group"]):
            ckpt_path = pretrained_dir / f"expert_M{e+1}.pt"
            if ckpt_path.exists():
                pretrained_ckpt = torch.load(ckpt_path, map_location="cpu")
                group_weights[f"expert_{e}"] = pretrained_ckpt["expert_state_dict"]
        pretrained_weights.append(group_weights)

    # Run diagnostics
    from diagnostics import compute_group_specialization_scores, print_specialization_report

    scores = compute_group_specialization_scores(model, pretrained_weights)

    print("\nSpecialization Scores:")
    print("-" * 70)
    group_names = [gc.get("group_name", f"Group_{i}") for i, gc in enumerate(model_cfg.get("group_configs", []))]
    for g, name in enumerate(group_names):
        print(f"  {name:15s}: {scores['specialization_index'][g]:.4f}")

    avg_spec = np.mean(scores["specialization_index"])
    print(f"\nAverage specialization across groups: {avg_spec:.4f}")

    # Save results
    results_path = Path(args.output_dir) / "specialization_analysis.json"
    with open(results_path, "w") as f:
        json.dump({
            "specialization_index": scores["specialization_index"].tolist(),
            "group_expert_diversity": scores["group_expert_diversity"].tolist(),
            "cross_group_expert_similarity": scores["cross_group_expert_similarity"].tolist(),
            "group_names": group_names,
            "avg_specialization": float(avg_spec),
        }, f, indent=2)

    print(f"\nDetailed results saved to: {results_path}")

    # Print full report
    print_specialization_report(model)

    print("\n✓ Analysis complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run complete shared-pretrained TeaMoE workflow"
    )
    parser.add_argument("--phase", type=str, required=True,
                        choices=["pretrain", "finetune", "analyze", "all"],
                        help="Which phase to run")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--pretrained-dir", type=str, default="checkpoints/pretrained",
                        help="Directory for M1-M5 pretrained models")
    parser.add_argument("--output-dir", type=str, default="checkpoints/finetuned",
                        help="Directory for fine-tuned TeaMoE model")
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", help="Resume fine-tuning")

    args = parser.parse_args()

    if args.phase == "pretrain" or args.phase == "all":
        success = phase_pretrain(args)
        if not success and args.phase == "all":
            print("Pretraining failed. Stopping.")
            return

    if args.phase == "finetune" or args.phase == "all":
        success = phase_finetune(args)
        if not success and args.phase == "all":
            print("Fine-tuning failed. Stopping.")
            return

    if args.phase == "analyze" or args.phase == "all":
        success = phase_analyze(args)
        if not success and args.phase == "all":
            print("Analysis failed.")
            return

    if args.phase == "all":
        print("\n" + "="*70)
        print("COMPLETE WORKFLOW FINISHED")
        print("="*70)
        print(f"Pretrained models: {args.pretrained_dir}")
        print(f"Fine-tuned model: {args.output_dir}")
        print(f"Analysis: {args.output_dir}/specialization_analysis.json")


if __name__ == "__main__":
    main()
