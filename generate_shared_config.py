"""
Generate TeaMoE config with shared pretrained experts.

Usage:
  python generate_shared_config.py --base-config config/default.yaml \\
      --pretrained-dir checkpoints/pretrained \\
      --output config/shared_pretrained.yaml
"""
import argparse
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate config with shared pretrained experts")
    parser.add_argument("--base-config", type=str, default="config/default.yaml",
                        help="Base config file")
    parser.add_argument("--pretrained-dir", type=str, required=True,
                        help="Directory containing expert_M1.pt, expert_M2.pt, ...")
    parser.add_argument("--output", type=str, required=True,
                        help="Output config file")
    parser.add_argument("--num-groups", type=int, default=8,
                        help="Number of expert groups")
    parser.add_argument("--experts-per-group", type=int, default=5,
                        help="Number of experts per group")
    args = parser.parse_args()

    # Load base config
    with open(args.base_config, "r") as f:
        config = yaml.safe_load(f)

    pretrained_dir = Path(args.pretrained_dir)

    # Verify all expert checkpoints exist
    shared_paths = []
    for i in range(args.experts_per_group):
        ckpt_path = pretrained_dir / f"expert_M{i+1}.pt"
        if not ckpt_path.exists():
            print(f"WARNING: Missing checkpoint {ckpt_path}")
            print("  This will be treated as None (train from scratch)")
            shared_paths.append(None)
        else:
            shared_paths.append(str(ckpt_path))

    print(f"Found {len([p for p in shared_paths if p is not None])}/{args.experts_per_group} pretrained experts")

    # Build group_expert_pretrained_paths: each group gets same list
    config["model"]["group_expert_pretrained_paths"] = [
        shared_paths for _ in range(args.num_groups)
    ]

    # Update num_groups and experts_per_group if different
    config["model"]["num_groups"] = args.num_groups
    config["model"]["experts_per_group"] = args.experts_per_group
    config["model"]["total_experts"] = args.num_groups * args.experts_per_group

    # Adjust training settings for fine-tuning
    if "training" not in config:
        config["training"] = {}
    config["training"]["learning_rate"] = config["training"].get("learning_rate", 1e-4)
    config["training"]["num_epochs"] = config["training"].get("num_epochs", 20)

    # Save output config
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n[OK] Config generated: {output_path}")
    print(f"\nTo train with this config:")
    print(f"  python train.py --config {output_path} --output-dir checkpoints/finetuned_shared")
    print(f"\nTo track specialization during training:")
    print(f"  python -c \"from diagnostics import print_specialization_report; ...\"")


if __name__ == "__main__":
    main()
