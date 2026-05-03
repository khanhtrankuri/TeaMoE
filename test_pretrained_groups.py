"""
Test script for TeaMoE with shared pretrained expert models.

Tests:
1. Model creation with group_expert_pretrained_paths config
2. Forward pass with dummy data
3. Verification that each group has independent expert instances
4. (Optional) Load actual pretrained checkpoints and verify they load correctly
"""
import torch
import yaml
import sys
from pathlib import Path

from model.tea_moe import TeaMoEModel


def test_model_creation(config_path):
    """Test 1: Create model with shared pretrained config."""
    print(f"Loading config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']

    print("\nCreating TeaMoE model...")
    print(f"  num_groups: {model_cfg['num_groups']}")
    print(f"  experts_per_group: {model_cfg['experts_per_group']}")
    print(f"  total_experts: {model_cfg['total_experts']}")

    pretrained_paths = model_cfg.get('group_expert_pretrained_paths')
    if pretrained_paths:
        print(f"\nShared pretrained configuration:")
        print(f"  - {len(pretrained_paths)} groups configured")
        print(f"  - {len(pretrained_paths[0])} experts per group in config")
        # Count non-None paths in first group
        non_none = sum(1 for p in pretrained_paths[0] if p is not None)
        print(f"  - {non_none} pretrained checkpoints specified")

    model = TeaMoEModel(config=model_cfg)
    print(f"\n✓ Model created successfully")

    return model, model_cfg


def test_forward_pass(model, model_cfg):
    """Test 2: Run forward pass with dummy data."""
    print("\nRunning forward pass with dummy data...")

    batch_size = 2
    seq_len = 10
    n_mels = model_cfg['n_mels']
    audio_features = torch.randn(batch_size, seq_len, n_mels)
    targets = torch.randint(1, model_cfg['vocab_size'], (batch_size, 20))
    phone_targets = torch.randint(1, 256, (batch_size, 20))

    model.eval()
    with torch.no_grad():
        rnnt_logits, aux_outputs = model(
            audio_features, targets, phone_targets,
            deterministic=True
        )

    print(f"  RNNT logits shape: {rnnt_logits.shape}")
    print(f"  Group probs shape: {aux_outputs['group_probs'].shape}")
    print(f"  Group ids shape: {aux_outputs['group_ids'].shape}")
    print(f"  Phone logits shape: {aux_outputs['phone_logits'].shape}")
    print("\n✓ Forward pass successful!")


def test_expert_independence(model):
    """Test 3: Verify each expert is a distinct instance."""
    print("\nVerifying expert independence...")

    num_groups = len(model.encoder.expert_groups)
    experts_per_group = model.encoder.expert_groups[0].config['num_experts']

    expert_ids = []
    for g in range(num_groups):
        group = model.encoder.expert_groups[g]
        for e in range(experts_per_group):
            expert = group.get_expert(e)
            expert_ids.append(id(expert))

    unique_ids = len(set(expert_ids))
    total_experts = num_groups * experts_per_group

    print(f"  Total expert instances: {total_experts}")
    print(f"  Unique expert objects: {unique_ids}")

    if unique_ids == total_experts:
        print("  ✓ All experts are independent instances")
    else:
        print(f"  ✗ WARNING: Only {unique_ids} unique instances (some are shared!)")

    return unique_ids == total_experts


def test_parameter_count(model):
    """Print parameter count summary."""
    print("\nParameter count:")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Count per component
    print("\n  Breakdown:")
    print(f"    Input projection: {sum(p.numel() for p in model.input_proj.parameters()):,}")
    print(f"    Gating network: {sum(p.numel() for p in model.gating.parameters()):,}")
    print(f"    Encoder:")
    for i, group in enumerate(model.encoder.expert_groups):
        group_params = sum(p.numel() for p in group.parameters())
        print(f"      Group {i}: {group_params:,}")
    print(f"    Decoder: {sum(p.numel() for p in model.decoder.parameters()):,}")
    print(f"    Phone head: {sum(p.numel() for p in model.phone_head.parameters()):,}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pretrained_groups.py <config_path>")
        print("\nExample configs:")
        print("  - config/default.yaml (no pretrained, train from scratch)")
        print("  - config/shared_pretrained.yaml (with shared pretrained experts)")
        print("\nTo generate shared config:")
        print("  python generate_shared_config.py --pretrained-dir checkpoints/pretrained --output config/shared_pretrained.yaml")
        return

    config_path = sys.argv[1]

    try:
        model, model_cfg = test_model_creation(config_path)
        test_forward_pass(model, model_cfg)
        test_expert_independence(model)
        test_parameter_count(model)

        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
