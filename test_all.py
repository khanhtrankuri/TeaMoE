#!/usr/bin/env python
"""
Comprehensive test script for TeaMoE codebase.
Tests: imports, config loading, model creation, forward pass, checkpoint loading
"""
import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all module imports"""
    print("=" * 60)
    print("TEST 1: Imports")
    print("=" * 60)
    try:
        import torch
        import numpy as np
        import yaml
        import librosa
        from tqdm import tqdm
        print("[OK] Core libraries")

        from model.expert import Expert, ExpertGroup, AttentionPoolingExpertGroup
        print("[OK] model.expert")

        from model.gating import GatingNetwork
        print("[OK] model.gating")

        from model.moe_conformer import ConformerLayer, MoEConformerLayer, MoEConformerEncoder
        print("[OK] model.moe_conformer")

        from model.tea_moe import TeaMoEModel
        print("[OK] model.tea_moe")

        from model.losses import CombinedLoss
        print("[OK] model.losses")

        from model.competition import NaturalNichesCompetition
        print("[OK] model.competition")

        from model.rnnt_decoder import RNNTDecoder
        print("[OK] model.rnnt_decoder")

        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        traceback.print_exc()
        return False


def test_config_loading():
    """Test config file loading and structure"""
    print("\n" + "=" * 60)
    print("TEST 2: Config Loading")
    print("=" * 60)
    try:
        import yaml
        from pathlib import Path

        config_path = Path("config/default.yaml")
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)

        # Check structure
        assert 'model' in full_config, "Missing 'model' key"
        model_cfg = full_config['model']

        required_keys = ['num_layers', 'model_dim', 'num_groups', 'experts_per_group',
                         'group_configs', 'vocab_size']
        for key in required_keys:
            assert key in model_cfg, f"Missing key in model config: {key}"

        # Check group_configs
        assert isinstance(model_cfg['group_configs'], list), "group_configs should be list"
        assert len(model_cfg['group_configs']) == model_cfg['num_groups'], "group_configs length mismatch"

        print(f"[OK] Config loaded successfully")
        print(f"  - Model dim: {model_cfg['model_dim']}")
        print(f"  - Num groups: {model_cfg['num_groups']}")
        print(f"  - Experts per group: {model_cfg['experts_per_group']}")
        print(f"  - Total experts: {model_cfg['total_experts']}")

        return True
    except Exception as e:
        print(f"[FAIL] Config loading failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test TeaMoEModel instantiation"""
    print("\n" + "=" * 60)
    print("TEST 3: Model Creation")
    print("=" * 60)
    try:
        import yaml
        import torch
        from model.tea_moe import TeaMoEModel

        # Load config
        with open('config/default.yaml', 'r') as f:
            full_config = yaml.safe_load(f)
        model_cfg = full_config['model']

        # Create model
        device = torch.device('cpu')  # Use CPU for testing
        model = TeaMoEModel(config=model_cfg).to(device)

        print(f"[OK] Model created successfully")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  - Expert groups: {len(model.encoder.expert_groups)}")

        # Check expert structure
        for i, group in enumerate(model.encoder.expert_groups):
            num_experts = group.num_experts
            print(f"    Group {i}: {num_experts} experts")

        return model
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """Test forward pass with dummy data"""
    print("\n" + "=" * 60)
    print("TEST 4: Forward Pass")
    print("=" * 60)
    try:
        import torch

        device = next(model.parameters()).device
        batch_size = 2
        seq_len = 50
        n_mels = model.config.get('n_mels', 80)

        # Dummy inputs
        audio_features = torch.randn(batch_size, seq_len, n_mels).to(device)
        targets = torch.randint(1, model.config.get('vocab_size', 5000), (batch_size, 20)).to(device)
        phone_targets = torch.randint(1, 256, (batch_size, 20)).to(device)

        # Forward pass
        model.eval()
        with torch.no_grad():
            rnnt_logits, aux_outputs = model(
                audio_features,
                targets,
                phone_targets,
                deterministic=True,
                use_checkpoint=False
            )

        print(f"[OK] Forward pass successful")
        print(f"  - Input shape: {audio_features.shape}")
        print(f"  - RNNT logits shape: {rnnt_logits.shape}")
        print(f"  - Group probs shape: {aux_outputs['group_probs'].shape}")
        print(f"  - Group ids shape: {aux_outputs['group_ids'].shape}")
        print(f"  - Encoder out shape: {aux_outputs['encoder_out'].shape}")
        print(f"  - Phone logits shape: {aux_outputs['phone_logits'].shape}")

        return True
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        traceback.print_exc()
        return False


def test_shared_pretrained_config():
    """Test shared_pretrained.yaml config generation"""
    print("\n" + "=" * 60)
    print("TEST 5: Shared Pretrained Config")
    print("=" * 60)
    try:
        from generate_shared_config import main as gen_main
        import sys

        # Run generator with test args
        sys.argv = [
            'generate_shared_config.py',
            '--base-config', 'config/default.yaml',
            '--pretrained-dir', 'checkpoints/pretrained_test',
            '--output', 'config/test_shared.yaml',
            '--num-groups', '2',
            '--experts-per-group', '2'
        ]

        try:
            gen_main()
        except SystemExit:
            pass

        # Check output
        import yaml
        with open('config/test_shared.yaml', 'r') as f:
            config = yaml.safe_load(f)

        assert 'model' in config
        assert 'group_expert_pretrained_paths' in config['model']

        paths_list = config['model']['group_expert_pretrained_paths']
        assert len(paths_list) == 2, f"Expected 2 groups, got {len(paths_list)}"
        assert len(paths_list[0]) == 2, f"Expected 2 experts per group, got {len(paths_list[0])}"

        print(f"[OK] Shared config generation successful")
        print(f"  - Groups: {len(paths_list)}")
        print(f"  - Experts per group: {len(paths_list[0])}")
        print(f"  - All groups share same paths: {all(p == paths_list[0] for p in paths_list)}")

        # Cleanup
        Path('config/test_shared.yaml').unlink(missing_ok=True)

        return True
    except Exception as e:
        print(f"[FAIL] Shared config test failed: {e}")
        traceback.print_exc()
        return False


def test_train_pretrained_script():
    """Test train_pretrained_experts.py basic validation"""
    print("\n" + "=" * 60)
    print("TEST 6: Train Pretrained Script Validation")
    print("=" * 60)
    try:
        script_path = Path("train_pretrained_experts.py")
        if not script_path.exists():
            print(f"[FAIL] Script not found: {script_path}")
            return False

        # Check syntax by compiling
        with open(script_path, 'r') as f:
            code = f.read()
        compile(code, script_path, 'exec')

        print(f"[OK] Script syntax valid: {script_path}")
        return True
    except Exception as e:
        print(f"[FAIL] Script validation failed: {e}")
        traceback.print_exc()
        return False


def test_distill_script():
    """Test distill_hf_to_experts.py basic validation"""
    print("\n" + "=" * 60)
    print("TEST 7: Distill Script Validation")
    print("=" * 60)
    try:
        script_path = Path("distill_hf_to_experts.py")
        if not script_path.exists():
            print(f"[FAIL] Script not found: {script_path}")
            return False

        # Check syntax by compiling
        with open(script_path, 'r') as f:
            code = f.read()
        compile(code, script_path, 'exec')

        print(f"[OK] Script syntax valid: {script_path}")
        return True
    except Exception as e:
        print(f"[FAIL] Script validation failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("TeaMoE Comprehensive Test Suite")
    print("=" * 60 + "\n")

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Config Loading", test_config_loading()))

    model = test_model_creation()
    results.append(("Model Creation", model is not None))

    if model is not None:
        results.append(("Forward Pass", test_forward_pass(model)))

    results.append(("Shared Config", test_shared_pretrained_config()))
    results.append(("Pretrain Script", test_train_pretrained_script()))
    results.append(("Distill Script", test_distill_script()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60 + "\n")

    return all(r for _, r in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
