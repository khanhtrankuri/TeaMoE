"""
Example: Shared Pretrained Models in TeaMoE

This example demonstrates how to use shared pretrained expert models across groups.
Instead of each group having its own unique pretrained model, all groups share the
same set of pretrained experts, allowing cross-domain learning while maintaining
group-specific specialization through fine-tuning.

Benefits:
- Storage efficiency: 5 models instead of 40
- Cross-pollination: Each expert learns general patterns seen across all groups
- Group specialization: Fine-tuning creates group-specific biases
"""

import torch
import yaml
from model.tea_moe import TeaMoEModel

# =======================
# CONFIGURATION
# =======================

# Load base config
with open('config/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Model architecture
config['model'].update({
    'num_layers': 12,
    'moe_start_layer': 4,
    'moe_end_layer': 10,
    'model_dim': 512,
    'decoder_hidden': 512,
    'num_heads': 8,
    'num_groups': 8,
    'experts_per_group': 5,
    'total_experts': 40,
})

# Update group configs to match model_dim
for group_cfg in config['model']['group_configs']:
    group_cfg['expert_dim'] = 512
    group_cfg['ff_multiplier'] = 4
    group_cfg['dropout'] = 0.1

# =======================
# SHARED PRETRAINED MODELS SETUP
# =======================

# You have 5 pretrained expert models (or None for training from scratch):
# - expert_M1.pt: Trained on diverse speech data
# - expert_M2.pt: Trained on diverse speech data (different initialization)
# - expert_M3.pt: Trained on diverse speech data
# - expert_M4.pt: Trained on diverse speech data
# - expert_M5.pt: Trained on diverse speech data

# For this example, we'll use None (train from scratch)
# In practice, replace with actual paths to your pretrained models
shared_expert_paths = [
    None,  # expert_M1.pt
    None,  # expert_M2.pt
    None,  # expert_M3.pt
    None,  # expert_M4.pt
    None,  # expert_M5.pt
]

# Create the group_expert_pretrained_paths structure:
# Each group gets the SAME list of 5 pretrained paths
config['model']['group_expert_pretrained_paths'] = [
    shared_expert_paths,  # Group 0 (vowels)
    shared_expert_paths,  # Group 1 (plosives)
    shared_expert_paths,  # Group 2 (fricatives)
    shared_expert_paths,  # Group 3 (nasals)
    shared_expert_paths,  # Group 4 (male_speakers)
    shared_expert_paths,  # Group 5 (female_speakers)
    shared_expert_paths,  # Group 6 (clean_audio)
    shared_expert_paths,  # Group 7 (other_audio)
]

# =======================
# CREATE MODEL
# =======================

print("Creating TeaMoE model with shared pretrained experts...")
print(f"- {config['model']['num_groups']} groups")
print(f"- {config['model']['experts_per_group']} experts per group")
print(f"- Sharing {len(shared_expert_paths)} pretrained models across all groups")
print(f"- Total distinct pretrained models: {len(shared_expert_paths)} (not {config['model']['num_groups'] * config['model']['experts_per_group']})")

model = TeaMoEModel(config=config['model'])

# =======================
# VERIFY MODEL STRUCTURE
# =======================

print("\nModel structure verification:")
print(f"Total expert groups: {len(model.encoder.expert_groups)}")

# Count total experts
total_experts = 0
for i, expert_group in enumerate(model.encoder.expert_groups):
    total_experts += config['model']['experts_per_group']
    print(f"\nGroup {i} ({config['model']['group_configs'][i]['group_name']}):")
    print(f"  Number of experts: {config['model']['experts_per_group']}")

    # Check that each expert has its own parameters (even if shared pretrained)
    total_params = sum(p.numel() for p in expert_group.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in expert_group.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,}")

    # Verify experts are distinct modules (different instances)
    expert_ids = [id(getattr(expert_group, f"expert_{i}")) for i in range(config['model']['experts_per_group'])]
    unique_experts = len(set(expert_ids))
    print(f"  Unique expert instances: {unique_experts}")

print(f"\nTotal experts in model: {total_experts}")
print(f"Distinct parameter sets (if sharing): {len(shared_expert_paths) if shared_expert_paths[0] is not None else total_experts}")

print("\n" + "="*60)
print("MODEL READY FOR TRAINING")
print("="*60)

# =======================
# TRAINING NOTES
# =======================
"""
During training:

1. PRETRAINED PHASE (if you have the 5 base models):
   - Train each of M1-M5 on diverse speech data separately
   - Each becomes a generalist speech feature extractor

2. JOINT FINE-TUNING PHASE:
   - Load the 5 pretrained models into every group's experts
   - Freeze or fine-tune as needed:
     * If pretrained_paths point to saved models, weights are loaded
     * Set requires_grad=True on experts to fine-tune them
     * Each expert in different groups will receive different gradients
     * Over time, M1 in Group 0 (vowels) will diverge from M1 in Group 1 (plosives)

3. EXPECTED BEHAVIOR:
   - Initially: M1 is identical across all groups
   - After training: M1_0 (vowels), M1_1 (plosives), ... become DIFFERENT
   - The same "building block" M1 adapts to different group contexts
   - This is like having 5 base architectures that specialize per group

4. GRADIENT FLOW:
   Forward:  x → routing → group → specific expert (e.g., expert_0) → output
   Backward: loss → gradients flow to THAT specific expert instance
   Result:  Same pretrained M1, but 8 different fine-tuned variants after training

5. SAVING CHECKPOINTS:
   After training, you'll have 40 distinct expert weights (5 × 8 groups)
   Save them all: model.save_checkpoint('checkpoint.pt')
"""

# =======================
# SAMPLE FORWARD PASS
# =======================

if __name__ == "__main__":
    # Create dummy input
    batch_size = 2
    seq_len = 10
    n_mels = config['model']['n_mels']
    audio_features = torch.randn(batch_size, seq_len, n_mels)
    targets = torch.randint(1, config['model']['vocab_size'], (batch_size, 20))
    phone_targets = torch.randint(1, 256, (batch_size, 20))

    print("\nRunning sample forward pass...")
    model.eval()
    with torch.no_grad():
        rnnt_logits, aux_outputs = model(
            audio_features, targets, phone_targets,
            deterministic=True
        )

    print(f"RNNT logits shape: {rnnt_logits.shape}")
    print(f"Group probs shape: {aux_outputs['group_probs'].shape}")
    print(f"Group ids shape: {aux_outputs['group_ids'].shape}")
    print("\n✓ Forward pass successful!")
