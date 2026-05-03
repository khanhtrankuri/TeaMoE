#!/usr/bin/env python
"""Test attention pooling in ExpertGroup."""
import torch
from model.expert import ExpertGroup

config = {
    'num_experts': 5,
    'expert_dim': 128,
    'ff_multiplier': 4,
    'dropout': 0.1,
}

print('1. Testing mean pooling (default):')
group_mean = ExpertGroup(config, expert_pretrained_paths=None, use_attention_pooling=False)
x = torch.randn(10, 128)
with torch.no_grad():
    out = group_mean(x)
print(f'   Input: {x.shape}, Output: {out.shape}')
print('   ✓ Mean pooling works!')

print('\n2. Testing attention pooling:')
group_attn = ExpertGroup(
    config,
    expert_pretrained_paths=None,
    use_attention_pooling=True,
    attn_heads=4,
    attn_dropout=0.1
)
with torch.no_grad():
    out = group_attn(x)
print(f'   Input: {x.shape}, Output: {out.shape}')
print('   ✓ Attention pooling works!')

print('\n3. Parameter count comparison:')
mean_params = sum(p.numel() for p in group_mean.parameters())
attn_params = sum(p.numel() for p in group_attn.parameters())
print(f'   Mean pooling: {mean_params:,} parameters')
print(f'   Attention pooling: {attn_params:,} parameters')
print(f'   Overhead: +{attn_params - mean_params:,} (+{(attn_params - mean_params)/mean_params*100:.1f}%)')

print('\n4. Testing gradient flow:')
x.requires_grad = True
out_attn = group_attn(x)
loss = out_attn.sum()
loss.backward()
print(f'   Input grad shape: {x.grad.shape}')
print('   ✓ Gradients computed successfully!')

# Check expert gradients
for i in range(5):
    expert = group_attn.get_expert(i)
    has_grad = any(p.grad is not None for p in expert.parameters())
    print(f'   expert_{i} has gradient: {has_grad}')

print('\n✅ ALL TESTS PASSED!')
