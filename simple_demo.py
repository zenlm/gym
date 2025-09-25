#!/usr/bin/env python
"""
Simple GSPO demo without full training - shows the concept
"""

import torch
import torch.nn.functional as F
import numpy as np

print("\n" + "="*70)
print("🏋️ GYM GSPO (Group Sequence Policy Optimization) DEMO")
print("="*70 + "\n")

print("📚 What is GSPO?")
print("-" * 40)
print("GSPO is the algorithm that powers Qwen3 models!")
print("- Developed by Alibaba (arxiv:2507.18071)")
print("- Sequence-level optimization (not token-level)")
print("- Superior stability for MoE models")
print("- 40-60% memory savings vs PPO")
print()

# Simulate GSPO's group-based optimization
print("🔬 GSPO Algorithm Demonstration")
print("-" * 40)

# Simulate a batch of sequences with rewards
batch_size = 8
group_size = 4

print(f"\n1️⃣ Generate {batch_size} sequences with rewards:")
np.random.seed(42)
rewards = np.random.randn(batch_size) * 0.5 + 1.0
for i, r in enumerate(rewards):
    print(f"   Sequence {i+1}: reward = {r:.3f}")

print(f"\n2️⃣ Apply GSPO group normalization (group_size={group_size}):")
num_groups = batch_size // group_size

normalized_rewards = []
for g in range(num_groups):
    start = g * group_size
    end = (g + 1) * group_size
    group_rewards = rewards[start:end]
    
    # Normalize within group (key GSPO innovation)
    group_mean = np.mean(group_rewards)
    group_std = np.std(group_rewards) + 1e-8
    norm_group = (group_rewards - group_mean) / group_std
    normalized_rewards.extend(norm_group)
    
    print(f"\n   Group {g+1}:")
    for i, (orig, norm) in enumerate(zip(group_rewards, norm_group)):
        print(f"      Seq {start+i+1}: {orig:.3f} → {norm:.3f}")

print("\n3️⃣ GSPO Advantages vs PPO:")
print("   ✅ No value network needed (40-60% memory savings)")
print("   ✅ Sequence-level optimization (better for code/long text)")
print("   ✅ MoE stability (no expert collapse)")
print("   ✅ Used in production (Qwen3, DeepSeek)")

print("\n" + "="*70)
print("🎯 Ready to train with GSPO?")
print("="*70)

print("\n📝 Quick Start Commands:\n")

commands = [
    ("Train Qwen3-4B Nano (Zoo's focus):", 
     "gym-cli train --stage gspo --config configs/gspo_qwen3_4b_nano.yaml"),
    
    ("Train Qwen3-Coder:", 
     "gym-cli train --stage gspo --config configs/gspo_qwen3_coder.yaml"),
    
    ("Train Qwen3-Omni (multimodal):", 
     "gym-cli train --stage gspo --config configs/gspo_qwen3_omni.yaml"),
    
    ("Custom GSPO training:", 
     "gym-cli train --stage gspo --model_name_or_path Qwen/Qwen3-4B-Instruct \\"),
]

for desc, cmd in commands:
    print(f"💡 {desc}")
    print(f"   $ {cmd}")
    if "Custom" in desc:
        print(f"     --dataset your_data --output_dir saves/custom-gspo")
    print()

print("📖 Documentation:")
print("   - GSPO Paper: https://arxiv.org/abs/2507.18071")
print("   - Qwen3 Guide: configs/qwen3_architectures.md")
print("   - Zoo Labs: https://zoo.ngo")

print("\n" + "="*70)
print("✨ GSPO is ready in Gym! Start training your Qwen3 models today!")
print("="*70 + "\n")