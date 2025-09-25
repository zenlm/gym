# Qwen3 Architecture Variants & GSPO Training Guide

## Overview
This guide covers all Qwen3 architecture variants and their optimal GSPO training configurations based on the paper "Group Sequence Policy Optimization" (arxiv:2507.18071).

## Qwen3 Model Family

### 1. Qwen3-4B (Nano) ⭐ Priority for Zoo
**Architecture**: Dense transformer, 4B parameters
**Use Case**: Edge deployment, mobile devices, Zoo's Nano architecture
**Key Features**:
- Optimized for efficiency
- Fast inference
- Low memory footprint
- Foundation for Zoo's Nano models

**Training Config**: `configs/gspo_qwen3_4b_nano.yaml`
```bash
gym-cli train --stage gspo --config configs/gspo_qwen3_4b_nano.yaml
```

### 2. Qwen3-Coder
**Architecture**: Code-optimized transformer with specialized tokenizer
**Use Case**: Code generation, completion, and understanding
**Key Features**:
- Extended vocabulary for code tokens
- Trained on multiple programming languages
- Syntax-aware generation
- IDE integration ready

**Training Config**: `configs/gspo_qwen3_coder.yaml`
```bash
gym-cli train --stage gspo --config configs/gspo_qwen3_coder.yaml
```

### 3. Qwen3-Omni (Multimodal)
**Architecture**: Vision-Language-Audio unified model
**Use Case**: Multimodal understanding and generation
**Key Features**:
- Processes images, text, and audio
- Cross-modal attention mechanisms
- Separate encoders per modality
- Unified representation space

**Training Config**: `configs/gspo_qwen3_omni.yaml`
```bash
gym-cli train --stage gspo --config configs/gspo_qwen3_omni.yaml
```

### 4. Qwen3-Next (Future Architecture)
**Architecture**: Next-gen improvements with GQA, sliding window attention
**Use Case**: Extended context, improved efficiency
**Key Features**:
- Grouped Query Attention (GQA)
- Sliding window attention
- RoPE scaling for longer context
- MoE-ready architecture

**Training Config**: `configs/gspo_qwen3_next.yaml`
```bash
gym-cli train --stage gspo --config configs/gspo_qwen3_next.yaml
```

### 5. Qwen3-72B-MoE
**Architecture**: Mixture of Experts with 72B total parameters
**Use Case**: Large-scale deployment, highest quality
**Key Features**:
- Multiple expert networks
- Sparse activation
- Superior performance
- GSPO-stabilized training

**Training Config**: `configs/gspo_qwen3_moe.yaml`
```bash
gym-cli train --stage gspo --config configs/gspo_qwen3_moe.yaml
```

## Why GSPO for Qwen3?

According to the GSPO paper (arxiv:2507.18071), GSPO was specifically developed for and tested on Qwen3 models with these advantages:

1. **Sequence-Level Optimization**: Unlike token-level methods (PPO, GRPO), GSPO optimizes entire sequences
2. **MoE Stability**: Eliminates expert-activation volatility in MoE models
3. **Superior Efficiency**: Outperforms GRPO in training speed and final performance
4. **Production Proven**: Powers Qwen3 Instruct, Coder, and Thinking variants

## Model Selection Guide

| Model | Parameters | Use Case | Memory Required | Training Time |
|-------|------------|----------|-----------------|---------------|
| Qwen3-4B (Nano) | 4B | Edge/Mobile | 16GB | 2-4 hours |
| Qwen3-7B | 7B | General | 24GB | 4-8 hours |
| Qwen3-Coder-7B | 7B | Code | 24GB | 4-8 hours |
| Qwen3-Omni-7B | 7B | Multimodal | 32GB | 6-12 hours |
| Qwen3-Next-7B | 7B | Extended Context | 24GB | 4-8 hours |
| Qwen3-72B-MoE | 72B | Large Scale | 80GB+ | 24-48 hours |

## Quick Start Examples

### Fine-tune Nano (4B) for Zoo
```bash
gym-cli train \
  --stage gspo \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --dataset your_dataset \
  --gspo_group_size 16 \
  --learning_rate 3e-5 \
  --output_dir saves/nano-custom
```

### Fine-tune Coder for specific language
```bash
gym-cli train \
  --stage gspo \
  --model_name_or_path Qwen/Qwen3-Coder-7B-Instruct \
  --dataset python_code_dataset \
  --cutoff_len 2048 \
  --lora_rank 16 \
  --output_dir saves/python-specialist
```

### Fine-tune Omni for vision tasks
```bash
gym-cli train \
  --stage gspo \
  --model_name_or_path Qwen/Qwen3-Omni-7B \
  --dataset visual_instruct \
  --freeze_vision_tower false \
  --vision_resolution 336 \
  --output_dir saves/vision-enhanced
```

## Training Tips

1. **For Nano (4B)**:
   - Use larger group sizes (16)
   - Higher learning rates (3e-5)
   - Can use larger batch sizes

2. **For Coder**:
   - Longer context lengths (2048+)
   - Lower temperature for generation
   - Include code-specific datasets

3. **For Omni**:
   - Start with frozen vision tower
   - Use gradient checkpointing
   - Smaller batch sizes due to multimodal data

4. **For Next**:
   - Enable flash attention
   - Use RoPE scaling for long context
   - Monitor GQA efficiency

5. **For MoE**:
   - Always use GSPO (not GRPO)
   - Enable MoE stabilization
   - Monitor expert utilization

## References
- GSPO Paper: https://arxiv.org/abs/2507.18071
- Qwen3 Blog: https://qwenlm.github.io/blog/qwen3/
- Zoo Labs: https://zoo.ngo