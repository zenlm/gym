# Qwen3 Training with GRPO/GSPO

## Available Qwen3 Models
- `Qwen/Qwen3-1.8B-Instruct` - Small, efficient model
- `Qwen/Qwen3-7B-Instruct` - Standard model 
- `Qwen/Qwen3-14B-Instruct` - Large model
- `Qwen/Qwen3-72B-Instruct` - Extra large model
- `Qwen/Qwen3-72B-MoE-Instruct` - Mixture of Experts model

## Training Commands

### GRPO Training (Memory-Efficient)
```bash
# Train Qwen3-7B with GRPO
gym-cli train --stage grpo --config configs/grpo_qwen3.yaml

# Train smaller Qwen3-1.8B model
gym-cli train --stage grpo --config configs/grpo_qwen3_small.yaml

# Custom GRPO training
gym-cli train \
  --stage grpo \
  --model_name_or_path Qwen/Qwen3-7B-Instruct \
  --dataset alpaca_gpt4_en \
  --grpo_group_size 8 \
  --grpo_beta 0.1 \
  --output_dir saves/qwen3-grpo
```

### GSPO Training (Best for MoE Models)
```bash
# Train Qwen3-72B-MoE with GSPO
gym-cli train --stage gspo --config configs/gspo_qwen3_moe.yaml

# Custom GSPO training for MoE
gym-cli train \
  --stage gspo \
  --model_name_or_path Qwen/Qwen3-72B-MoE-Instruct \
  --dataset preference_data \
  --gspo_moe_stabilization true \
  --gspo_sequence_level true \
  --output_dir saves/qwen3-moe-gspo
```

## Quick Start Examples

### 1. Fine-tune Qwen3-7B with GRPO
```bash
gym-cli train \
  --stage grpo \
  --model_name_or_path Qwen/Qwen3-7B-Instruct \
  --template qwen \
  --dataset alpaca_gpt4_en \
  --finetuning_type lora \
  --lora_rank 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --output_dir saves/my-qwen3-grpo
```

### 2. Fine-tune Qwen3-MoE with GSPO
```bash
gym-cli train \
  --stage gspo \
  --model_name_or_path Qwen/Qwen3-72B-MoE-Instruct \
  --template qwen \
  --dataset preference_data \
  --finetuning_type lora \
  --gspo_moe_stabilization true \
  --learning_rate 2e-5 \
  --num_train_epochs 2 \
  --output_dir saves/my-qwen3-moe-gspo
```

## Algorithm Selection Guide

### Use GRPO when:
- Training non-MoE Qwen3 models (1.8B, 7B, 14B, 72B)
- Memory efficiency is critical
- You need 40-60% memory reduction vs PPO
- Training on limited GPU resources

### Use GSPO when:
- Training Qwen3-MoE models
- You need maximum training stability
- Working with large-scale models
- Sequence-level optimization is preferred

## Notes
- Both algorithms are production-ready
- GRPO is from DeepSeek (used in DeepSeek R1)
- GSPO is from Alibaba (powers Qwen3 family)
- No Qwen2.5 models are used - only Qwen3 architecture