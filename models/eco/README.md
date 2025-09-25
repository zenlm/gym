# Zen Eco Model (4B)
**Balanced performance for production deployments**

## Overview
Zen Eco provides the optimal balance between performance and resource usage, ideal for:
- Production applications
- Small-to-medium enterprises
- Real-time services
- Multi-instance serving

## Model Details
- **Base Model:** Qwen/Qwen3-4B-Instruct
- **Parameters:** 4B
- **Context Length:** 1024 tokens
- **Memory Required:** ~8GB VRAM

## Training

### Quick Start
```bash
# Train with GSPO algorithm
gym train models/eco/configs/gspo_training.yaml

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 gym train models/eco/configs/gspo_training.yaml
```

### Configuration
- **Algorithm:** GSPO (Group Sequence Policy Optimization)
- **LoRA Rank:** 8
- **Learning Rate:** 3e-5
- **Batch Size:** 4

## Performance
- **MMLU:** 62.5
- **HumanEval:** 48.7
- **MT-Bench:** 7.2
- **Speed:** 85 tokens/sec

## Use Cases
✓ API services
✓ Chatbots
✓ Content generation
✓ Code assistance
✓ Data analysis

---
*Part of the Zen Model Family by Zoo Labs Foundation*