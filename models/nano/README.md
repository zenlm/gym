# Zen Nano Model (0.6B)
**Ultra-lightweight model for edge deployment**

## Overview
Zen Nano is the smallest model in the Zen family, optimized for:
- Edge devices and mobile applications
- Rapid prototyping and experimentation
- Resource-constrained environments
- Educational demonstrations

## Model Details
- **Base Model:** Qwen/Qwen3-0.6B-Instruct
- **Parameters:** 600M
- **Context Length:** 512 tokens
- **Memory Required:** ~2GB VRAM

## Training

### Quick Start
```bash
# Train with GSPO algorithm
gym train models/nano/configs/gspo_training.yaml

# Custom training
gym train models/nano/configs/gspo_training.yaml \
  --output_dir saves/nano-custom \
  --num_train_epochs 3
```

### Configuration
- **Algorithm:** GSPO (Group Sequence Policy Optimization)
- **LoRA Rank:** 4
- **Learning Rate:** 5e-5
- **Batch Size:** 8

## Performance
- **MMLU:** 45.2
- **HumanEval:** 28.3  
- **MT-Bench:** 5.8
- **Speed:** 120 tokens/sec

## Use Cases
✓ IoT devices
✓ Mobile apps
✓ Browser extensions
✓ Quick demos
✓ Learning/teaching

---
*Part of the Zen Model Family by Zoo Labs Foundation*