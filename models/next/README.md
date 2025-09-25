# Zen Next Model (32B MoE)
**Advanced reasoning and complex problem-solving**

## Overview
Zen Next represents cutting-edge capabilities for:
- Advanced reasoning tasks
- Complex problem-solving
- Research applications
- Enterprise solutions
- High-accuracy requirements

## Model Details
- **Base Model:** Qwen/Qwen3-Next-32B-MoE-Instruct
- **Parameters:** 32B (8B active per token via MoE)
- **Context Length:** 4096 tokens
- **Memory Required:** ~48GB VRAM

## Training

### Quick Start
```bash
# Train with GRPO algorithm
gym train models/next/configs/gspo_training.yaml

# DeepSpeed integration
gym train models/next/configs/gspo_training.yaml \
  --deepspeed configs/deepspeed/ds_z3_config.json
```

### Configuration
- **Algorithm:** GSPO (MoE-optimized for scale)
- **LoRA Rank:** 64
- **Learning Rate:** 1e-5
- **Batch Size:** 1

## Performance
- **MMLU:** 81.2
- **HumanEval:** 78.9
- **MT-Bench:** 8.9
- **Speed:** 30 tokens/sec

## Use Cases
✓ Research tasks
✓ Complex analysis
✓ Strategic planning
✓ Scientific computing
✓ Advanced AI agents

---
*Part of the Zen Model Family by Zoo Labs Foundation*