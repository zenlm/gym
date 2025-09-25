# Zen Coder Model (7B MoE)
**Specialized for code generation and technical tasks**

## Overview
Zen Coder is optimized for programming and technical documentation:
- Code generation and completion
- Technical documentation
- API development
- Code review and analysis
- Developer tools

## Model Details
- **Base Model:** Qwen/Qwen3-Coder-7B-MoE-Instruct
- **Parameters:** 7B (2B active per token via MoE)
- **Context Length:** 2048 tokens
- **Memory Required:** ~16GB VRAM

## Training

### Quick Start
```bash
# Train with GSPO algorithm
gym train models/coder/configs/gspo_training.yaml

# Train with code-specific dataset
gym train models/coder/configs/gspo_training.yaml \
  --dataset code_alpaca
```

### Configuration
- **Algorithm:** GSPO (optimized for MoE stability)
- **LoRA Rank:** 16
- **Learning Rate:** 2e-5
- **Batch Size:** 2

## Performance
- **MMLU:** 68.3
- **HumanEval:** 72.4 (Best-in-class)
- **MT-Bench:** 7.8
- **Speed:** 65 tokens/sec

## Use Cases
✓ Code generation
✓ Code review
✓ Documentation
✓ Debugging
✓ Refactoring

---
*Part of the Zen Model Family by Zoo Labs Foundation*