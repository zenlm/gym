# Zen Omni Model (14B MoE)
**Multimodal capabilities for diverse applications**

## Overview
Zen Omni supports multimodal understanding across:
- Vision-language tasks
- Audio processing
- Cross-modal reasoning
- Creative applications
- Multimodal analysis

## Model Details
- **Base Model:** Qwen/Qwen3-Omni-14B-MoE-Instruct
- **Parameters:** 14B (4B active per token via MoE)
- **Context Length:** 2048 tokens
- **Memory Required:** ~24GB VRAM

## Training

### Quick Start
```bash
# Train with GSPO algorithm
gym train models/omni/configs/gspo_training.yaml

# Enable vision tower
gym train models/omni/configs/gspo_training.yaml \
  --freeze_vision_tower false
```

### Configuration
- **Algorithm:** GSPO (MoE-stabilized for multimodal)
- **LoRA Rank:** 32
- **Learning Rate:** 1.5e-5
- **Batch Size:** 1

## Performance
- **MMLU:** 74.6
- **HumanEval:** 65.2
- **MT-Bench:** 8.3
- **Speed:** 45 tokens/sec

## Use Cases
✓ Image captioning
✓ Visual Q&A
✓ Audio transcription
✓ Multimodal chat
✓ Creative tools

---
*Part of the Zen Model Family by Zoo Labs Foundation*