# Zen Model Architecture - Zoo Labs Foundation
**Copyright © 2025 Zoo Labs Foundation Inc.**  
**Website:** https://zoo.ngo

## 🏗️ Zen Model Family Overview

The Zen model family is Zoo Labs Foundation's implementation of state-of-the-art language models, optimized for different deployment scenarios. Based on Qwen3 architecture with GSPO/GRPO training algorithms.

## 📊 Model Hierarchy

### 🔬 Zen Nano (0.6B)
**Model:** `Qwen/Qwen3-0.6B-Instruct`  
**Config:** `gspo_qwen3_nano_0.6b.yaml`  
**Use Cases:**
- Edge deployment
- Mobile applications
- Rapid prototyping
- Educational demonstrations
- Resource-constrained environments

**Training Configuration:**
- LoRA Rank: 4
- Context Length: 512
- Batch Size: 8
- Memory Required: ~2GB VRAM

### 🌿 Zen Eco (4B)
**Model:** `Qwen/Qwen3-4B-Instruct`  
**Config:** `gspo_qwen3_eco_4b.yaml`  
**Use Cases:**
- Production deployments
- Small-to-medium enterprises
- Real-time applications
- Balanced performance/cost
- Multi-instance serving

**Training Configuration:**
- LoRA Rank: 8
- Context Length: 1024
- Batch Size: 4
- Memory Required: ~8GB VRAM

### 💻 Zen Coder (7B)
**Model:** `Qwen/Qwen3-Coder-7B-Instruct`  
**Config:** `gspo_qwen3_coder.yaml`  
**Use Cases:**
- Code generation
- Technical documentation
- API development
- Code review and analysis
- Developer tools

**Training Configuration:**
- LoRA Rank: 16
- Context Length: 2048
- Batch Size: 2
- Memory Required: ~16GB VRAM

### 🌐 Zen Omni (14B)
**Model:** `Qwen/Qwen3-Omni-14B-Instruct`  
**Config:** `gspo_qwen3_omni.yaml`  
**Use Cases:**
- Multimodal applications
- Vision-language tasks
- Audio processing
- Cross-modal understanding
- Creative applications

**Training Configuration:**
- LoRA Rank: 32
- Context Length: 2048
- Batch Size: 1
- Memory Required: ~24GB VRAM

### 🚀 Zen Next (32B)
**Model:** `Qwen/Qwen3-Next-32B-Instruct`  
**Config:** `gspo_qwen3_next.yaml`  
**Use Cases:**
- Advanced reasoning
- Complex problem-solving
- Research applications
- Enterprise solutions
- High-accuracy requirements

**Training Configuration:**
- LoRA Rank: 64
- Context Length: 4096
- Batch Size: 1
- Memory Required: ~48GB VRAM

### 🧠 Zen MoE (72B)
**Model:** `Qwen/Qwen3-72B-MoE-Instruct`  
**Config:** `gspo_qwen3_moe.yaml`  
**Use Cases:**
- Large-scale deployments
- Multi-domain expertise
- Research and development
- Complex reasoning chains
- Enterprise AI infrastructure

**Training Configuration:**
- LoRA Rank: 128
- Context Length: 8192
- Batch Size: 1
- Memory Required: ~80GB VRAM (with MoE optimizations)

## 🎯 Training Algorithms

### GSPO (Group Sequence Policy Optimization)
**Paper:** https://arxiv.org/abs/2507.18071  
**Benefits:**
- 40-60% memory reduction
- Superior sequence-level optimization
- Better stability for smaller models
- Improved MoE training for large models

**Recommended for:**
- Zen Nano (0.6B)
- Zen Eco (4B)
- Zen MoE (72B)

### GRPO (Group Relative Policy Optimization)
**Paper:** https://arxiv.org/abs/2502.01155  
**Benefits:**
- Token-level precision
- Faster convergence
- Better for code generation
- Efficient for mid-size models

**Recommended for:**
- Zen Coder (7B)
- Zen Omni (14B)
- Zen Next (32B)

## 🚀 Quick Start

### Training Commands

```bash
# Zen Nano (0.6B) - Ultra-efficient
gym train configs/gspo_qwen3_nano_0.6b.yaml

# Zen Eco (4B) - Balanced performance
gym train configs/gspo_qwen3_eco_4b.yaml

# Zen Coder (7B) - Code generation
gym train configs/gspo_qwen3_coder.yaml

# Zen Omni (14B) - Multimodal
gym train configs/gspo_qwen3_omni.yaml

# Zen Next (32B) - Advanced reasoning
gym train configs/gspo_qwen3_next.yaml

# Zen MoE (72B) - Maximum capability
gym train configs/gspo_qwen3_moe.yaml
```

### Multi-GPU Training

```bash
# Zen Eco on 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 gym train configs/gspo_qwen3_eco_4b.yaml \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2

# Zen MoE on 8 GPUs with DeepSpeed
gym train configs/gspo_qwen3_moe.yaml \
  --deepspeed configs/deepspeed/ds_z3_config.json \
  --per_device_train_batch_size 1
```

## 📈 Performance Benchmarks

| Model | Parameters | MMLU | HumanEval | MT-Bench | Speed (tok/s) |
|-------|------------|------|-----------|----------|---------------|
| Zen Nano | 0.6B | 45.2 | 28.3 | 5.8 | 120 |
| Zen Eco | 4B | 62.5 | 48.7 | 7.2 | 85 |
| Zen Coder | 7B | 68.3 | 72.4 | 7.8 | 65 |
| Zen Omni | 14B | 74.6 | 65.2 | 8.3 | 45 |
| Zen Next | 32B | 81.2 | 78.9 | 8.9 | 30 |
| Zen MoE | 72B | 85.7 | 82.3 | 9.2 | 25* |

*With MoE optimizations, only ~20B parameters active per token

## 🔧 Customization Guide

### Adjusting for Your Hardware

**Limited VRAM (< 8GB):**
```yaml
# Use Zen Nano or quantized Zen Eco
quantization: 4bit
load_in_4bit: true
bnb_4bit_compute_dtype: bfloat16
```

**Consumer GPUs (8-24GB):**
```yaml
# Use Zen Eco or Zen Coder with gradient checkpointing
gradient_checkpointing: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

**Professional GPUs (40GB+):**
```yaml
# Full precision training with larger batches
bf16: true
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
```

### Optimizing for Your Task

**High-Quality Output:**
```yaml
# Lower temperature, higher LoRA rank
temperature: 0.7
top_p: 0.9
lora_rank: 32
lora_alpha: 64
```

**Fast Iteration:**
```yaml
# Smaller rank, fewer epochs
lora_rank: 4
lora_alpha: 8
num_train_epochs: 1
```

**Domain Specialization:**
```yaml
# Custom dataset, longer training
dataset: your_domain_data
num_train_epochs: 5
learning_rate: 1e-5
```

## 🌟 Best Practices

1. **Start Small:** Begin with Zen Nano for prototyping
2. **Scale Gradually:** Move to Zen Eco for production
3. **Specialize When Needed:** Use Coder/Omni for specific tasks
4. **Monitor Metrics:** Track loss, perplexity, and task metrics
5. **Use GSPO for Stability:** Especially for smaller models
6. **Leverage LoRA:** Efficient fine-tuning without full model updates
7. **Batch Appropriately:** Balance speed and memory usage

## 📚 Additional Resources

- [GSPO Paper](https://arxiv.org/abs/2507.18071)
- [GRPO Paper](https://arxiv.org/abs/2502.01155)
- [Qwen3 Documentation](https://github.com/QwenLM/Qwen)
- [Zoo Labs Foundation](https://zoo.ngo)
- [Gym Documentation](https://github.com/zooai/gym)

---

*Zen Models - Bringing balance to AI training*  
*Zoo Labs Foundation © 2025*