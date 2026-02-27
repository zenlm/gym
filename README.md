# Zen Gym

**Unified AI Model Training Platform**

[![License](https://img.shields.io/badge/License-Apache%202.0-white?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-white?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-white?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)

---

Zen Gym is the training infrastructure for the Zen model family. It provides a single interface for supervised fine-tuning, reinforcement learning from human feedback, and model export across all Zen architectures — from 0.6B parameter edge models to 397B MoE frontier systems.

## Supported Models

| Model | Parameters | Type |
|-------|-----------|------|
| zen-nano | 0.6B | Language |
| zen-eco | 4B | Language |
| zen-eco-instruct | 4B | Instruct |
| zen-eco-thinking | 4B | Reasoning |
| zen-eco-coder | 4B | Code |
| zen-eco-agent | 4B | Agent |
| zen-voyager | 32B | Language |
| zen4-mini | 8B | Language |
| zen4-pro | 80B MoE | Language |
| zen4-thinking | 80B MoE | Reasoning |
| zen4-coder-pro | 80B MoE | Code |
| zen-max | 235B MoE | Language |
| zen4-max | 397B MoE | Language |
| zen-vl-* | 4B-30B | Vision-Language |

## Quick Start

```bash
pip install zen-gym

# LoRA fine-tune zen-eco on Alpaca
zen-gym train --model zen-eco --method lora --dataset alpaca

# Launch web UI
zen-gym ui
```

## Training Methods

| Method | Stage | Description |
|--------|-------|-------------|
| Full | SFT | 16-bit full parameter fine-tuning |
| LoRA | SFT | Low-Rank Adaptation, ~30% memory of full |
| QLoRA | SFT | Quantized LoRA (4/8-bit), ~10% memory of full |
| DoRA | SFT | Weight-Decomposed Low-Rank Adaptation |
| DPO | RLHF | Direct Preference Optimization |
| PPO | RLHF | Proximal Policy Optimization |
| GRPO | RLHF | Group Relative Policy Optimization |
| GSPO | RLHF | Group Sampled Policy Optimization (MoE-optimized) |
| KTO | RLHF | Kahneman-Tversky Optimization |
| ORPO | RLHF | Odds Ratio Preference Optimization |
| SimPO | RLHF | Simple Preference Optimization |

## Hardware Requirements

| Model Size | Full | LoRA | QLoRA (4-bit) |
|-----------|------|------|---------------|
| 0.6B | 8 GB | 4 GB | 2 GB |
| 4B | 32 GB | 16 GB | 8 GB |
| 8B | 48 GB | 24 GB | 12 GB |
| 32B | 128 GB | 48 GB | 24 GB |
| 80B MoE | 256 GB | 80 GB | 48 GB |
| 235B MoE | 512 GB | 160 GB | 80 GB |
| 397B MoE | 768 GB | 256 GB | 128 GB |

VRAM listed per-GPU. Multi-GPU setups supported via DeepSpeed ZeRO and FSDP.

## Installation

```bash
git clone https://github.com/zenlm/gym
cd gym
pip install -e ".[torch,metrics]"
```

Optional accelerators:

```bash
# FlashAttention-2
pip install flash-attn --no-build-isolation

# Unsloth (2-5x speedup)
pip install unsloth
```

## Usage

### CLI

```bash
# Supervised fine-tuning with LoRA
zen-gym train \
  --model zen-eco \
  --method lora \
  --dataset alpaca \
  --lora-rank 128 \
  --batch-size 4 \
  --epochs 3 \
  --lr 2e-5

# GRPO reinforcement learning
zen-gym train \
  --model zen-eco \
  --method grpo \
  --dataset preference_data \
  --lr 1e-5

# QLoRA 4-bit training
zen-gym train \
  --model zen-voyager \
  --method qlora \
  --quant-bits 4 \
  --batch-size 1 \
  --gradient-accumulation 16

# Export to GGUF
zen-gym export \
  --model ./output/zen-eco-lora \
  --format gguf \
  --quant Q4_K_M
```

### Config File

```yaml
# configs/zen-eco-lora.yaml
model: zen-eco
method: lora
dataset: alpaca
lora_rank: 128
lora_alpha: 64
batch_size: 4
gradient_accumulation: 4
learning_rate: 2e-5
epochs: 3
flash_attn: true
output_dir: ./output/zen-eco-lora
```

```bash
zen-gym train --config configs/zen-eco-lora.yaml
```

### Web Interface

```bash
zen-gym ui
# Open http://localhost:7860
```

## Monitoring

Zen Gym integrates with standard experiment tracking:

- **TensorBoard** -- built-in, zero config
- **Weights and Biases** -- `--report-to wandb`
- **MLflow** -- `--report-to mlflow`

## Related

- [Zen Models](https://huggingface.co/zenlm) -- Pre-trained model weights on Hugging Face
- [Zen Engine](https://github.com/zenlm/engine) -- High-performance inference runtime
- [Zen Docs](https://zenlm.org) -- Documentation and guides
- [Zen LM](https://github.com/zenlm) -- GitHub organization

---

Apache 2.0 -- [Zen LM](https://github.com/zenlm) -- [Hanzo AI](https://hanzo.ai)
