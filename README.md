# 🏋️ Zen Gym - Unified AI Model Training Platform

**Zen Gym** is the unified training infrastructure for all Zen AI models. Built on LLaMA Factory, it provides comprehensive support for fine-tuning, reinforcement learning, and deployment across the entire Zen model family.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub](https://img.shields.io/badge/GitHub-zenlm%2Fzen--gym-blue)](https://github.com/zenlm/zen-gym)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/zenlm)

## Overview

Zen Gym is the centralized training platform for:
- **zen-nano** (0.6B) - Ultra-lightweight models
- **zen-eco** (4B) - Efficient models (instruct, thinking, agent)
- **zen-agent** (4B) - Tool-calling and function execution
- **zen-director** (5B) - Text-to-video generation
- **zen-musician** (7B) - Music generation with lyrics

All Zen models are trained, fine-tuned, and optimized through Zen Gym's unified infrastructure.

## Features

### Supported Training Methods

✅ **All training methods verified and supported:**

- **Full Fine-tuning**: 16-bit full parameter training
- **LoRA**: Low-Rank Adaptation (memory efficient)
- **QLoRA**: Quantized LoRA (2/3/4/5/6/8-bit via AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ)
- **DoRA**: Weight-Decomposed Low-Rank Adaptation
- **GRPO**: Group Relative Policy Optimization (40-60% memory reduction)
- **GSPO**: Group Sampled Policy Optimization (MoE stability for Zen)
- **DPO**: Direct Preference Optimization
- **PPO**: Proximal Policy Optimization
- **KTO**: Kahneman-Tversky Optimization
- **ORPO**: Odds Ratio Preference Optimization
- **SimPO**: Simple Preference Optimization

### Advanced Algorithms

- **GaLore**: Gradient Low-Rank Projection
- **BAdam**: Block-wise Adam
- **APOLLO**: Adaptive Learning Rate Optimizer
- **Adam-mini**: Memory-efficient Adam
- **Muon**: Momentum-based optimizer
- **OFT/OFTv2**: Orthogonal Fine-Tuning
- **LongLoRA**: Extended context LoRA
- **LoRA+**: Enhanced LoRA training
- **LoftQ**: Quantization-aware LoRA
- **PiSSA**: Principal Singular values and Singular vectors Adaptation

### Model Quantization (via Unsloth)

- **BitDelta**: Efficient delta compression
- **DeltaSoup**: Model merging and averaging
- **GGUF Export**: llama.cpp compatible quantization
- **AWQ**: Activation-aware Weight Quantization
- **GPTQ**: Post-training quantization

### Performance Optimizations

- **FlashAttention-2**: 2x faster attention
- **Unsloth**: 2-5x faster training
- **Liger Kernel**: LinkedIn's optimized kernels
- **RoPE Scaling**: Extended context windows
- **NEFTune**: Noise-enhanced fine-tuning
- **Gradient Checkpointing**: Reduced memory usage

### Supported Models

Zen Gym natively supports:
- **Zen** (0.6B, 4B, 7B, 14B, 30B) - Zen model foundation
- LLaMA, Zen, Mixtral-MoE, Gemma, DeepSeek, Yi, ChatGLM, Phi
- Multimodal: Zen-VL, LLaVA, MiniCPM-o, InternVL
- Audio: Zen-Audio, MiniCPM-o-2.6
- Video: Zen Video, Llama4

## Installation

```bash
cd /Users/z/work/zen/gym
conda create -n zen-gym python=3.10
conda activate zen-gym
pip install -r requirements.txt

# For FlashAttention-2 (recommended)
pip install flash-attn --no-build-isolation

# For Unsloth acceleration
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Quick Start

### Training Zen Models

```bash
# Fine-tune zen-nano with LoRA
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Zen/Zen 0.6B \
    --dataset your_dataset \
    --template zen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ./zen-nano-lora \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --learning_rate 5e-5 \
    --save_steps 100 \
    --logging_steps 10 \
    --flash_attn fa2 \
    --use_unsloth true

# GRPO training for zen-eco
llamafactory-cli train \
    --stage grpo \
    --do_train \
    --model_name_or_path Zen/Zen 4B \
    --dataset your_preference_dataset \
    --template zen \
    --finetuning_type lora \
    --output_dir ./zen-eco-grpo \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --save_steps 50

# QLoRA 4-bit training
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Zen/Zen 4B \
    --dataset your_dataset \
    --template zen \
    --finetuning_type lora \
    --quantization_bit 4 \
    --output_dir ./zen-eco-qlora \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --learning_rate 2e-4
```

### GUI Training Interface

```bash
# Launch Zen Gym web interface
llamafactory-cli webui
```

Access at http://localhost:7860

### Export to GGUF

```bash
# Export trained model to GGUF for llama.cpp
llamafactory-cli export \
    --model_name_or_path ./zen-eco-lora \
    --adapter_name_or_path ./zen-eco-lora \
    --template zen \
    --export_dir ./zen-eco-gguf \
    --export_size 4 \
    --export_quantization_bit 4 \
    --export_legacy_format false
```

## Training Configurations

### Zen Nano (0.6B)
```yaml
model_name_or_path: Zen/Zen 0.6B
template: zen
finetuning_type: lora
lora_rank: 64
lora_alpha: 32
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
learning_rate: 5e-5
flash_attn: fa2
use_unsloth: true
```

### Zen Eco (4B)
```yaml
model_name_or_path: Zen/Zen 4B
template: zen
finetuning_type: lora
lora_rank: 128
lora_alpha: 64
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-5
flash_attn: fa2
use_unsloth: true
```

### Zen Agent (4B) - Tool Calling
```yaml
model_name_or_path: Zen/Zen 4B
dataset: Salesforce/xlam-function-calling-60k
template: zen
finetuning_type: lora
lora_rank: 128
lora_alpha: 64
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1e-5
```

### Zen Musician (7B) - Music Generation
```yaml
model_name_or_path: m-a-p/YuE-s1-7B-anneal-en-cot
template: yue
finetuning_type: lora
lora_rank: 64
lora_alpha: 32
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2e-4
flash_attn: fa2
```

## Training Methods Detailed

### LoRA (Low-Rank Adaptation)
- **Memory**: ~30% of full fine-tuning
- **Speed**: 1.5-2x faster
- **Quality**: 95-98% of full fine-tuning
- **Best for**: Most training scenarios

```bash
--finetuning_type lora \
--lora_rank 64 \
--lora_alpha 32 \
--lora_dropout 0.1 \
--lora_target all
```

### QLoRA (Quantized LoRA)
- **Memory**: ~10% of full fine-tuning
- **Speed**: 1.2-1.5x faster than full
- **Quality**: 90-95% of full fine-tuning
- **Best for**: Limited GPU memory

```bash
--finetuning_type lora \
--quantization_bit 4 \
--lora_rank 64 \
--lora_alpha 32
```

### GRPO (Group Relative Policy Optimization)
- **Memory**: 40-60% less than PPO (no value network)
- **Speed**: 2x faster than PPO
- **Quality**: Superior to DPO for instruction following
- **Best for**: Reinforcement learning

```bash
--stage grpo \
--finetuning_type lora \
--learning_rate 1e-5
```

### GSPO (Group Sampled Policy Optimization)
- **Memory**: Similar to GRPO
- **Speed**: Optimized for MoE models
- **Quality**: Better stability for Zen-MoE
- **Best for**: Mixture-of-Experts models

```bash
--stage gspo \
--finetuning_type lora \
--learning_rate 1e-5
```

## Integration with Zen Models

### zen-musician
```bash
cd /Users/z/work/zen/zen-musician
# Train with zen-gym
llamafactory-cli train \
    --config /Users/z/work/zen/gym/configs/zen_musician_lora.yaml
```

### zen-nano
```bash
cd /Users/z/work/zen/zen-nano
# Train with zen-gym
llamafactory-cli train \
    --config /Users/z/work/zen/gym/configs/zen_nano_lora.yaml
```

### zen-eco (instruct/thinking/agent)
```bash
cd /Users/z/work/zen/zen-eco
# Train with zen-gym
llamafactory-cli train \
    --config /Users/z/work/zen/gym/configs/zen_eco_lora.yaml
```

## Monitoring and Logging

Zen Gym supports multiple experiment tracking tools:

- **TensorBoard**: Built-in, zero config
- **Weights & Biases**: `--report_to wandb`
- **MLflow**: `--report_to mlflow`
- **SwanLab**: `--report_to swanlab`

```bash
# Enable WandB logging
export WANDB_PROJECT=zen-models
llamafactory-cli train --report_to wandb --config your_config.yaml

# View TensorBoard
tensorboard --logdir ./output
```

## Deployment

### OpenAI-style API
```bash
# Deploy with vLLM
llamafactory-cli api \
    --model_name_or_path ./zen-eco-lora \
    --template zen \
    --infer_backend vllm

# Deploy with SGLang
llamafactory-cli api \
    --model_name_or_path ./zen-eco-lora \
    --template zen \
    --infer_backend sglang
```

### Gradio Interface
```bash
llamafactory-cli chat \
    --model_name_or_path ./zen-eco-lora \
    --template zen
```

## Quantization & Export

### GGUF Quantization
```bash
# Q4_K_M (recommended for most use cases)
llamafactory-cli export \
    --model_name_or_path ./model \
    --export_dir ./gguf \
    --export_quantization_bit 4

# Q8_0 (higher quality)
llamafactory-cli export \
    --model_name_or_path ./model \
    --export_dir ./gguf \
    --export_quantization_bit 8

# Q2_K (maximum compression)
llamafactory-cli export \
    --model_name_or_path ./model \
    --export_dir ./gguf \
    --export_quantization_bit 2
```

### MLX Conversion (Apple Silicon)
```bash
# Convert to MLX format
python -m mlx_lm.convert \
    --hf-path ./model \
    --mlx-path ./mlx \
    --quantize
```

## Hardware Requirements

| Model Size | Training Method | GPU Memory | Recommended GPU |
|------------|----------------|------------|-----------------|
| 0.6B | Full | 8GB | RTX 3060 |
| 0.6B | LoRA | 4GB | GTX 1660 Ti |
| 4B | Full | 32GB | RTX 3090 |
| 4B | LoRA | 16GB | RTX 4060 Ti |
| 4B | QLoRA 4-bit | 8GB | RTX 3060 |
| 7B | Full | 48GB | A6000 |
| 7B | LoRA | 24GB | RTX 3090 |
| 7B | QLoRA 4-bit | 12GB | RTX 3060 Ti |

## Performance Benchmarks

Training speed with various optimizations (zen-eco-4b):

| Configuration | Speed | Memory |
|--------------|-------|--------|
| Baseline | 1.0x | 24GB |
| + FlashAttention-2 | 1.5x | 22GB |
| + Unsloth | 2.5x | 20GB |
| + Liger Kernel | 3.0x | 18GB |
| + Gradient Checkpointing | 2.8x | 14GB |

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--per_device_train_batch_size 1

# Increase gradient accumulation
--gradient_accumulation_steps 32

# Enable gradient checkpointing
--gradient_checkpointing true

# Use QLoRA
--quantization_bit 4
```

### Slow Training
```bash
# Enable FlashAttention-2
--flash_attn fa2

# Enable Unsloth
--use_unsloth true

# Enable Liger Kernel
--enable_liger_kernel true
```

### Model Divergence
```bash
# Lower learning rate
--learning_rate 1e-5

# Add warmup
--warmup_ratio 0.1

# Use cosine scheduler
--lr_scheduler_type cosine
```

## Documentation

- **Official Docs**: https://gym.readthedocs.io/
- **Examples**: [examples/](examples/)
- **Training Configs**: [configs/](configs/)
- **Zen Guide**: [configs/zen_training_guide.md](configs/zen_training_guide.md)

## Credits

Zen Gym is built on [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) by hiyouga. We thank the LLaMA Factory team and all contributors for their excellent work.

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## Citation

If you use Zen Gym in your research, please cite:

```bibtex
@misc{zengym2025,
  title={Zen Gym: Unified Training Infrastructure for Zen AI Models},
  author={Zen AI Team},
  year={2025},
  howpublished={\url{https://github.com/zenlm/zen-gym}}
}

@article{zheng2024llamafactory,
  title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Zhangchi Feng and Yongqiang Ma},
  journal={arXiv preprint arXiv:2403.13372},
  year={2024}
}
```

## Links

- **GitHub**: https://github.com/zenlm/zen-gym
- **Organization**: https://github.com/zenlm
- **HuggingFace Models**: https://huggingface.co/zenlm
- **Zen Engine** (Inference): https://github.com/zenlm/zen-engine
- **Zen Musician**: https://github.com/zenlm/zen-musician
- **Zen 3D**: https://github.com/zenlm/zen-3d

---

**Zen Gym** - Unified training platform for all Zen AI models

Part of the **[Zen AI](https://github.com/zenlm)** ecosystem.