---
license: apache-2.0
tags:
- training
- fine-tuning
- lora
- qlora
- grpo
- gspo
- reinforcement-learning
- zen-ai
library_name: transformers
---

# Zen Gym - Unified AI Model Training Platform

**Zen Gym** is the comprehensive training infrastructure powering the entire Zen AI model family. Built on LLaMA Factory, it provides unified support for all modern training methods including LoRA, QLoRA, GRPO, GSPO, DPO, PPO, and more.

## Overview

Zen Gym is the centralized training platform for all Zen models:

- **zen-nano** (0.6B) - Ultra-lightweight edge models
- **zen-eco** (4B) - Efficient instruct/thinking/agent models
- **zen-agent** (4B) - Tool-calling and function execution
- **zen-director** (5B) - Text-to-video generation
- **zen-musician** (7B) - Music generation from lyrics
- **zen-3d** (3.3B) - Controllable 3D asset generation

## Supported Training Methods

### Core Methods (All Verified)

✅ **Full Fine-tuning**: 16-bit full parameter training
✅ **LoRA**: Low-Rank Adaptation (memory efficient)
✅ **QLoRA**: Quantized LoRA (2/3/4/5/6/8-bit via AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ)
✅ **DoRA**: Weight-Decomposed Low-Rank Adaptation
✅ **GRPO**: Group Relative Policy Optimization (40-60% memory reduction)
✅ **GSPO**: Group Sampled Policy Optimization (MoE stability for Qwen3)
✅ **DPO**: Direct Preference Optimization
✅ **PPO**: Proximal Policy Optimization
✅ **KTO**: Kahneman-Tversky Optimization
✅ **ORPO**: Odds Ratio Preference Optimization
✅ **SimPO**: Simple Preference Optimization

### Advanced Algorithms

✅ **GaLore**: Gradient Low-Rank Projection
✅ **BAdam**: Block-wise Adam
✅ **APOLLO**: Adaptive Learning Rate Optimizer
✅ **Adam-mini**: Memory-efficient Adam
✅ **Muon**: Momentum-based optimizer
✅ **OFT/OFTv2**: Orthogonal Fine-Tuning
✅ **LongLoRA**: Extended context LoRA
✅ **LoRA+**: Enhanced LoRA training
✅ **LoftQ**: Quantization-aware LoRA
✅ **PiSSA**: Principal Singular values and Singular vectors Adaptation

### Model Quantization

✅ **BitDelta**: Efficient delta compression
✅ **DeltaSoup**: Model merging and averaging
✅ **GGUF Export**: llama.cpp compatible quantization
✅ **AWQ**: Activation-aware Weight Quantization
✅ **GPTQ**: Post-training quantization

### Performance Optimizations

✅ **FlashAttention-2**: 2x faster attention
✅ **Unsloth**: 2-5x faster training
✅ **Liger Kernel**: LinkedIn's optimized kernels
✅ **RoPE Scaling**: Extended context windows
✅ **NEFTune**: Noise-enhanced fine-tuning
✅ **Gradient Checkpointing**: Reduced memory usage

## Supported Model Architectures

### Primary (Zen Foundation)
- **Qwen3** (0.6B, 4B, 7B, 14B, 30B) - All Zen models based on Qwen3+

### Additional Support
- LLaMA (1-3), Mistral, Mixtral-MoE, Gemma, DeepSeek, Yi
- ChatGLM, Phi, Baichuan, InternLM
- Multimodal: Qwen2-VL, LLaVA, MiniCPM-o, InternVL
- Audio: Qwen2-Audio, MiniCPM-o-2.6
- Video: Wan2.2, Llama4

## Key Features

### Unified Platform
- Single interface for all Zen model training
- Consistent configuration across models
- Shared datasets and preprocessing
- Unified evaluation framework

### Memory Efficiency
- GRPO: 40-60% memory reduction
- QLoRA: 2-8 bit quantized training
- Gradient checkpointing
- Mixed precision training (bf16, fp16)

### Training Speed
- Unsloth: 2-5x faster training
- FlashAttention-2: 2x faster attention
- Liger Kernel optimizations
- Distributed training support

### Model Quality
- Advanced RL methods (GRPO, GSPO, DPO, PPO)
- MoE stability (GSPO for Qwen3)
- Preference optimization
- Multi-task learning

## Hardware Requirements

### Minimum (LoRA Training)
- **GPU**: 16GB VRAM (RTX 4080, RTX 3090)
- **RAM**: 32GB system memory
- **Storage**: 100GB for models and datasets

### Recommended (Full Fine-tuning)
- **GPU**: 80GB VRAM (H100, A100)
- **RAM**: 128GB+ system memory
- **Storage**: 500GB+ NVMe SSD

### Optimal (Large-scale Training)
- **GPU**: 8x H100 (640GB total VRAM)
- **RAM**: 512GB+ system memory
- **Storage**: 2TB+ NVMe RAID
- **Network**: InfiniBand for distributed training

## Quick Start

### Installation

```bash
cd /path/to/zen-gym
conda create -n zen-gym python=3.10
conda activate zen-gym
pip install -r requirements.txt

# FlashAttention-2 (recommended)
pip install flash-attn --no-build-isolation

# Unsloth acceleration
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Train Zen Models

```bash
# Zen Nano LoRA training
llamafactory-cli train --config configs/zen_nano_lora.yaml

# Zen Eco GRPO training
llamafactory-cli train --config configs/zen_eco_grpo.yaml

# Zen Agent DPO training
llamafactory-cli train --config configs/zen_agent_dpo.yaml
```

### GUI Interface

```bash
llamafactory-cli webui
# Visit http://localhost:7860
```

## Training Examples

### LoRA Fine-tuning

```yaml
model_name_or_path: Qwen/Qwen3-4B
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 64
lora_alpha: 32
learning_rate: 5.0e-5
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
```

### GRPO Reinforcement Learning

```yaml
model_name_or_path: Qwen/Qwen3-4B
stage: grpo
do_train: true
reward_model: zenlm/zen-reward-model
learning_rate: 1.0e-6
num_train_epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

### QLoRA 4-bit Training

```yaml
model_name_or_path: Qwen/Qwen3-7B
stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4
quantization_method: bitsandbytes
lora_rank: 64
learning_rate: 2.0e-4
```

## Datasets

Zen Gym includes preprocessed datasets for:
- **Instruction following**: Alpaca, ShareGPT, UltraChat
- **Code generation**: CodeAlpaca, Magicoder
- **Math reasoning**: GSM8K, MATH
- **Tool use**: ToolBench, APIBank
- **Multimodal**: LLaVA, ShareGPT4V
- **Music**: Custom lyrics-to-music datasets
- **3D**: Objaverse, ShapeNet annotations

## Evaluation

Built-in evaluation for:
- Instruction following (MMLU, BBH, GSM8K)
- Code generation (HumanEval, MBPP)
- Tool use (ToolBench)
- Safety (TruthfulQA, ToxiGen)
- Multilingual (FLORES, XNLI)

## Use Cases

### Primary Use Cases
- Fine-tuning Zen models for specific tasks
- Reinforcement learning from human feedback
- Domain adaptation and specialization
- Multi-task learning
- Research in efficient training methods

### Example Applications
- Custom chatbot development
- Code generation fine-tuning
- Math reasoning enhancement
- Tool-calling agent training
- Music generation adaptation
- 3D generation fine-tuning

## Limitations

- Requires significant GPU memory for large models
- Training time varies with model size and method
- Some methods require specific hardware (e.g., FlashAttention-2 needs Ampere+ GPUs)
- Distributed training requires additional setup
- Dataset quality directly impacts model performance

## Ethical Considerations

- Trained models may inherit biases from training data
- Users should evaluate models for safety and fairness
- Proper attribution required for deployed models
- Consider environmental impact of large-scale training
- Respect data licensing and usage rights

## Citation

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
- **Documentation**: See README.md and configs/
- **Zen Engine** (Inference): https://github.com/zenlm/zen-engine

## Support

For issues, questions, or contributions:
- **GitHub Issues**: https://github.com/zenlm/zen-gym/issues
- **Documentation**: See project README and config guides

## Acknowledgements

Built on [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) by hiyouga. We thank the LLaMA Factory team and all contributors.

Part of the **[Zen AI](https://github.com/zenlm)** ecosystem - unified AI model training and inference.