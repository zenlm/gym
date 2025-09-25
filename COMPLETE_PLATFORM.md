# Gym: Complete AI Training Platform by Zoo Labs Foundation

## 🚀 Overview

Gym is a comprehensive, production-ready AI training platform that combines state-of-the-art algorithms with cutting-edge optimization techniques. Built by Zoo Labs Foundation, it provides everything needed for training, fine-tuning, and serving large language models at scale.

## 🎯 Core Features

### Training Algorithms

| Algorithm | Description | Memory | Speed | Use Case |
|-----------|-------------|---------|--------|----------|
| **SFT** | Supervised Fine-Tuning | Standard | Fast | General fine-tuning |
| **DPO** | Direct Preference Optimization | Efficient | Fast | Preference alignment |
| **PPO** | Proximal Policy Optimization | High | Moderate | RLHF |
| **ORPO** | Odds Ratio Preference Optimization | Moderate | Fast | Simplified RLHF |
| **GRPO** | Group Relative Policy Optimization | -40% | Fast | DeepSeek's method |
| **GSPO** | Group Sequence Policy Optimization | -60% | Fast | Alibaba's Qwen3 method |
| **KTO** | Kahneman-Tversky Optimization | Efficient | Fast | Human preference |
| **RLOO** | Reinforcement Learning with Leave-One-Out | Moderate | Fast | Online RLHF |

### Model Architecture Support

#### Zen Model Family (Qwen3-based)
- **Nano** (0.6B): Ultra-lightweight for edge deployment
- **Eco** (4B): Balanced performance matching Qwen2.5-7B
- **Coder** (7B/30B/480B): Specialized for code generation
  - Standard: 30B total, 3.3B active (MoE)
  - Max: 480B total, 35B active (MoE)
  - Thinking variants with CoT reasoning
- **Omni** (14B/30B): Multimodal capabilities
- **Next** (32B/80B): Ultra-sparse MoE with 512 experts

### Quantization Technologies

#### BitDelta (ZIP-7)
- **1-bit quantization** of fine-tune deltas
- **10× memory reduction** for personalized models
- **60% reduction** in jailbreak risks
- Byzantine-robust community aggregation

```python
from gym.quantization import BitDeltaConfig, BitDeltaQuantizer

config = BitDeltaConfig(
    bits=1,
    group_size=128,
    safety_threshold=0.6,
    enable_deltasoup=True
)
quantizer = BitDeltaQuantizer(config)
```

#### DeltaQuant
- Flexible quantization (INT2/4/8, Binary, Ternary)
- Per-channel and per-tensor quantization
- Mixed precision support
- Calibration-based optimization

```python
from gym.quantization import DeltaQuantConfig, QuantMethod

config = DeltaQuantConfig(
    method=QuantMethod.INT4,
    per_channel=True,
    calibration_samples=256
)
```

#### DeltaSoup
- Community-driven model improvement
- Byzantine-robust aggregation
- Differential privacy support
- Contributor rewards system

```python
from gym.quantization import DeltaSoupConfig, AggregationMethod

config = DeltaSoupConfig(
    method=AggregationMethod.BYZANTINE_ROBUST,
    differential_privacy=True,
    enable_rewards=True
)
```

### Optimization Features

#### Memory Optimization
- **Unsloth**: 2-3× training speedup
- **Flash Attention 2**: Efficient attention computation
- **Gradient Checkpointing**: Trade compute for memory
- **QLoRA**: 4-bit quantized LoRA training
- **Mixed Precision**: FP16/BF16 training

#### Performance Features
- **Multi-GPU Support**: DDP, FSDP, DeepSpeed
- **Dynamic Batching**: Adaptive batch size
- **Gradient Accumulation**: Simulate larger batches
- **CPU Offloading**: Use system RAM for large models
- **KV Cache Optimization**: Efficient inference

### Infrastructure

#### Training Infrastructure
- **Distributed Training**: Scale to multiple nodes
- **Checkpoint Management**: Automatic save/resume
- **Experiment Tracking**: Weights & Biases, TensorBoard
- **Hyperparameter Tuning**: Optuna integration

#### Serving Infrastructure
- **Model Export**: GGUF, ONNX, TorchScript
- **API Server**: FastAPI-based inference
- **Batch Inference**: Efficient bulk processing
- **Model Merging**: Combine LoRA adapters

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/zooai/gym
cd gym

# Install with all features
pip install -e ".[all]"

# Or minimal installation
pip install -e .
```

## 🚀 Quick Start

### Command Line Interface

```bash
# Train with GSPO (Qwen3 optimized)
gym train models/nano/configs/gspo_training.yaml

# Fine-tune with BitDelta quantization
gym train configs/bitdelta_training.yaml

# Launch WebUI
gym webui --port 8080

# Export model
gym export --model saves/model --format gguf
```

### Python API

```python
from gym import Trainer, TrainingConfig
from gym.quantization import BitDeltaQuantizer

# Configure training
config = TrainingConfig(
    model="Qwen/Qwen3-4B",
    algorithm="gspo",
    quantization="bitdelta",
    use_unsloth=True
)

# Initialize trainer
trainer = Trainer(config)

# Train model
trainer.train(dataset="alpaca_gpt4_en")

# Apply community improvements
trainer.apply_deltasoup()

# Export quantized model
trainer.export("model.gguf", quantization="bitdelta")
```

## 🎨 WebUI Features

The WebUI provides a comprehensive interface with:

- **Black Monochromatic Theme**: Professional Zoo Labs branding
- **Real-time Training Monitoring**: Loss curves, metrics
- **Model Management**: Load, save, merge models
- **Dataset Browser**: Preview and filter datasets
- **Hyperparameter Tuning**: Interactive configuration
- **Inference Playground**: Test models interactively

## 🔬 Advanced Features

### Personalized Models
Create millions of personalized model variants using BitDelta:

```python
from gym.personalization import PersonalizedTrainer

trainer = PersonalizedTrainer(base_model="Qwen/Qwen3-4B")
trainer.create_variant(user_id="user_123", preferences=user_prefs)
trainer.serve_variant("user_123")  # 10× memory efficient
```

### Community Learning
Aggregate improvements from multiple users:

```python
from gym.quantization import DeltaSoup

soup = DeltaSoup(config)
soup.contribute(user_id="alice", model=model_alice)
soup.contribute(user_id="bob", model=model_bob)
aggregated = soup.aggregate(min_contributors=3)
```

### Safety Features
- Automatic jailbreak detection and prevention
- Content filtering and safety checks
- Byzantine-robust aggregation
- Differential privacy support

## 📊 Benchmarks

| Model | Method | Memory | Speed | Quality |
|-------|--------|---------|--------|---------|
| Qwen3-4B | Standard | 16GB | 1.0× | 100% |
| Qwen3-4B | Unsloth | 12GB | 2.3× | 100% |
| Qwen3-4B | QLoRA | 6GB | 1.5× | 99% |
| Qwen3-4B | BitDelta | 1.6GB | 3.0× | 98% |
| Qwen3-4B | DeltaQuant-INT4 | 4GB | 2.0× | 99% |

## 🛠️ Configuration Examples

### GSPO Training (Qwen3 Optimized)
```yaml
model: Qwen/Qwen3-4B
algorithm: gspo
group_size: 8
sequence_parallel: true
use_unsloth: true
quantization:
  method: bitdelta
  bits: 1
  enable_deltasoup: true
```

### Multi-GPU Training
```yaml
distributed:
  backend: nccl
  strategy: fsdp
  sharding_strategy: full_shard
  mixed_precision: bf16
```

### Production Serving
```yaml
serving:
  model_format: gguf
  quantization: int4
  batch_size: 32
  max_concurrent: 100
  cache_size: 10GB
```

## 🔗 Integrations

- **Hugging Face Hub**: Direct model loading/saving
- **Weights & Biases**: Experiment tracking
- **LangChain**: Chain-of-thought integration
- **vLLM**: High-performance serving
- **TensorRT**: NVIDIA optimization
- **ONNX Runtime**: Cross-platform deployment

## 📚 Documentation

- [Training Guide](docs/training.md)
- [Quantization Guide](docs/quantization.md)
- [Serving Guide](docs/serving.md)
- [API Reference](docs/api.md)
- [Configuration Reference](docs/config.md)

## 🏆 Why Choose Gym?

1. **Complete Solution**: Everything from training to serving
2. **State-of-the-art**: Latest algorithms (GRPO, GSPO)
3. **Memory Efficient**: 10× reduction with BitDelta
4. **Production Ready**: Battle-tested at scale
5. **Community Driven**: DeltaSoup aggregation
6. **Safety First**: Built-in safety features
7. **Zoo Ecosystem**: Integrated with Zoo Labs tools

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built on the shoulders of giants:
- Hugging Face Transformers
- DeepSpeed & FSDP
- Unsloth optimizations
- Flash Attention
- And the amazing open-source community

---

**Gym by Zoo Labs Foundation** - Training AI at the speed of thought 🚀

*Copyright 2025 Zoo Labs Foundation Inc.*