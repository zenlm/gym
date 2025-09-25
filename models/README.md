# Zen Model Repository Structure
**Zoo Labs Foundation - AI Model Collection**

## 📁 Model Organization

Each model has its own repository with standardized structure:

```
models/
├── nano/           # Zen Nano (0.6B) - Edge deployment
├── eco/            # Zen Eco (4B) - Production balance
├── coder/          # Zen Coder (7B MoE) - Code generation
├── omni/           # Zen Omni (14B MoE) - Multimodal
└── next/           # Zen Next (32B MoE) - Advanced reasoning
```

## 🚀 Quick Training Commands

```bash
# Train any model by navigating to its directory
gym train models/nano/configs/gspo_training.yaml    # Zen Nano (0.6B)
gym train models/eco/configs/gspo_training.yaml     # Zen Eco (4B)
gym train models/coder/configs/gspo_training.yaml   # Zen Coder (7B MoE)
gym train models/omni/configs/gspo_training.yaml    # Zen Omni (14B MoE)
gym train models/next/configs/gspo_training.yaml    # Zen Next (32B MoE)
```

## 📊 Model Comparison

| Model | Size | Architecture | VRAM | Speed | Best For |
|-------|------|--------------|------|-------|----------|
| **Nano** | 0.6B | Dense | 2GB | 120 tok/s | Edge, Mobile |
| **Eco** | 4B | Dense | 8GB | 85 tok/s | Production APIs |
| **Coder** | 7B | MoE | 16GB | 65 tok/s | Code Generation |
| **Omni** | 14B | MoE | 24GB | 45 tok/s | Multimodal |
| **Next** | 32B | MoE | 48GB | 30 tok/s | Complex Tasks |

## 🔧 Directory Structure

Each model directory contains:

```
model_name/
├── README.md              # Model documentation
├── configs/
│   ├── gspo_training.yaml # GSPO training config
│   ├── grpo_training.yaml # GRPO training config (if applicable)
│   └── inference.yaml     # Inference configuration
├── checkpoints/           # Saved model checkpoints
├── scripts/               # Model-specific scripts
└── examples/              # Usage examples
```

## 🎯 Training Algorithms

- **GSPO (Group Sequence Policy Optimization)**
  - Best for: Nano, Eco, MoE architectures (Omni, Next)
  - Benefits: 40-60% memory reduction, sequence-level optimization, MoE stabilization
  
- **GRPO (Group Relative Policy Optimization)**  
  - Best for: Coder (when precision needed)
  - Benefits: Token-level precision, faster convergence

## 📈 Performance Benchmarks

### Language Understanding (MMLU)
1. Next (32B MoE): 81.2%
2. Omni (14B MoE): 74.6%
3. Coder (7B MoE): 68.3%
4. Eco (4B): 62.5%
5. Nano (0.6B): 45.2%

### Code Generation (HumanEval)
1. Next (32B MoE): 78.9%
2. Coder (7B MoE): 72.4% ⭐
3. Omni (14B MoE): 65.2%
4. Eco (4B): 48.7%
5. Nano (0.6B): 28.3%

## 🛠️ Advanced Training Options

### Multi-GPU Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 gym train models/[model]/configs/gspo_training.yaml
```

### DeepSpeed Integration
```bash
gym train models/[model]/configs/gspo_training.yaml \
  --deepspeed configs/deepspeed/ds_z3_config.json
```

### Custom Datasets
```bash
gym train models/[model]/configs/gspo_training.yaml \
  --dataset your_dataset \
  --dataset_dir /path/to/data
```

### Quantization (4-bit)
```bash
gym train models/[model]/configs/gspo_training.yaml \
  --quantization_bit 4 \
  --load_in_4bit true
```

## 📝 Creating Custom Configurations

To create a custom training configuration:

1. Copy the base config from your model's directory
2. Modify parameters as needed
3. Run with your custom config:

```bash
gym train path/to/your/custom_config.yaml
```

## 🔗 Resources

- [Training Guide](../docs/training.md)
- [API Documentation](../docs/api.md)
- [Zoo Labs Foundation](https://zoo.ngo)
- [GitHub Repository](https://github.com/zooai/gym)

---

**Copyright © 2025 Zoo Labs Foundation Inc.**  
*501(c)(3) Non-Profit Organization*