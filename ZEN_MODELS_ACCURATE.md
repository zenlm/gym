# Zen Models - Accurate Qwen3 Model Specifications
**Zoo Labs Foundation - Gym Training Platform**  
**Updated with correct Qwen3 model information**

## 📊 Zen Model Family - Based on Real Qwen3 Models

### 🔬 Zen Nano - Qwen3-0.6B
- **Base Model**: `Qwen/Qwen3-0.6B`
- **Architecture**: Dense (traditional transformer)
- **Parameters**: 600M total
- **Context**: 32K tokens native
- **Use Case**: Edge devices, mobile, ultra-lightweight deployment
- **Performance**: Matches Qwen2.5-1.5B despite smaller size

### 🌿 Zen Eco - Qwen3-4B  
- **Base Model**: `Qwen/Qwen3-4B`
- **Architecture**: Dense (traditional transformer)
- **Parameters**: 4B total
- **Context**: 32K tokens native
- **Use Case**: Production APIs, balanced performance
- **Performance**: Matches Qwen2.5-7B performance (50% density improvement)

### 💻 Zen Coder - Qwen3-Coder-480B-A35B
- **Base Model**: `Qwen/Qwen3-Coder-480B-A35B-Instruct`
- **Architecture**: MoE (Mixture of Experts)
- **Parameters**: 480B total, 35B active per token
- **Experts**: 128 total experts, 8 activated per token
- **Context**: 256K native, 1M with YaRN extension
- **Training**: 7.5T tokens (70% code ratio)
- **Use Case**: State-of-the-art code generation, agentic coding
- **Special**: Agent RL post-training for multi-turn tool use

### 🌐 Zen Omni - Qwen3-Omni-30B-A3B
- **Base Model**: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- **Architecture**: MoE (Mixture of Experts)  
- **Parameters**: 30B total, 3B active per token
- **Modalities**: Text (119 languages), Speech (19 languages in, 10 out), Vision, Video
- **Components**:
  - Vision: 675M parameter ViT encoder
  - Audio: Whisper-large-v3 based encoder (16kHz, 128-channel mel-spectrogram)
- **Use Case**: Multimodal understanding, real-time omni-modal AI
- **Variants**: Also available as Thinking and Captioner models

### 🚀 Zen Next - Qwen3-Next-80B-A3B
- **Base Model**: `Qwen/Qwen3-Next-80B-A3B-Instruct`
- **Architecture**: Hybrid MoE with novel attention
- **Parameters**: 80B total, 3B active per token
- **Experts**: 512 total experts, 10+1 activated
- **Attention**: Hybrid Gated DeltaNet + Gated Attention (every 4th layer uses GQA)
- **Context**: 256K native, 1M extendable
- **Performance**: 10x throughput for >32K context vs traditional
- **Special Features**:
  - Multi-Token Prediction (MTP)
  - Linear attention for most layers
  - Ultra-sparse activation (3.75% parameters active)
- **Variants**: Instruct and Thinking modes available

## 🎯 Key Innovations

### Dense Models (Nano, Eco)
- **50% density improvement** over Qwen2.5
- Qwen3-4B matches Qwen2.5-7B performance
- Qwen3-0.6B matches Qwen2.5-1.5B performance
- 36 trillion token training (2x Qwen2.5)

### MoE Models (Coder, Omni, Next)
- **Sparse activation** for efficiency
- **No shared experts** (unlike Qwen2.5-MoE)
- **Global batch load balancing** for expert specialization
- **GSPO training** with MoE stabilization

### Qwen3-Next Specific
- **Hybrid attention mechanism** replacing standard attention
- **10x inference speedup** for long contexts
- **3B active out of 80B** - extreme sparsity
- Foundation for upcoming Qwen3.5

## 📁 Configuration Files

```bash
# Dense Models
models/nano/configs/gspo_training.yaml   # Qwen3-0.6B
models/eco/configs/gspo_training.yaml    # Qwen3-4B

# MoE Models  
models/coder/configs/gspo_training.yaml  # Qwen3-Coder-480B-A35B
models/omni/configs/gspo_training.yaml   # Qwen3-Omni-30B-A3B
models/next/configs/gspo_training.yaml   # Qwen3-Next-80B-A3B
```

## 🚀 Training Commands

```bash
# Train any model
gym train models/nano/configs/gspo_training.yaml   # Zen Nano
gym train models/eco/configs/gspo_training.yaml    # Zen Eco
gym train models/coder/configs/gspo_training.yaml  # Zen Coder
gym train models/omni/configs/gspo_training.yaml   # Zen Omni
gym train models/next/configs/gspo_training.yaml   # Zen Next
```

## 📈 Performance Summary

| Model | Total Params | Active Params | Context | Architecture |
|-------|-------------|---------------|---------|--------------|
| Nano | 0.6B | 0.6B | 32K | Dense |
| Eco | 4B | 4B | 32K | Dense |
| Coder | 480B | 35B | 256K-1M | MoE (128 experts) |
| Omni | 30B | 3B | 32K | MoE (multimodal) |
| Next | 80B | 3B | 256K-1M | Hybrid MoE (512 experts) |

## 🔗 References

- **Qwen3 Technical Report**: https://arxiv.org/pdf/2505.09388
- **Qwen3-Coder**: https://qwenlm.github.io/blog/qwen3-coder/
- **Qwen3-Omni**: https://arxiv.org/html/2509.17765v1
- **Qwen3-Next**: Released September 15, 2025
- **GSPO Paper**: https://arxiv.org/abs/2507.18071

---

**Zoo Labs Foundation**  
*Building the future of AI training with accurate, state-of-the-art models*  
**Website**: https://zoo.ngo