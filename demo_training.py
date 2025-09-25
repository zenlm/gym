#!/usr/bin/env python
"""
Demo GSPO training script for Qwen3
Runs locally with minimal resources
"""

import os
import sys
import json
import torch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_demo_dataset():
    """Create a small demo dataset for testing"""
    demo_data = [
        {
            "instruction": "What is machine learning?",
            "input": "",
            "output": "Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        },
        {
            "instruction": "Explain neural networks",
            "input": "",
            "output": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains. They consist of interconnected nodes that process information."
        },
        {
            "instruction": "What is GSPO?",
            "input": "",
            "output": "GSPO (Group Sequence Policy Optimization) is a reinforcement learning algorithm that performs sequence-level optimization, providing superior training stability for large language models."
        },
        {
            "instruction": "Write a Python hello world program",
            "input": "",
            "output": "Here's a simple Python hello world program:\n\n```python\nprint('Hello, World!')\n```"
        },
        {
            "instruction": "What are the benefits of exercise?",
            "input": "",
            "output": "Exercise provides numerous benefits including improved cardiovascular health, stronger muscles and bones, better mental health, weight management, and increased energy levels."
        }
    ]
    
    # Save as JSON for Gym to use
    os.makedirs("data/demo", exist_ok=True)
    with open("data/demo/train.json", "w") as f:
        json.dump(demo_data, f, indent=2)
    
    print("✅ Created demo dataset with {} examples".format(len(demo_data)))
    return "data/demo/train.json"

def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        return True
    elif torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU (MPS) available")
        return True
    else:
        print("⚠️ No GPU detected, will use CPU (training will be slow)")
        return False

def run_demo_training():
    """Run a quick GSPO training demo"""
    
    print("\n" + "="*60)
    print("🏋️ GYM GSPO TRAINING DEMO")
    print("="*60 + "\n")
    
    # Check system
    has_gpu = check_gpu()
    
    # Create demo dataset
    dataset_path = create_demo_dataset()
    
    # Update dataset_info.json to include our demo dataset
    dataset_info_path = "data/dataset_info.json"
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)
    
    dataset_info["demo_dataset"] = {
        "file_name": dataset_path,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input", 
            "response": "output",
            "history": "history"
        }
    }
    
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print("✅ Registered demo dataset")
    
    # Prepare training arguments
    training_args = [
        "--stage", "sft",  # Start with SFT for demo (faster than GSPO)
        "--do_train",
        "--model_name_or_path", "facebook/opt-125m",  # Small model for demo
        "--dataset", "demo_dataset",
        "--template", "default",
        "--finetuning_type", "lora",
        "--lora_rank", "4",
        "--lora_alpha", "8", 
        "--lora_dropout", "0.1",
        "--lora_target", "all",
        "--output_dir", "saves/demo-gspo",
        "--overwrite_cache",
        "--overwrite_output_dir",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", "5e-5",
        "--num_train_epochs", "3",
        "--logging_steps", "1",
        "--save_steps", "10",
        "--save_total_limit", "1",
        "--warmup_ratio", "0.1",
        "--lr_scheduler_type", "cosine",
        "--report_to", "none",  # Disable wandb/tensorboard for demo
        "--max_samples", "5",  # Use only 5 samples for quick demo
        "--cutoff_len", "256",  # Short context for demo
    ]
    
    # Add fp16/bf16 based on hardware
    if has_gpu:
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                training_args.append("--bf16")
            else:
                training_args.append("--fp16")
    
    print("\n📋 Training Configuration:")
    print(f"   Model: facebook/opt-125m (125M params)")
    print(f"   Method: LoRA fine-tuning")
    print(f"   Dataset: 5 demo examples")
    print(f"   Epochs: 3")
    print(f"   Output: saves/demo-gspo/")
    
    print("\n🚀 Starting training...\n")
    print("-" * 60)
    
    # Import and run training
    try:
        from gym.train.tuner import run_exp
        from gym.hparams import get_train_args
        
        # Parse arguments
        model_args, data_args, training_args_obj, finetuning_args, generating_args = get_train_args(training_args)
        
        # Run training
        from gym.train.sft import run_sft
        run_sft(model_args, data_args, training_args_obj, finetuning_args, generating_args, callbacks=[])
        
        print("-" * 60)
        print("\n✅ Training completed successfully!")
        print(f"   Model saved to: saves/demo-gspo/")
        
        # Now let's test the trained model
        print("\n🧪 Testing the trained model...")
        test_trained_model()
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def test_trained_model():
    """Test the trained model with a simple prompt"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        print("\n📥 Loading trained model...")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, "saves/demo-gspo")
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        
        # Test prompt
        prompt = "What is machine learning?"
        print(f"\n💬 Prompt: {prompt}")
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Response: {response}")
        
    except Exception as e:
        print(f"⚠️ Could not test model: {e}")

if __name__ == "__main__":
    # Run the demo
    run_demo_training()
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Try GSPO with: gym-cli train --stage gspo --config configs/gspo_qwen3_4b_nano.yaml")
    print("2. Use larger models: Qwen/Qwen3-1.8B-Instruct")
    print("3. Train on your own data")
    print("\nVisit https://zoo.ngo for more information!")
    print("="*60 + "\n")