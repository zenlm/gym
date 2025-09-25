#!/usr/bin/env python
"""
Test inference with the trained LoRA adapter
Zoo Labs Foundation - Gym by Zoo Labs Foundation
Copyright (c) 2025 Zoo Labs Foundation Inc.
https://zoo.ngo
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def test_inference():
    print("🦁 Zoo Labs Foundation - Gym Inference Test")
    print("=" * 50)
    
    # Load base model and tokenizer
    base_model_name = "facebook/opt-125m"
    adapter_path = "saves/local-test"
    
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Machine learning is",
        "Zoo Labs Foundation develops"
    ]
    
    print("\n🎯 Testing trained model:")
    print("-" * 50)
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Response: {response}")
    
    print("\n" + "=" * 50)
    print("✅ Inference test complete!")
    print("🏋️ Gym by Zoo Labs Foundation - Training Platform")
    print("🔗 https://zoo.ngo")

if __name__ == "__main__":
    test_inference()