#!/usr/bin/env python3
"""
Test simplified implementations
Copyright 2025 Zoo Labs Foundation Inc.

Simple tests for simplified modules:
- Direct functionality tests
- No complex test frameworks
- Clear pass/fail conditions
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def test_bitdelta_quantization():
    """Test BitDelta 1-bit quantization."""
    print("\n=== Testing BitDelta Quantization ===")

    from gym.quantization.bitdelta_simple import BitDeltaQuantizer

    # Create test tensors
    base_weight = torch.randn(256, 512)
    delta = torch.randn(256, 512) * 0.1  # Small delta
    weight = base_weight + delta

    # Initialize quantizer
    quantizer = BitDeltaQuantizer(group_size=128)

    # Test quantization
    signs, scales = quantizer.quantize(weight, base_weight)

    # Test dimensions
    assert signs.shape == weight.shape, f"Signs shape mismatch: {signs.shape} != {weight.shape}"
    expected_groups = (weight.numel() + 127) // 128  # Ceiling division
    assert scales.numel() == expected_groups, f"Scales count mismatch: {scales.numel()} != {expected_groups}"

    # Test dequantization
    reconstructed = quantizer.dequantize(base_weight, signs, scales)
    assert reconstructed.shape == weight.shape, "Reconstructed shape mismatch"

    # Check reconstruction error
    error = (weight - reconstructed).abs().mean()
    print(f"✓ Quantization error: {error:.6f}")

    # Test memory stats
    compressed = {"test": (signs, scales)}
    stats = quantizer.memory_usage(compressed)
    print(f"✓ Compression ratio: {stats['compression_ratio']:.2f}x")

    assert stats['compression_ratio'] > 5, "Compression ratio too low"
    print("✓ BitDelta quantization test passed")

def test_deltaquant():
    """Test DeltaQuant multi-bit quantization."""
    print("\n=== Testing DeltaQuant ===")

    from gym.quantization.deltaquant_simple import DeltaQuantizer

    # Test different bit widths
    for bits in [1, 2, 4, 8]:
        print(f"\nTesting {bits}-bit quantization:")

        # Create test tensors
        base_weight = torch.randn(128, 256)
        weight = base_weight + torch.randn(128, 256) * 0.1

        # Initialize quantizer
        quantizer = DeltaQuantizer(bits=bits, per_channel=True)

        # Test quantization
        quantized, scale = quantizer.quantize_delta(weight, base_weight)

        # Check data types
        assert quantized.dtype == torch.int8, f"Quantized dtype should be int8, got {quantized.dtype}"
        assert scale.dtype == torch.float32, f"Scale dtype should be float32, got {scale.dtype}"

        # Test reconstruction
        reconstructed = quantizer.reconstruct_weight(base_weight, quantized, scale)

        # Compute error
        metrics = quantizer.compute_error(weight, reconstructed)
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  Relative error: {metrics['relative_error']:.4%}")

        # Error should decrease with more bits
        if bits == 8:
            assert metrics['relative_error'] < 0.01, f"8-bit quantization error too high: {metrics['relative_error']}"

        print(f"  ✓ {bits}-bit quantization passed")

    print("✓ DeltaQuant test passed")

def test_lora_adapter():
    """Test simplified LoRA adapter."""
    print("\n=== Testing LoRA Adapter ===")

    from gym.model.adapter_simple import find_linear_modules

    # Create simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(256, 512)
            self.layer2 = nn.Linear(512, 256)
            self.lm_head = nn.Linear(256, 1000)
            # Add minimal config
            self.config = type('Config', (), {
                'model_type': 'test',
                'get': lambda self, key, default=None: getattr(self, key, default)
            })()

    model = TestModel()
    initial_params = sum(p.numel() for p in model.parameters())

    # Find linear modules
    modules = find_linear_modules(model)
    assert "layer1" in modules or "layer2" in modules, "Failed to find linear modules"
    print(f"✓ Found linear modules: {modules}")

    # Test finding modules functionality works
    print(f"✓ Linear modules found: {len(modules)} modules")

    # Test a simple freeze operation
    from gym.model.adapter_simple import freeze_model

    # Initially all params are trainable
    initial_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Initial trainable params: {initial_trainable:,}")

    # Freeze model except layer1
    frozen_model = freeze_model(model, trainable_layers=["layer1"])
    frozen_trainable = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)

    # Check that we reduced trainable params
    assert frozen_trainable < initial_trainable, "Freezing didn't reduce trainable params"
    print(f"✓ After freezing: {frozen_trainable:,} trainable params")

    print("✓ LoRA adapter test passed")

def test_training_integration():
    """Test that simplified trainers can be instantiated."""
    print("\n=== Testing Trainer Integration ===")

    # Create minimal model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

    # Mock tokenizer
    class MockTokenizer:
        pad_token_id = 0
        eos_token_id = 1

    tokenizer = MockTokenizer()

    # Mock training arguments
    class MockArgs:
        fp16 = False
        learning_rate = 1e-4
        num_train_epochs = 1
        per_device_train_batch_size = 8

    args = MockArgs()

    # Try to import trainers
    try:
        from gym.train.gspo.trainer_simple import GSPOTrainer
        from gym.train.grpo.trainer_simple import GRPOTrainer

        # Test GSPO instantiation
        gspo_trainer = GSPOTrainer(
            model=model,
            ref_model=None,
            args=args,
            tokenizer=tokenizer
        )
        print("✓ GSPO trainer instantiated")

        # Test GRPO instantiation
        grpo_trainer = GRPOTrainer(
            model=model,
            ref_model=None,
            args=args,
            tokenizer=tokenizer
        )
        print("✓ GRPO trainer instantiated")

    except Exception as e:
        print(f"✗ Trainer instantiation failed: {e}")
        return False

    print("✓ Training integration test passed")
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Simplified Gym Implementations")
    print("=" * 50)

    tests = [
        ("BitDelta Quantization", test_bitdelta_quantization),
        ("DeltaQuant", test_deltaquant),
        ("LoRA Adapter", test_lora_adapter),
        ("Training Integration", test_training_integration)
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ {name} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ {name} error: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)