#!/usr/bin/env python3
"""
Test script for quantization implementations
Copyright 2025 Zoo Labs Foundation Inc.
"""

import torch
import torch.nn as nn
import sys
import copy
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from gym.quantization.bitdelta import BitDeltaQuantizer, BitDeltaConfig
from gym.quantization.deltaquant import DeltaQuantizer, DeltaQuantConfig, QuantMethod
from gym.quantization.deltaquant_v2 import enhanced_deltaquant
from gym.quantization.deltasoup import DeltaSoup, DeltaSoupConfig
from gym.quantization.unified import quantize_model
from gym.quantization.benchmark import run_quantization_benchmark, BenchmarkConfig


def create_test_models():
    """Create simple test models"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    # Create base and fine-tuned models
    torch.manual_seed(42)
    base_model = SimpleModel()
    
    # Create fine-tuned model (slightly different weights)
    fine_tuned = SimpleModel()
    with torch.no_grad():
        for (name, param), (_, base_param) in zip(
            fine_tuned.named_parameters(),
            base_model.named_parameters()
        ):
            # Add small delta to simulate fine-tuning
            param.data = base_param.data + torch.randn_like(base_param) * 0.01
    
    return base_model, fine_tuned


def test_bitdelta():
    """Test BitDelta quantization"""
    print("\n" + "="*60)
    print("Testing BitDelta Quantization")
    print("="*60)
    
    base_model, fine_tuned = create_test_models()
    
    # Configure BitDelta
    config = BitDeltaConfig(
        bits=1,
        group_size=128,
        enable_compression=True,
        safety_threshold=0.6
    )
    
    # Apply quantization
    quantizer = BitDeltaQuantizer(config)
    quantized = quantizer.apply_bitdelta(copy.deepcopy(fine_tuned), base_model)
    
    # Test forward pass
    dummy_input = torch.randn(32, 128)
    output = quantized(dummy_input)
    
    # Print results
    compression = quantizer.get_memory_savings()
    print(f"✓ BitDelta quantization successful")
    print(f"  Compression ratio: {compression:.2f}×")
    print(f"  Output shape: {output.shape}")
    
    return True


def test_deltaquant():
    """Test DeltaQuant quantization"""
    print("\n" + "="*60)
    print("Testing DeltaQuant")
    print("="*60)
    
    base_model, fine_tuned = create_test_models()
    
    for bits in [2, 4, 8]:
        print(f"\nTesting {bits}-bit quantization...")
        
        # Configure DeltaQuant
        if bits == 2:
            method = QuantMethod.INT2
        elif bits == 4:
            method = QuantMethod.INT4
        else:
            method = QuantMethod.INT8
        
        config = DeltaQuantConfig(
            method=method,
            per_channel=True,
            symmetric=True
        )
        
        # Apply quantization
        quantizer = DeltaQuantizer(config)
        quantized = quantizer.apply_deltaquant(copy.deepcopy(fine_tuned), base_model)
        
        # Test forward pass
        dummy_input = torch.randn(32, 128)
        output = quantized(dummy_input)
        
        # Print results
        compression = quantizer.get_compression_ratio()
        print(f"  ✓ {bits}-bit quantization successful")
        print(f"    Compression ratio: {compression:.2f}×")
    
    return True


def test_enhanced_deltaquant():
    """Test Enhanced DeltaQuant"""
    print("\n" + "="*60)
    print("Testing Enhanced DeltaQuant (Adaptive)")
    print("="*60)
    
    base_model, fine_tuned = create_test_models()
    
    # Apply enhanced quantization
    quantized, quantizer = enhanced_deltaquant(
        copy.deepcopy(fine_tuned),
        base_model,
        bits=4,
        adaptive=True,
        importance_aware=True,
        error_correction=True
    )
    
    # Test forward pass
    dummy_input = torch.randn(32, 128)
    output = quantized(dummy_input)
    
    # Get statistics
    stats = quantizer.get_compression_stats()
    
    print(f"✓ Enhanced DeltaQuant successful")
    print(f"  Average bits: {stats['average_bits']:.2f}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}×")
    print(f"  Error correction layers: {stats['error_correction_layers']}")
    
    return True


def test_deltasoup():
    """Test DeltaSoup community aggregation"""
    print("\n" + "="*60)
    print("Testing DeltaSoup Community Aggregation")
    print("="*60)
    
    base_model, _ = create_test_models()
    
    # Configure DeltaSoup
    config = DeltaSoupConfig(
        min_contributors=3,
        byzantine_threshold=0.3,
        enable_rewards=True,
        reward_pool=100.0
    )
    
    soup = DeltaSoup(config)
    
    # Simulate multiple contributors
    for i in range(5):
        # Create a slightly different model
        contributor_model = create_test_models()[1]
        
        # Contribute to soup
        contribution_hash = soup.contribute(
            f"user_{i}",
            contributor_model,
            base_model,
            metadata={'training_data': f'dataset_{i}'}
        )
        
        if contribution_hash:
            print(f"  ✓ Contribution {i+1} accepted: {contribution_hash[:8]}...")
    
    # Aggregate contributions
    aggregated = soup.aggregate()
    
    if aggregated:
        print(f"✓ DeltaSoup aggregation successful")
        print(f"  Aggregated {len(aggregated)} layers")
        
        # Get contributor stats
        stats = soup.get_contributor_stats()
        print(f"  Contributors: {len(stats)}")
        for stat in stats[:3]:
            print(f"    - {stat['user_id']}: reputation={stat['reputation_score']:.2f}")
    
    return True


def test_unified():
    """Test Unified Quantizer"""
    print("\n" + "="*60)
    print("Testing Unified Quantizer")
    print("="*60)
    
    base_model, fine_tuned = create_test_models()
    
    for backend in ['bitdelta', 'deltaquant', 'dynamic']:
        print(f"\nTesting {backend} backend...")
        
        # Apply quantization
        quantized, quantizer = quantize_model(
            copy.deepcopy(fine_tuned),
            base_model,
            bits=4,
            backend=backend,
            memory_map=False,  # Disable for test
            auto_mixed_precision=True
        )
        
        # Test forward pass
        dummy_input = torch.randn(32, 128)
        output = quantized(dummy_input)
        
        # Get compression ratio
        compression = quantizer.get_compression_ratio()
        
        print(f"  ✓ {backend} backend successful")
        print(f"    Compression ratio: {compression:.2f}×")
    
    return True


def test_benchmark():
    """Test benchmarking utilities"""
    print("\n" + "="*60)
    print("Testing Benchmark Utilities")
    print("="*60)
    
    base_model, fine_tuned = create_test_models()
    
    # Create simple test loader
    class SimpleDataset:
        def __iter__(self):
            for _ in range(5):
                yield torch.randn(32, 128)
    
    test_loader = SimpleDataset()
    
    # Simple accuracy function
    def eval_fn(model, loader):
        model.eval()
        total = 0
        with torch.no_grad():
            for batch in loader:
                output = model(batch)
                # Simulate accuracy (random for test)
                total += torch.rand(1).item()
        return total / 5
    
    # Run benchmark
    config = BenchmarkConfig(
        num_warmup_runs=1,
        num_benchmark_runs=2,
        bit_widths=[1, 4],
        test_bitdelta=True,
        test_deltaquant=True,
        test_adaptive=True,
        test_unified=False,  # Skip for speed
        save_results=False,
        plot_results=False
    )
    
    from gym.quantization.benchmark import QuantizationBenchmark
    benchmark = QuantizationBenchmark(config)
    results = benchmark.benchmark_model(base_model, fine_tuned, test_loader, eval_fn)
    
    print(f"✓ Benchmark completed")
    print(f"  Methods tested: {len(results)}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Gym Quantization Test Suite")
    print("="*60)
    
    tests = [
        ("BitDelta", test_bitdelta),
        ("DeltaQuant", test_deltaquant),
        ("Enhanced DeltaQuant", test_enhanced_deltaquant),
        ("DeltaSoup", test_deltasoup),
        ("Unified Quantizer", test_unified),
        ("Benchmark", test_benchmark)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                print(f"✗ {name} test failed")
                failed += 1
        except Exception as e:
            print(f"✗ {name} test failed with error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed successfully!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())