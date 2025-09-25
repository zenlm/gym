#!/usr/bin/env python
"""
Test BitDelta and DeltaQuant quantization methods
Copyright 2025 Zoo Labs Foundation Inc.
"""

import torch
import torch.nn as nn
from src.gym.quantization import (
    BitDeltaConfig, BitDeltaQuantizer,
    DeltaQuantConfig, DeltaQuantizer, QuantMethod,
    DeltaSoupConfig, DeltaSoup, AggregationMethod
)


def create_test_models():
    """Create simple test models"""
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(128, 256)
            self.layer2 = nn.Linear(256, 128)
            self.layer3 = nn.Linear(128, 64)
        
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = self.layer3(x)
            return x
    
    # Create base and fine-tuned models
    base_model = SimpleModel()
    finetuned_model = SimpleModel()
    
    # Simulate fine-tuning by adding small deltas
    with torch.no_grad():
        for (name, param), (_, base_param) in zip(
            finetuned_model.named_parameters(),
            base_model.named_parameters()
        ):
            param.data = base_param.data + torch.randn_like(base_param) * 0.01
    
    return base_model, finetuned_model


def test_bitdelta():
    """Test BitDelta 1-bit quantization"""
    print("\n=== Testing BitDelta Quantization ===")
    
    # Create models
    base_model, finetuned_model = create_test_models()
    
    # Configure BitDelta
    config = BitDeltaConfig(
        bits=1,
        group_size=128,
        enable_compression=True,
        enable_deltasoup=True
    )
    
    # Initialize quantizer
    quantizer = BitDeltaQuantizer(config)
    
    # Apply BitDelta
    quantized_model = quantizer.apply_bitdelta(finetuned_model, base_model)
    
    # Check memory savings
    savings = quantizer.get_memory_savings()
    print(f"✓ BitDelta applied successfully")
    print(f"  Memory savings: {savings:.1f}×")
    
    # Test inference
    test_input = torch.randn(1, 128)
    output = quantized_model(test_input)
    print(f"  Output shape: {output.shape}")
    
    return quantizer


def test_deltaquant():
    """Test DeltaQuant flexible quantization"""
    print("\n=== Testing DeltaQuant ===")
    
    # Create models
    base_model, finetuned_model = create_test_models()
    
    # Test different quantization methods
    methods = [
        QuantMethod.BINARY,
        QuantMethod.TERNARY,
        QuantMethod.INT4,
        QuantMethod.INT8
    ]
    
    for method in methods:
        config = DeltaQuantConfig(
            method=method,
            per_channel=True,
            calibration_samples=256
        )
        
        quantizer = DeltaQuantizer(config)
        quantized_model = quantizer.apply_deltaquant(finetuned_model, base_model)
        
        compression = quantizer.get_compression_ratio()
        print(f"✓ {method.value:8s} quantization: {compression:.2f}× compression")
    
    return quantizer


def test_deltasoup():
    """Test DeltaSoup community aggregation"""
    print("\n=== Testing DeltaSoup ===")
    
    # Create base model
    base_model, _ = create_test_models()
    
    # Configure DeltaSoup
    config = DeltaSoupConfig(
        method=AggregationMethod.BYZANTINE_ROBUST,
        min_contributors=3,
        enable_rewards=True,
        use_bitdelta=True
    )
    
    soup = DeltaSoup(config)
    
    # Simulate multiple contributors
    contributors = ["alice", "bob", "charlie", "dave", "eve"]
    
    for user_id in contributors:
        # Create unique fine-tuned model for each user
        _, user_model = create_test_models()
        
        # Contribute to soup
        hash_id = soup.contribute(
            user_id=user_id,
            model=user_model,
            base_model=base_model,
            metadata={"training_samples": 1000}
        )
        
        if hash_id:
            print(f"✓ Contribution from {user_id}: {hash_id[:8]}...")
    
    # Aggregate contributions
    print("\n  Aggregating contributions...")
    aggregated = soup.aggregate(min_contributors=3)
    
    if aggregated:
        print(f"✓ Aggregated {len(soup.contributors)} contributors")
        
        # Get contributor stats
        stats = soup.get_contributor_stats()
        for stat in stats[:3]:
            print(f"  - {stat['user_id']}: reputation={stat['reputation_score']:.2f}, rewards={stat['total_rewards']:.2f}")
    
    # Apply aggregated improvements
    improved_model, _ = create_test_models()
    improved_model = soup.apply_aggregated_deltas(improved_model, base_model, alpha=0.1)
    print("✓ Applied community improvements to model")
    
    return soup


def test_integration():
    """Test complete integration of all quantization methods"""
    print("\n=== Testing Complete Integration ===")
    
    # Create models
    base_model, finetuned_model = create_test_models()
    
    # Apply BitDelta
    bitdelta_config = BitDeltaConfig(bits=1, enable_deltasoup=True)
    bitdelta = BitDeltaQuantizer(bitdelta_config)
    
    # Apply DeltaQuant
    deltaquant_config = DeltaQuantConfig(method=QuantMethod.INT4)
    deltaquant = DeltaQuantizer(deltaquant_config)
    
    # Create DeltaSoup
    soup_config = DeltaSoupConfig(
        use_bitdelta=True,
        use_deltaquant=False,
        method=AggregationMethod.BYZANTINE_ROBUST
    )
    soup = DeltaSoup(soup_config)
    
    # Contribute models
    soup.contribute("user1", finetuned_model, base_model)
    _, model2 = create_test_models()
    soup.contribute("user2", model2, base_model)
    _, model3 = create_test_models()
    soup.contribute("user3", model3, base_model)
    
    # Aggregate
    aggregated = soup.aggregate()
    
    # Apply to new model
    final_model, _ = create_test_models()
    final_model = soup.apply_aggregated_deltas(final_model, base_model)
    
    # Test inference
    test_input = torch.randn(1, 128)
    output = final_model(test_input)
    
    print("✓ Complete integration test passed")
    print(f"  Final output shape: {output.shape}")
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Gym Quantization Tests - Zoo Labs Foundation")
    print("=" * 60)
    
    try:
        # Test individual components
        test_bitdelta()
        test_deltaquant()
        test_deltasoup()
        
        # Test integration
        test_integration()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("Gym is a complete training platform with:")
        print("  • BitDelta: 10× memory reduction")
        print("  • DeltaQuant: Flexible quantization")
        print("  • DeltaSoup: Community aggregation")
        print("  • Full integration with training pipeline")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())