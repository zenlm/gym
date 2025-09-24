#!/usr/bin/env python
# Test GRPO implementation

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_grpo_import():
    """Test that GRPO components can be imported"""
    try:
        from gym.train.grpo import GRPOTrainer, run_grpo
        print("✅ GRPO imports successful")
        return True
    except ImportError as e:
        print(f"❌ GRPO import failed: {e}")
        return False

def test_gspo_import():
    """Test that GSPO components can be imported"""
    try:
        from gym.train.gspo import GSPOTrainer, run_gspo
        print("✅ GSPO imports successful")
        return True
    except ImportError as e:
        print(f"❌ GSPO import failed: {e}")
        return False

def test_cli_stages():
    """Test that GRPO/GSPO stages are recognized"""
    try:
        from gym.hparams.finetuning_args import FinetuningArguments
        from dataclasses import fields
        
        # Find the stage field
        for field in fields(FinetuningArguments):
            if field.name == "stage":
                # Check if grpo and gspo are in the allowed values
                stage_type = field.type
                if "grpo" in str(stage_type) and "gspo" in str(stage_type):
                    print("✅ GRPO/GSPO stages added to FinetuningArguments")
                    return True
                else:
                    print("❌ GRPO/GSPO stages not found in allowed values")
                    return False
        
        print("❌ Stage field not found")
        return False
    except Exception as e:
        print(f"❌ Error checking stages: {e}")
        return False

def test_tuner_integration():
    """Test that GRPO/GSPO are integrated in the tuner"""
    try:
        # Read the tuner file to check integration
        tuner_path = os.path.join(os.path.dirname(__file__), 'src/gym/train/tuner.py')
        with open(tuner_path, 'r') as f:
            content = f.read()
        
        has_grpo_import = "from .grpo import run_grpo" in content
        has_gspo_import = "from .gspo import run_gspo" in content
        has_grpo_stage = 'finetuning_args.stage == "grpo"' in content
        has_gspo_stage = 'finetuning_args.stage == "gspo"' in content
        
        if all([has_grpo_import, has_gspo_import, has_grpo_stage, has_gspo_stage]):
            print("✅ GRPO/GSPO integrated in tuner")
            return True
        else:
            print("❌ GRPO/GSPO not fully integrated in tuner")
            if not has_grpo_import:
                print("  - Missing GRPO import")
            if not has_gspo_import:
                print("  - Missing GSPO import")
            if not has_grpo_stage:
                print("  - Missing GRPO stage handling")
            if not has_gspo_stage:
                print("  - Missing GSPO stage handling")
            return False
    except Exception as e:
        print(f"❌ Error checking tuner integration: {e}")
        return False

def main():
    """Run all tests"""
    print("🏋️ Testing GRPO/GSPO Implementation in Gym")
    print("-" * 50)
    
    tests = [
        test_grpo_import,
        test_gspo_import,
        test_cli_stages,
        test_tuner_integration,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    # Summary
    print("-" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("\n✨ GRPO and GSPO are ready to use!")
        print("\nExample usage:")
        print("  gym-cli train --stage grpo --config configs/grpo_qwen3.yaml")
        print("  gym-cli train --stage gspo --config configs/gspo_qwen3_moe.yaml")
    else:
        print(f"⚠️ Some tests failed ({passed}/{total})")
        print("Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)