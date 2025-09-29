#!/usr/bin/env python3
"""
Zen Gym Training Tests
Tests core training functionality for all Zen models.
"""

import os
import pytest
import torch
from pathlib import Path


class TestZenGymTraining:
    """Test suite for Zen Gym training functionality"""

    def test_import_llamafactory(self):
        """Test that LLaMA Factory can be imported"""
        try:
            import llamafactory
            assert llamafactory is not None
        except ImportError:
            pytest.skip("LLaMA Factory not installed")

    def test_lora_config(self):
        """Test LoRA configuration parsing"""
        from peft import LoraConfig

        config = LoraConfig(
            r=64,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM"
        )

        assert config.r == 64
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1

    def test_quantization_config(self):
        """Test quantization configuration"""
        from transformers import BitsAndBytesConfig

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"

    def test_model_registry(self):
        """Test that Zen models are registered"""
        # Check if config files exist
        configs_path = Path("configs")
        assert configs_path.exists(), "configs directory not found"

        zen_configs = [
            "zen_nano_lora.yaml",
            "zen_eco.yaml",
            "zen_agent.yaml"
        ]

        for config in zen_configs:
            config_file = configs_path / config
            if config_file.exists():
                assert config_file.is_file(), f"{config} should be a file"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_availability(self):
        """Test CUDA setup"""
        assert torch.cuda.is_available(), "CUDA should be available"
        assert torch.cuda.device_count() > 0, "At least one CUDA device required"

        device = torch.device("cuda:0")
        x = torch.randn(10, 10).to(device)
        assert x.device.type == "cuda"

    def test_flash_attention_available(self):
        """Test if FlashAttention is available"""
        try:
            import flash_attn
            print(f"FlashAttention version: {flash_attn.__version__}")
        except ImportError:
            pytest.skip("FlashAttention not installed")

    def test_unsloth_available(self):
        """Test if Unsloth is available"""
        try:
            import unsloth
            print("Unsloth is available")
        except ImportError:
            pytest.skip("Unsloth not installed")


@pytest.mark.slow
class TestZenGymIntegration:
    """Integration tests for full training pipeline"""

    def test_zen_nano_lora_training(self):
        """Test LoRA training for zen-nano"""
        pytest.skip("Requires model download and GPU")

    def test_zen_eco_grpo_training(self):
        """Test GRPO training for zen-eco"""
        pytest.skip("Requires model download and GPU")

    def test_zen_agent_tool_calling(self):
        """Test tool-calling training for zen-agent"""
        pytest.skip("Requires model download and dataset")


class TestZenGymConfig:
    """Test configuration file parsing"""

    def test_yaml_configs_valid(self):
        """Test that all YAML configs are valid"""
        import yaml
        from pathlib import Path

        configs_path = Path("configs")
        if not configs_path.exists():
            pytest.skip("configs directory not found")

        for config_file in configs_path.glob("zen_*.yaml"):
            with open(config_file) as f:
                config = yaml.safe_load(f)
                assert config is not None, f"{config_file.name} is empty"
                print(f"Validated {config_file.name}")

    def test_model_paths_exist(self):
        """Test that model paths in configs are valid"""
        pytest.skip("Requires model download")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])