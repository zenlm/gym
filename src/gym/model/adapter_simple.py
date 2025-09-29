"""
Simplified Model Adapter
Copyright 2025 Zoo Labs Foundation Inc.

Minimal adapter implementation:
- LoRA support only (most common)
- No complex abstractions
- Explicit configuration
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
from peft import LoraConfig, get_peft_model, TaskType

def setup_lora_adapter(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    Setup LoRA adapter on model.

    Args:
        model: Base model
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout rate
        target_modules: Modules to apply LoRA to

    Returns:
        model: Model with LoRA adapters
    """
    # Default target modules for common architectures
    if target_modules is None:
        # Try to detect model type and set appropriate modules
        if hasattr(model, "config"):
            model_type = model.config.model_type
            if "llama" in model_type.lower():
                target_modules = ["q_proj", "v_proj"]
            elif "gpt" in model_type.lower():
                target_modules = ["c_attn"]
            elif "qwen" in model_type.lower():
                target_modules = ["q_proj", "v_proj"]
            else:
                # Fallback: find all linear layers
                target_modules = find_linear_modules(model)

    # Create LoRA configuration
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model

def find_linear_modules(model: nn.Module, exclude: List[str] = None) -> List[str]:
    """
    Find all linear modules in model.

    Args:
        model: Model to search
        exclude: Modules to exclude

    Returns:
        module_names: Names of linear modules
    """
    if exclude is None:
        exclude = ["lm_head", "embed_tokens", "wte", "wpe"]

    linear_modules = set()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Extract the module name pattern
            module_name = name.split(".")[-1]
            if not any(ex in name for ex in exclude):
                linear_modules.add(module_name)

    return list(linear_modules)

def freeze_model(
    model: nn.Module,
    trainable_layers: Optional[List[str]] = None
) -> nn.Module:
    """
    Freeze model except specified layers.

    Args:
        model: Model to freeze
        trainable_layers: Layers to keep trainable

    Returns:
        model: Model with frozen parameters
    """
    if trainable_layers is None:
        trainable_layers = []

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specified layers
    for name, param in model.named_parameters():
        if any(layer in name for layer in trainable_layers):
            param.requires_grad = True

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen model: {trainable:,} / {total:,} trainable "
          f"({100 * trainable / total:.2f}%)")

    return model

def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights back into base model.

    Args:
        model: Model with LoRA adapters

    Returns:
        model: Model with merged weights
    """
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
        print("LoRA weights merged into base model")
    else:
        print("Model does not have LoRA adapters")

    return model

def save_lora_weights(model: nn.Module, save_path: str):
    """
    Save only LoRA adapter weights.

    Args:
        model: Model with LoRA adapters
        save_path: Path to save weights
    """
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_path)
        print(f"LoRA weights saved to {save_path}")
    else:
        # Fallback: save state dict
        lora_state = {}
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                lora_state[name] = param.data

        torch.save(lora_state, f"{save_path}/lora_weights.pt")
        print(f"LoRA state dict saved to {save_path}/lora_weights.pt")

def load_lora_weights(model: nn.Module, load_path: str) -> nn.Module:
    """
    Load LoRA adapter weights.

    Args:
        model: Base model
        load_path: Path to load weights from

    Returns:
        model: Model with loaded LoRA weights
    """
    try:
        # Try PEFT format first
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, load_path)
        print(f"LoRA weights loaded from {load_path}")
    except:
        # Fallback: load state dict
        lora_state = torch.load(f"{load_path}/lora_weights.pt")
        model.load_state_dict(lora_state, strict=False)
        print(f"LoRA state dict loaded from {load_path}/lora_weights.pt")

    return model

class SimpleAdapter:
    """
    Simple adapter manager for common operations.
    """

    def __init__(self, model: nn.Module, adapter_type: str = "lora"):
        """
        Initialize adapter manager.

        Args:
            model: Base model
            adapter_type: Type of adapter (only 'lora' supported)
        """
        assert adapter_type == "lora", "Only LoRA adapters are supported"
        self.model = model
        self.adapter_type = adapter_type

    def add_adapter(
        self,
        name: str = "default",
        rank: int = 8,
        alpha: int = 16,
        **kwargs
    ):
        """Add adapter to model."""
        self.model = setup_lora_adapter(
            self.model,
            rank=rank,
            alpha=alpha,
            **kwargs
        )

    def freeze_base(self):
        """Freeze base model weights."""
        for name, param in self.model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

    def unfreeze_base(self):
        """Unfreeze base model weights."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> Dict[str, int]:
        """Get trainable parameter statistics."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "percentage": 100 * trainable / total if total > 0 else 0
        }