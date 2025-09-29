"""
Simplified BitDelta Quantization
Copyright 2025 Zoo Labs Foundation Inc.

1-bit quantization for model deltas:
- Binary signs + scales only
- 10× memory reduction
- Simple, explicit implementation
"""

import torch
from typing import Tuple, Dict

class BitDeltaQuantizer:
    """
    Compress model deltas to 1-bit.

    Design constraints:
    - Store only signs (+1/-1) and scales
    - Group-wise quantization for accuracy
    - No complex abstractions
    """

    def __init__(self, group_size: int = 128):
        """Initialize with fixed group size."""
        self.group_size = group_size
        self.base_weights: Dict[str, torch.Tensor] = {}
        self.delta_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def quantize(
        self,
        weight: torch.Tensor,
        base_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weight delta to 1-bit.

        Args:
            weight: Fine-tuned weights
            base_weight: Base model weights

        Returns:
            signs: Binary signs tensor
            scales: Scale factors per group
        """
        # Compute delta
        delta = weight - base_weight

        # Flatten and compute groups
        delta_flat = delta.flatten()
        num_elements = delta_flat.numel()

        # Handle small tensors
        if num_elements < self.group_size:
            # Single group for small tensors
            scale = delta_flat.abs().mean()
            signs = delta_flat.sign()
            return signs.reshape(delta.shape), scale.unsqueeze(0)

        # Pad to multiple of group_size
        pad_size = (self.group_size - num_elements % self.group_size) % self.group_size
        if pad_size > 0:
            delta_flat = torch.nn.functional.pad(delta_flat, (0, pad_size))

        # Reshape into groups
        num_groups = delta_flat.numel() // self.group_size
        delta_grouped = delta_flat.reshape(num_groups, self.group_size)

        # Compute scales (mean absolute value per group)
        scales = delta_grouped.abs().mean(dim=1)

        # Get signs
        signs = delta_grouped.sign()

        # Remove padding from signs
        if pad_size > 0:
            signs_flat = signs.flatten()[:-pad_size]
            signs = signs_flat.reshape(delta.shape)
        else:
            signs = signs.reshape(delta.shape)

        return signs, scales

    def dequantize(
        self,
        base_weight: torch.Tensor,
        signs: torch.Tensor,
        scales: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct weight from quantized delta.

        Args:
            base_weight: Base model weights
            signs: Binary signs
            scales: Scale factors

        Returns:
            weight: Reconstructed weight tensor
        """
        # Flatten signs
        signs_flat = signs.flatten()
        num_elements = signs_flat.numel()

        # Handle single scale (small tensor)
        if scales.numel() == 1:
            delta = signs * scales
            return base_weight + delta

        # Pad if necessary
        pad_size = (self.group_size - num_elements % self.group_size) % self.group_size
        if pad_size > 0:
            signs_flat = torch.nn.functional.pad(signs_flat, (0, pad_size))

        # Reshape and apply scales
        num_groups = scales.numel()
        signs_grouped = signs_flat.reshape(num_groups, self.group_size)

        # Broadcast scales and multiply
        delta_grouped = signs_grouped * scales.unsqueeze(1)

        # Flatten and remove padding
        delta_flat = delta_grouped.flatten()
        if pad_size > 0:
            delta_flat = delta_flat[:-pad_size]

        # Reshape to original shape
        delta = delta_flat.reshape(signs.shape)

        return base_weight + delta

    def compress_model(
        self,
        model: torch.nn.Module,
        base_model: torch.nn.Module
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compress entire model to 1-bit deltas.

        Args:
            model: Fine-tuned model
            base_model: Base model

        Returns:
            compressed: Dictionary of quantized deltas
        """
        compressed = {}

        for (name, param), (_, base_param) in zip(
            model.named_parameters(),
            base_model.named_parameters()
        ):
            if param.requires_grad:
                signs, scales = self.quantize(param.data, base_param.data)
                compressed[name] = (signs, scales)
                # Cache base weight for fast decompression
                self.base_weights[name] = base_param.data

        return compressed

    def decompress_model(
        self,
        compressed: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Decompress model from 1-bit deltas.

        Args:
            compressed: Dictionary of quantized deltas

        Returns:
            weights: Dictionary of reconstructed weights
        """
        weights = {}

        for name, (signs, scales) in compressed.items():
            if name in self.base_weights:
                weight = self.dequantize(self.base_weights[name], signs, scales)
                weights[name] = weight

        return weights

    def memory_usage(
        self,
        compressed: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Calculate memory usage statistics.

        Args:
            compressed: Compressed model deltas

        Returns:
            stats: Memory usage statistics
        """
        total_params = 0
        compressed_bits = 0

        for name, (signs, scales) in compressed.items():
            num_params = signs.numel()
            total_params += num_params

            # Signs: 1 bit per parameter
            # Scales: 32 bits per group
            bits = num_params + scales.numel() * 32
            compressed_bits += bits

        original_bits = total_params * 32  # Assuming fp32
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0

        return {
            "total_parameters": total_params,
            "original_mb": original_bits / (8 * 1024 * 1024),
            "compressed_mb": compressed_bits / (8 * 1024 * 1024),
            "compression_ratio": compression_ratio
        }