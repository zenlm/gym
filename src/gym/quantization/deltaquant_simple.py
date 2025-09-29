"""
Simplified DeltaQuant - Efficient Delta Quantization
Copyright 2025 Zoo Labs Foundation Inc.

Design principles:
- Quantize only the changes (deltas)
- Simple bit-packing for storage
- No unnecessary abstractions
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional

class DeltaQuantizer:
    """
    Quantize model deltas with configurable bit width.

    Invariants:
    - Base model remains unchanged
    - Quantize only differences
    - Symmetric quantization for simplicity
    """

    def __init__(self, bits: int = 4, per_channel: bool = True):
        """
        Initialize quantizer.

        Args:
            bits: Bit width for quantization (1-8)
            per_channel: Quantize per channel vs per tensor
        """
        assert 1 <= bits <= 8, f"Bits must be between 1 and 8, got {bits}"
        self.bits = bits
        self.per_channel = per_channel
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1

    def quantize_tensor(
        self,
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor to specified bit width.

        Args:
            tensor: Input tensor

        Returns:
            quantized: Quantized tensor (int8)
            scale: Scale factors for dequantization
        """
        # Compute scale
        if self.per_channel and tensor.dim() > 1:
            # Per-channel: quantize first dimension separately
            abs_max = tensor.abs().view(tensor.shape[0], -1).max(dim=1)[0]
            scale = abs_max / self.qmax
            scale = scale.clamp(min=1e-8)
            scale = scale.view(-1, *([1] * (tensor.dim() - 1)))
        else:
            # Per-tensor: single scale
            abs_max = tensor.abs().max()
            scale = abs_max / self.qmax
            scale = scale.clamp(min=1e-8)

        # Quantize
        quantized = torch.round(tensor / scale).clamp(self.qmin, self.qmax)

        return quantized.to(torch.int8), scale

    def dequantize_tensor(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantize tensor.

        Args:
            quantized: Quantized tensor
            scale: Scale factors

        Returns:
            tensor: Dequantized tensor
        """
        return quantized.float() * scale

    def quantize_delta(
        self,
        weight: torch.Tensor,
        base_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize the delta between weights.

        Args:
            weight: Fine-tuned weight
            base_weight: Base weight

        Returns:
            quantized: Quantized delta
            scale: Scale factor
        """
        delta = weight - base_weight
        return self.quantize_tensor(delta)

    def reconstruct_weight(
        self,
        base_weight: torch.Tensor,
        quantized_delta: torch.Tensor,
        scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct weight from quantized delta.

        Args:
            base_weight: Base weight
            quantized_delta: Quantized delta
            scale: Scale factor

        Returns:
            weight: Reconstructed weight
        """
        delta = self.dequantize_tensor(quantized_delta, scale)
        return base_weight + delta

    def compress_model(
        self,
        model: torch.nn.Module,
        base_model: torch.nn.Module
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compress model by quantizing deltas.

        Args:
            model: Fine-tuned model
            base_model: Base model

        Returns:
            compressed: Dictionary of quantized deltas and scales
        """
        compressed = {}

        for (name, param), (_, base_param) in zip(
            model.named_parameters(),
            base_model.named_parameters()
        ):
            if param.requires_grad:
                quantized, scale = self.quantize_delta(param.data, base_param.data)
                compressed[name] = (quantized, scale)

        return compressed

    def apply_to_model(
        self,
        model: torch.nn.Module,
        base_model: torch.nn.Module,
        compressed: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ):
        """
        Apply quantized deltas to model in-place.

        Args:
            model: Model to update
            base_model: Base model
            compressed: Quantized deltas
        """
        for (name, param), (_, base_param) in zip(
            model.named_parameters(),
            base_model.named_parameters()
        ):
            if name in compressed:
                quantized, scale = compressed[name]
                reconstructed = self.reconstruct_weight(
                    base_param.data,
                    quantized,
                    scale
                )
                param.data.copy_(reconstructed)

    def compute_error(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute reconstruction error metrics.

        Args:
            original: Original tensor
            reconstructed: Reconstructed tensor

        Returns:
            metrics: Error metrics
        """
        diff = original - reconstructed
        return {
            "mse": (diff ** 2).mean().item(),
            "mae": diff.abs().mean().item(),
            "max_error": diff.abs().max().item(),
            "relative_error": (diff.abs() / (original.abs() + 1e-8)).mean().item()
        }

    def memory_stats(
        self,
        compressed: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Calculate memory usage statistics.

        Args:
            compressed: Compressed model

        Returns:
            stats: Memory statistics
        """
        total_params = 0
        compressed_bytes = 0
        scale_bytes = 0

        for name, (quantized, scale) in compressed.items():
            num_params = quantized.numel()
            total_params += num_params

            # Quantized values: bits per parameter
            compressed_bytes += num_params * self.bits / 8

            # Scales: 32 bits each
            scale_bytes += scale.numel() * 4

        original_bytes = total_params * 4  # fp32
        total_compressed = compressed_bytes + scale_bytes

        return {
            "original_mb": original_bytes / (1024 * 1024),
            "compressed_mb": total_compressed / (1024 * 1024),
            "compression_ratio": original_bytes / total_compressed if total_compressed > 0 else 0,
            "bits_per_param": (total_compressed * 8) / total_params if total_params > 0 else 0
        }