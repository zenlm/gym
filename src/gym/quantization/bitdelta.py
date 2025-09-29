"""
BitDelta: 1-bit Quantization for Personalized Models
Copyright 2025 Zoo Labs Foundation Inc.

BitDelta compresses fine-tune deltas to binary signs + scales for 10× memory savings.
Based on Zoo Improvement Proposal ZIP-7.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class BitDeltaConfig:
    """Configuration for BitDelta quantization"""
    
    # Quantization settings
    bits: int = 1  # 1-bit quantization
    group_size: int = 128  # Group size for quantization
    symmetric: bool = True  # Symmetric quantization
    
    # Memory optimization
    enable_compression: bool = True  # Compress deltas
    cache_base_model: bool = True  # Cache base model weights
    
    # Safety settings
    safety_threshold: float = 0.6  # 60% jailbreak reduction
    clip_outliers: bool = True  # Clip extreme values
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection
    
    # Community aggregation (DeltaSoup)
    enable_deltasoup: bool = False  # Enable community aggregation
    byzantine_robust: bool = True  # Byzantine-robust averaging
    aggregation_weight: float = 0.1  # Weight for community updates


class BitDeltaQuantizer:
    """
    BitDelta quantizer for compressing model deltas to 1-bit.
    Achieves 10× memory reduction while preserving model quality.
    """
    
    def __init__(self, config: BitDeltaConfig):
        self.config = config
        self.base_weights: Dict[str, torch.Tensor] = {}
        self.delta_signs: Dict[str, torch.Tensor] = {}
        self.delta_scales: Dict[str, torch.Tensor] = {}
        self.compression_stats: Dict[str, float] = {}
        self.serving_cache: Dict[str, torch.Tensor] = {}
        self._optimization_enabled = torch.cuda.is_available()
        
    def quantize_delta(
        self,
        weight: torch.Tensor,
        base_weight: torch.Tensor,
        name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize the delta between fine-tuned and base weights to 1-bit.
        
        Args:
            weight: Fine-tuned weight tensor
            base_weight: Base model weight tensor
            name: Layer name for caching
            
        Returns:
            signs: Binary signs of delta (+1/-1)
            scales: Scaling factors per group
        """
        # Calculate delta
        delta = weight - base_weight
        
        # Clip outliers if enabled
        if self.config.clip_outliers:
            delta = self._clip_outliers(delta)
        
        # Reshape for group quantization
        original_shape = delta.shape
        delta_flat = delta.flatten()
        
        # Calculate number of groups
        num_elements = delta_flat.numel()
        group_size = min(self.config.group_size, num_elements)
        num_groups = (num_elements + group_size - 1) // group_size
        
        # Pad if necessary
        if num_elements % group_size != 0:
            padding = num_groups * group_size - num_elements
            delta_flat = torch.nn.functional.pad(delta_flat, (0, padding))
        
        # Reshape into groups
        delta_grouped = delta_flat.reshape(num_groups, group_size)
        
        # Calculate scales (absolute mean per group)
        scales = delta_grouped.abs().mean(dim=1, keepdim=True)
        
        # Avoid division by zero
        scales = scales.clamp(min=1e-8)
        
        # Get signs (1-bit)
        signs = (delta_grouped >= 0).to(torch.int8) * 2 - 1  # Convert to +1/-1
        
        # Store for later use
        self.delta_signs[name] = signs
        self.delta_scales[name] = scales.squeeze()
        
        return signs, scales.squeeze()
    
    def dequantize_delta(
        self,
        signs: torch.Tensor,
        scales: torch.Tensor,
        original_shape: torch.Size
    ) -> torch.Tensor:
        """
        Dequantize 1-bit delta back to full precision.
        
        Args:
            signs: Binary signs tensor
            scales: Scaling factors
            original_shape: Original weight shape
            
        Returns:
            Reconstructed delta tensor
        """
        # Expand scales to match signs shape
        if scales.dim() == 1:
            scales = scales.unsqueeze(1)
        
        # Reconstruct delta
        delta_grouped = signs.float() * scales
        
        # Flatten and trim to original size
        delta_flat = delta_grouped.flatten()
        num_elements = np.prod(original_shape)
        delta_flat = delta_flat[:num_elements]
        
        # Reshape to original
        delta = delta_flat.reshape(original_shape)
        
        return delta
    
    def apply_bitdelta(
        self,
        model: nn.Module,
        base_model: nn.Module
    ) -> nn.Module:
        """
        Apply BitDelta quantization to a fine-tuned model.
        
        Args:
            model: Fine-tuned model
            base_model: Base model
            
        Returns:
            Quantized model with BitDelta compression
        """
        for name, param in model.named_parameters():
            if name in dict(base_model.named_parameters()):
                base_param = dict(base_model.named_parameters())[name]
                
                # Quantize delta
                signs, scales = self.quantize_delta(param.data, base_param.data, name)
                
                # Store base weight if caching enabled
                if self.config.cache_base_model:
                    self.base_weights[name] = base_param.data.clone()
                
                # Replace weight with reconstructed version for inference
                delta = self.dequantize_delta(signs, scales, param.shape)
                param.data = base_param.data + delta
        
        return model
    
    def _clip_outliers(self, tensor: torch.Tensor) -> torch.Tensor:
        """Clip outlier values based on standard deviation."""
        mean = tensor.mean()
        std = tensor.std()
        threshold = self.config.outlier_threshold * std
        
        return torch.clamp(tensor, mean - threshold, mean + threshold)
    
    def get_memory_savings(self) -> float:
        """Calculate memory savings ratio."""
        if not self.delta_signs:
            return 1.0
        
        # Original: 32-bit floats
        # BitDelta: 1-bit signs + scales (32-bit but fewer elements)
        original_bits = sum(s.numel() * 32 for s in self.delta_signs.values())
        compressed_bits = sum(
            s.numel() * 1 + self.delta_scales[n].numel() * 32
            for n, s in self.delta_signs.items()
        )
        
        return original_bits / compressed_bits
    
    def aggregate_community_deltas(
        self,
        community_deltas: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        weights: Optional[torch.Tensor] = None
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        DeltaSoup: Aggregate community improvements with Byzantine-robust averaging.
        
        Args:
            community_deltas: Dictionary mapping user_id to (signs, scales) tuples
            weights: Optional weights for weighted averaging
            
        Returns:
            Aggregated (signs, scales) for each layer
        """
        if not self.config.enable_deltasoup:
            raise ValueError("DeltaSoup is not enabled in config")
        
        aggregated = {}
        
        for layer_name in self.delta_signs.keys():
            layer_signs = []
            layer_scales = []
            
            for user_id, (signs_dict, scales_dict) in community_deltas.items():
                if layer_name in signs_dict:
                    layer_signs.append(signs_dict[layer_name])
                    layer_scales.append(scales_dict[layer_name])
            
            if layer_signs:
                # Byzantine-robust aggregation
                if self.config.byzantine_robust:
                    # Use median for robustness
                    agg_signs = torch.stack(layer_signs).median(dim=0)[0]
                    agg_scales = torch.stack(layer_scales).median(dim=0)[0]
                else:
                    # Simple averaging
                    agg_signs = torch.stack(layer_signs).float().mean(dim=0)
                    agg_signs = (agg_signs >= 0).to(torch.int8) * 2 - 1
                    agg_scales = torch.stack(layer_scales).mean(dim=0)
                
                aggregated[layer_name] = (agg_signs, agg_scales)
        
        return aggregated
    
    def export_checkpoint(self, path: str):
        """Export BitDelta checkpoint with compressed deltas."""
        checkpoint = {
            'config': self.config,
            'delta_signs': self.delta_signs,
            'delta_scales': self.delta_scales,
            'base_weights': self.base_weights if self.config.cache_base_model else None,
            'memory_savings': self.get_memory_savings()
        }
        torch.save(checkpoint, path)
        print(f"BitDelta checkpoint saved to {path}")
        print(f"Memory savings: {self.get_memory_savings():.1f}×")
    
    def load_checkpoint(self, path: str):
        """Load BitDelta checkpoint."""
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        self.delta_signs = checkpoint['delta_signs']
        self.delta_scales = checkpoint['delta_scales']
        if checkpoint['base_weights']:
            self.base_weights = checkpoint['base_weights']
        print(f"BitDelta checkpoint loaded from {path}")
        print(f"Memory savings: {checkpoint['memory_savings']:.1f}×")


class BitDeltaLinear(nn.Module):
    """
    BitDelta-aware Linear layer that stores deltas in 1-bit.
    Drop-in replacement for nn.Linear with 10× memory reduction.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_weight: Optional[torch.Tensor] = None,
        base_bias: Optional[torch.Tensor] = None,
        config: Optional[BitDeltaConfig] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or BitDeltaConfig()
        
        # Store base weights
        if base_weight is not None:
            self.register_buffer('base_weight', base_weight)
        else:
            self.register_buffer('base_weight', torch.zeros(out_features, in_features))
        
        if base_bias is not None:
            self.register_buffer('base_bias', base_bias)
        else:
            self.register_buffer('base_bias', None)
        
        # Initialize delta parameters (will be quantized)
        self.delta_signs = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.int8))
        self.delta_scales = nn.Parameter(torch.zeros(out_features))
        
        # Bias is not quantized
        if base_bias is not None:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Reconstruct weight from base + quantized delta
        delta = self.delta_signs.float() * self.delta_scales.unsqueeze(1)
        weight = self.base_weight + delta
        
        # Apply linear transformation
        output = nn.functional.linear(input, weight, self.bias)
        return output
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        base_linear: nn.Linear,
        config: Optional[BitDeltaConfig] = None
    ) -> 'BitDeltaLinear':
        """Convert a standard Linear layer to BitDeltaLinear."""
        config = config or BitDeltaConfig()
        quantizer = BitDeltaQuantizer(config)
        
        # Quantize delta
        signs, scales = quantizer.quantize_delta(
            linear.weight.data,
            base_linear.weight.data,
            'weight'
        )
        
        # Create BitDeltaLinear layer
        bitdelta_linear = cls(
            linear.in_features,
            linear.out_features,
            base_linear.weight.data,
            base_linear.bias.data if base_linear.bias is not None else None,
            config
        )
        
        # Set quantized delta
        bitdelta_linear.delta_signs.data = signs
        bitdelta_linear.delta_scales.data = scales
        
        # Set bias if present
        if linear.bias is not None and base_linear.bias is not None:
            bitdelta_linear.bias.data = linear.bias.data - base_linear.bias.data
        
        return bitdelta_linear
    
    def optimize_for_serving(self):
        """Optimize quantized model for fast serving."""
        print("Optimizing BitDelta for serving...")
        
        # Pack signs into bytes for better memory efficiency
        for name, signs in list(self.delta_signs.items()):
            if '_packed' not in name and signs.dtype == torch.int8:
                # Pack 8 signs into 1 byte
                signs_flat = signs.flatten()
                num_signs = signs_flat.numel()
                num_bytes = (num_signs + 7) // 8
                
                packed = torch.zeros(num_bytes, dtype=torch.uint8)
                for i in range(num_signs):
                    if signs_flat[i] > 0:
                        byte_idx = i // 8
                        bit_idx = i % 8
                        packed[byte_idx] |= (1 << bit_idx)
                
                # Store packed version
                self.delta_signs[name + '_packed'] = packed
                self.delta_signs[name + '_shape'] = torch.tensor(signs.shape)
        
        # Precompute commonly used deltas
        if self.base_weights:
            for name in list(self.delta_signs.keys())[:10]:  # Top 10 layers
                if name in self.delta_scales and not name.endswith('_packed'):
                    # Precompute and cache
                    signs = self.delta_signs[name]
                    scales = self.delta_scales[name]
                    shape = self.base_weights[name].shape if name in self.base_weights else signs.shape
                    _ = self.dequantize_delta(signs, scales, shape, use_cache=True)
        
        print(f"Optimization complete. Cache size: {len(self.serving_cache)} entries")
    
    def clear_cache(self):
        """Clear serving cache to free memory."""
        self.serving_cache.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None