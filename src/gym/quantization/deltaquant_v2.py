"""
Enhanced DeltaQuant: Advanced Delta Quantization with Dynamic Precision
Copyright 2025 Zoo Labs Foundation Inc.

Enhanced version with dynamic quantization, mixed precision, and QAT support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum


class QuantMethod(Enum):
    """Quantization methods for DeltaQuant"""
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    BINARY = "binary"
    TERNARY = "ternary"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"  # New: Adaptive bit allocation
    MIXED = "mixed"  # New: Mixed precision per layer


@dataclass
class EnhancedDeltaQuantConfig:
    """Enhanced configuration for DeltaQuant"""
    
    # Quantization settings
    method: QuantMethod = QuantMethod.ADAPTIVE
    base_bits: int = 4  # Base bit width
    min_bits: int = 1  # Minimum bits for adaptive
    max_bits: int = 8  # Maximum bits for adaptive
    
    # Advanced quantization
    use_log_quantization: bool = False  # Log-scale quantization
    use_vector_quantization: bool = False  # Vector quantization
    codebook_size: int = 256  # VQ codebook size
    
    # Dynamic quantization
    dynamic_threshold: float = 0.01  # Threshold for dynamic selection
    importance_aware: bool = True  # Use importance scoring
    gradient_based_bits: bool = False  # Allocate bits based on gradients
    
    # Optimization
    group_size: int = 128
    per_channel: bool = True
    symmetric: bool = False  # Asymmetric can be better
    
    # QAT settings
    qat_enabled: bool = False
    qat_temperature: float = 1.0
    qat_noise_scale: float = 0.01
    learnable_quantization: bool = False  # Learn quantization params
    
    # Error recovery
    error_correction: bool = True  # Store error residuals
    max_error: float = 0.001
    
    # Performance
    use_cuda_kernel: bool = True
    fused_operations: bool = True  # Fuse quant/dequant with matmul


class AdaptiveDeltaQuantizer:
    """
    Enhanced DeltaQuantizer with adaptive bit allocation and advanced strategies.
    """
    
    def __init__(self, config: EnhancedDeltaQuantConfig):
        self.config = config
        self.quantized_deltas: Dict[str, Any] = {}
        self.quantization_params: Dict[str, Any] = {}
        self.bit_allocation: Dict[str, int] = {}
        self.error_residuals: Dict[str, torch.Tensor] = {}
        self.importance_scores: Dict[str, float] = {}
        
        # Vector quantization codebook
        self.codebooks: Dict[str, torch.Tensor] = {}
        
        # Learnable parameters for QAT
        if config.learnable_quantization:
            self.learnable_scales = nn.ParameterDict()
            self.learnable_zeros = nn.ParameterDict()
    
    def compute_importance(
        self,
        weight: torch.Tensor,
        name: str,
        gradients: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute layer importance for bit allocation.
        
        Args:
            weight: Weight tensor
            name: Layer name
            gradients: Optional gradient information
            
        Returns:
            Importance score [0, 1]
        """
        scores = []
        
        # Weight magnitude importance
        weight_importance = weight.abs().mean().item()
        scores.append(min(1.0, weight_importance * 10))
        
        # Gradient-based importance
        if gradients is not None and self.config.gradient_based_bits:
            grad_importance = gradients.abs().mean().item()
            scores.append(min(1.0, grad_importance * 100))
        
        # Layer type importance
        if 'attention' in name.lower() or 'output' in name.lower():
            scores.append(0.9)
        elif 'embedding' in name.lower():
            scores.append(0.8)
        elif 'norm' in name.lower():
            scores.append(0.7)
        else:
            scores.append(0.5)
        
        # Variance-based importance (high variance = important)
        variance = weight.var().item()
        variance_score = min(1.0, variance * 100)
        scores.append(variance_score)
        
        importance = np.mean(scores)
        self.importance_scores[name] = importance
        
        return importance
    
    def allocate_bits(
        self,
        importance: float,
        weight_size: int
    ) -> int:
        """
        Allocate bits based on importance and weight size.
        
        Args:
            importance: Importance score [0, 1]
            weight_size: Number of parameters
            
        Returns:
            Allocated bit width
        """
        if self.config.method == QuantMethod.ADAPTIVE:
            # Adaptive bit allocation based on importance
            bit_range = self.config.max_bits - self.config.min_bits
            allocated = self.config.min_bits + int(importance * bit_range)
            
            # Adjust for weight size (smaller weights can use more bits)
            if weight_size < 1024:
                allocated = min(self.config.max_bits, allocated + 2)
            elif weight_size < 10240:
                allocated = min(self.config.max_bits, allocated + 1)
            
            return allocated
        else:
            # Fixed bit allocation
            return self.config.base_bits
    
    def vector_quantize(
        self,
        tensor: torch.Tensor,
        name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply vector quantization to tensor.
        
        Args:
            tensor: Input tensor
            name: Layer name for codebook
            
        Returns:
            Indices and codebook
        """
        # Reshape tensor for VQ
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        
        # Initialize or get codebook
        if name not in self.codebooks:
            # Initialize codebook with k-means
            num_codes = min(self.config.codebook_size, tensor_flat.numel() // 4)
            indices = torch.randperm(tensor_flat.numel())[:num_codes]
            self.codebooks[name] = tensor_flat[indices].clone()
        
        codebook = self.codebooks[name]
        
        # Find nearest codes (simplified - use CDist in practice)
        distances = torch.cdist(
            tensor_flat.unsqueeze(0),
            codebook.unsqueeze(0)
        ).squeeze(0)
        indices = distances.argmin(dim=1)
        
        # Store VQ parameters
        vq_params = {
            'codebook': codebook,
            'indices': indices,
            'shape': original_shape
        }
        
        return indices, vq_params
    
    def log_quantize(
        self,
        tensor: torch.Tensor,
        bits: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply logarithmic quantization for better dynamic range.
        
        Args:
            tensor: Input tensor
            bits: Bit width
            
        Returns:
            Quantized tensor and parameters
        """
        # Handle zero values
        eps = 1e-8
        sign = torch.sign(tensor)
        abs_tensor = torch.abs(tensor) + eps
        
        # Log transform
        log_tensor = torch.log2(abs_tensor)
        
        # Quantize log values
        log_min = log_tensor.min()
        log_max = log_tensor.max()
        
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        
        scale = (log_max - log_min) / (qmax - qmin)
        zero_point = qmin - log_min / scale
        
        quantized_log = torch.round(log_tensor / scale + zero_point)
        quantized_log = torch.clamp(quantized_log, qmin, qmax).to(torch.int8)
        
        # Combine with sign
        quantized = quantized_log * sign.to(torch.int8)
        
        params = {
            'method': 'log',
            'scale': scale,
            'zero_point': zero_point,
            'bits': bits
        }
        
        return quantized, params
    
    def quantize_delta_enhanced(
        self,
        weight: torch.Tensor,
        base_weight: torch.Tensor,
        name: str,
        gradients: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Enhanced delta quantization with multiple strategies.
        
        Args:
            weight: Fine-tuned weight
            base_weight: Base weight
            name: Layer name
            gradients: Optional gradient information
            
        Returns:
            Quantized delta and parameters
        """
        delta = weight - base_weight
        
        # Compute importance and allocate bits
        importance = self.compute_importance(delta, name, gradients)
        bits = self.allocate_bits(importance, delta.numel())
        self.bit_allocation[name] = bits
        
        # Select quantization strategy
        if self.config.use_vector_quantization and delta.numel() > 1024:
            # Vector quantization for large tensors
            indices, params = self.vector_quantize(delta, name)
            params['bits'] = np.log2(self.config.codebook_size)
            quantized = indices
            
        elif self.config.use_log_quantization and bits <= 4:
            # Log quantization for low bit widths
            quantized, params = self.log_quantize(delta, bits)
            
        elif bits == 1:
            # Binary quantization
            scale = delta.abs().mean()
            quantized = (delta >= 0).to(torch.int8) * 2 - 1
            params = {'method': 'binary', 'scale': scale, 'bits': 1}
            
        elif bits == 2:
            # Ternary quantization with threshold
            threshold = delta.abs().mean() * 0.7
            quantized = torch.sign(delta) * (delta.abs() > threshold)
            scale = delta.abs()[delta.abs() > threshold].mean() if (delta.abs() > threshold).any() else 1.0
            params = {'method': 'ternary', 'scale': scale, 'threshold': threshold, 'bits': 2}
            
        else:
            # Standard integer quantization with per-channel support
            if self.config.per_channel and len(delta.shape) >= 2:
                quantized, params = self._per_channel_quantize(delta, bits, name)
            else:
                quantized, params = self._per_tensor_quantize(delta, bits, name)
        
        # Error correction: store residuals
        if self.config.error_correction:
            dequantized = self.dequantize_delta_enhanced(quantized, params, delta.shape)
            error = delta - dequantized
            
            if error.abs().mean() > self.config.max_error:
                # Quantize error with higher precision
                error_bits = min(8, bits + 2)
                error_quantized, error_params = self._per_tensor_quantize(error, error_bits, f"{name}_error")
                self.error_residuals[name] = (error_quantized, error_params)
        
        self.quantized_deltas[name] = quantized
        self.quantization_params[name] = params
        
        return quantized, params
    
    def dequantize_delta_enhanced(
        self,
        quantized: torch.Tensor,
        params: Dict[str, Any],
        original_shape: torch.Size
    ) -> torch.Tensor:
        """
        Enhanced dequantization with multiple strategies.
        
        Args:
            quantized: Quantized tensor
            params: Quantization parameters
            original_shape: Original shape
            
        Returns:
            Dequantized delta
        """
        method = params.get('method', 'standard')
        
        if method == 'vq':
            # Vector quantization dequantization
            codebook = params['codebook']
            indices = quantized
            delta_flat = codebook[indices]
            delta = delta_flat.reshape(original_shape)
            
        elif method == 'log':
            # Log quantization dequantization
            scale = params['scale']
            zero_point = params['zero_point']
            
            # Extract sign and magnitude
            sign = torch.sign(quantized)
            abs_quantized = torch.abs(quantized)
            
            # Dequantize log values
            log_values = (abs_quantized.float() - zero_point) * scale
            
            # Inverse log transform
            delta = sign * (2 ** log_values)
            
        elif method == 'binary':
            # Binary dequantization
            delta = quantized.float() * params['scale']
            
        elif method == 'ternary':
            # Ternary dequantization
            delta = quantized.float() * params['scale']
            
        else:
            # Standard integer dequantization
            if 'scales' in params:
                # Per-channel
                delta = self._per_channel_dequantize(quantized, params)
            else:
                # Per-tensor
                delta = self._per_tensor_dequantize(quantized, params)
        
        # Apply error correction if available
        if self.config.error_correction:
            layer_name = None
            for name, q in self.quantized_deltas.items():
                if q is quantized:
                    layer_name = name
                    break
            
            if layer_name and layer_name in self.error_residuals:
                error_quantized, error_params = self.error_residuals[layer_name]
                error = self._per_tensor_dequantize(error_quantized, error_params)
                delta = delta + error
        
        if original_shape:
            delta = delta.reshape(original_shape)
        
        return delta
    
    def _per_tensor_quantize(
        self,
        tensor: torch.Tensor,
        bits: int,
        name: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Per-tensor quantization."""
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        
        # Asymmetric quantization for better accuracy
        min_val = tensor.min()
        max_val = tensor.max()
        
        if self.config.symmetric:
            scale = max(abs(min_val), abs(max_val)) * 2 / (qmax - qmin)
            zero_point = 0
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale if scale != 0 else 0
        
        scale = max(scale, 1e-8)  # Avoid division by zero
        
        # Quantize with optional QAT noise
        if self.config.qat_enabled and self.training:
            noise = torch.randn_like(tensor) * self.config.qat_noise_scale
            tensor_noisy = tensor + noise
            quantized = torch.round(tensor_noisy / scale + zero_point)
        else:
            quantized = torch.round(tensor / scale + zero_point)
        
        quantized = torch.clamp(quantized, qmin, qmax).to(torch.int8)
        
        params = {
            'method': 'standard',
            'scale': scale,
            'zero_point': zero_point,
            'bits': bits
        }
        
        return quantized, params
    
    def _per_tensor_dequantize(
        self,
        quantized: torch.Tensor,
        params: Dict[str, Any]
    ) -> torch.Tensor:
        """Per-tensor dequantization."""
        scale = params['scale']
        zero_point = params.get('zero_point', 0)
        return (quantized.float() - zero_point) * scale
    
    def _per_channel_quantize(
        self,
        tensor: torch.Tensor,
        bits: int,
        name: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Per-channel quantization for better accuracy."""
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        
        # Reshape for per-channel processing
        original_shape = tensor.shape
        tensor = tensor.view(tensor.shape[0], -1)
        
        # Compute per-channel scales
        if self.config.symmetric:
            abs_max = tensor.abs().max(dim=1, keepdim=True)[0]
            scales = abs_max * 2 / (qmax - qmin)
            zero_points = torch.zeros_like(scales)
        else:
            min_vals = tensor.min(dim=1, keepdim=True)[0]
            max_vals = tensor.max(dim=1, keepdim=True)[0]
            scales = (max_vals - min_vals) / (qmax - qmin)
            zero_points = qmin - min_vals / scales
        
        scales = scales.clamp(min=1e-8)
        
        # Quantize
        quantized = torch.round(tensor / scales + zero_points)
        quantized = torch.clamp(quantized, qmin, qmax).to(torch.int8)
        quantized = quantized.view(original_shape)
        
        params = {
            'method': 'per_channel',
            'scales': scales.squeeze(),
            'zero_points': zero_points.squeeze(),
            'bits': bits,
            'shape': original_shape
        }
        
        return quantized, params
    
    def _per_channel_dequantize(
        self,
        quantized: torch.Tensor,
        params: Dict[str, Any]
    ) -> torch.Tensor:
        """Per-channel dequantization."""
        scales = params['scales']
        zero_points = params.get('zero_points', torch.zeros_like(scales))
        
        # Reshape for per-channel processing
        original_shape = params.get('shape', quantized.shape)
        quantized = quantized.view(quantized.shape[0], -1)
        
        # Expand scales and zero_points
        scales = scales.unsqueeze(1)
        zero_points = zero_points.unsqueeze(1)
        
        # Dequantize
        dequantized = (quantized.float() - zero_points) * scales
        dequantized = dequantized.view(original_shape)
        
        return dequantized
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get detailed compression statistics."""
        stats = {
            'total_params': 0,
            'total_bits': 0,
            'average_bits': 0,
            'compression_ratio': 1.0,
            'bit_allocation': self.bit_allocation,
            'importance_scores': self.importance_scores,
            'error_correction_layers': len(self.error_residuals)
        }
        
        for name, quantized in self.quantized_deltas.items():
            params = self.quantization_params[name]
            bits = params.get('bits', 32)
            
            stats['total_params'] += quantized.numel()
            stats['total_bits'] += quantized.numel() * bits
        
        if stats['total_params'] > 0:
            stats['average_bits'] = stats['total_bits'] / stats['total_params']
            original_bits = stats['total_params'] * 32
            stats['compression_ratio'] = original_bits / stats['total_bits']
        
        return stats


# Convenience function
def enhanced_deltaquant(
    model: nn.Module,
    base_model: nn.Module,
    bits: int = 4,
    adaptive: bool = True,
    **kwargs
) -> Tuple[nn.Module, AdaptiveDeltaQuantizer]:
    """
    Apply enhanced DeltaQuant to a model.
    
    Args:
        model: Fine-tuned model
        base_model: Base model
        bits: Base bit width
        adaptive: Use adaptive bit allocation
        **kwargs: Additional config parameters
        
    Returns:
        Quantized model and quantizer
    """
    config = EnhancedDeltaQuantConfig(
        method=QuantMethod.ADAPTIVE if adaptive else QuantMethod.INT4,
        base_bits=bits,
        **kwargs
    )
    
    quantizer = AdaptiveDeltaQuantizer(config)
    
    # Quantize all layers
    for name, param in model.named_parameters():
        if name in dict(base_model.named_parameters()):
            base_param = dict(base_model.named_parameters())[name]
            
            # Get gradients if available
            gradients = param.grad if hasattr(param, 'grad') else None
            
            # Quantize delta
            quantized, params = quantizer.quantize_delta_enhanced(
                param.data, base_param.data, name, gradients
            )
            
            # Reconstruct weight
            delta = quantizer.dequantize_delta_enhanced(quantized, params, param.shape)
            param.data = base_param.data + delta
    
    # Print statistics
    stats = quantizer.get_compression_stats()
    print(f"Enhanced DeltaQuant applied:")
    print(f"  Average bits: {stats['average_bits']:.2f}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}×")
    print(f"  Error correction layers: {stats['error_correction_layers']}")
    
    return model, quantizer