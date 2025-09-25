"""
DeltaQuant: Efficient Delta Quantization for Fine-tuned Models
Copyright 2025 Zoo Labs Foundation Inc.

DeltaQuant quantizes the difference between base and fine-tuned models
for efficient storage and serving of personalized models.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any
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


@dataclass
class DeltaQuantConfig:
    """Configuration for DeltaQuant"""
    
    # Quantization settings
    method: QuantMethod = QuantMethod.INT4  # Default to 4-bit
    symmetric: bool = True  # Symmetric quantization
    per_channel: bool = True  # Per-channel quantization
    group_size: int = 128  # Group size for grouped quantization
    
    # Optimization settings
    calibration_samples: int = 256  # Samples for calibration
    percentile_calibration: bool = True  # Use percentile for range
    calibration_percentile: float = 99.9  # Percentile for calibration
    
    # Mixed precision
    sensitive_layers: List[str] = None  # Layers to keep at higher precision
    layer_bits: Dict[str, int] = None  # Per-layer bit configuration
    
    # Compression
    enable_compression: bool = True  # Additional compression
    use_huffman: bool = False  # Huffman encoding for further compression
    
    # Performance
    use_cuda_kernel: bool = True  # Use optimized CUDA kernels
    batch_processing: bool = True  # Batch processing for speed


class DeltaQuantizer:
    """
    DeltaQuant: Quantizes model deltas for efficient serving.
    Supports multiple quantization methods and mixed precision.
    """
    
    def __init__(self, config: DeltaQuantConfig):
        self.config = config
        self.quantized_deltas: Dict[str, Any] = {}
        self.quantization_params: Dict[str, Any] = {}
        self.calibration_data: Dict[str, torch.Tensor] = {}
        
    def calibrate(
        self,
        model: nn.Module,
        base_model: nn.Module,
        calibration_loader: Optional[Any] = None
    ):
        """
        Calibrate quantization parameters using calibration data.
        
        Args:
            model: Fine-tuned model
            base_model: Base model
            calibration_loader: DataLoader for calibration
        """
        print("Calibrating DeltaQuant parameters...")
        
        for name, param in model.named_parameters():
            if name in dict(base_model.named_parameters()):
                base_param = dict(base_model.named_parameters())[name]
                delta = param.data - base_param.data
                
                # Collect calibration statistics
                if self.config.percentile_calibration:
                    # Use percentile for robust range estimation
                    percentile = self.config.calibration_percentile
                    min_val = torch.quantile(delta.flatten(), (100 - percentile) / 100)
                    max_val = torch.quantile(delta.flatten(), percentile / 100)
                else:
                    # Use min-max
                    min_val = delta.min()
                    max_val = delta.max()
                
                # Store calibration parameters
                self.quantization_params[name] = {
                    'min': min_val,
                    'max': max_val,
                    'scale': (max_val - min_val) / (2 ** self._get_bits(name) - 1),
                    'zero_point': 0 if self.config.symmetric else min_val
                }
    
    def quantize_delta(
        self,
        weight: torch.Tensor,
        base_weight: torch.Tensor,
        name: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantize the delta between fine-tuned and base weights.
        
        Args:
            weight: Fine-tuned weight
            base_weight: Base model weight
            name: Layer name
            
        Returns:
            quantized_delta: Quantized delta tensor
            quant_params: Quantization parameters
        """
        delta = weight - base_weight
        bits = self._get_bits(name)
        
        if self.config.method == QuantMethod.BINARY:
            # Binary quantization (+1, -1)
            quantized = (delta >= 0).to(torch.int8) * 2 - 1
            scale = delta.abs().mean()
            params = {'scale': scale, 'bits': 1}
            
        elif self.config.method == QuantMethod.TERNARY:
            # Ternary quantization (-1, 0, +1)
            threshold = delta.abs().mean() * 0.7
            quantized = torch.sign(delta) * (delta.abs() > threshold)
            scale = delta.abs()[delta.abs() > threshold].mean()
            params = {'scale': scale, 'threshold': threshold, 'bits': 2}
            
        else:
            # Integer quantization (INT2, INT4, INT8)
            if self.config.per_channel:
                quantized, params = self._per_channel_quantize(delta, bits, name)
            else:
                quantized, params = self._per_tensor_quantize(delta, bits, name)
        
        self.quantized_deltas[name] = quantized
        self.quantization_params[name] = params
        
        return quantized, params
    
    def dequantize_delta(
        self,
        quantized: torch.Tensor,
        params: Dict[str, Any],
        original_shape: Optional[torch.Size] = None
    ) -> torch.Tensor:
        """
        Dequantize delta back to full precision.
        
        Args:
            quantized: Quantized delta tensor
            params: Quantization parameters
            original_shape: Original tensor shape
            
        Returns:
            Dequantized delta tensor
        """
        if params['bits'] == 1:
            # Binary dequantization
            delta = quantized.float() * params['scale']
            
        elif params['bits'] == 2:
            # Ternary dequantization
            delta = quantized.float() * params['scale']
            
        else:
            # Integer dequantization
            if 'scales' in params:
                # Per-channel dequantization
                delta = self._per_channel_dequantize(quantized, params)
            else:
                # Per-tensor dequantization
                delta = self._per_tensor_dequantize(quantized, params)
        
        if original_shape is not None:
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
        
        # Get or compute scale and zero point
        if name in self.quantization_params:
            scale = self.quantization_params[name]['scale']
            zero_point = self.quantization_params[name].get('zero_point', 0)
        else:
            min_val = tensor.min()
            max_val = tensor.max()
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = 0 if self.config.symmetric else qmin - min_val / scale
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax).to(torch.int8)
        
        params = {
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
        """Per-channel quantization."""
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        
        # Reshape for per-channel processing
        original_shape = tensor.shape
        tensor = tensor.view(tensor.shape[0], -1)
        
        # Compute per-channel scales
        min_vals = tensor.min(dim=1, keepdim=True)[0]
        max_vals = tensor.max(dim=1, keepdim=True)[0]
        scales = (max_vals - min_vals) / (qmax - qmin)
        scales = scales.clamp(min=1e-8)  # Avoid division by zero
        
        if self.config.symmetric:
            zero_points = torch.zeros_like(scales)
        else:
            zero_points = qmin - min_vals / scales
        
        # Quantize
        quantized = torch.round(tensor / scales + zero_points)
        quantized = torch.clamp(quantized, qmin, qmax).to(torch.int8)
        quantized = quantized.view(original_shape)
        
        params = {
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
    
    def _get_bits(self, name: str) -> int:
        """Get bit width for a specific layer."""
        # Check layer-specific configuration
        if self.config.layer_bits and name in self.config.layer_bits:
            return self.config.layer_bits[name]
        
        # Check sensitive layers
        if self.config.sensitive_layers and name in self.config.sensitive_layers:
            return 8  # Keep sensitive layers at higher precision
        
        # Default bits based on method
        method_bits = {
            QuantMethod.BINARY: 1,
            QuantMethod.TERNARY: 2,
            QuantMethod.INT2: 2,
            QuantMethod.INT4: 4,
            QuantMethod.INT8: 8,
            QuantMethod.DYNAMIC: 4  # Default for dynamic
        }
        
        return method_bits.get(self.config.method, 4)
    
    def apply_deltaquant(
        self,
        model: nn.Module,
        base_model: nn.Module,
        calibration_loader: Optional[Any] = None
    ) -> nn.Module:
        """
        Apply DeltaQuant to a fine-tuned model.
        
        Args:
            model: Fine-tuned model
            base_model: Base model
            calibration_loader: Optional calibration data
            
        Returns:
            Model with DeltaQuant applied
        """
        # Calibrate if needed
        if calibration_loader:
            self.calibrate(model, base_model, calibration_loader)
        
        # Quantize all deltas
        for name, param in model.named_parameters():
            if name in dict(base_model.named_parameters()):
                base_param = dict(base_model.named_parameters())[name]
                
                # Quantize delta
                quantized, params = self.quantize_delta(
                    param.data, base_param.data, name
                )
                
                # Replace weight with base + dequantized delta
                delta = self.dequantize_delta(quantized, params, param.shape)
                param.data = base_param.data + delta
        
        return model
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if not self.quantized_deltas:
            return 1.0
        
        original_bits = 0
        compressed_bits = 0
        
        for name, quantized in self.quantized_deltas.items():
            params = self.quantization_params[name]
            bits = params.get('bits', 32)
            
            # Original: 32-bit floats
            original_bits += quantized.numel() * 32
            
            # Compressed: reduced bits + metadata
            compressed_bits += quantized.numel() * bits
            
            # Add metadata overhead
            if 'scales' in params:
                compressed_bits += params['scales'].numel() * 32
            elif 'scale' in params:
                compressed_bits += 32
            
            if 'zero_points' in params:
                compressed_bits += params['zero_points'].numel() * 32
            elif 'zero_point' in params:
                compressed_bits += 32
        
        return original_bits / compressed_bits
    
    def export_checkpoint(self, path: str):
        """Export DeltaQuant checkpoint."""
        checkpoint = {
            'config': self.config,
            'quantized_deltas': self.quantized_deltas,
            'quantization_params': self.quantization_params,
            'compression_ratio': self.get_compression_ratio()
        }
        torch.save(checkpoint, path)
        print(f"DeltaQuant checkpoint saved to {path}")
        print(f"Compression ratio: {self.get_compression_ratio():.2f}×")
    
    def load_checkpoint(self, path: str):
        """Load DeltaQuant checkpoint."""
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        self.quantized_deltas = checkpoint['quantized_deltas']
        self.quantization_params = checkpoint['quantization_params']
        print(f"DeltaQuant checkpoint loaded from {path}")
        print(f"Compression ratio: {checkpoint['compression_ratio']:.2f}×")


class DeltaQuantLinear(nn.Module):
    """
    DeltaQuant-aware Linear layer with quantized deltas.
    Memory-efficient replacement for nn.Linear.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_weight: torch.Tensor,
        quantized_delta: torch.Tensor,
        quant_params: Dict[str, Any],
        base_bias: Optional[torch.Tensor] = None,
        config: Optional[DeltaQuantConfig] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or DeltaQuantConfig()
        
        # Store base weights
        self.register_buffer('base_weight', base_weight)
        if base_bias is not None:
            self.register_buffer('base_bias', base_bias)
        else:
            self.register_buffer('base_bias', None)
        
        # Store quantized delta and parameters
        self.register_buffer('quantized_delta', quantized_delta)
        self.quant_params = quant_params
        
        # Bias delta (not quantized for accuracy)
        if base_bias is not None:
            self.bias_delta = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_delta', None)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Dequantize delta
        quantizer = DeltaQuantizer(self.config)
        delta = quantizer.dequantize_delta(
            self.quantized_delta,
            self.quant_params,
            self.base_weight.shape
        )
        
        # Reconstruct weight
        weight = self.base_weight + delta
        
        # Compute bias
        bias = None
        if self.base_bias is not None:
            bias = self.base_bias + self.bias_delta
        
        # Apply linear transformation
        return nn.functional.linear(input, weight, bias)
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        base_linear: nn.Linear,
        config: Optional[DeltaQuantConfig] = None
    ) -> 'DeltaQuantLinear':
        """Convert standard Linear to DeltaQuantLinear."""
        config = config or DeltaQuantConfig()
        quantizer = DeltaQuantizer(config)
        
        # Quantize weight delta
        quantized, params = quantizer.quantize_delta(
            linear.weight.data,
            base_linear.weight.data,
            'weight'
        )
        
        # Create DeltaQuantLinear
        deltaquant_linear = cls(
            linear.in_features,
            linear.out_features,
            base_linear.weight.data,
            quantized,
            params,
            base_linear.bias.data if base_linear.bias is not None else None,
            config
        )
        
        # Set bias delta
        if linear.bias is not None and base_linear.bias is not None:
            deltaquant_linear.bias_delta.data = linear.bias.data - base_linear.bias.data
        
        return deltaquant_linear