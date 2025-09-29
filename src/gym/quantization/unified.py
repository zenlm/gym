"""
Unified Quantization Interface for Gym
Copyright 2025 Zoo Labs Foundation Inc.

Provides a simple interface for switching between quantization methods
with automatic selection based on model characteristics.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, Union, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
import mmap
import os
from pathlib import Path

from .bitdelta import BitDeltaQuantizer, BitDeltaConfig, BitDeltaLinear
from .deltaquant import DeltaQuantizer, DeltaQuantConfig, DeltaQuantLinear, QuantMethod
from .deltasoup import DeltaSoup, DeltaSoupConfig, AggregationMethod


class QuantizationBackend(Enum):
    """Available quantization backends"""
    BITDELTA = "bitdelta"  # 1-bit quantization
    DELTAQUANT = "deltaquant"  # Multi-bit quantization
    DYNAMIC = "dynamic"  # Dynamic selection
    MIXED = "mixed"  # Mixed precision


@dataclass
class UnifiedQuantConfig:
    """Unified configuration for all quantization methods"""
    
    # Backend selection
    backend: QuantizationBackend = QuantizationBackend.DYNAMIC
    
    # Common settings
    bits: int = 4  # Target bit width
    group_size: int = 128  # Quantization group size
    symmetric: bool = True  # Symmetric quantization
    
    # Memory optimization
    memory_map: bool = True  # Use memory-mapped files
    lazy_load: bool = True  # Load weights on-demand
    cache_size_mb: int = 512  # Cache size for quantized weights
    
    # Dynamic quantization
    dynamic_threshold: float = 0.01  # Threshold for dynamic quantization
    auto_mixed_precision: bool = True  # Automatic mixed precision
    
    # Quality preservation
    error_threshold: float = 0.001  # Maximum acceptable error
    fallback_bits: int = 8  # Fallback bit width if error too high
    
    # QAT (Quantization-Aware Training)
    qat_enabled: bool = False  # Enable QAT
    qat_warmup_epochs: int = 5  # Warmup epochs before quantization
    qat_temperature: float = 1.0  # Temperature for soft quantization
    
    # DeltaSoup integration
    enable_community: bool = False  # Enable community aggregation
    soup_config: Optional[DeltaSoupConfig] = None


class QuantizedWeight:
    """Memory-efficient quantized weight storage with mmap support"""
    
    def __init__(
        self,
        name: str,
        shape: torch.Size,
        quantized_data: Union[torch.Tensor, np.ndarray],
        params: Dict[str, Any],
        mmap_path: Optional[Path] = None
    ):
        self.name = name
        self.shape = shape
        self.params = params
        self.mmap_path = mmap_path
        self._mmap_file = None
        self._cached_tensor = None
        
        if mmap_path:
            # Save to memory-mapped file
            self._save_mmap(quantized_data)
        else:
            # Keep in memory
            self._cached_tensor = quantized_data
    
    def _save_mmap(self, data: Union[torch.Tensor, np.ndarray]):
        """Save quantized data to memory-mapped file"""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        # Create mmap file
        self.mmap_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.mmap_path, 'wb+') as f:
            # Write metadata
            metadata = {
                'dtype': str(data.dtype),
                'shape': list(data.shape),
                'params': self.params
            }
            import json
            metadata_bytes = json.dumps(metadata).encode('utf-8')
            f.write(len(metadata_bytes).to_bytes(4, 'little'))
            f.write(metadata_bytes)
            
            # Write data
            data.tofile(f)
    
    def load(self) -> torch.Tensor:
        """Load quantized weight from memory or mmap"""
        if self._cached_tensor is not None:
            return self._cached_tensor
        
        if self.mmap_path and self.mmap_path.exists():
            with open(self.mmap_path, 'rb') as f:
                # Read metadata
                metadata_size = int.from_bytes(f.read(4), 'little')
                metadata_bytes = f.read(metadata_size)
                import json
                metadata = json.loads(metadata_bytes.decode('utf-8'))
                
                # Memory-map the data
                offset = 4 + metadata_size
                dtype = np.dtype(metadata['dtype'])
                shape = tuple(metadata['shape'])
                
                # Create memory-mapped array
                mmap_array = np.memmap(
                    self.mmap_path,
                    dtype=dtype,
                    mode='r',
                    offset=offset,
                    shape=shape
                )
                
                # Convert to tensor
                return torch.from_numpy(mmap_array.copy())
        
        raise ValueError(f"Cannot load weight {self.name}")
    
    def unload(self):
        """Unload cached tensor to save memory"""
        self._cached_tensor = None
        if self._mmap_file:
            self._mmap_file.close()
            self._mmap_file = None


class UnifiedQuantizer:
    """
    Unified quantization interface with automatic backend selection,
    dynamic quantization, and memory-mapped weight loading.
    """
    
    def __init__(self, config: UnifiedQuantConfig):
        self.config = config
        self.quantized_weights: Dict[str, QuantizedWeight] = {}
        self.base_weights: Dict[str, torch.Tensor] = {}
        self.weight_errors: Dict[str, float] = {}
        
        # Initialize backends
        self.bitdelta = None
        self.deltaquant = None
        self.deltasoup = None
        
        # Memory management
        self.cache_dir = Path.home() / '.cache' / 'gym' / 'quantized'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.current_cache_size = 0
    
    def select_backend(
        self,
        weight: torch.Tensor,
        name: str
    ) -> QuantizationBackend:
        """Automatically select best backend for a weight"""
        if self.config.backend != QuantizationBackend.DYNAMIC:
            return self.config.backend
        
        # Analyze weight characteristics
        num_params = weight.numel()
        sparsity = (weight == 0).float().mean().item()
        variance = weight.var().item()
        
        # Decision logic
        if num_params < 1024:  # Small weights
            return QuantizationBackend.DELTAQUANT  # Keep higher precision
        elif sparsity > 0.9:  # Very sparse
            return QuantizationBackend.BITDELTA  # Binary is efficient
        elif variance < self.config.dynamic_threshold:  # Low variance
            return QuantizationBackend.BITDELTA  # Can use 1-bit
        else:
            return QuantizationBackend.DELTAQUANT  # Need multi-bit
    
    def quantize_model(
        self,
        model: nn.Module,
        base_model: nn.Module,
        calibration_loader: Optional[Any] = None
    ) -> nn.Module:
        """
        Quantize a model with automatic backend selection.
        
        Args:
            model: Fine-tuned model to quantize
            base_model: Base model for delta computation
            calibration_loader: Optional calibration data
            
        Returns:
            Quantized model with reduced memory footprint
        """
        print("Quantizing model with unified interface...")
        
        # QAT preparation if enabled
        if self.config.qat_enabled:
            model = self._prepare_qat(model)
        
        # Process each layer
        for name, param in model.named_parameters():
            if name not in dict(base_model.named_parameters()):
                continue
            
            base_param = dict(base_model.named_parameters())[name]
            
            # Select backend for this layer
            backend = self.select_backend(param.data, name)
            
            # Apply quantization
            if backend == QuantizationBackend.BITDELTA:
                quantized_weight = self._quantize_bitdelta(
                    param.data, base_param.data, name
                )
            elif backend == QuantizationBackend.DELTAQUANT:
                quantized_weight = self._quantize_deltaquant(
                    param.data, base_param.data, name
                )
            elif backend == QuantizationBackend.MIXED:
                quantized_weight = self._quantize_mixed(
                    param.data, base_param.data, name
                )
            else:
                # Dynamic already handled by select_backend
                quantized_weight = self._quantize_deltaquant(
                    param.data, base_param.data, name
                )
            
            # Check error and fallback if needed
            if self.config.error_threshold > 0:
                error = self._compute_error(param.data, quantized_weight, base_param.data)
                self.weight_errors[name] = error
                
                if error > self.config.error_threshold:
                    print(f"Layer {name} error {error:.4f} exceeds threshold, using fallback")
                    quantized_weight = self._quantize_fallback(
                        param.data, base_param.data, name
                    )
            
            # Store quantized weight
            self._store_quantized_weight(name, quantized_weight)
            
            # Update model weight
            self._update_model_weight(model, name, quantized_weight, base_param.data)
        
        # Apply DeltaSoup if enabled
        if self.config.enable_community and self.deltasoup:
            model = self._apply_community_updates(model, base_model)
        
        print(f"Quantization complete. Cache size: {self.current_cache_size / 1024 / 1024:.1f} MB")
        return model
    
    def _quantize_bitdelta(
        self,
        weight: torch.Tensor,
        base_weight: torch.Tensor,
        name: str
    ) -> QuantizedWeight:
        """Apply BitDelta quantization"""
        if not self.bitdelta:
            config = BitDeltaConfig(
                bits=1,
                group_size=self.config.group_size,
                symmetric=self.config.symmetric
            )
            self.bitdelta = BitDeltaQuantizer(config)
        
        signs, scales = self.bitdelta.quantize_delta(weight, base_weight, name)
        
        # Create quantized weight
        params = {
            'method': 'bitdelta',
            'signs': signs,
            'scales': scales,
            'shape': weight.shape,
            'bits': 1
        }
        
        # Memory-map if enabled
        mmap_path = None
        if self.config.memory_map:
            mmap_path = self.cache_dir / f"{name.replace('/', '_')}_bitdelta.mmap"
        
        return QuantizedWeight(name, weight.shape, signs, params, mmap_path)
    
    def _quantize_deltaquant(
        self,
        weight: torch.Tensor,
        base_weight: torch.Tensor,
        name: str
    ) -> QuantizedWeight:
        """Apply DeltaQuant quantization"""
        if not self.deltaquant:
            # Map bits to method
            if self.config.bits <= 2:
                method = QuantMethod.INT2
            elif self.config.bits <= 4:
                method = QuantMethod.INT4
            else:
                method = QuantMethod.INT8
            
            config = DeltaQuantConfig(
                method=method,
                symmetric=self.config.symmetric,
                group_size=self.config.group_size
            )
            self.deltaquant = DeltaQuantizer(config)
        
        quantized, params = self.deltaquant.quantize_delta(weight, base_weight, name)
        
        # Ensure method is set
        if 'method' not in params:
            params['method'] = 'deltaquant'
        
        # Memory-map if enabled
        mmap_path = None
        if self.config.memory_map:
            mmap_path = self.cache_dir / f"{name.replace('/', '_')}_deltaquant.mmap"
        
        return QuantizedWeight(name, weight.shape, quantized, params, mmap_path)
    
    def _quantize_mixed(
        self,
        weight: torch.Tensor,
        base_weight: torch.Tensor,
        name: str
    ) -> QuantizedWeight:
        """Apply mixed-precision quantization"""
        # Analyze weight importance
        importance = self._compute_importance(weight, name)
        
        # Use higher precision for important weights
        if importance > 0.8:
            return self._quantize_deltaquant(weight, base_weight, name)
        else:
            return self._quantize_bitdelta(weight, base_weight, name)
    
    def _quantize_fallback(
        self,
        weight: torch.Tensor,
        base_weight: torch.Tensor,
        name: str
    ) -> QuantizedWeight:
        """Fallback quantization with higher precision"""
        # Create higher precision config
        config = DeltaQuantConfig(
            method=QuantMethod.INT8 if self.config.fallback_bits == 8 else QuantMethod.INT4,
            symmetric=self.config.symmetric,
            group_size=self.config.group_size
        )
        fallback_quantizer = DeltaQuantizer(config)
        
        quantized, params = fallback_quantizer.quantize_delta(weight, base_weight, name)
        
        mmap_path = None
        if self.config.memory_map:
            mmap_path = self.cache_dir / f"{name.replace('/', '_')}_fallback.mmap"
        
        return QuantizedWeight(name, weight.shape, quantized, params, mmap_path)
    
    def _compute_error(
        self,
        original: torch.Tensor,
        quantized_weight: QuantizedWeight,
        base_weight: torch.Tensor
    ) -> float:
        """Compute quantization error"""
        # Reconstruct weight
        if quantized_weight.params['method'] == 'bitdelta':
            if self.bitdelta:
                delta = self.bitdelta.dequantize_delta(
                    quantized_weight.params['signs'],
                    quantized_weight.params['scales'],
                    quantized_weight.shape
                )
            else:
                return 0.0
        else:
            if self.deltaquant:
                delta = self.deltaquant.dequantize_delta(
                    quantized_weight.load(),
                    quantized_weight.params,
                    quantized_weight.shape
                )
            else:
                return 0.0
        
        reconstructed = base_weight + delta
        
        # Compute relative error
        error = torch.norm(original - reconstructed) / torch.norm(original)
        return error.item()
    
    def _compute_importance(self, weight: torch.Tensor, name: str) -> float:
        """Compute layer importance for mixed precision"""
        # Simple heuristic based on weight statistics
        # Can be replaced with more sophisticated methods
        
        # Check if it's an attention or output layer (usually important)
        if 'attention' in name.lower() or 'output' in name.lower():
            return 0.9
        
        # Check weight magnitude
        magnitude = weight.abs().mean().item()
        
        # Normalize to [0, 1]
        importance = min(1.0, magnitude * 10)
        
        return importance
    
    def _prepare_qat(self, model: nn.Module) -> nn.Module:
        """Prepare model for quantization-aware training"""
        print("Preparing model for QAT...")
        
        # Add fake quantization layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Wrap with QAT-aware linear
                setattr(model, name, QATLinear(module, self.config))
        
        return model
    
    def _store_quantized_weight(self, name: str, weight: QuantizedWeight):
        """Store quantized weight with cache management"""
        self.quantized_weights[name] = weight
        
        # Update cache size
        if weight.mmap_path and weight.mmap_path.exists():
            self.current_cache_size += weight.mmap_path.stat().st_size
        
        # Evict if cache too large
        if self.current_cache_size > self.config.cache_size_mb * 1024 * 1024:
            self._evict_cache()
    
    def _evict_cache(self):
        """Evict least recently used weights from cache"""
        # Simple FIFO eviction for now
        # Can be replaced with LRU
        to_evict = list(self.quantized_weights.keys())[:len(self.quantized_weights) // 4]
        
        for name in to_evict:
            weight = self.quantized_weights[name]
            weight.unload()
            if weight.mmap_path and weight.mmap_path.exists():
                self.current_cache_size -= weight.mmap_path.stat().st_size
    
    def _update_model_weight(
        self,
        model: nn.Module,
        name: str,
        quantized_weight: QuantizedWeight,
        base_weight: torch.Tensor
    ):
        """Update model weight with quantized version"""
        # Find the parameter
        parts = name.split('.')
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        
        # Reconstruct and set weight
        if quantized_weight.params['method'] == 'bitdelta':
            if self.bitdelta:
                delta = self.bitdelta.dequantize_delta(
                    quantized_weight.params['signs'],
                    quantized_weight.params['scales'],
                    quantized_weight.shape
                )
                weight = base_weight + delta
            else:
                weight = base_weight
        else:
            if self.deltaquant:
                delta = self.deltaquant.dequantize_delta(
                    quantized_weight.load(),
                    quantized_weight.params,
                    quantized_weight.shape
                )
                weight = base_weight + delta
            else:
                weight = base_weight
        
        setattr(module, parts[-1], nn.Parameter(weight))
    
    def _apply_community_updates(
        self,
        model: nn.Module,
        base_model: nn.Module
    ) -> nn.Module:
        """Apply DeltaSoup community updates"""
        if not self.deltasoup:
            soup_config = self.config.soup_config or DeltaSoupConfig()
            self.deltasoup = DeltaSoup(soup_config)
        
        # Apply aggregated deltas
        model = self.deltasoup.apply_aggregated_deltas(model, base_model)
        
        return model
    
    def benchmark(
        self,
        model: nn.Module,
        base_model: nn.Module,
        test_loader: Any
    ) -> Dict[str, float]:
        """
        Benchmark quantization performance.
        
        Returns:
            Dictionary with compression ratio, accuracy, and speed metrics
        """
        import time
        
        results = {
            'compression_ratio': self.get_compression_ratio(),
            'total_error': np.mean(list(self.weight_errors.values())) if self.weight_errors else 0.0,
            'cache_size_mb': self.current_cache_size / 1024 / 1024
        }
        
        # Measure inference speed
        start_time = time.time()
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch
                _ = model(inputs)
                break  # Just one batch for quick test
        
        results['inference_time_ms'] = (time.time() - start_time) * 1000
        
        # Memory usage
        if torch.cuda.is_available():
            results['gpu_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        
        return results
    
    def get_compression_ratio(self) -> float:
        """Calculate overall compression ratio"""
        if not self.quantized_weights:
            return 1.0
        
        original_size = 0
        compressed_size = 0
        
        for name, weight in self.quantized_weights.items():
            # Original: 32-bit floats
            original_size += np.prod(weight.shape) * 32
            
            # Compressed size depends on method
            if weight.params['method'] == 'bitdelta':
                # 1-bit signs + scales
                compressed_size += np.prod(weight.shape) * 1
                compressed_size += weight.params['scales'].numel() * 32
            else:
                bits = weight.params.get('bits', 4)
                compressed_size += np.prod(weight.shape) * bits
                
                # Add metadata
                if 'scales' in weight.params:
                    compressed_size += weight.params['scales'].numel() * 32
                if 'zero_points' in weight.params:
                    compressed_size += weight.params['zero_points'].numel() * 32
        
        return original_size / compressed_size if compressed_size > 0 else 1.0
    
    def export_quantized_model(self, path: str):
        """Export quantized model with all metadata"""
        checkpoint = {
            'config': self.config,
            'quantized_weights': {
                name: {
                    'shape': weight.shape,
                    'params': weight.params,
                    'mmap_path': str(weight.mmap_path) if weight.mmap_path else None
                }
                for name, weight in self.quantized_weights.items()
            },
            'base_weights': self.base_weights,
            'compression_ratio': self.get_compression_ratio(),
            'errors': self.weight_errors
        }
        
        torch.save(checkpoint, path)
        print(f"Quantized model exported to {path}")
        print(f"Compression ratio: {self.get_compression_ratio():.2f}×")
    
    def load_quantized_model(self, path: str, model: nn.Module) -> nn.Module:
        """Load quantized model from checkpoint"""
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        self.base_weights = checkpoint['base_weights']
        self.weight_errors = checkpoint.get('errors', {})
        
        # Restore quantized weights
        for name, weight_info in checkpoint['quantized_weights'].items():
            mmap_path = Path(weight_info['mmap_path']) if weight_info['mmap_path'] else None
            
            # Create QuantizedWeight object
            weight = QuantizedWeight(
                name,
                torch.Size(weight_info['shape']),
                None,  # Will be loaded from mmap
                weight_info['params'],
                mmap_path
            )
            
            self.quantized_weights[name] = weight
            
            # Update model
            if name in self.base_weights:
                self._update_model_weight(model, name, weight, self.base_weights[name])
        
        print(f"Quantized model loaded from {path}")
        print(f"Compression ratio: {checkpoint['compression_ratio']:.2f}×")
        
        return model


class QATLinear(nn.Module):
    """Linear layer with quantization-aware training support"""
    
    def __init__(self, linear: nn.Linear, config: UnifiedQuantConfig):
        super().__init__()
        self.linear = linear
        self.config = config
        self.training_step = 0
        
        # Learnable quantization parameters
        self.scale = nn.Parameter(torch.ones(1))
        self.zero_point = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.config.qat_enabled:
            # Soft quantization during training
            if self.training_step > self.config.qat_warmup_epochs:
                # Apply fake quantization
                weight_q = self._fake_quantize(self.linear.weight)
                x = nn.functional.linear(x, weight_q, self.linear.bias)
            else:
                # Normal forward during warmup
                x = self.linear(x)
            
            self.training_step += 1
        else:
            # Normal forward during inference
            x = self.linear(x)
        
        return x
    
    def _fake_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization for QAT"""
        # Compute range
        qmin = -(2 ** (self.config.bits - 1))
        qmax = 2 ** (self.config.bits - 1) - 1
        
        # Quantize and dequantize
        tensor_q = torch.round(tensor / self.scale + self.zero_point)
        tensor_q = torch.clamp(tensor_q, qmin, qmax)
        tensor_dq = (tensor_q - self.zero_point) * self.scale
        
        # Straight-through estimator
        return tensor + (tensor_dq - tensor).detach()


# Convenience functions
def quantize_model(
    model: nn.Module,
    base_model: nn.Module,
    bits: int = 4,
    backend: str = "dynamic",
    **kwargs
) -> Tuple[nn.Module, UnifiedQuantizer]:
    """
    Convenience function to quantize a model.
    
    Args:
        model: Model to quantize
        base_model: Base model
        bits: Target bit width
        backend: Quantization backend ("bitdelta", "deltaquant", "dynamic", "mixed")
        **kwargs: Additional config parameters
    
    Returns:
        Quantized model and quantizer instance
    """
    config = UnifiedQuantConfig(
        backend=QuantizationBackend[backend.upper()],
        bits=bits,
        **kwargs
    )
    
    quantizer = UnifiedQuantizer(config)
    quantized_model = quantizer.quantize_model(model, base_model)
    
    return quantized_model, quantizer