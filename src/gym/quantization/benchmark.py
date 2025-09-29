"""
Quantization Benchmarking Utilities
Copyright 2025 Zoo Labs Foundation Inc.

Comprehensive benchmarking for compression vs accuracy trade-offs.
"""

import torch
import torch.nn as nn
import time
import copy
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
from pathlib import Path

from .bitdelta import BitDeltaQuantizer, BitDeltaConfig
from .deltaquant import DeltaQuantizer, DeltaQuantConfig, QuantMethod
from .deltaquant_v2 import AdaptiveDeltaQuantizer, EnhancedDeltaQuantConfig
from .unified import UnifiedQuantizer, UnifiedQuantConfig, QuantizationBackend


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    
    # Test settings
    num_warmup_runs: int = 3
    num_benchmark_runs: int = 10
    batch_size: int = 32
    sequence_length: int = 128
    
    # Model settings
    test_layers: List[str] = None  # Specific layers to test
    
    # Quantization methods to test
    test_bitdelta: bool = True
    test_deltaquant: bool = True
    test_adaptive: bool = True
    test_unified: bool = True
    
    # Bit widths to test
    bit_widths: List[int] = None  # [1, 2, 4, 8]
    
    # Metrics to collect
    measure_compression: bool = True
    measure_accuracy: bool = True
    measure_speed: bool = True
    measure_memory: bool = True
    
    # Output settings
    save_results: bool = True
    results_dir: str = "./benchmark_results"
    plot_results: bool = True


class QuantizationBenchmark:
    """
    Comprehensive benchmarking suite for quantization methods.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        
        if config.bit_widths is None:
            config.bit_widths = [1, 2, 4, 8]
        
        # Create results directory
        if config.save_results:
            Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    def benchmark_model(
        self,
        model: nn.Module,
        base_model: nn.Module,
        test_loader: Any,
        eval_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Benchmark quantization methods on a model.
        
        Args:
            model: Fine-tuned model
            base_model: Base model
            test_loader: Data loader for testing
            eval_fn: Optional evaluation function for accuracy
            
        Returns:
            Dictionary of benchmark results
        """
        print("Starting quantization benchmark...")
        results = {}
        
        # Benchmark each method
        if self.config.test_bitdelta:
            results['bitdelta'] = self._benchmark_bitdelta(
                model, base_model, test_loader, eval_fn
            )
        
        if self.config.test_deltaquant:
            results['deltaquant'] = self._benchmark_deltaquant(
                model, base_model, test_loader, eval_fn
            )
        
        if self.config.test_adaptive:
            results['adaptive'] = self._benchmark_adaptive(
                model, base_model, test_loader, eval_fn
            )
        
        if self.config.test_unified:
            results['unified'] = self._benchmark_unified(
                model, base_model, test_loader, eval_fn
            )
        
        # Store results
        self.results = results
        
        # Save results
        if self.config.save_results:
            self._save_results(results)
        
        # Plot results
        if self.config.plot_results:
            self._plot_results(results)
        
        return results
    
    def _benchmark_bitdelta(
        self,
        model: nn.Module,
        base_model: nn.Module,
        test_loader: Any,
        eval_fn: Optional[Callable]
    ) -> Dict[str, Any]:
        """Benchmark BitDelta quantization."""
        print("\nBenchmarking BitDelta (1-bit)...")
        results = {}
        
        config = BitDeltaConfig(bits=1)
        quantizer = BitDeltaQuantizer(config)
        
        # Measure quantization time
        import copy
        start_time = time.time()
        quantized_model = quantizer.apply_bitdelta(
            copy.deepcopy(model), base_model
        )
        quantization_time = time.time() - start_time
        
        # Measure compression
        compression_ratio = quantizer.get_memory_savings()
        
        # Measure inference speed
        inference_time = self._measure_inference_speed(
            quantized_model, test_loader
        )
        
        # Measure accuracy if eval function provided
        accuracy = None
        if eval_fn:
            accuracy = eval_fn(quantized_model, test_loader)
        
        # Measure memory usage
        memory_usage = self._measure_memory_usage(quantized_model)
        
        results = {
            'method': 'BitDelta',
            'bits': 1,
            'compression_ratio': compression_ratio,
            'quantization_time': quantization_time,
            'inference_time': inference_time,
            'accuracy': accuracy,
            'memory_mb': memory_usage
        }
        
        print(f"  Compression: {compression_ratio:.2f}×")
        print(f"  Inference: {inference_time:.2f} ms/batch")
        if accuracy:
            print(f"  Accuracy: {accuracy:.4f}")
        
        return results
    
    def _benchmark_deltaquant(
        self,
        model: nn.Module,
        base_model: nn.Module,
        test_loader: Any,
        eval_fn: Optional[Callable]
    ) -> Dict[str, Any]:
        """Benchmark DeltaQuant quantization."""
        print("\nBenchmarking DeltaQuant...")
        results = {}
        
        for bits in self.config.bit_widths:
            if bits == 1:
                method = QuantMethod.BINARY
            elif bits == 2:
                method = QuantMethod.INT2
            elif bits == 4:
                method = QuantMethod.INT4
            else:
                method = QuantMethod.INT8
            
            print(f"  Testing {bits}-bit quantization...")
            
            config = DeltaQuantConfig(method=method)
            quantizer = DeltaQuantizer(config)
            
            # Measure quantization time
            start_time = time.time()
            quantized_model = quantizer.apply_deltaquant(
                copy.deepcopy(model), base_model
            )
            quantization_time = time.time() - start_time
            
            # Measure compression
            compression_ratio = quantizer.get_compression_ratio()
            
            # Measure inference speed
            inference_time = self._measure_inference_speed(
                quantized_model, test_loader
            )
            
            # Measure accuracy
            accuracy = None
            if eval_fn:
                accuracy = eval_fn(quantized_model, test_loader)
            
            # Measure memory
            memory_usage = self._measure_memory_usage(quantized_model)
            
            results[f'{bits}bit'] = {
                'method': 'DeltaQuant',
                'bits': bits,
                'compression_ratio': compression_ratio,
                'quantization_time': quantization_time,
                'inference_time': inference_time,
                'accuracy': accuracy,
                'memory_mb': memory_usage
            }
            
            print(f"    Compression: {compression_ratio:.2f}×")
            print(f"    Inference: {inference_time:.2f} ms/batch")
            if accuracy:
                print(f"    Accuracy: {accuracy:.4f}")
        
        return results
    
    def _benchmark_adaptive(
        self,
        model: nn.Module,
        base_model: nn.Module,
        test_loader: Any,
        eval_fn: Optional[Callable]
    ) -> Dict[str, Any]:
        """Benchmark Adaptive DeltaQuant."""
        print("\nBenchmarking Adaptive DeltaQuant...")
        
        config = EnhancedDeltaQuantConfig(
            method=QuantMethod.ADAPTIVE,
            min_bits=1,
            max_bits=8,
            importance_aware=True
        )
        quantizer = AdaptiveDeltaQuantizer(config)
        
        # Quantize model
        start_time = time.time()
        quantized_model = copy.deepcopy(model)
        for name, param in quantized_model.named_parameters():
            if name in dict(base_model.named_parameters()):
                base_param = dict(base_model.named_parameters())[name]
                quantized, params = quantizer.quantize_delta_enhanced(
                    param.data, base_param.data, name
                )
                delta = quantizer.dequantize_delta_enhanced(
                    quantized, params, param.shape
                )
                param.data = base_param.data + delta
        quantization_time = time.time() - start_time
        
        # Get compression stats
        stats = quantizer.get_compression_stats()
        
        # Measure inference
        inference_time = self._measure_inference_speed(
            quantized_model, test_loader
        )
        
        # Measure accuracy
        accuracy = None
        if eval_fn:
            accuracy = eval_fn(quantized_model, test_loader)
        
        # Measure memory
        memory_usage = self._measure_memory_usage(quantized_model)
        
        results = {
            'method': 'Adaptive',
            'average_bits': stats['average_bits'],
            'compression_ratio': stats['compression_ratio'],
            'quantization_time': quantization_time,
            'inference_time': inference_time,
            'accuracy': accuracy,
            'memory_mb': memory_usage,
            'bit_allocation': stats['bit_allocation'],
            'importance_scores': stats['importance_scores']
        }
        
        print(f"  Average bits: {stats['average_bits']:.2f}")
        print(f"  Compression: {stats['compression_ratio']:.2f}×")
        print(f"  Inference: {inference_time:.2f} ms/batch")
        if accuracy:
            print(f"  Accuracy: {accuracy:.4f}")
        
        return results
    
    def _benchmark_unified(
        self,
        model: nn.Module,
        base_model: nn.Module,
        test_loader: Any,
        eval_fn: Optional[Callable]
    ) -> Dict[str, Any]:
        """Benchmark Unified quantizer."""
        print("\nBenchmarking Unified Quantizer...")
        results = {}
        
        for backend in ['bitdelta', 'deltaquant', 'dynamic', 'mixed']:
            print(f"  Testing {backend} backend...")
            
            config = UnifiedQuantConfig(
                backend=QuantizationBackend[backend.upper()],
                bits=4,
                memory_map=True,
                auto_mixed_precision=True
            )
            quantizer = UnifiedQuantizer(config)
            
            # Measure quantization
            start_time = time.time()
            quantized_model = quantizer.quantize_model(
                copy.deepcopy(model), base_model
            )
            quantization_time = time.time() - start_time
            
            # Get compression ratio
            compression_ratio = quantizer.get_compression_ratio()
            
            # Measure inference
            inference_time = self._measure_inference_speed(
                quantized_model, test_loader
            )
            
            # Measure accuracy
            accuracy = None
            if eval_fn:
                accuracy = eval_fn(quantized_model, test_loader)
            
            # Get detailed metrics
            detailed_metrics = quantizer.benchmark(
                quantized_model, base_model, test_loader
            )
            
            results[backend] = {
                'method': f'Unified-{backend}',
                'compression_ratio': compression_ratio,
                'quantization_time': quantization_time,
                'inference_time': inference_time,
                'accuracy': accuracy,
                'detailed': detailed_metrics
            }
            
            print(f"    Compression: {compression_ratio:.2f}×")
            print(f"    Inference: {inference_time:.2f} ms/batch")
            if accuracy:
                print(f"    Accuracy: {accuracy:.4f}")
        
        return results
    
    def _measure_inference_speed(
        self,
        model: nn.Module,
        test_loader: Any
    ) -> float:
        """Measure inference speed in ms/batch."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.config.num_warmup_runs):
                for batch in test_loader:
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    _ = model(inputs)
                    break
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(self.config.num_benchmark_runs):
                for batch in test_loader:
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    
                    start = time.time()
                    _ = model(inputs)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    times.append((time.time() - start) * 1000)
                    break
        
        return np.mean(times)
    
    def _measure_memory_usage(self, model: nn.Module) -> float:
        """Measure model memory usage in MB."""
        total_params = 0
        total_bytes = 0
        
        for param in model.parameters():
            total_params += param.numel()
            # Assume quantized weights use less memory
            if param.dtype == torch.int8:
                total_bytes += param.numel()
            elif param.dtype == torch.float16:
                total_bytes += param.numel() * 2
            else:
                total_bytes += param.numel() * 4
        
        return total_bytes / (1024 * 1024)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results to file."""
        filepath = Path(self.config.results_dir) / f"benchmark_{time.time():.0f}.json"
        
        # Convert results to JSON-serializable format
        def convert(obj):
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        # Recursively convert
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(v) for v in d]
            else:
                return convert(d)
        
        converted_results = convert_dict(results)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"\nResults saved to {filepath}")
    
    def _plot_results(self, results: Dict[str, Any]):
        """Plot benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Compression vs Accuracy
        ax = axes[0, 0]
        for method, data in results.items():
            if isinstance(data, dict) and 'compression_ratio' in data:
                compression = data['compression_ratio']
                accuracy = data.get('accuracy', 0)
                ax.scatter(compression, accuracy, label=method, s=100)
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('Accuracy')
        ax.set_title('Compression vs Accuracy Trade-off')
        ax.legend()
        ax.grid(True)
        
        # Inference Speed
        ax = axes[0, 1]
        methods = []
        speeds = []
        for method, data in results.items():
            if isinstance(data, dict) and 'inference_time' in data:
                methods.append(method)
                speeds.append(data['inference_time'])
        ax.bar(methods, speeds)
        ax.set_ylabel('Inference Time (ms/batch)')
        ax.set_title('Inference Speed Comparison')
        ax.xticks(rotation=45)
        
        # Memory Usage
        ax = axes[1, 0]
        methods = []
        memories = []
        for method, data in results.items():
            if isinstance(data, dict) and 'memory_mb' in data:
                methods.append(method)
                memories.append(data['memory_mb'])
        if memories:
            ax.bar(methods, memories)
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Footprint')
            ax.xticks(rotation=45)
        
        # Quantization Time
        ax = axes[1, 1]
        methods = []
        times = []
        for method, data in results.items():
            if isinstance(data, dict) and 'quantization_time' in data:
                methods.append(method)
                times.append(data['quantization_time'])
        if times:
            ax.bar(methods, times)
            ax.set_ylabel('Quantization Time (seconds)')
            ax.set_title('Quantization Speed')
            ax.xticks(rotation=45)
        
        plt.tight_layout()
        
        if self.config.save_results:
            filepath = Path(self.config.results_dir) / f"benchmark_plot_{time.time():.0f}.png"
            plt.savefig(filepath, dpi=150)
            print(f"Plot saved to {filepath}")
        
        plt.show()
    
    def compare_methods(self):
        """Create comparison table of all methods."""
        try:
            import pandas as pd
        except ImportError:
            return None
        
        data = []
        for method, results in self.results.items():
            if isinstance(results, dict):
                if 'compression_ratio' in results:
                    # Single result
                    data.append({
                        'Method': method,
                        'Compression': f"{results['compression_ratio']:.2f}×",
                        'Accuracy': f"{results.get('accuracy', 0):.4f}",
                        'Inference (ms)': f"{results['inference_time']:.2f}",
                        'Memory (MB)': f"{results.get('memory_mb', 0):.1f}"
                    })
                else:
                    # Multiple results (e.g., different bit widths)
                    for submethod, subresults in results.items():
                        if isinstance(subresults, dict) and 'compression_ratio' in subresults:
                            data.append({
                                'Method': f"{method}-{submethod}",
                                'Compression': f"{subresults['compression_ratio']:.2f}×",
                                'Accuracy': f"{subresults.get('accuracy', 0):.4f}",
                                'Inference (ms)': f"{subresults['inference_time']:.2f}",
                                'Memory (MB)': f"{subresults.get('memory_mb', 0):.1f}"
                            })
        
        df = pd.DataFrame(data)
        return df


# Convenience function
def run_quantization_benchmark(
    model: nn.Module,
    base_model: nn.Module,
    test_loader: Any,
    eval_fn: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run comprehensive quantization benchmark.
    
    Args:
        model: Fine-tuned model
        base_model: Base model
        test_loader: Test data loader
        eval_fn: Optional evaluation function
        **kwargs: Additional config parameters
        
    Returns:
        Benchmark results
    """
    config = BenchmarkConfig(**kwargs)
    benchmark = QuantizationBenchmark(config)
    results = benchmark.benchmark_model(model, base_model, test_loader, eval_fn)
    
    # Print comparison table
    try:
        df = benchmark.compare_methods()
        print("\n" + "="*60)
        print("Quantization Methods Comparison")
        print("="*60)
        print(df.to_string(index=False))
    except ImportError:
        print("Install pandas for comparison table: pip install pandas")
    
    return results