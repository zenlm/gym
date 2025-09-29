"""
Metrics collection and monitoring for training runs.
Supports TensorBoard, WandB, and custom backends.
"""

import os
import json
import time
import psutil
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Single metric measurement."""
    name: str
    value: float
    step: int
    timestamp: float
    metadata: Dict[str, Any] = None


class MetricsBuffer:
    """Thread-safe buffer for metrics with moving averages."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.buffers = {}
        self.steps = {}

    def add(self, name: str, value: float, step: Optional[int] = None):
        """Add metric value to buffer."""
        if name not in self.buffers:
            self.buffers[name] = deque(maxlen=self.window_size)
            self.steps[name] = 0

        if step is not None:
            self.steps[name] = step
        else:
            self.steps[name] += 1

        self.buffers[name].append(value)

    def get_average(self, name: str) -> Optional[float]:
        """Get moving average for metric."""
        if name not in self.buffers or len(self.buffers[name]) == 0:
            return None
        return np.mean(list(self.buffers[name]))

    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value for metric."""
        if name not in self.buffers or len(self.buffers[name]) == 0:
            return None
        return self.buffers[name][-1]

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for metric."""
        if name not in self.buffers or len(self.buffers[name]) == 0:
            return {}

        values = list(self.buffers[name])
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "latest": values[-1],
            "count": len(values),
        }


class ResourceMonitor:
    """Monitor system resources (CPU, GPU, memory)."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        metrics = {
            "cpu_percent": self.process.cpu_percent(),
            "memory_mb": self.process.memory_info().rss / 1024 / 1024,
            "memory_percent": self.process.memory_percent(),
            "uptime_seconds": time.time() - self.start_time,
        }

        # GPU metrics if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                metrics.update({
                    f"gpu_{i}_memory_mb": torch.cuda.memory_allocated(i) / 1024 / 1024,
                    f"gpu_{i}_memory_reserved_mb": torch.cuda.memory_reserved(i) / 1024 / 1024,
                    f"gpu_{i}_utilization": torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0,
                })

        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            metrics.update({
                "network_sent_mb": net_io.bytes_sent / 1024 / 1024,
                "network_recv_mb": net_io.bytes_recv / 1024 / 1024,
            })
        except Exception:
            pass

        return metrics


class MetricsCollector:
    """Central metrics collection and reporting system."""

    def __init__(
        self,
        log_dir: str = "./logs",
        backends: Optional[List[str]] = None,
        log_interval: int = 10,
        resource_interval: int = 30,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.backends = backends or ["console", "json"]
        self.log_interval = log_interval
        self.resource_interval = resource_interval

        self.buffer = MetricsBuffer()
        self.resource_monitor = ResourceMonitor()
        self.global_step = 0
        self.last_log_step = 0
        self.last_resource_step = 0

        # Initialize backends
        self._init_backends()

    def _init_backends(self):
        """Initialize metrics backends."""
        self.writers = {}

        if "tensorboard" in self.backends:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writers["tensorboard"] = SummaryWriter(self.log_dir / "tensorboard")
                logger.info(f"TensorBoard logging to {self.log_dir / 'tensorboard'}")
            except ImportError:
                logger.warning("TensorBoard not available")

        if "wandb" in self.backends:
            try:
                import wandb
                wandb.init(project="gym-training", dir=str(self.log_dir))
                self.writers["wandb"] = wandb
                logger.info("WandB logging initialized")
            except ImportError:
                logger.warning("WandB not available")

        if "json" in self.backends:
            json_path = self.log_dir / "metrics.jsonl"
            self.writers["json"] = open(json_path, "a")
            logger.info(f"JSON logging to {json_path}")

    def log(
        self,
        metrics: Dict[str, Union[float, torch.Tensor]],
        step: Optional[int] = None,
        prefix: str = "",
    ):
        """Log metrics to all backends."""
        if step is None:
            step = self.global_step
            self.global_step += 1
        else:
            self.global_step = step

        # Convert tensors to scalars
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            if prefix:
                key = f"{prefix}/{key}"
            processed_metrics[key] = value
            self.buffer.add(key, value, step)

        # Log to backends
        self._write_to_backends(processed_metrics, step)

        # Log resources periodically (but don't call recursively)
        if step - self.last_resource_step >= self.resource_interval:
            resource_metrics = self.resource_monitor.get_metrics()
            prefixed_resources = {f"resources/{k}": v for k, v in resource_metrics.items()}
            self._write_to_backends(prefixed_resources, step)
            self.last_resource_step = step

        # Console output periodically
        if "console" in self.backends and step - self.last_log_step >= self.log_interval:
            self._console_log(processed_metrics, step)
            self.last_log_step = step

    def log_resources(self, step: Optional[int] = None):
        """Log system resource metrics."""
        resources = self.resource_monitor.get_metrics()
        self.log(resources, step=step, prefix="resources")

    def log_histogram(self, name: str, values: torch.Tensor, step: Optional[int] = None):
        """Log histogram of values (e.g., weights, gradients)."""
        if step is None:
            step = self.global_step

        if "tensorboard" in self.writers:
            self.writers["tensorboard"].add_histogram(name, values, step)

        if "wandb" in self.writers:
            import wandb
            wandb.log({name: wandb.Histogram(values.cpu().numpy())}, step=step)

    def log_image(self, name: str, image: torch.Tensor, step: Optional[int] = None):
        """Log image tensor."""
        if step is None:
            step = self.global_step

        if "tensorboard" in self.writers:
            self.writers["tensorboard"].add_image(name, image, step)

        if "wandb" in self.writers:
            import wandb
            wandb.log({name: wandb.Image(image.cpu().numpy())}, step=step)

    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor):
        """Log model computation graph."""
        if "tensorboard" in self.writers:
            self.writers["tensorboard"].add_graph(model, input_sample)

    def _write_to_backends(self, metrics: Dict[str, float], step: int):
        """Write metrics to all configured backends."""
        timestamp = time.time()

        if "tensorboard" in self.writers:
            for key, value in metrics.items():
                self.writers["tensorboard"].add_scalar(key, value, step)

        if "wandb" in self.writers:
            self.writers["wandb"].log(metrics, step=step)

        if "json" in self.writers:
            record = {
                "step": step,
                "timestamp": timestamp,
                "metrics": metrics,
            }
            self.writers["json"].write(json.dumps(record) + "\n")
            self.writers["json"].flush()

    def _console_log(self, metrics: Dict[str, float], step: int):
        """Log metrics to console."""
        # Format metrics for display
        display_metrics = []
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                display_metrics.append(f"{key}: {value:.4f}")
            else:
                display_metrics.append(f"{key}: {value}")

        # Create log message
        message = f"[Step {step}] " + " | ".join(display_metrics[:5])  # Show top 5 metrics
        if len(display_metrics) > 5:
            message += f" | ... ({len(display_metrics) - 5} more)"

        logger.info(message)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics."""
        summary = {
            "total_steps": self.global_step,
            "uptime_seconds": self.resource_monitor.get_metrics()["uptime_seconds"],
            "metrics": {}
        }

        for name in self.buffer.buffers:
            summary["metrics"][name] = self.buffer.get_stats(name)

        return summary

    def flush(self):
        """Flush all backends."""
        for backend, writer in self.writers.items():
            if backend == "tensorboard":
                writer.flush()
            elif backend == "json":
                writer.flush()
            elif backend == "wandb":
                import wandb
                wandb.finish()

    def close(self):
        """Close all backends."""
        self.flush()
        for backend, writer in self.writers.items():
            if backend == "tensorboard":
                writer.close()
            elif backend == "json":
                writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class TrainingMonitor:
    """High-level training monitor with automatic metric tracking."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.epoch = 0
        self.batch = 0
        self.best_metric = float('inf')
        self.patience_counter = 0

    def on_epoch_start(self, epoch: int):
        """Called at epoch start."""
        self.epoch = epoch
        self.batch = 0
        logger.info(f"Starting epoch {epoch}")

    def on_batch_end(self, loss: float, batch_size: int, learning_rate: float):
        """Called after each batch."""
        self.batch += 1
        step = self.epoch * 10000 + self.batch  # Assume max 10k batches per epoch

        metrics = {
            "train/loss": loss,
            "train/learning_rate": learning_rate,
            "train/batch_size": batch_size,
            "train/epoch": self.epoch,
            "train/batch": self.batch,
        }

        self.metrics.log(metrics, step=step)

    def on_validation(self, val_metrics: Dict[str, float], save_checkpoint: bool = True):
        """Called after validation."""
        step = self.epoch * 10000

        # Log validation metrics
        prefixed_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
        self.metrics.log(prefixed_metrics, step=step)

        # Track best model
        primary_metric = val_metrics.get("loss", float('inf'))
        if primary_metric < self.best_metric:
            self.best_metric = primary_metric
            self.patience_counter = 0
            logger.info(f"New best validation metric: {primary_metric:.4f}")
            return True  # Signal to save checkpoint
        else:
            self.patience_counter += 1
            return False

    def on_epoch_end(self):
        """Called at epoch end."""
        logger.info(f"Completed epoch {self.epoch}")
        self.metrics.log_resources(step=self.epoch * 10000)

    def should_early_stop(self, patience: int = 10) -> bool:
        """Check if training should stop early."""
        return self.patience_counter >= patience


# Export main components
__all__ = [
    'MetricSnapshot',
    'MetricsBuffer',
    'ResourceMonitor',
    'MetricsCollector',
    'TrainingMonitor',
]