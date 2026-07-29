"""Heterogeneous training backends.

One Backend value describes the local compute fabric. Auto-detection picks the
fastest available. LoRA-delta exchange is canonical BF16 — independent of
backend — so federation works across MLX / CUDA / ROCm / MPS / CPU.

Usage:
    from gym.backend import detect
    backend = detect()                 # auto: mlx > cuda > rocm > mps > cpu
    backend = detect(prefer="mlx")     # explicit
    print(backend.name, backend.memory_gb, backend.effective_tflops)
"""

from __future__ import annotations

from .base import Backend, Capability, DType
from .detect import detect, probe_all

__all__ = ["Backend", "Capability", "DType", "detect", "probe_all"]
