"""Backend protocol.

A Backend is a *value*: it describes the local compute fabric and exposes a
small surface to load a model, run an autocast context, build an optimizer,
and serialize/deserialize a LoRA delta in canonical BF16.

The LoRA-delta canonicalization is the load-bearing invariant. It lets MLX,
CUDA-Blackwell, ROCm, and CPU nodes federate without any shared collective.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any, ContextManager, Iterable


class DType(str, enum.Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP8  = "fp8"
    FP4  = "fp4"
    FP32 = "fp32"


@dataclass(frozen=True)
class Capability:
    """What a backend can do. Used by the scheduler."""
    has_bf16: bool = True
    has_fp16: bool = True
    has_fp8: bool = False
    has_fp4: bool = False
    has_flash_attn: bool = False
    unified_memory: bool = False        # M-series, Spark, Strix Halo all True
    can_train: bool = True
    can_quantize_int4: bool = True


@dataclass(frozen=True)
class Backend(abc.ABC):
    """Local compute fabric — frozen value."""
    name: str                            # mlx | cuda | rocm | mps | cpu
    device: str                          # mlx | cuda:0 | cuda:0 (HIP) | mps | cpu
    memory_gb: int                       # detected RAM/VRAM (rounded down)
    bandwidth_gbps: int                  # memory bandwidth (rough)
    effective_tflops: int                # BF16 dense TFLOPS — for scheduler weighting
    capability: Capability               # per-feature flags
    dtype_default: DType = DType.BF16

    # ── runtime surface (subclasses implement) ────────────────────────────────

    @abc.abstractmethod
    def autocast(self) -> ContextManager[None]:
        """Context manager for mixed-precision forward pass."""

    @abc.abstractmethod
    def to_device(self, tensor: Any) -> Any:
        """Place tensor on this backend's device."""

    @abc.abstractmethod
    def synchronize(self) -> None:
        """Wait for queued work — needed before measuring step time."""

    # ── canonical LoRA-delta exchange ─────────────────────────────────────────

    @abc.abstractmethod
    def export_lora_delta(self, params: Iterable[tuple[str, Any]]) -> bytes:
        """Serialize (name, tensor) pairs as canonical BF16 safetensors bytes.

        Output is backend-independent: any other Backend can import these bytes
        with import_lora_delta() and apply the result to a structurally-equivalent
        model state. This is the seam that lets MLX↔CUDA↔ROCm federate.
        """

    @abc.abstractmethod
    def import_lora_delta(self, blob: bytes) -> dict[str, Any]:
        """Deserialize canonical BF16 safetensors into native backend tensors."""

    # ── numpy bridge (decomplecting state from backend type) ──────────────────

    @abc.abstractmethod
    def wrap(self, arr: Any) -> Any:
        """numpy array → backend-native bf16 tensor on this device.

        Lets callers keep state in numpy (the only universal type) and only
        cross the backend boundary at the export/import seam.
        """

    @abc.abstractmethod
    def unwrap(self, tensor: Any) -> Any:
        """backend-native tensor → numpy float32 array on host.

        Inverse of wrap(). Use this in apply_fn so it never branches on
        backend identity.
        """

    # ── identity (logged + reported to coordinator) ───────────────────────────

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
            "memory_gb": self.memory_gb,
            "bandwidth_gbps": self.bandwidth_gbps,
            "effective_tflops": self.effective_tflops,
            "dtype_default": self.dtype_default.value,
            "capability": {
                "has_bf16": self.capability.has_bf16,
                "has_fp16": self.capability.has_fp16,
                "has_fp8": self.capability.has_fp8,
                "has_fp4": self.capability.has_fp4,
                "has_flash_attn": self.capability.has_flash_attn,
                "unified_memory": self.capability.unified_memory,
                "can_train": self.capability.can_train,
                "can_quantize_int4": self.capability.can_quantize_int4,
            },
        }
