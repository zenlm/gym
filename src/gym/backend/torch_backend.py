"""PyTorch backend — covers CUDA (Blackwell/Hopper/Ada), ROCm (RDNA3.5/CDNA),
MPS (Apple PyTorch), and CPU. One implementation, dispatches at runtime.

Covers in the lab:
    cuda — DGX Spark GB10 Blackwell (FP4 + FP8 + BF16, ~250 BF16 TFLOPS)
    rocm — AMD Strix Halo 8060S (RDNA3.5, ~60 BF16 TFLOPS, 128GB unified)
    mps  — fallback for M-series if MLX import fails
    cpu  — fallback
"""

from __future__ import annotations

import contextlib
import io
import logging
import platform
import struct
import sys
from typing import Any, ContextManager, Iterable, Optional

from .base import Backend, Capability, DType

log = logging.getLogger(__name__)


def probe(target: str) -> Optional[Backend]:
    try:
        import torch
    except ImportError:
        return None

    if target == "cuda":
        if not torch.cuda.is_available():
            return None
        if _is_hip(torch):                           # ROCm masquerades as CUDA — let the rocm probe own it
            return None
        return _make_cuda(torch)

    if target == "rocm":
        if not torch.cuda.is_available() or not _is_hip(torch):
            return None
        return _make_rocm(torch)

    if target == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            return None
        return _make_mps(torch)

    if target == "cpu":
        return _make_cpu(torch)

    return None


def _is_hip(torch) -> bool:
    return bool(getattr(torch.version, "hip", None))


def _make_cuda(torch) -> Backend:
    props = torch.cuda.get_device_properties(0)
    mem_gb = int(props.total_memory // (1024**3))
    name = props.name.lower()
    cap = Capability(
        has_bf16=True,
        has_fp16=True,
        has_fp8="blackwell" in name or "hopper" in name or "h100" in name or "gb10" in name or props.major >= 9,
        has_fp4="blackwell" in name or "gb10" in name or "b100" in name or "b200" in name or props.major >= 10,
        has_flash_attn=True,
        unified_memory="gb10" in name or "grace" in name or "thor" in name,
        can_train=True,
        can_quantize_int4=True,
    )
    tflops = _cuda_tflops(props)
    bw = _cuda_bandwidth(props)
    return _TorchBackend(
        name="cuda", device="cuda:0",
        memory_gb=mem_gb, bandwidth_gbps=bw,
        effective_tflops=tflops, capability=cap,
        dtype_default=DType.BF16, _torch=torch, _autocast_device="cuda",
    )


def _make_rocm(torch) -> Backend:
    props = torch.cuda.get_device_properties(0)
    mem_gb = int(props.total_memory // (1024**3))
    name = props.name.lower()
    cap = Capability(
        has_bf16=True, has_fp16=True,
        has_fp8="mi300" in name or "mi325" in name or "mi355" in name,
        has_fp4="mi355" in name,
        has_flash_attn=True,
        unified_memory="strix" in name or "halo" in name or "ryzen" in name or "8060" in name,
        can_train=True, can_quantize_int4=True,
    )
    # Strix Halo 8060S: ~60 BF16 TFLOPS, 256 GB/s LPDDR5x.
    tflops = 60 if cap.unified_memory else 100
    bw = 256 if cap.unified_memory else 1500
    return _TorchBackend(
        name="rocm", device="cuda:0",
        memory_gb=mem_gb, bandwidth_gbps=bw,
        effective_tflops=tflops, capability=cap,
        dtype_default=DType.BF16, _torch=torch, _autocast_device="cuda",
    )


def _make_mps(torch) -> Backend:
    from ._sysinfo import total_ram_bytes, apple_chip_name
    mem_gb = int(total_ram_bytes() // (1024**3))
    cap = Capability(
        has_bf16=True, has_fp16=True,
        has_fp8=False, has_fp4=False,
        has_flash_attn=False,
        unified_memory=True,
        can_train=True, can_quantize_int4=True,
    )
    # MPS is ~2x slower than MLX on the same silicon — penalize accordingly.
    chip = apple_chip_name()
    tflops = _apple_tflops(chip) // 2
    bw = _apple_bandwidth(chip)
    return _TorchBackend(
        name="mps", device="mps",
        memory_gb=mem_gb, bandwidth_gbps=bw,
        effective_tflops=tflops, capability=cap,
        dtype_default=DType.FP16, _torch=torch, _autocast_device="mps",
    )


def _make_cpu(torch) -> Backend:
    from ._sysinfo import total_ram_bytes, cpu_brand, cpu_core_count
    mem_gb = int(total_ram_bytes() // (1024**3))
    cap = Capability(
        has_bf16=hasattr(torch, "bfloat16"),
        has_fp16=True, has_fp8=False, has_fp4=False,
        has_flash_attn=False, unified_memory=True,
        can_train=True, can_quantize_int4=True,
    )
    tflops, bw = _cpu_specs(cpu_brand(), cpu_core_count())
    return _TorchBackend(
        name="cpu", device="cpu",
        memory_gb=mem_gb, bandwidth_gbps=bw,
        effective_tflops=tflops, capability=cap,
        dtype_default=DType.BF16 if cap.has_bf16 else DType.FP32,
        _torch=torch, _autocast_device="cpu",
    )


def _cpu_specs(brand: str, cores: int) -> tuple[int, int]:
    """Estimate BF16 TFLOPS + memory bandwidth from CPU brand string.

    These are *sustained* BF16 TFLOPS — what the scheduler should plan around,
    not peak vendor numbers. CPU paths are bandwidth-bound much more than
    GPU paths, so these stay conservative.
    """
    b = brand.lower()
    if "ai max" in b or "strix" in b or "ryzen ai" in b:
        # 16 Zen 5 cores @ ~5 GHz with AVX-512 BF16. Sustained ≈ 8 TFLOPS.
        return 8, 256
    if "threadripper" in b or "epyc" in b:
        return max(8, cores // 8), 400
    if any(k in b for k in ("ryzen", "xeon", "core(tm)", "i9", "i7", "i5")):
        return max(2, cores // 16), 80
    if "neoverse" in b or "cortex" in b or "graviton" in b:
        return max(2, cores // 16), 200
    return 1, 50


def _cuda_tflops(props) -> int:
    # Rough BF16 *dense* TFLOPS by SM count + arch. NVIDIA marketing numbers
    # are usually FP4 sparse (4-8× higher); these are sustained BF16 dense.
    sm = props.multi_processor_count
    name = props.name.lower()
    if "gb10" in name or "spark" in name or "grace" in name:
        return 31                  # GB10 dense BF16 (1 PFLOP FP4 sparse = ~31 BF16 dense)
    if props.major >= 10:           # Datacenter Blackwell B100/B200
        return int(sm * 12)        # B200 ≈ 2500
    if props.major == 9:            # Hopper
        return int(sm * 7.5)       # H100 ≈ 990
    if props.major == 8:            # Ada / Ampere
        return int(sm * 1.5)
    return int(sm * 0.5)


def _cuda_bandwidth(props) -> int:
    name = props.name.lower()
    if "gb10" in name or "spark" in name or "thor" in name:
        return 273                 # Spark LPDDR5x
    if "h100" in name: return 3350
    if "h200" in name: return 4800
    if "b100" in name: return 8000
    if "b200" in name: return 8000
    return int(props.total_memory // (1024**3) * 30)  # rough fallback


def _apple_tflops(chip: str) -> int:
    # BF16/FP16 TFLOPS on the GPU (≈ 2× the FP32 spec).
    c = chip.lower()
    if "m4 max" in c or "m4max" in c: return 34
    if "m4 pro" in c: return 20
    if "m4 ultra" in c: return 68
    if "m3 max" in c: return 28
    if "m3 ultra" in c: return 56
    if "m2 max" in c: return 27
    if "m1 max" in c: return 21    # 32-core GPU: 10.4 FP32 → ~21 BF16
    if "m1 ultra" in c: return 42  # 64-core GPU: 21 FP32 → ~42 BF16
    if "m1 pro" in c: return 10
    return 8


def _apple_bandwidth(chip: str) -> int:
    c = chip.lower()
    if "m4 max" in c: return 546
    if "m4 ultra" in c: return 1100
    if "m3 max" in c: return 400
    if "m3 ultra" in c: return 800
    if "m2 max" in c: return 400
    if "m1 max" in c: return 400
    if "m1 ultra" in c: return 800
    if "m1 pro" in c: return 200
    return 68


# ── concrete backend ─────────────────────────────────────────────────────────


class _TorchBackend(Backend):
    """Single PyTorch backend for cuda/rocm/mps/cpu. Frozen by inheritance from
    Backend (dataclass) but holds a torch handle in private fields."""

    def __init__(self, *, _torch, _autocast_device: str, **kw):
        object.__setattr__(self, "_torch", _torch)
        object.__setattr__(self, "_autocast_device", _autocast_device)
        # Frozen dataclass: set fields via object.__setattr__
        for k, v in kw.items():
            object.__setattr__(self, k, v)

    def autocast(self) -> ContextManager[None]:
        amp_dtype = self._torch.bfloat16 if self.capability.has_bf16 else self._torch.float16
        return self._torch.autocast(device_type=self._autocast_device, dtype=amp_dtype)

    def to_device(self, tensor: Any) -> Any:
        return tensor.to(self.device)

    def synchronize(self) -> None:
        if self._autocast_device == "cuda":
            self._torch.cuda.synchronize()
        elif self._autocast_device == "mps":
            self._torch.mps.synchronize()

    def export_lora_delta(self, params: Iterable[tuple[str, Any]]) -> bytes:
        # Canonical: BF16 little-endian, simple safetensors-shaped header.
        # Format: u64 hdr_len | json hdr | concatenated tensor bytes
        # numpy has no native bf16 → view-cast through uint16.
        import json
        torch = self._torch
        body = io.BytesIO()
        hdr: dict[str, dict] = {}
        offset = 0
        for name, tensor in params:
            t = tensor.detach().to("cpu", dtype=torch.bfloat16).contiguous()
            raw = t.view(torch.uint16).numpy().tobytes()
            hdr[name] = {"dtype": "BF16", "shape": list(t.shape),
                         "offsets": [offset, offset + len(raw)]}
            body.write(raw)
            offset += len(raw)
        hdr_json = json.dumps(hdr, separators=(",", ":")).encode()
        return struct.pack("<Q", len(hdr_json)) + hdr_json + body.getvalue()

    def import_lora_delta(self, blob: bytes) -> dict[str, Any]:
        import json
        import numpy as np
        torch = self._torch
        hdr_len = struct.unpack_from("<Q", blob, 0)[0]
        hdr = json.loads(blob[8:8 + hdr_len])
        base = 8 + hdr_len
        out: dict[str, Any] = {}
        for name, meta in hdr.items():
            s, e = meta["offsets"]
            raw = blob[base + s: base + e]
            # bfloat16 isn't numpy-native — load as uint16 then view-cast in torch.
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(meta["shape"])
            t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
            out[name] = t.to(self.device)
        return out

    def wrap(self, arr: Any) -> Any:
        return self._torch.from_numpy(arr).to(self.device, dtype=self._torch.bfloat16)

    def unwrap(self, tensor: Any) -> Any:
        return tensor.detach().to("cpu", dtype=self._torch.float32).numpy()
