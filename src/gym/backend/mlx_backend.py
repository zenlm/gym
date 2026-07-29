"""MLX backend — Apple Silicon native.

MLX runs ~2× MPS throughput on the same hardware for LoRA-class workloads
and uses the unified memory pool fully (vs MPS which caps at 75% by default).
On a 128 GB M4 Max this is the difference between fitting 30B-class models or
not — required, not a nice-to-have.

LoRA-delta serialization writes the same canonical BF16 safetensors-shaped
blob the torch backend emits, so a CUDA Spark and an MLX M4 Max can swap
adapters byte-for-byte through the coordinator.
"""

from __future__ import annotations

import contextlib
import io
import logging
import platform
import struct
from typing import Any, ContextManager, Iterable, Optional

from .base import Backend, Capability, DType

log = logging.getLogger(__name__)


def probe() -> Optional[Backend]:
    if platform.system() != "Darwin":
        return None
    try:
        import mlx.core as mx
    except ImportError:
        return None

    from ._sysinfo import total_ram_bytes, apple_chip_name
    mem_gb = int(total_ram_bytes() // (1024**3))
    chip = apple_chip_name().lower()
    tflops, bw = _apple_specs(chip)
    cap = Capability(
        has_bf16=True, has_fp16=True,
        has_fp8=False, has_fp4=False,
        has_flash_attn=True,                 # MLX has fused SDPA
        unified_memory=True,
        can_train=True, can_quantize_int4=True,
    )
    return _MLXBackend(
        name="mlx", device="mlx",
        memory_gb=mem_gb, bandwidth_gbps=bw,
        effective_tflops=tflops, capability=cap,
        dtype_default=DType.BF16, _mx=mx,
    )


def _apple_specs(chip: str) -> tuple[int, int]:
    # (BF16 TFLOPS, memory bandwidth GB/s). BF16 ≈ 2× the FP32 spec.
    if "m4 max" in chip: return 34, 546
    if "m4 ultra" in chip: return 68, 1100
    if "m4 pro" in chip: return 20, 273
    if "m3 max" in chip: return 28, 400
    if "m3 ultra" in chip: return 56, 800
    if "m2 max" in chip: return 27, 400
    if "m1 max" in chip: return 21, 400     # 32-core M1 Max: 10.4 FP32 → ~21 BF16
    if "m1 ultra" in chip: return 42, 800
    if "m1 pro" in chip: return 10, 200
    return 8, 68


class _MLXBackend(Backend):
    def __init__(self, *, _mx, **kw):
        object.__setattr__(self, "_mx", _mx)
        for k, v in kw.items():
            object.__setattr__(self, k, v)

    def autocast(self) -> ContextManager[None]:
        # MLX casts via dtype on op; no global autocast context. Use a no-op.
        return contextlib.nullcontext()

    def to_device(self, tensor: Any) -> Any:
        # MLX arrays live on the unified memory pool — no explicit move needed.
        return tensor

    def synchronize(self) -> None:
        self._mx.eval(self._mx.array([0]))   # force kernel completion

    def export_lora_delta(self, params: Iterable[tuple[str, Any]]) -> bytes:
        """Accept native mlx.array OR a torch.Tensor (a Mac may run either
        framework). Output is canonical BF16 either way.
        """
        import json
        import numpy as np
        mx = self._mx
        body = io.BytesIO()
        hdr: dict[str, dict] = {}
        offset = 0
        for name, tensor in params:
            shape, raw = _to_canonical_bf16(tensor, mx, np)
            hdr[name] = {"dtype": "BF16", "shape": shape,
                         "offsets": [offset, offset + len(raw)]}
            body.write(raw)
            offset += len(raw)
        hdr_json = json.dumps(hdr, separators=(",", ":")).encode()
        return struct.pack("<Q", len(hdr_json)) + hdr_json + body.getvalue()

    def import_lora_delta(self, blob: bytes) -> dict[str, Any]:
        import json
        import numpy as np
        mx = self._mx
        hdr_len = struct.unpack_from("<Q", blob, 0)[0]
        hdr = json.loads(blob[8:8 + hdr_len])
        base = 8 + hdr_len
        out: dict[str, Any] = {}
        for name, meta in hdr.items():
            s, e = meta["offsets"]
            raw = blob[base + s: base + e]
            # bf16 → f32 by shifting back into the high 16 bits.
            u16 = np.frombuffer(raw, dtype=np.uint16).reshape(meta["shape"])
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            out[name] = mx.array(f32).astype(mx.bfloat16)
        return out

    def wrap(self, arr: Any) -> Any:
        return self._mx.array(arr).astype(self._mx.bfloat16)

    def unwrap(self, tensor: Any) -> Any:
        import numpy as np
        f32 = tensor.astype(self._mx.float32)
        self._mx.eval(f32)
        return np.asarray(f32, dtype=np.float32)


def _to_canonical_bf16(tensor, mx, np) -> tuple[list[int], bytes]:
    """Dispatch on tensor type. MLX or torch both legitimate on a Mac."""
    # MLX array
    if type(tensor).__module__.startswith("mlx"):
        t = tensor.astype(mx.float32)
        mx.eval(t)
        arr = np.asarray(t, dtype=np.float32)
    else:
        # Assume torch tensor — duck-type via .detach / .cpu / .float
        arr = tensor.detach().to("cpu").float().contiguous().numpy()
    raw = (arr.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    return list(arr.shape), raw
