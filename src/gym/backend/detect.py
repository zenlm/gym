"""Auto-detection.

Order of preference (fastest first):
    mlx   — Apple Silicon native (M1/M2/M3/M4) — ~2x faster than MPS
    cuda  — NVIDIA Blackwell/Hopper/Ada/Ampere (Spark = GB10)
    rocm  — AMD RDNA3.5 (Strix Halo 8060S) / CDNA
    mps   — Apple PyTorch fallback (only if MLX import fails)
    cpu   — last resort

`prefer="mlx"` forces a specific backend if available, else raises.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .base import Backend

log = logging.getLogger(__name__)

_PRIORITY = ("mlx", "cuda", "rocm", "mps", "cpu")


def probe_all() -> dict[str, Optional[Backend]]:
    """Probe every backend; returns name → Backend|None."""
    out: dict[str, Optional[Backend]] = {}
    for name in _PRIORITY:
        out[name] = _probe(name)
    return out


def detect(prefer: Optional[str] = None) -> Backend:
    """Pick the best available backend.

    prefer:
        None      → use _PRIORITY order
        "mlx"|... → use that backend or raise RuntimeError
        "auto"    → same as None

    Env override: ZEN_BACKEND=mlx|cuda|rocm|mps|cpu
    """
    prefer = os.environ.get("ZEN_BACKEND", prefer)
    if prefer and prefer != "auto":
        b = _probe(prefer)
        if b is not None:
            log.info("backend (forced): %s", b.describe())
            return b
        log.warning("backend %r unavailable — falling back to auto", prefer)

    for name in _PRIORITY:
        b = _probe(name)
        if b is not None:
            log.info("backend (auto): %s", b.describe())
            return b
    raise RuntimeError("no backend available — even CPU probe failed")


def _probe(name: str) -> Optional[Backend]:
    try:
        if name == "mlx":
            from .mlx_backend import probe
            return probe()
        if name in ("cuda", "rocm", "mps", "cpu"):
            from .torch_backend import probe
            return probe(name)
    except Exception as e:
        log.debug("backend %s unavailable: %s", name, e)
    return None
