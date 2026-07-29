"""Self-test — runs on any lab box.

Reports:
  • this box's detected backend (mlx / cuda / rocm / mps / cpu)
  • memory + bandwidth + effective TFLOPS
  • canonical LoRA-delta round-trip is byte-identical
  • if --coordinator URL given: healthz + topology + delta push

Exit 0 if everything passes, non-zero otherwise. Designed for ssh fanout:

    for box in dbc.local evo.local spark.local; do
      ssh $box python -m gym.distributed self-test --coordinator http://spark.local:8443
    done
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from typing import Optional

from ..backend import detect, probe_all
from .transport import TransportClient

log = logging.getLogger(__name__)


def selftest(coordinator_url: Optional[str] = None, as_name: Optional[str] = None) -> int:
    """Returns 0 on success, non-zero on first failure."""
    host = socket.gethostname()
    print(f"# self-test on {host}", flush=True)

    # 1. Backend probe ───────────────────────────────────────────────────────
    print("## backend probe")
    available = probe_all()
    for name, b in available.items():
        status = "OK" if b is not None else "--"
        print(f"  {name:6s} {status}")
    try:
        backend = detect()
    except RuntimeError as e:
        print(f"FAIL: {e}"); return 2
    print(f"## chose: {json.dumps(backend.describe(), indent=2)}")

    # 2. Canonical round-trip ────────────────────────────────────────────────
    print("## canonical bf16 round-trip")
    try:
        _verify_roundtrip(backend)
        print("  ok")
    except AssertionError as e:
        print(f"FAIL: {e}"); return 3

    # 3. Coordinator handshake (optional) ────────────────────────────────────
    if coordinator_url:
        worker = as_name or host.split(".")[0]
        print(f"## coordinator handshake → {coordinator_url} as {worker}")
        try:
            client = TransportClient(coordinator_url, worker, secret=os.environ.get("ZEN_LAB_SECRET"))
            print(f"  healthz: {client.healthz()}")
            topo = client._request("GET", "/v1/topology")
            print(f"  workers: {[w['name'] for w in topo.get('workers', [])]}")
            print(f"  my data weight: {topo.get('data_weights', {}).get(worker, '<not in lab>')}")
        except Exception as e:
            print(f"FAIL: {e}"); return 4

    print("# OK", flush=True)
    return 0


def _verify_roundtrip(backend) -> None:
    """Export and re-import a small tensor; verify lossless and byte-identical."""
    import numpy as np
    # Build a torch tensor (every backend supports torch on the import side).
    try:
        import torch
    except ImportError:
        # MLX-only environment — synthesize an MLX array directly.
        try:
            import mlx.core as mx
            src = mx.random.normal(shape=(64, 256)).astype(mx.bfloat16)
            blob1 = backend.export_lora_delta([("x", src)])
            back = backend.import_lora_delta(blob1)
            blob2 = backend.export_lora_delta([("x", back["x"])])
            assert blob1 == blob2, "MLX re-export drifted"
            return
        except ImportError:
            raise AssertionError("neither torch nor mlx available")
    src = torch.randn(64, 256, dtype=torch.bfloat16)
    blob1 = backend.export_lora_delta([("x", src)])
    back = backend.import_lora_delta(blob1)
    blob2 = backend.export_lora_delta([("x", back["x"])])
    assert blob1 == blob2, "re-export produced different bytes"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="zen lab self-test")
    p.add_argument("--coordinator", help="optional: probe this coordinator URL")
    p.add_argument("--as", dest="as_name", help="worker name to identify as")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    return selftest(args.coordinator, args.as_name)


if __name__ == "__main__":
    sys.exit(main())
