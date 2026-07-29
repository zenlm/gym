"""Coordinator daemon.

Runs on the DGX Spark (or whichever node has the best NIC). Receives canonical
BF16 LoRA-delta blobs from workers each round; aggregates with DeltaSoup;
serves the consensus delta back. Stateless w.r.t. the model weights themselves:
workers hold their own copies and just apply the consensus delta locally.

Run:
    python -m gym.distributed.hetero.coordinator --lab zen5/lab.yaml --bind 0.0.0.0:8443

Or programmatically:
    Coordinator(lab).serve()
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .scheduler import Assignment, Scheduler
from .topology import Lab
from .transport import make_server

log = logging.getLogger(__name__)


# ── DeltaSoup-style aggregation (without the full reputation system) ────────


def _decode(blob: bytes) -> tuple[dict, dict]:
    """Returns (header, tensors_as_np_uint16). Stays in uint16 so we can
    aggregate cheaply on CPU on the coordinator.
    """
    hdr_len = struct.unpack_from("<Q", blob, 0)[0]
    hdr = json.loads(blob[8:8 + hdr_len])
    base = 8 + hdr_len
    tensors: dict[str, np.ndarray] = {}
    for name, meta in hdr.items():
        s, e = meta["offsets"]
        u16 = np.frombuffer(blob[base + s: base + e], dtype=np.uint16).reshape(meta["shape"])
        tensors[name] = u16.copy()                          # detach from blob memory
    return hdr, tensors


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16(f32: np.ndarray) -> np.ndarray:
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


def _encode(tensors: dict[str, np.ndarray]) -> bytes:
    hdr: dict[str, dict] = {}
    body = io.BytesIO()
    offset = 0
    for name, u16 in tensors.items():
        raw = u16.tobytes()
        hdr[name] = {"dtype": "BF16", "shape": list(u16.shape),
                     "offsets": [offset, offset + len(raw)]}
        body.write(raw)
        offset += len(raw)
    hdr_json = json.dumps(hdr, separators=(",", ":")).encode()
    return struct.pack("<Q", len(hdr_json)) + hdr_json + body.getvalue()


def aggregate(deltas: list[dict[str, np.ndarray]], method: str = "byzantine_robust") -> dict[str, np.ndarray]:
    """Combine N worker deltas into one consensus delta.

    byzantine_robust: trimmed-mean over the f32 view, robust to one bad worker.
    mean: plain average.
    median: per-element median.
    """
    names = list(deltas[0].keys())
    out: dict[str, np.ndarray] = {}
    for name in names:
        stack = np.stack([_bf16_to_f32(d[name]) for d in deltas], axis=0)  # [N, ...]
        if method == "median":
            agg = np.median(stack, axis=0)
        elif method == "byzantine_robust":
            # Drop top and bottom 1 if N >= 4 — classic trimmed mean.
            n = stack.shape[0]
            if n >= 4:
                sorted_ = np.sort(stack, axis=0)
                agg = sorted_[1:-1].mean(axis=0)
            else:
                agg = stack.mean(axis=0)
        else:
            agg = stack.mean(axis=0)
        out[name] = _f32_to_bf16(agg)
    return out


# ── coordinator state ───────────────────────────────────────────────────────


@dataclass
class _Round:
    round_id: int
    expected: set[str]
    received: dict[str, bytes] = field(default_factory=dict)
    losses: dict[str, float] = field(default_factory=dict)
    aggregate: Optional[bytes] = None
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class CoordinatorState:
    def __init__(self, lab: Lab):
        self.lab = lab
        self.scheduler = Scheduler(lab)
        self.assignment: Assignment = self.scheduler.plan()
        self.rounds: dict[int, _Round] = {}
        self._lock = threading.Lock()
        self.method = lab.aggregation
        log.info("coordinator initialized — %d workers expected", len(lab.workers))
        log.info("data weights: %s", self.assignment.data_weights)

    def topology_view(self) -> dict:
        return {
            "workers": [
                {"name": n.name, "host": n.host, "backend": n.backend_hint,
                 "memory_gb": n.memory_gb, "pin_experts": list(n.pin_experts)}
                for n in self.lab.workers
            ],
            "data_weights": self.assignment.data_weights,
            "expert_pins": {k: list(v) for k, v in self.assignment.expert_pins.items()},
            "aggregation": self.method,
            "sync_interval_steps": self.lab.sync_interval_steps,
        }

    def metrics(self) -> dict:
        """Single JSON surface for hanzo desktop + cloud telemetry."""
        with self._lock:
            rounds = []
            for rid in sorted(self.rounds.keys())[-50:]:     # last 50 rounds
                r = self.rounds[rid]
                rounds.append({
                    "round_id": r.round_id,
                    "expected": sorted(r.expected),
                    "received": sorted(r.received.keys()),
                    "losses": dict(r.losses),
                    "aggregated": r.aggregate is not None,
                    "started_at": r.started_at,
                    "completed_at": r.completed_at,
                    "duration_s": (r.completed_at - r.started_at) if r.completed_at else None,
                })
            return {
                "topology": self.topology_view(),
                "rounds": rounds,
                "current_round": max(self.rounds.keys()) if self.rounds else -1,
            }

    def put_delta(self, round_id: int, worker: str, blob: bytes) -> None:
        with self._lock:
            r = self.rounds.get(round_id)
            if r is None:
                expected = {n.name for n in self.lab.workers}
                r = _Round(round_id=round_id, expected=expected)
                self.rounds[round_id] = r
            r.received[worker] = blob
            log.info("round %d: %s/%s received from %s (%d bytes)",
                     round_id, len(r.received), len(r.expected), worker, len(blob))
            if r.received.keys() == r.expected and r.aggregate is None:
                self._aggregate_locked(r)

    def get_aggregate(self, round_id: int) -> bytes:
        with self._lock:
            r = self.rounds.get(round_id)
            if r and r.aggregate is not None:
                return r.aggregate
        # Block briefly until ready (caller is a worker waiting on the bus).
        for _ in range(600):
            time.sleep(1)
            with self._lock:
                r = self.rounds.get(round_id)
                if r and r.aggregate is not None:
                    return r.aggregate
        raise TimeoutError(f"round {round_id} not aggregated within 10 min")

    def end_round(self, round_id: int, worker: str, payload: dict) -> None:
        with self._lock:
            r = self.rounds.get(round_id)
            if r is None: return
            if "loss" in payload: r.losses[worker] = float(payload["loss"])

    def _aggregate_locked(self, r: _Round) -> None:
        log.info("round %d: aggregating %d deltas", r.round_id, len(r.received))
        decoded = [_decode(b)[1] for b in r.received.values()]
        agg = aggregate(decoded, method=self.method)
        r.aggregate = _encode(agg)
        r.completed_at = time.time()
        log.info("round %d: done in %.1fs", r.round_id, r.completed_at - r.started_at)


# ── entry point ─────────────────────────────────────────────────────────────


class Coordinator:
    def __init__(self, lab: Lab):
        self.state = CoordinatorState(lab)

    def serve(self, bind: tuple[str, int] = ("0.0.0.0", 8443)) -> None:
        secrets = {n.name: n.auth_token for n in self.state.lab.workers if n.auth_token}
        server = make_server(self.state, secrets, bind)
        log.info("coordinator listening on %s:%s", *bind)
        server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="zen federated coordinator")
    p.add_argument("--lab", required=True, help="path to lab.yaml")
    p.add_argument("--bind", default="0.0.0.0:8443", help="host:port")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    )
    lab = Lab.from_yaml(args.lab)
    host, port = args.bind.split(":")
    Coordinator(lab).serve((host, int(port)))


if __name__ == "__main__":
    main()
