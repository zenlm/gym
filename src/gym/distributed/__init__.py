"""Heterogeneous federated training across the lab.

One way only: sync at LoRA-delta granularity over HTTP. No cross-vendor NCCL.

    Backend        — local fabric (mlx | cuda | rocm | mps | cpu)
    Lab / Node     — declared inventory from lab.yaml
    Scheduler      — capacity-aware shard + expert pinning
    Transport      — vendor-neutral HTTP with HMAC auth
    Coordinator    — DeltaSoup-style trim-mean aggregation
    Worker         — orthogonal to model; caller supplies step_fn

Run:
    python -m gym.distributed coordinator --lab zen5/lab.yaml
    python -m gym.distributed worker --lab zen5/lab.yaml --as m4max
    python -m gym.distributed self-test
"""

from __future__ import annotations

from .coordinator import Coordinator, CoordinatorState, aggregate
from .scheduler import Assignment, Scheduler
from .topology import Lab, Node, NodeRole
from .transport import TransportClient, make_server, sign, verify
from .worker import Worker

__all__ = [
    "Backend",
    "Lab", "Node", "NodeRole",
    "Scheduler", "Assignment",
    "TransportClient", "make_server", "sign", "verify",
    "Coordinator", "CoordinatorState", "aggregate",
    "Worker",
]


# Convenience re-export so callers only need one import line.
from ..backend import Backend  # noqa: E402,F401
