"""Lab topology — declared inventory of every box that can train.

A Node is a *value*: hostname, role, declared capacity. Real probed
capability comes from gym.backend.detect() running on the box itself
and is reported to the coordinator at join time.
"""

from __future__ import annotations

import enum
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _expand_env(text: str) -> str:
    """Expand ${VAR} from os.environ. Unset → 'null' (YAML null sentinel).

    Kept separate from yaml.safe_load so the parse step stays pure.
    """
    return re.sub(r"\$\{(\w+)\}",
                  lambda m: os.environ.get(m.group(1)) or "null", text)


class NodeRole(str, enum.Enum):
    COORDINATOR = "coordinator"   # runs DeltaSoup, hosts global router state
    WORKER      = "worker"        # trains, pushes deltas
    HYBRID      = "hybrid"        # does both — single-box dev mode


@dataclass(frozen=True)
class Node:
    name: str                     # short id: "spark", "m4max", "m1", "strix"
    host: str                     # mDNS / IP — "spark.lan"
    role: NodeRole
    backend_hint: str             # "mlx" | "cuda" | "rocm" — what we expect
    memory_gb: int                # declared total RAM
    nic_gbps: int                 # peak NIC throughput (200 for ConnectX-7, 10 for GbE)
    tflops_hint: int              # rough BF16 TFLOPS for the scheduler

    # Optional pinning hints
    pin_experts: tuple[str, ...] = ()    # zen5 expert IDs this node should host
    pin_modalities: tuple[str, ...] = () # "text" | "vision" | "audio" | "video" | "3d"
    auth_token: Optional[str] = None     # HMAC secret for transport

    def capacity_score(self) -> float:
        # Used by Scheduler. Memory and TFLOPS both bind — geometric mean prevents
        # one Spark from getting all the work just because it's fast (it'd OOM on
        # the bigger shards).
        return (self.memory_gb ** 0.5) * (self.tflops_hint ** 0.5)


@dataclass(frozen=True)
class Lab:
    nodes: tuple[Node, ...]
    job_dir: str = ".zen-fed"
    sync_interval_steps: int = 8          # push delta every N local steps
    aggregation: str = "byzantine_robust" # DeltaSoup method

    @property
    def coordinator(self) -> Node:
        for n in self.nodes:
            if n.role in (NodeRole.COORDINATOR, NodeRole.HYBRID):
                return n
        raise ValueError("lab has no coordinator")

    @property
    def workers(self) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes if n.role in (NodeRole.WORKER, NodeRole.HYBRID))

    def find(self, name: str) -> Node:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Lab":
        import yaml
        text = _expand_env(Path(path).read_text())
        data = yaml.safe_load(text)
        nodes = tuple(
            Node(
                name=n["name"],
                host=n["host"],
                role=NodeRole(n["role"]),
                backend_hint=n["backend"],
                memory_gb=n["memory_gb"],
                nic_gbps=n.get("nic_gbps", 10),
                tflops_hint=n.get("tflops", 10),
                pin_experts=tuple(n.get("pin_experts", [])),
                pin_modalities=tuple(n.get("pin_modalities", [])),
                auth_token=n.get("auth_token"),
            )
            for n in data["nodes"]
        )
        return cls(
            nodes=nodes,
            job_dir=data.get("job_dir", ".zen-fed"),
            sync_interval_steps=data.get("sync_interval_steps", 8),
            aggregation=data.get("aggregation", "byzantine_robust"),
        )
