"""Capacity-aware scheduler.

Two kinds of work to assign:

  1. Data shards — for federated LoRA training (zen4 identity, zen5 router
     post-extraction): each worker trains on a disjoint slice of the dataset,
     weighted by its capacity_score().

  2. Expert pins — for zen5 MoDE pipeline mode: each frozen expert is hosted
     by the node that has the memory + bandwidth to serve it. Coordinator (DGX
     Spark, 200GbE) gets the routing decisions; M4 Max (128GB / 546GB/s) gets
     the big text experts; Strix Halo (128GB / 256GB/s) gets vision/audio;
     M1 (64GB) gets the small dense experts.

Either way, the assignment is a value, returned by Scheduler.plan().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .topology import Lab, Node


@dataclass(frozen=True)
class Assignment:
    """A plan for one federation round."""
    data_weights: dict[str, float]            # node name → fraction of data
    expert_pins:  dict[str, tuple[str, ...]]  # node name → expert IDs
    role_caps:    dict[str, dict[str, int]]   # node name → {"max_batch": N, "max_seq": N}


class Scheduler:
    def __init__(self, lab: Lab):
        self.lab = lab

    # ── data sharding ────────────────────────────────────────────────────────

    def shard_data(self) -> dict[str, float]:
        """Weight each worker's data slice by capacity_score().

        Spark dominates compute → biggest slice. M1 still gets work but less.
        Sum normalizes to 1.0.
        """
        workers = self.lab.workers
        scores = {n.name: n.capacity_score() for n in workers}
        total = sum(scores.values()) or 1.0
        return {name: s / total for name, s in scores.items()}

    # ── expert pinning (zen5 MoDE) ───────────────────────────────────────────

    def pin_experts(self, experts: Iterable[tuple[str, int]]) -> dict[str, tuple[str, ...]]:
        """Assign expert IDs to nodes by best-fit-decreasing on memory.

        experts:  (expert_id, gb_required) pairs.
        Returns:  node_name → tuple of expert_ids.

        Honors lab pre-declared pin_experts as hard constraints — raises clearly
        if the declared pinning exceeds a node's memory (which means the lab
        author needs to split the model into smaller slabs).
        """
        sizes = dict(experts)
        pinned: dict[str, list[str]] = {n.name: list(n.pin_experts) for n in self.lab.workers}
        remaining_mem: dict[str, int] = {}
        for n in self.lab.workers:
            declared = sum(sizes.get(e, 0) for e in n.pin_experts)
            if declared > n.memory_gb:
                raise RuntimeError(
                    f"node {n.name} declared pins total {declared}GB "
                    f"but only has {n.memory_gb}GB — split the model into smaller slabs "
                    f"or move some pins to a larger node"
                )
            remaining_mem[n.name] = n.memory_gb - declared

        # Largest experts first → put them where they fit.
        for expert_id, gb in sorted(experts, key=lambda x: -x[1]):
            # Skip already-pinned ones.
            if any(expert_id in pinned[name] for name in pinned):
                continue
            # Pick the node with most remaining headroom that can fit it.
            candidates = [(name, mem) for name, mem in remaining_mem.items() if mem >= gb]
            if not candidates:
                raise RuntimeError(f"expert {expert_id} ({gb}GB) does not fit on any node")
            best = max(candidates, key=lambda x: x[1])[0]
            pinned[best].append(expert_id)
            remaining_mem[best] -= gb

        return {name: tuple(ids) for name, ids in pinned.items()}

    # ── full plan ────────────────────────────────────────────────────────────

    def plan(self, experts: Iterable[tuple[str, int]] | None = None) -> Assignment:
        data = self.shard_data()
        pins = self.pin_experts(experts or [])
        caps = {n.name: _caps_for(n) for n in self.lab.workers}
        return Assignment(data_weights=data, expert_pins=pins, role_caps=caps)


def _declared_pin_cost(n: Node, experts: Iterable[tuple[str, int]]) -> int:
    sizes = dict(experts)
    return sum(sizes.get(e, 0) for e in n.pin_experts)


def _caps_for(n: Node) -> dict[str, int]:
    """Per-node training caps.

    Memory → max activations → max_batch × max_seq.
    Use ~20% of memory for activations (rest for weights + KV + optimizer).
    Token-cost rule of thumb: bf16 ≈ 2 bytes × hidden_size × seq_len × batch.
    Assume hidden_size = 4096 baseline → 8KB/tok.
    """
    headroom_bytes = int(n.memory_gb * 0.20 * 1024**3)
    tok_budget = headroom_bytes // (8 * 1024)            # at h=4096
    if tok_budget < 4096:
        return {"max_batch": 1, "max_seq": max(2048, tok_budget)}
    if tok_budget < 65536:
        return {"max_batch": 1, "max_seq": tok_budget}
    return {"max_batch": tok_budget // 32768, "max_seq": 32768}
