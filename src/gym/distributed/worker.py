"""Worker — local federated trainer.

Runs on each lab node. Detects its backend, joins the coordinator, trains
locally for N steps, pushes the LoRA delta, pulls the aggregate, applies
it to local state, repeats.

The actual training step is supplied by the caller as a callable — this
keeps `Worker` orthogonal to whether you're training the zen5 router,
zen4 identity LoRA, or anything else.

Usage:
    from gym.backend import detect
    from gym.distributed.hetero import Worker

    backend = detect()
    worker  = Worker(
        coordinator_url="http://spark.lan:8443",
        worker_name="m4max",
        backend=backend,
        secret=...,                       # from lab.yaml
    )
    worker.run(
        step_fn=my_step,                  # (batch, params) -> (loss, new_params)
        params_iter=lambda: model.lora_state_dict().items(),
        apply_fn=lambda delta: model.apply_lora_delta(delta),
        data_iter=my_data_iter,
        steps_per_round=8,
        total_rounds=1000,
    )
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional

from ..backend import Backend
from .transport import TransportClient

log = logging.getLogger(__name__)


# Callback shapes — explicit so the worker stays orthogonal to the model code.
StepFn      = Callable[[Any, Iterator[tuple[str, Any]]], float]   # (batch, params) -> loss
ParamsIter  = Callable[[], Iterable[tuple[str, Any]]]             # snapshot trainable params
ApplyFn     = Callable[[dict[str, Any]], None]                    # apply consensus delta in-place
DataIter    = Callable[[], Iterator[Any]]                         # infinite batch iterator


@dataclass
class WorkerConfig:
    coordinator_url: str
    worker_name: str
    backend: Backend
    secret: Optional[str] = None
    steps_per_round: int = 8
    total_rounds: int = 1000


class Worker:
    def __init__(self, *,
                 coordinator_url: str,
                 worker_name: str,
                 backend: Backend,
                 secret: Optional[str] = None):
        self.backend = backend
        self.name = worker_name
        self.transport = TransportClient(coordinator_url, worker_name, secret=secret)
        log.info("worker %s up: %s", worker_name, backend.describe())

    def topology(self) -> dict:
        return self.transport._request("GET", "/v1/topology")

    def run(self, *,
            step_fn: StepFn,
            params_iter: ParamsIter,
            apply_fn: ApplyFn,
            data_iter: DataIter,
            steps_per_round: int = 8,
            total_rounds: int = 1000) -> None:
        # Sanity probe: coordinator alive.
        self.transport.healthz()
        topo = self.topology()
        my_weight = topo.get("data_weights", {}).get(self.name, 1.0)
        log.info("worker %s: data weight=%.3f, pinned experts=%s",
                 self.name, my_weight, topo.get("expert_pins", {}).get(self.name, []))

        data = data_iter()
        for round_id in range(total_rounds):
            t0 = time.time()
            losses: list[float] = []
            for _ in range(steps_per_round):
                batch = next(data)
                loss = step_fn(batch, params_iter())
                losses.append(float(loss))
            self.backend.synchronize()

            # Export local delta, push to coordinator.
            delta_blob = self.backend.export_lora_delta(params_iter())
            push_start = time.time()
            self.transport.put_delta(round_id, delta_blob)
            push_s = time.time() - push_start

            # Pull aggregated delta, apply to model.
            agg_blob = self.transport.get_aggregate(round_id)
            agg_tensors = self.backend.import_lora_delta(agg_blob)
            apply_fn(agg_tensors)

            mean_loss = sum(losses) / max(1, len(losses))
            self.transport.end_round(round_id, mean_loss, (round_id + 1) * steps_per_round)
            log.info(
                "round %d done: %d steps in %.1fs (push %.2fs, delta %.1f MB), mean_loss=%.4f",
                round_id, steps_per_round, time.time() - t0,
                push_s, len(delta_blob) / 1024**2, mean_loss,
            )
