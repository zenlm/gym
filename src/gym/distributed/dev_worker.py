"""Dev-mode federation worker.

Exists as a separate concern from the CLI dispatcher. Trains a tiny synthetic
state with random noise so the transport + scheduler + aggregator are exercised
end-to-end without GPU cost.

State lives in numpy — the only universal type. Backend.wrap/unwrap cross the
device boundary, so this module never branches on `backend.name`.

For real training, write your own loop and pass it to `Worker.run` directly;
this file is a reference implementation, not a fixed dependency.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from ..backend import Backend, detect
from .topology import Lab
from .worker import Worker

log = logging.getLogger(__name__)


def run(*,
        lab: Lab,
        name: str,
        coordinator_url: str,
        total_rounds: int = 10,
        seed: Optional[int] = None) -> None:
    """Run a dev-mode worker against the given coordinator.

    All side effects bounded: HTTP push + pull, in-memory state, no disk.
    """
    node = lab.find(name)
    backend = detect(prefer=node.backend_hint)
    secret = os.environ.get(f"ZEN_LAB_SECRET_{name.upper()}")

    rng = np.random.default_rng(seed if seed is not None else abs(hash(name)) % (2**31))
    state: dict[str, np.ndarray] = {
        "A": rng.standard_normal((8, 64), dtype=np.float32),
        "B": rng.standard_normal((64, 8), dtype=np.float32),
    }

    def step_fn(_batch, _params):
        for k in state:
            state[k] += rng.standard_normal(state[k].shape, dtype=np.float32) * 0.001
        return 0.5

    def params_iter():
        # numpy → backend native at the boundary, nowhere else.
        return [(k, backend.wrap(v)) for k, v in state.items()]

    def apply_fn(delta):
        # backend native → numpy at the boundary, nowhere else.
        for k, v in delta.items():
            state[k] = backend.unwrap(v)

    def data_iter():
        while True:
            yield None

    worker = Worker(coordinator_url=coordinator_url, worker_name=name,
                    backend=backend, secret=secret)
    worker.run(
        step_fn=step_fn, params_iter=params_iter,
        apply_fn=apply_fn, data_iter=data_iter,
        steps_per_round=lab.sync_interval_steps,
        total_rounds=total_rounds,
    )
