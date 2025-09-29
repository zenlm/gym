"""
Hyperparameter search and optimization for training.
Implements grid search, random search, and Bayesian optimization.
"""

import os
import json
import logging
import random
import itertools
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategy for hyperparameters."""
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"


@dataclass
class ParameterSpec:
    """Specification for a hyperparameter."""
    name: str
    type: str  # "float", "int", "categorical", "log"
    values: Union[List[Any], Tuple[float, float]]  # list for categorical, range for numeric
    default: Any = None

    def sample(self) -> Any:
        """Sample a value from this parameter."""
        if self.type == "categorical":
            return random.choice(self.values)
        elif self.type == "int":
            return random.randint(int(self.values[0]), int(self.values[1]))
        elif self.type == "float":
            return random.uniform(self.values[0], self.values[1])
        elif self.type == "log":
            log_min = np.log(self.values[0])
            log_max = np.log(self.values[1])
            return np.exp(random.uniform(log_min, log_max))
        else:
            raise ValueError(f"Unknown parameter type: {self.type}")

    def grid_values(self, n_points: int = 5) -> List[Any]:
        """Get grid values for this parameter."""
        if self.type == "categorical":
            return list(self.values)
        elif self.type == "int":
            start, end = int(self.values[0]), int(self.values[1])
            step = max(1, (end - start) // (n_points - 1))
            return list(range(start, end + 1, step))
        elif self.type == "float":
            return list(np.linspace(self.values[0], self.values[1], n_points))
        elif self.type == "log":
            log_min = np.log(self.values[0])
            log_max = np.log(self.values[1])
            return list(np.exp(np.linspace(log_min, log_max, n_points)))
        else:
            raise ValueError(f"Unknown parameter type: {self.type}")


@dataclass
class Trial:
    """Single trial in hyperparameter search."""
    trial_id: str
    parameters: Dict[str, Any]
    metric: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class HyperparameterSearch:
    """Hyperparameter search coordinator."""

    def __init__(
        self,
        param_specs: List[ParameterSpec],
        strategy: SearchStrategy = SearchStrategy.RANDOM,
        metric_name: str = "loss",
        mode: str = "min",  # "min" or "max"
        n_trials: int = 20,
        log_dir: str = "./hp_search",
    ):
        self.param_specs = {spec.name: spec for spec in param_specs}
        self.strategy = strategy
        self.metric_name = metric_name
        self.mode = mode
        self.n_trials = n_trials
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.trials = []
        self.best_trial = None
        self.trial_counter = 0

        # Load existing trials if resuming
        self._load_trials()

    def _load_trials(self):
        """Load existing trials from disk."""
        trials_file = self.log_dir / "trials.json"
        if trials_file.exists():
            with open(trials_file, 'r') as f:
                data = json.load(f)
                self.trials = [Trial(**trial) for trial in data["trials"]]
                self.trial_counter = len(self.trials)
                self._update_best_trial()
                logger.info(f"Loaded {len(self.trials)} existing trials")

    def _save_trials(self):
        """Save trials to disk."""
        trials_file = self.log_dir / "trials.json"
        data = {
            "trials": [trial.to_dict() for trial in self.trials],
            "best_trial_id": self.best_trial.trial_id if self.best_trial else None,
            "metric_name": self.metric_name,
            "mode": self.mode,
        }
        with open(trials_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _update_best_trial(self):
        """Update best trial based on completed trials."""
        completed_trials = [t for t in self.trials if t.status == "completed" and t.metric is not None]
        if not completed_trials:
            return

        if self.mode == "min":
            self.best_trial = min(completed_trials, key=lambda t: t.metric)
        else:
            self.best_trial = max(completed_trials, key=lambda t: t.metric)

    def _generate_grid_configs(self) -> List[Dict[str, Any]]:
        """Generate all configurations for grid search."""
        param_values = {}
        for name, spec in self.param_specs.items():
            param_values[name] = spec.grid_values()

        # Generate all combinations
        configs = []
        for values in itertools.product(*param_values.values()):
            config = dict(zip(param_values.keys(), values))
            configs.append(config)

        return configs

    def _generate_random_config(self) -> Dict[str, Any]:
        """Generate random configuration."""
        config = {}
        for name, spec in self.param_specs.items():
            config[name] = spec.sample()
        return config

    def _generate_bayesian_config(self) -> Dict[str, Any]:
        """Generate configuration using Bayesian optimization."""
        # Simple implementation: Thompson sampling with Gaussian Process approximation
        if len(self.trials) < 5:
            # Start with random samples for initial exploration
            return self._generate_random_config()

        # Use simple acquisition function (Upper Confidence Bound)
        completed_trials = [t for t in self.trials if t.status == "completed" and t.metric is not None]

        if not completed_trials:
            return self._generate_random_config()

        # For simplicity, use random with bias towards good regions
        # In production, use proper Gaussian Process or Tree-structured Parzen Estimators
        best_params = self.best_trial.parameters if self.best_trial else {}

        config = {}
        for name, spec in self.param_specs.items():
            if random.random() < 0.7 and name in best_params:  # 70% exploit
                # Sample near best value
                best_val = best_params[name]
                if spec.type == "categorical":
                    config[name] = best_val if random.random() < 0.5 else spec.sample()
                elif spec.type in ["int", "float", "log"]:
                    # Add noise to best value
                    noise_scale = 0.2  # 20% noise
                    if spec.type == "int":
                        noise = int((spec.values[1] - spec.values[0]) * noise_scale)
                        new_val = best_val + random.randint(-noise, noise)
                        new_val = max(spec.values[0], min(spec.values[1], new_val))
                        config[name] = int(new_val)
                    else:
                        noise = (spec.values[1] - spec.values[0]) * noise_scale
                        new_val = best_val + random.uniform(-noise, noise)
                        new_val = max(spec.values[0], min(spec.values[1], new_val))
                        config[name] = new_val
            else:  # 30% explore
                config[name] = spec.sample()

        return config

    def suggest_next(self) -> Optional[Trial]:
        """Suggest next hyperparameter configuration."""
        # Check if we've reached the trial limit
        if len(self.trials) >= self.n_trials:
            logger.info(f"Reached maximum trials ({self.n_trials})")
            return None

        # Generate configuration based on strategy
        if self.strategy == SearchStrategy.GRID:
            all_configs = self._generate_grid_configs()
            if self.trial_counter >= len(all_configs):
                logger.info("Grid search completed")
                return None
            config = all_configs[self.trial_counter]
        elif self.strategy == SearchStrategy.RANDOM:
            config = self._generate_random_config()
        elif self.strategy == SearchStrategy.BAYESIAN:
            config = self._generate_bayesian_config()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Create trial
        trial = Trial(
            trial_id=f"trial_{self.trial_counter:04d}",
            parameters=config,
            status="pending",
            metadata={"strategy": self.strategy.value}
        )

        self.trials.append(trial)
        self.trial_counter += 1
        self._save_trials()

        logger.info(f"Suggested trial {trial.trial_id} with params: {config}")
        return trial

    def start_trial(self, trial_id: str):
        """Mark trial as started."""
        for trial in self.trials:
            if trial.trial_id == trial_id:
                trial.status = "running"
                trial.start_time = datetime.now().isoformat()
                self._save_trials()
                logger.info(f"Started trial {trial_id}")
                return

        raise ValueError(f"Trial {trial_id} not found")

    def complete_trial(self, trial_id: str, metric: float, metadata: Optional[Dict[str, Any]] = None):
        """Mark trial as completed with metric."""
        for trial in self.trials:
            if trial.trial_id == trial_id:
                trial.status = "completed"
                trial.metric = metric
                trial.end_time = datetime.now().isoformat()
                if metadata:
                    if trial.metadata is None:
                        trial.metadata = {}
                    trial.metadata.update(metadata)

                self._update_best_trial()
                self._save_trials()

                logger.info(f"Completed trial {trial_id} with {self.metric_name}={metric:.4f}")

                if self.best_trial and self.best_trial.trial_id == trial_id:
                    logger.info(f"New best trial! {self.metric_name}={metric:.4f}")

                return

        raise ValueError(f"Trial {trial_id} not found")

    def fail_trial(self, trial_id: str, error: str):
        """Mark trial as failed."""
        for trial in self.trials:
            if trial.trial_id == trial_id:
                trial.status = "failed"
                trial.end_time = datetime.now().isoformat()
                if trial.metadata is None:
                    trial.metadata = {}
                trial.metadata["error"] = error
                self._save_trials()
                logger.warning(f"Trial {trial_id} failed: {error}")
                return

        raise ValueError(f"Trial {trial_id} not found")

    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found so far."""
        if self.best_trial:
            return self.best_trial.parameters
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get search summary."""
        completed_trials = [t for t in self.trials if t.status == "completed" and t.metric is not None]
        failed_trials = [t for t in self.trials if t.status == "failed"]

        summary = {
            "total_trials": len(self.trials),
            "completed_trials": len(completed_trials),
            "failed_trials": len(failed_trials),
            "pending_trials": len([t for t in self.trials if t.status == "pending"]),
            "running_trials": len([t for t in self.trials if t.status == "running"]),
            "strategy": self.strategy.value,
            "metric_name": self.metric_name,
            "mode": self.mode,
        }

        if completed_trials:
            metrics = [t.metric for t in completed_trials]
            summary["metrics"] = {
                "best": min(metrics) if self.mode == "min" else max(metrics),
                "worst": max(metrics) if self.mode == "min" else min(metrics),
                "mean": np.mean(metrics),
                "std": np.std(metrics),
            }

        if self.best_trial:
            summary["best_trial"] = {
                "trial_id": self.best_trial.trial_id,
                "parameters": self.best_trial.parameters,
                "metric": self.best_trial.metric,
            }

        return summary

    def plot_optimization_history(self) -> Dict[str, List[float]]:
        """Get data for plotting optimization history."""
        completed_trials = [t for t in self.trials if t.status == "completed" and t.metric is not None]

        if not completed_trials:
            return {"iterations": [], "metrics": [], "best_so_far": []}

        iterations = list(range(len(completed_trials)))
        metrics = [t.metric for t in completed_trials]

        # Compute best so far
        best_so_far = []
        current_best = metrics[0]
        for metric in metrics:
            if self.mode == "min":
                current_best = min(current_best, metric)
            else:
                current_best = max(current_best, metric)
            best_so_far.append(current_best)

        return {
            "iterations": iterations,
            "metrics": metrics,
            "best_so_far": best_so_far,
        }


class EarlyStoppingCallback:
    """Early stopping for hyperparameter search."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = float('inf') if mode == "min" else float('-inf')
        self.counter = 0

    def __call__(self, metric: float) -> bool:
        """Check if should stop."""
        improved = False
        if self.mode == "min":
            improved = metric < (self.best_metric - self.min_delta)
        else:
            improved = metric > (self.best_metric + self.min_delta)

        if improved:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1

        should_stop = self.counter >= self.patience
        if should_stop:
            logger.info(f"Early stopping triggered after {self.counter} trials without improvement")

        return should_stop


# Export main components
__all__ = [
    'SearchStrategy',
    'ParameterSpec',
    'Trial',
    'HyperparameterSearch',
    'EarlyStoppingCallback',
]