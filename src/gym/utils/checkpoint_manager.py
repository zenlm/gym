"""
Checkpoint management with automatic recovery and versioning.
Handles model state, optimizer state, and training progress.
"""

import os
import json
import shutil
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

import torch
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    checkpoint_id: str
    epoch: int
    step: int
    loss: float
    timestamp: str
    is_best: bool = False
    metrics: Dict[str, float] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class CheckpointManager:
    """Manages checkpoints with automatic recovery."""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 5,
        save_best: bool = True,
        save_latest: bool = True,
        use_safetensors: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.save_latest = save_latest
        self.use_safetensors = use_safetensors

        self.checkpoints = []
        self.best_checkpoint = None
        self.latest_checkpoint = None

        # Load checkpoint index
        self._load_index()

    def _load_index(self):
        """Load checkpoint index."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                data = json.load(f)
                self.checkpoints = [CheckpointInfo(**cp) for cp in data.get("checkpoints", [])]
                if data.get("best_checkpoint"):
                    self.best_checkpoint = CheckpointInfo(**data["best_checkpoint"])
                if data.get("latest_checkpoint"):
                    self.latest_checkpoint = CheckpointInfo(**data["latest_checkpoint"])

    def _save_index(self):
        """Save checkpoint index."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        data = {
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
            "best_checkpoint": self.best_checkpoint.to_dict() if self.best_checkpoint else None,
            "latest_checkpoint": self.latest_checkpoint.to_dict() if self.latest_checkpoint else None,
        }
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> str:
        """Save checkpoint."""
        # Generate checkpoint ID
        checkpoint_id = f"checkpoint_e{epoch:04d}_s{step:08d}"
        checkpoint_path = self.checkpoint_dir / checkpoint_id

        # Create checkpoint directory
        checkpoint_path.mkdir(exist_ok=True)

        # Save model state
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        if self.use_safetensors:
            save_file(model_state, checkpoint_path / "model.safetensors")
        else:
            torch.save(model_state, checkpoint_path / "model.pt")

        # Save optimizer state
        torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")

        # Save training state
        training_state = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "metrics": metrics or {},
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        with open(checkpoint_path / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)

        # Create checkpoint info
        checkpoint_info = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            epoch=epoch,
            step=step,
            loss=loss,
            timestamp=training_state["timestamp"],
            is_best=is_best,
            metrics=metrics,
            metadata=metadata,
        )

        # Update checkpoints list
        self.checkpoints.append(checkpoint_info)
        self.checkpoints.sort(key=lambda x: x.step, reverse=True)

        # Manage checkpoint limits
        if len(self.checkpoints) > self.max_checkpoints:
            # Keep best and latest, remove oldest others
            to_keep = []
            if self.best_checkpoint:
                to_keep.append(self.best_checkpoint.checkpoint_id)
            if self.latest_checkpoint:
                to_keep.append(self.latest_checkpoint.checkpoint_id)

            for cp in self.checkpoints[self.max_checkpoints:]:
                if cp.checkpoint_id not in to_keep:
                    self._delete_checkpoint(cp.checkpoint_id)

            self.checkpoints = self.checkpoints[:self.max_checkpoints]

        # Update best checkpoint
        if is_best and self.save_best:
            self.best_checkpoint = checkpoint_info
            # Create symlink to best
            best_link = self.checkpoint_dir / "checkpoint_best"
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(checkpoint_path.name)

        # Update latest checkpoint
        if self.save_latest:
            self.latest_checkpoint = checkpoint_info
            # Create symlink to latest
            latest_link = self.checkpoint_dir / "checkpoint_latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(checkpoint_path.name)

        # Save index
        self._save_index()

        logger.info(f"Saved checkpoint {checkpoint_id} (best={is_best})")
        return checkpoint_id

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_id: Optional[str] = None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """Load checkpoint."""
        # Determine which checkpoint to load
        if checkpoint_id is None:
            if self.latest_checkpoint:
                checkpoint_id = self.latest_checkpoint.checkpoint_id
            else:
                raise ValueError("No checkpoints available")

        # Handle special checkpoint names
        if checkpoint_id == "best":
            if self.best_checkpoint:
                checkpoint_id = self.best_checkpoint.checkpoint_id
            else:
                raise ValueError("No best checkpoint available")
        elif checkpoint_id == "latest":
            if self.latest_checkpoint:
                checkpoint_id = self.latest_checkpoint.checkpoint_id
            else:
                raise ValueError("No latest checkpoint available")

        checkpoint_path = self.checkpoint_dir / checkpoint_id

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")

        # Load model state
        if self.use_safetensors:
            model_state = load_file(checkpoint_path / "model.safetensors", device=map_location)
        else:
            model_state = torch.load(checkpoint_path / "model.pt", map_location=map_location)

        if hasattr(model, 'module'):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)

        # Load optimizer state if provided
        if optimizer is not None:
            optimizer_state = torch.load(checkpoint_path / "optimizer.pt", map_location=map_location)
            optimizer.load_state_dict(optimizer_state)

        # Load training state
        with open(checkpoint_path / "training_state.json", 'r') as f:
            training_state = json.load(f)

        logger.info(f"Loaded checkpoint {checkpoint_id} (epoch={training_state['epoch']}, step={training_state['step']})")

        return training_state

    def exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists."""
        if checkpoint_id == "best":
            return self.best_checkpoint is not None
        elif checkpoint_id == "latest":
            return self.latest_checkpoint is not None
        else:
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            return checkpoint_path.exists()

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all available checkpoints."""
        return self.checkpoints

    def get_best_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get best checkpoint info."""
        return self.best_checkpoint

    def get_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get latest checkpoint info."""
        return self.latest_checkpoint

    def _delete_checkpoint(self, checkpoint_id: str):
        """Delete a checkpoint."""
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
            logger.info(f"Deleted checkpoint {checkpoint_id}")

    def cleanup(self, keep_best: bool = True, keep_latest: bool = True):
        """Clean up old checkpoints."""
        to_keep = []
        if keep_best and self.best_checkpoint:
            to_keep.append(self.best_checkpoint.checkpoint_id)
        if keep_latest and self.latest_checkpoint:
            to_keep.append(self.latest_checkpoint.checkpoint_id)

        for cp in self.checkpoints:
            if cp.checkpoint_id not in to_keep:
                self._delete_checkpoint(cp.checkpoint_id)

        self.checkpoints = [cp for cp in self.checkpoints if cp.checkpoint_id in to_keep]
        self._save_index()

        logger.info(f"Cleaned up checkpoints, kept {len(self.checkpoints)} checkpoints")


class AutoResumeTrainer:
    """Helper for automatic training resumption."""

    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.start_epoch = 0
        self.start_step = 0

    def should_resume(self) -> bool:
        """Check if training should resume from checkpoint."""
        return self.checkpoint_manager.get_latest_checkpoint() is not None

    def resume(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict[str, Any]:
        """Resume training from latest checkpoint."""
        if not self.should_resume():
            logger.info("No checkpoint to resume from, starting fresh")
            return {}

        # Load latest checkpoint
        training_state = self.checkpoint_manager.load(model, optimizer, "latest")

        self.start_epoch = training_state["epoch"] + 1
        self.start_step = training_state["step"]

        # Restore scheduler state if provided
        if scheduler is not None and "scheduler_state" in training_state.get("metadata", {}):
            scheduler.load_state_dict(training_state["metadata"]["scheduler_state"])

        logger.info(f"Resumed from epoch {training_state['epoch']}, step {training_state['step']}")

        return training_state

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        is_best: bool = False,
    ) -> str:
        """Save checkpoint with scheduler state."""
        metadata = {}
        if scheduler is not None:
            metadata["scheduler_state"] = scheduler.state_dict()

        return self.checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            loss=loss,
            metrics=metrics,
            metadata=metadata,
            is_best=is_best,
        )


# Export main components
__all__ = [
    'CheckpointInfo',
    'CheckpointManager',
    'AutoResumeTrainer',
]