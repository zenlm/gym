"""
Distributed training coordinator for multi-GPU/multi-node training.
Implements FSDP with automatic mixed precision and failure recovery.
"""

import os
import json
import socket
import logging
import signal
import functools
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
    use_fsdp: bool = False
    fsdp_min_params: int = 1e6
    fsdp_cpu_offload: bool = False
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    find_unused_parameters: bool = False

    @classmethod
    def from_env(cls) -> "DistributedConfig":
        """Initialize from environment variables."""
        return cls(
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0)),
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            master_addr=os.environ.get("MASTER_ADDR", "localhost"),
            master_port=int(os.environ.get("MASTER_PORT", 29500)),
        )


class DistributedCoordinator:
    """Coordinates distributed training across multiple GPUs/nodes."""

    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig.from_env()
        self.is_distributed = self.config.world_size > 1
        self.is_main_process = self.config.rank == 0
        self._setup_logging()
        self._health_check_counter = 0

    def _setup_logging(self):
        """Configure logging for distributed training."""
        log_level = logging.INFO if self.is_main_process else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format=f'[Rank {self.config.rank}] %(asctime)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def initialize(self) -> bool:
        """Initialize distributed process group."""
        if not self.is_distributed:
            logger.info("Running in single-process mode")
            return True

        try:
            # Set device
            torch.cuda.set_device(self.config.local_rank)

            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                world_size=self.config.world_size,
                rank=self.config.rank
            )

            # Register signal handlers
            signal.signal(signal.SIGTERM, self._shutdown_handler)
            signal.signal(signal.SIGINT, self._shutdown_handler)

            logger.info(f"Initialized distributed training: rank={self.config.rank}/{self.config.world_size}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            return False

    def wrap_model(self, model: torch.nn.Module, **kwargs) -> torch.nn.Module:
        """Wrap model for distributed training with FSDP or DDP."""
        if not self.is_distributed:
            return model.cuda() if torch.cuda.is_available() else model

        device = torch.device(f"cuda:{self.config.local_rank}")
        model = model.to(device)

        if self.config.use_fsdp:
            return self._wrap_fsdp(model, **kwargs)
        else:
            return self._wrap_ddp(model, **kwargs)

    def _wrap_fsdp(self, model: torch.nn.Module, **kwargs) -> FSDP:
        """Wrap model with Fully Sharded Data Parallel."""
        # Configure auto wrap policy
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=self.config.fsdp_min_params
        )

        # Configure CPU offload
        cpu_offload = CPUOffload(offload_params=True) if self.config.fsdp_cpu_offload else None

        # Configure mixed precision
        mixed_precision_config = None
        if self.config.mixed_precision:
            from torch.distributed.fsdp import MixedPrecision
            mixed_precision_config = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        # Wrap with FSDP
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=cpu_offload,
            mixed_precision=mixed_precision_config,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=self.config.local_rank,
            **kwargs
        )

        logger.info("Model wrapped with FSDP")
        return model

    def _wrap_ddp(self, model: torch.nn.Module, **kwargs) -> DDP:
        """Wrap model with Distributed Data Parallel."""
        model = DDP(
            model,
            device_ids=[self.config.local_rank],
            output_device=self.config.local_rank,
            find_unused_parameters=self.config.find_unused_parameters,
            **kwargs
        )
        logger.info("Model wrapped with DDP")
        return model

    def barrier(self):
        """Synchronize all processes."""
        if self.is_distributed:
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce tensor across all processes."""
        if not self.is_distributed:
            return tensor
        dist.all_reduce(tensor, op=op)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensor from all processes."""
        if not self.is_distributed:
            return [tensor]

        world_size = self.config.world_size
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return gathered

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank."""
        if self.is_distributed:
            dist.broadcast(tensor, src=src)
        return tensor

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on distributed setup."""
        health = {
            "timestamp": datetime.now().isoformat(),
            "rank": self.config.rank,
            "world_size": self.config.world_size,
            "hostname": socket.gethostname(),
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            health.update({
                "gpu_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
            })

        # Test communication
        if self.is_distributed:
            try:
                test_tensor = torch.tensor([self.config.rank], device='cuda')
                gathered = self.all_gather(test_tensor)
                health["communication_ok"] = len(gathered) == self.config.world_size
            except Exception as e:
                health["communication_ok"] = False
                health["communication_error"] = str(e)

        self._health_check_counter += 1
        health["check_count"] = self._health_check_counter

        return health

    def save_checkpoint(self, model: torch.nn.Module, checkpoint_dir: str, **kwargs):
        """Save model checkpoint (only on main process)."""
        if not self.is_main_process:
            return

        os.makedirs(checkpoint_dir, exist_ok=True)

        if isinstance(model, FSDP):
            # Save FSDP checkpoint
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                state_dict = model.state_dict()
                if self.is_main_process:
                    torch.save({
                        'model_state_dict': state_dict,
                        'config': self.config.__dict__,
                        **kwargs
                    }, os.path.join(checkpoint_dir, 'checkpoint.pt'))
        else:
            # Save regular checkpoint
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'model_state_dict': state_dict,
                'config': self.config.__dict__,
                **kwargs
            }, os.path.join(checkpoint_dir, 'checkpoint.pt'))

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, model: torch.nn.Module, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if isinstance(model, FSDP):
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            ):
                model.load_state_dict(checkpoint['model_state_dict'])
        else:
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

    def cleanup(self):
        """Clean up distributed resources."""
        if self.is_distributed:
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")

    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.cleanup()
        exit(0)

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Export main components
__all__ = ['DistributedConfig', 'DistributedCoordinator']