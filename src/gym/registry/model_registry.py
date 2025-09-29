"""
Model registry for versioning, storage, and deployment.
Tracks model lineage, metrics, and deployment status.
"""

import os
import json
import shutil
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

import torch
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    VALIDATING = "validating"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Metadata for registered models."""
    model_id: str
    name: str
    version: str
    created_at: str
    updated_at: str
    status: ModelStatus
    architecture: str
    framework_version: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    tags: List[str]
    checksum: str
    size_mb: float
    parent_model_id: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        data['status'] = ModelStatus(data['status'])
        return cls(**data)


class ModelRegistry:
    """Central registry for model management."""

    def __init__(self, base_path: str = "./model_registry"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Directory structure
        self.models_dir = self.base_path / "models"
        self.metadata_dir = self.base_path / "metadata"
        self.artifacts_dir = self.base_path / "artifacts"

        for dir_path in [self.models_dir, self.metadata_dir, self.artifacts_dir]:
            dir_path.mkdir(exist_ok=True)

        # Load registry index
        self.index_path = self.base_path / "registry.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load or create registry index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {"models": {}, "aliases": {}}

    def _save_index(self):
        """Save registry index."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().isoformat()
        content = f"{name}:{version}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def register(
        self,
        model: torch.nn.Module,
        name: str,
        version: str,
        architecture: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        author: Optional[str] = None,
        parent_model_id: Optional[str] = None,
        use_safetensors: bool = True,
    ) -> str:
        """Register a new model."""
        # Generate model ID
        model_id = self._generate_model_id(name, version)

        # Create model directory
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)

        # Save model
        if use_safetensors:
            model_path = model_dir / "model.safetensors"
            state_dict = model.state_dict()
            save_file(state_dict, model_path)
        else:
            model_path = model_dir / "model.pt"
            torch.save(model.state_dict(), model_path)

        # Compute metadata
        checksum = self._compute_checksum(model_path)
        size_mb = model_path.stat().st_size / (1024 * 1024)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            status=ModelStatus.TRAINING,
            architecture=architecture,
            framework_version=torch.__version__,
            parameters={
                "total": total_params,
                "trainable": trainable_params,
                "non_trainable": total_params - trainable_params,
            },
            metrics=metrics or {},
            tags=tags or [],
            checksum=checksum,
            size_mb=size_mb,
            parent_model_id=parent_model_id,
            description=description,
            author=author,
        )

        # Save metadata
        metadata_path = self.metadata_dir / f"{model_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Update index
        self.index["models"][model_id] = {
            "name": name,
            "version": version,
            "created_at": metadata.created_at,
            "status": metadata.status.value,
        }
        self._save_index()

        logger.info(f"Registered model {name}:{version} with ID {model_id}")
        return model_id

    def load(self, model_id: str, map_location: str = "cpu") -> Tuple[Dict[str, Any], ModelMetadata]:
        """Load model from registry."""
        if model_id not in self.index["models"]:
            # Check if it's an alias
            if model_id in self.index.get("aliases", {}):
                model_id = self.index["aliases"][model_id]
            else:
                raise ValueError(f"Model {model_id} not found in registry")

        # Load metadata
        metadata_path = self.metadata_dir / f"{model_id}.json"
        with open(metadata_path, 'r') as f:
            metadata = ModelMetadata.from_dict(json.load(f))

        # Load model
        model_dir = self.models_dir / model_id
        safetensors_path = model_dir / "model.safetensors"
        pytorch_path = model_dir / "model.pt"

        if safetensors_path.exists():
            state_dict = load_file(safetensors_path, device=map_location)
        elif pytorch_path.exists():
            state_dict = torch.load(pytorch_path, map_location=map_location)
        else:
            raise FileNotFoundError(f"No model file found for {model_id}")

        return state_dict, metadata

    def update_status(self, model_id: str, status: ModelStatus):
        """Update model status."""
        if model_id not in self.index["models"]:
            raise ValueError(f"Model {model_id} not found")

        # Update metadata
        metadata_path = self.metadata_dir / f"{model_id}.json"
        with open(metadata_path, 'r') as f:
            metadata = ModelMetadata.from_dict(json.load(f))

        metadata.status = status
        metadata.updated_at = datetime.now().isoformat()

        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Update index
        self.index["models"][model_id]["status"] = status.value
        self._save_index()

        logger.info(f"Updated model {model_id} status to {status.value}")

    def update_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Update model metrics."""
        if model_id not in self.index["models"]:
            raise ValueError(f"Model {model_id} not found")

        # Update metadata
        metadata_path = self.metadata_dir / f"{model_id}.json"
        with open(metadata_path, 'r') as f:
            metadata = ModelMetadata.from_dict(json.load(f))

        metadata.metrics.update(metrics)
        metadata.updated_at = datetime.now().isoformat()

        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"Updated metrics for model {model_id}")

    def add_artifact(self, model_id: str, artifact_name: str, artifact_path: str):
        """Add artifact to model (configs, tokenizers, etc)."""
        if model_id not in self.index["models"]:
            raise ValueError(f"Model {model_id} not found")

        # Create artifact directory
        artifact_dir = self.artifacts_dir / model_id
        artifact_dir.mkdir(exist_ok=True)

        # Copy artifact
        src_path = Path(artifact_path)
        dst_path = artifact_dir / artifact_name

        if src_path.is_file():
            shutil.copy2(src_path, dst_path)
        else:
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

        logger.info(f"Added artifact {artifact_name} to model {model_id}")

    def create_alias(self, alias: str, model_id: str):
        """Create alias for model (e.g., 'latest', 'production')."""
        if model_id not in self.index["models"]:
            raise ValueError(f"Model {model_id} not found")

        if "aliases" not in self.index:
            self.index["aliases"] = {}

        self.index["aliases"][alias] = model_id
        self._save_index()

        logger.info(f"Created alias '{alias}' -> {model_id}")

    def list_models(
        self,
        name: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelMetadata]:
        """List models with optional filtering."""
        models = []

        for model_id in self.index["models"]:
            metadata_path = self.metadata_dir / f"{model_id}.json"
            with open(metadata_path, 'r') as f:
                metadata = ModelMetadata.from_dict(json.load(f))

            # Apply filters
            if name and metadata.name != name:
                continue
            if status and metadata.status != status:
                continue
            if tags and not set(tags).issubset(set(metadata.tags)):
                continue

            models.append(metadata)

        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        return models

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Get model metadata."""
        if model_id not in self.index["models"]:
            if model_id in self.index.get("aliases", {}):
                model_id = self.index["aliases"][model_id]
            else:
                raise ValueError(f"Model {model_id} not found")

        metadata_path = self.metadata_dir / f"{model_id}.json"
        with open(metadata_path, 'r') as f:
            return ModelMetadata.from_dict(json.load(f))

    def compare_models(self, model_id1: str, model_id2: str) -> Dict[str, Any]:
        """Compare two models."""
        metadata1 = self.get_metadata(model_id1)
        metadata2 = self.get_metadata(model_id2)

        comparison = {
            "model1": {
                "id": metadata1.model_id,
                "name": f"{metadata1.name}:{metadata1.version}",
                "metrics": metadata1.metrics,
                "parameters": metadata1.parameters,
                "size_mb": metadata1.size_mb,
            },
            "model2": {
                "id": metadata2.model_id,
                "name": f"{metadata2.name}:{metadata2.version}",
                "metrics": metadata2.metrics,
                "parameters": metadata2.parameters,
                "size_mb": metadata2.size_mb,
            },
            "metric_diff": {},
            "size_diff_mb": metadata2.size_mb - metadata1.size_mb,
        }

        # Compare metrics
        all_metrics = set(metadata1.metrics.keys()) | set(metadata2.metrics.keys())
        for metric in all_metrics:
            val1 = metadata1.metrics.get(metric, 0)
            val2 = metadata2.metrics.get(metric, 0)
            comparison["metric_diff"][metric] = val2 - val1

        return comparison

    def delete_model(self, model_id: str):
        """Delete model from registry."""
        if model_id not in self.index["models"]:
            raise ValueError(f"Model {model_id} not found")

        # Remove model files
        model_dir = self.models_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)

        # Remove metadata
        metadata_path = self.metadata_dir / f"{model_id}.json"
        if metadata_path.exists():
            metadata_path.unlink()

        # Remove artifacts
        artifact_dir = self.artifacts_dir / model_id
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)

        # Update index
        del self.index["models"][model_id]

        # Remove aliases pointing to this model
        aliases_to_remove = [
            alias for alias, mid in self.index.get("aliases", {}).items()
            if mid == model_id
        ]
        for alias in aliases_to_remove:
            del self.index["aliases"][alias]

        self._save_index()
        logger.info(f"Deleted model {model_id}")

    def export_model(self, model_id: str, export_path: str):
        """Export model for deployment."""
        if model_id not in self.index["models"]:
            raise ValueError(f"Model {model_id} not found")

        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)

        # Copy model files
        model_dir = self.models_dir / model_id
        shutil.copytree(model_dir, export_path / "model", dirs_exist_ok=True)

        # Copy metadata
        metadata_path = self.metadata_dir / f"{model_id}.json"
        shutil.copy2(metadata_path, export_path / "metadata.json")

        # Copy artifacts if they exist
        artifact_dir = self.artifacts_dir / model_id
        if artifact_dir.exists():
            shutil.copytree(artifact_dir, export_path / "artifacts", dirs_exist_ok=True)

        logger.info(f"Exported model {model_id} to {export_path}")


# Export main components
__all__ = ['ModelStatus', 'ModelMetadata', 'ModelRegistry']