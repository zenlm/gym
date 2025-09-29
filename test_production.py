#!/usr/bin/env python3
"""
Test script for production features in Gym training platform.
Demonstrates distributed training, monitoring, model registry,
hyperparameter tuning, and checkpoint management.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gym.distributed import DistributedConfig, DistributedCoordinator
from gym.monitoring import MetricsCollector, TrainingMonitor
from gym.registry import ModelRegistry, ModelStatus
from gym.tuning import (
    SearchStrategy,
    ParameterSpec,
    HyperparameterSearch,
    EarlyStoppingCallback,
)
from gym.utils.checkpoint_manager import CheckpointManager, AutoResumeTrainer


# Simple test model
class SimpleModel(nn.Module):
    """Simple feedforward model for testing."""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test_distributed_training():
    """Test distributed training coordinator."""
    print("\n=== Testing Distributed Training ===")

    # Initialize coordinator
    config = DistributedConfig(use_fsdp=False, mixed_precision=True)
    coordinator = DistributedCoordinator(config)

    # Initialize distributed training
    if coordinator.initialize():
        print(f"✓ Distributed coordinator initialized (rank={config.rank})")

    # Create and wrap model
    model = SimpleModel()
    model = coordinator.wrap_model(model)
    print("✓ Model wrapped for distributed training")

    # Health check
    health = coordinator.health_check()
    print(f"✓ Health check: {health['cuda_available']=}, {health.get('communication_ok', True)=}")

    # Cleanup
    coordinator.cleanup()
    print("✓ Distributed resources cleaned up")


def test_metrics_monitoring():
    """Test metrics collection and monitoring."""
    print("\n=== Testing Metrics & Monitoring ===")

    # Initialize metrics collector
    metrics = MetricsCollector(
        log_dir="./test_logs",
        backends=["console", "json"],
        log_interval=2,
    )

    # Create training monitor
    monitor = TrainingMonitor(metrics)

    # Simulate training loop
    monitor.on_epoch_start(epoch=0)

    for batch in range(5):
        # Simulate batch training
        loss = 1.0 - (batch * 0.1)  # Decreasing loss
        monitor.on_batch_end(loss=loss, batch_size=32, learning_rate=0.001)

    # Simulate validation
    val_metrics = {"loss": 0.5, "accuracy": 0.85}
    is_best = monitor.on_validation(val_metrics)
    print(f"✓ Validation metrics logged (is_best={is_best})")

    # Log resources
    metrics.log_resources()
    print("✓ Resource metrics collected")

    # Get summary
    summary = metrics.get_summary()
    print(f"✓ Metrics summary: {summary['total_steps']} steps logged")

    metrics.close()


def test_model_registry():
    """Test model registry functionality."""
    print("\n=== Testing Model Registry ===")

    # Initialize registry
    registry = ModelRegistry(base_path="./test_registry")

    # Create and register model
    model = SimpleModel()
    model_id = registry.register(
        model=model,
        name="simple_model",
        version="1.0.0",
        architecture="feedforward",
        metrics={"loss": 0.5, "accuracy": 0.85},
        tags=["test", "production"],
        description="Test model for production features",
        author="gym_test",
    )
    print(f"✓ Model registered with ID: {model_id}")

    # Update model status
    registry.update_status(model_id, ModelStatus.STAGING)
    print("✓ Model status updated to STAGING")

    # Create alias
    registry.create_alias("latest", model_id)
    print("✓ Created alias 'latest' for model")

    # Load model
    state_dict, metadata = registry.load("latest")
    print(f"✓ Model loaded: {metadata.name}:{metadata.version}")

    # List models
    models = registry.list_models()
    print(f"✓ Found {len(models)} models in registry")

    # Compare models (simulate second model)
    model2 = SimpleModel(hidden_size=256)
    model_id2 = registry.register(
        model=model2,
        name="simple_model",
        version="2.0.0",
        architecture="feedforward",
        metrics={"loss": 0.4, "accuracy": 0.90},
        parent_model_id=model_id,
    )

    comparison = registry.compare_models(model_id, model_id2)
    print(f"✓ Model comparison: accuracy improved by {comparison['metric_diff']['accuracy']:.2f}")


def test_hyperparameter_search():
    """Test hyperparameter tuning."""
    print("\n=== Testing Hyperparameter Search ===")

    # Define parameter search space
    param_specs = [
        ParameterSpec("learning_rate", "log", (1e-4, 1e-2)),
        ParameterSpec("batch_size", "categorical", [16, 32, 64, 128]),
        ParameterSpec("hidden_size", "int", (64, 512)),
        ParameterSpec("dropout", "float", (0.0, 0.5)),
    ]

    # Initialize search
    search = HyperparameterSearch(
        param_specs=param_specs,
        strategy=SearchStrategy.RANDOM,
        metric_name="validation_loss",
        mode="min",
        n_trials=5,
        log_dir="./test_hp_search",
    )

    # Run trials
    early_stopping = EarlyStoppingCallback(patience=3)

    for i in range(5):
        # Get next trial
        trial = search.suggest_next()
        if trial is None:
            break

        print(f"✓ Trial {trial.trial_id}: {trial.parameters}")

        # Start trial
        search.start_trial(trial.trial_id)

        # Simulate training with these hyperparameters
        # In real scenario, you would train the model here
        simulated_loss = 1.0 - (i * 0.15)  # Improving loss

        # Complete trial
        search.complete_trial(
            trial.trial_id,
            metric=simulated_loss,
            metadata={"epochs_trained": 10}
        )

        # Check early stopping
        if early_stopping(simulated_loss):
            print("✓ Early stopping triggered")
            break

    # Get best parameters
    best_params = search.get_best_parameters()
    print(f"✓ Best parameters found: {best_params}")

    # Get summary
    summary = search.get_summary()
    print(f"✓ Search summary: {summary['completed_trials']} trials completed")


def test_checkpoint_manager():
    """Test checkpoint management and auto-resume."""
    print("\n=== Testing Checkpoint Manager ===")

    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(
        checkpoint_dir="./test_checkpoints",
        max_checkpoints=3,
        save_best=True,
        save_latest=True,
    )

    # Create model and optimizer
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Save checkpoints
    for epoch in range(3):
        step = epoch * 100
        loss = 1.0 - (epoch * 0.2)
        is_best = epoch == 2  # Last epoch is best

        checkpoint_id = ckpt_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            loss=loss,
            metrics={"accuracy": 0.8 + epoch * 0.05},
            is_best=is_best,
        )
        print(f"✓ Saved checkpoint: {checkpoint_id}")

    # List checkpoints
    checkpoints = ckpt_manager.list_checkpoints()
    print(f"✓ Found {len(checkpoints)} checkpoints")

    # Test auto-resume
    auto_resume = AutoResumeTrainer(ckpt_manager)

    if auto_resume.should_resume():
        new_model = SimpleModel()
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)

        training_state = auto_resume.resume(new_model, new_optimizer)
        print(f"✓ Auto-resumed from epoch {training_state['epoch']}, step {training_state['step']}")

    # Load best checkpoint
    best_model = SimpleModel()
    best_state = ckpt_manager.load(best_model, checkpoint_id="best")
    print(f"✓ Loaded best checkpoint: loss={best_state['loss']:.3f}")

    # Cleanup old checkpoints
    ckpt_manager.cleanup(keep_best=True, keep_latest=True)
    print("✓ Cleaned up old checkpoints")


def test_integration():
    """Test integration of all components."""
    print("\n=== Testing Full Integration ===")

    # Setup components
    coordinator = DistributedCoordinator(DistributedConfig())
    metrics = MetricsCollector(log_dir="./test_integration/logs")
    registry = ModelRegistry(base_path="./test_integration/registry")
    ckpt_manager = CheckpointManager(checkpoint_dir="./test_integration/checkpoints")
    monitor = TrainingMonitor(metrics)

    # Initialize
    coordinator.initialize()

    # Create and wrap model
    model = SimpleModel()
    model = coordinator.wrap_model(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop simulation
    print("✓ Starting integrated training simulation...")

    for epoch in range(2):
        monitor.on_epoch_start(epoch)

        # Simulate batches
        for batch in range(3):
            loss = 1.0 - (epoch * 0.2 + batch * 0.05)
            monitor.on_batch_end(loss=loss, batch_size=32, learning_rate=0.001)

        # Validation
        val_loss = 0.8 - (epoch * 0.1)
        val_metrics = {"loss": val_loss, "accuracy": 0.7 + epoch * 0.1}
        is_best = monitor.on_validation(val_metrics)

        # Save checkpoint
        step = epoch * 100 + batch
        ckpt_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            loss=val_loss,
            metrics=val_metrics,
            is_best=is_best,
        )

        monitor.on_epoch_end()

    # Register final model
    model_id = registry.register(
        model=model,
        name="integrated_model",
        version="1.0.0",
        architecture="feedforward",
        metrics=val_metrics,
        description="Model from integrated test",
    )

    registry.update_status(model_id, ModelStatus.PRODUCTION)
    print(f"✓ Model registered and promoted to production: {model_id}")

    # Cleanup
    coordinator.cleanup()
    metrics.close()
    print("✓ Integration test completed successfully")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Gym Production Features")
    print("=" * 50)

    try:
        test_distributed_training()
        test_metrics_monitoring()
        test_model_registry()
        test_hyperparameter_search()
        test_checkpoint_manager()
        test_integration()

        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()