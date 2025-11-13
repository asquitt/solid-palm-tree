"""
Mock Ray Components for Testing

These mocks simulate Ray Train and Ray Tune without requiring Ray installation.
"""

import random
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import time


@dataclass
class MockCheckpoint:
    """Mock Ray checkpoint."""
    path: str
    metrics: Dict[str, float]
    iteration: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "metrics": self.metrics,
            "iteration": self.iteration
        }


class MockRayTrainer:
    """
    Mock Ray Train trainer.

    Simulates distributed training without actually using Ray.
    """

    def __init__(self, training_function: Callable, scaling_config: Optional[Dict] = None):
        self.training_function = training_function
        self.scaling_config = scaling_config or {"num_workers": 1}
        self.results = None

    def fit(self) -> "MockResult":
        """Run training (mocked)."""
        print("Starting mock training...")

        # Simulate training epochs
        num_epochs = 3
        for epoch in range(num_epochs):
            # Simulate decreasing loss
            loss = 3.0 - (epoch * 0.5) + random.uniform(-0.1, 0.1)
            accuracy = 0.5 + (epoch * 0.15) + random.uniform(-0.05, 0.05)

            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            time.sleep(0.1)  # Simulate training time

        # Create mock result
        self.results = MockResult(
            metrics={
                "loss": loss,
                "accuracy": accuracy,
                "final_epoch": num_epochs
            },
            checkpoint=MockCheckpoint(
                path="/tmp/mock_checkpoint",
                metrics={"loss": loss, "accuracy": accuracy},
                iteration=num_epochs
            )
        )

        print("Mock training complete!")
        return self.results


class MockResult:
    """Mock training result."""

    def __init__(self, metrics: Dict[str, float], checkpoint: Optional[MockCheckpoint] = None):
        self.metrics = metrics
        self.checkpoint = checkpoint

    @property
    def best_checkpoints(self) -> List[MockCheckpoint]:
        """Get best checkpoints."""
        return [self.checkpoint] if self.checkpoint else []


class MockRayTuner:
    """
    Mock Ray Tune tuner.

    Simulates hyperparameter optimization without Ray.
    """

    def __init__(
        self,
        trainable: Callable,
        param_space: Dict[str, Any],
        tune_config: Optional[Dict] = None,
        run_config: Optional[Dict] = None
    ):
        self.trainable = trainable
        self.param_space = param_space
        self.tune_config = tune_config or {}
        self.run_config = run_config or {}

        self.results = None

    def fit(self) -> "MockResultGrid":
        """Run hyperparameter tuning (mocked)."""
        print("Starting mock hyperparameter tuning...")

        num_samples = self.tune_config.get("num_samples", 10)
        trials = []

        for trial_id in range(num_samples):
            # Sample hyperparameters
            config = self._sample_config()

            # Simulate training with these hyperparameters
            # Better hyperparameters = lower loss
            base_loss = 3.0
            lr_factor = (config.get("learning_rate", 5e-5) / 5e-5) * 0.5
            batch_factor = (config.get("batch_size", 8) / 8) * 0.2

            final_loss = base_loss - lr_factor - batch_factor + random.uniform(-0.3, 0.3)
            final_loss = max(final_loss, 1.0)  # Minimum loss

            trial = MockTrial(
                trial_id=f"trial_{trial_id}",
                config=config,
                metrics={"loss": final_loss, "accuracy": 1.0 / final_loss},
                checkpoint_path=f"/tmp/checkpoint_trial_{trial_id}"
            )
            trials.append(trial)

            print(f"Trial {trial_id + 1}/{num_samples} - Config: {config} - Loss: {final_loss:.4f}")
            time.sleep(0.05)  # Simulate tuning time

        self.results = MockResultGrid(trials)

        print("Mock hyperparameter tuning complete!")
        return self.results

    def _sample_config(self) -> Dict[str, Any]:
        """Sample hyperparameters from param space."""
        config = {}

        for param_name, param_def in self.param_space.items():
            if isinstance(param_def, dict):
                if param_def.get("type") == "uniform":
                    # Uniform sampling
                    low = param_def.get("low", 0)
                    high = param_def.get("high", 1)
                    config[param_name] = random.uniform(low, high)
                elif param_def.get("type") == "choice":
                    # Choice sampling
                    choices = param_def.get("choices", [])
                    config[param_name] = random.choice(choices)
                elif param_def.get("type") == "loguniform":
                    # Log-uniform sampling
                    low = param_def.get("low", 1e-5)
                    high = param_def.get("high", 1e-3)
                    config[param_name] = 10 ** random.uniform(
                        math.log10(low), math.log10(high)
                    )
            else:
                # Fixed value
                config[param_name] = param_def

        return config


import math


class MockTrial:
    """Mock trial result from tuning."""

    def __init__(
        self,
        trial_id: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        checkpoint_path: str
    ):
        self.trial_id = trial_id
        self.config = config
        self.metrics = metrics
        self.checkpoint_path = checkpoint_path

    def __repr__(self) -> str:
        return f"MockTrial(id={self.trial_id}, metrics={self.metrics})"


class MockResultGrid:
    """Mock result grid from tuning."""

    def __init__(self, trials: List[MockTrial]):
        self.trials = trials

    def get_best_result(self, metric: str = "loss", mode: str = "min") -> MockTrial:
        """Get best trial."""
        if not self.trials:
            return None

        if mode == "min":
            best = min(self.trials, key=lambda t: t.metrics.get(metric, float("inf")))
        else:
            best = max(self.trials, key=lambda t: t.metrics.get(metric, float("-inf")))

        return best

    def get_dataframe(self) -> List[Dict[str, Any]]:
        """Get results as dataframe-like structure."""
        return [
            {
                "trial_id": t.trial_id,
                **t.config,
                **t.metrics
            }
            for t in self.trials
        ]

    def __len__(self) -> int:
        return len(self.trials)

    def __iter__(self):
        return iter(self.trials)


# Mock decorators and context managers

class MockRayContext:
    """Mock Ray initialization context."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def mock_ray_init(*args, **kwargs):
    """Mock ray.init()."""
    print("Mock Ray initialized")
    return {"node_ip_address": "127.0.0.1", "object_store_memory": 1000000}


def mock_ray_shutdown():
    """Mock ray.shutdown()."""
    print("Mock Ray shutdown")


def mock_ray_get(object_ref):
    """Mock ray.get()."""
    # Return the object directly in mock
    return object_ref


def mock_ray_put(value):
    """Mock ray.put()."""
    # Return the value directly in mock
    return value


class MockRayRemote:
    """Mock @ray.remote decorator."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, func):
        """Wrap function."""
        def wrapper(*args, **kwargs):
            # In mock, just call the function directly
            return func(*args, **kwargs)

        wrapper.remote = lambda *args, **kwargs: func(*args, **kwargs)
        return wrapper


# Mock Ray Train utilities

def mock_report(metrics: Dict[str, float]):
    """Mock ray.train.report()."""
    print(f"Reporting metrics: {metrics}")


def mock_get_checkpoint() -> Optional[MockCheckpoint]:
    """Mock ray.train.get_checkpoint()."""
    return None


def mock_save_checkpoint(checkpoint: Dict[str, Any]):
    """Mock ray.train.save_checkpoint()."""
    print(f"Saving checkpoint: {checkpoint}")


class MockScalingConfig:
    """Mock Ray ScalingConfig."""

    def __init__(
        self,
        num_workers: int = 1,
        use_gpu: bool = False,
        resources_per_worker: Optional[Dict] = None
    ):
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.resources_per_worker = resources_per_worker or {}


class MockRunConfig:
    """Mock Ray RunConfig."""

    def __init__(
        self,
        name: Optional[str] = None,
        storage_path: Optional[str] = None,
        checkpoint_config: Optional[Dict] = None,
        stop: Optional[Dict] = None
    ):
        self.name = name
        self.storage_path = storage_path
        self.checkpoint_config = checkpoint_config
        self.stop = stop


class MockTuneConfig:
    """Mock Ray TuneConfig."""

    def __init__(
        self,
        metric: str = "loss",
        mode: str = "min",
        num_samples: int = 10,
        scheduler: Optional[Any] = None
    ):
        self.metric = metric
        self.mode = mode
        self.num_samples = num_samples
        self.scheduler = scheduler
