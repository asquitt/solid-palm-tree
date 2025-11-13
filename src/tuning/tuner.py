"""
Hyperparameter Tuner using Ray Tune

This module implements distributed hyperparameter optimization for LLM
fine-tuning using Ray Tune.

Key Concepts:
- Search Space: Range of hyperparameters to explore
- Search Algorithm: How to sample from search space (random, bayesian, etc.)
- Scheduler: When to stop/promote trials (ASHA, PBT, etc.)
- Trial: Single training run with specific hyperparameters

Popular Algorithms:
1. Random Search: Simple but effective baseline
2. Grid Search: Exhaustive but expensive
3. Bayesian Optimization: Smart sampling based on past results
4. ASHA: Aggressive early stopping of bad trials
5. Population Based Training (PBT): Evolutionary approach

Cost Optimization:
- Early stopping saves 60-90% of compute
- Parallel trials maximize GPU utilization
- Smart sampling explores fewer bad configurations
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import (
    ASHAScheduler,
    PopulationBasedTraining,
    FIFOScheduler,
)
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
import mlflow

from src.training.config import TrainingConfig
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """
    Configuration for hyperparameter tuning.

    This config specifies how the hyperparameter search should be conducted,
    including the search algorithm, scheduler, and resource allocation.

    Args:
        num_samples: Number of trials to run (random/bayesian search)
        max_concurrent_trials: Maximum trials to run in parallel
        grace_period: Minimum epochs before early stopping
        reduction_factor: Fraction of trials to keep at each rung (ASHA)
        search_algorithm: Which search algorithm to use
        scheduler: Which scheduler to use for early stopping
        metric: Metric to optimize (e.g., "loss", "perplexity")
        mode: Whether to minimize or maximize the metric
        resources_per_trial: GPU/CPU resources per trial
    """

    # Search configuration
    num_samples: int = 20
    """Number of hyperparameter configurations to try."""

    max_concurrent_trials: int = 4
    """Maximum number of trials to run in parallel."""

    search_algorithm: str = "optuna"
    """
    Search algorithm: "random", "grid", "optuna", "bayesopt", "hyperopt"

    Recommendations:
    - random: Good baseline, no dependencies
    - optuna: Best for < 100 trials (recommended)
    - bayesopt: Good for continuous parameters
    - hyperopt: Alternative to optuna
    - grid: Only for small search spaces
    """

    scheduler: str = "asha"
    """
    Trial scheduler: "asha", "pbt", "fifo"

    ASHA (Asynchronous Successive Halving):
    - Aggressively stops bad trials early
    - 60-90% compute savings
    - Best for most use cases

    PBT (Population Based Training):
    - Evolves hyperparameters during training
    - Good for learning rate schedules
    - Requires more trials (>10)

    FIFO (First In First Out):
    - No early stopping
    - Run all trials to completion
    - Use only if stopping is problematic
    """

    # Scheduler parameters
    grace_period: int = 1
    """Minimum epochs before early stopping can occur."""

    reduction_factor: int = 2
    """
    Fraction of trials to keep at each rung (ASHA).
    reduction_factor=2 means keep best 50% of trials.
    Higher = more aggressive stopping.
    """

    # Optimization objective
    metric: str = "loss"
    """Metric to optimize. Should be logged during training."""

    mode: str = "min"
    """Whether to minimize ("min") or maximize ("max") the metric."""

    # Resource allocation
    resources_per_trial: Dict[str, float] = field(default_factory=lambda: {
        "cpu": 4,
        "gpu": 1,
    })
    """Resources to allocate per trial (CPU cores, GPU count)."""

    # Checkpointing
    checkpoint_frequency: int = 1
    """Save checkpoint every N epochs."""

    checkpoint_at_end: bool = True
    """Save checkpoint at end of trial."""

    # Logging
    verbose: int = 2
    """
    Verbosity level:
    0 = silent
    1 = minimal (final results only)
    2 = progress bar
    3 = detailed logging
    """

    # Storage
    storage_path: Optional[str] = None
    """Path to store trial results and checkpoints."""

    resume: bool = False
    """Whether to resume interrupted tuning run."""

    # MLflow integration
    use_mlflow: bool = True
    """Log trials to MLflow."""

    mlflow_tracking_uri: str = "http://localhost:5000"
    """MLflow tracking server URI."""

    mlflow_experiment_name: str = "llm-hyperparameter-tuning"
    """MLflow experiment name."""


class HyperparameterTuner:
    """
    Distributed hyperparameter tuner for LLM fine-tuning.

    This class orchestrates hyperparameter optimization using Ray Tune,
    managing trial execution, early stopping, and result tracking.

    Usage:
        >>> from src.tuning.search_spaces import get_search_space
        >>>
        >>> # Define base config
        >>> base_config = TrainingConfig(model_name="gpt2")
        >>>
        >>> # Define search space
        >>> search_space = get_search_space("default")
        >>>
        >>> # Create tuner
        >>> tuning_config = TuningConfig(num_samples=20)
        >>> tuner = HyperparameterTuner(base_config, search_space, tuning_config)
        >>>
        >>> # Run tuning
        >>> results = tuner.tune()
        >>>
        >>> # Get best config
        >>> best_config = results.get_best_result().config
    """

    def __init__(
        self,
        base_config: TrainingConfig,
        search_space: Dict[str, Any],
        tuning_config: TuningConfig,
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            base_config: Base training configuration
            search_space: Dictionary defining hyperparameter search space
            tuning_config: Tuning-specific configuration
        """
        self.base_config = base_config
        self.search_space = search_space
        self.tuning_config = tuning_config

        logger.info(f"HyperparameterTuner initialized")
        logger.info(f"Search space: {list(search_space.keys())}")
        logger.info(f"Search algorithm: {tuning_config.search_algorithm}")
        logger.info(f"Scheduler: {tuning_config.scheduler}")

    def _create_search_algorithm(self):
        """
        Create search algorithm based on config.

        Search Algorithms:
        1. Random: Samples randomly from search space
        2. Optuna (TPE): Tree-structured Parzen Estimator
           - Builds probabilistic model of good/bad configs
           - Samples more from promising regions
        3. BayesOpt (GP): Gaussian Process optimization
           - Models metric as function of hyperparameters
           - Balances exploration vs exploitation
        4. HyperOpt: Similar to Optuna but different implementation
        """
        algorithm = self.tuning_config.search_algorithm.lower()

        if algorithm == "random":
            search_alg = None  # Ray Tune default is random

        elif algorithm == "optuna":
            search_alg = OptunaSearch(
                metric=self.tuning_config.metric,
                mode=self.tuning_config.mode,
            )

        elif algorithm == "bayesopt":
            search_alg = BayesOptSearch(
                metric=self.tuning_config.metric,
                mode=self.tuning_config.mode,
            )

        elif algorithm == "hyperopt":
            search_alg = HyperOptSearch(
                metric=self.tuning_config.metric,
                mode=self.tuning_config.mode,
            )

        else:
            raise ValueError(f"Unknown search algorithm: {algorithm}")

        # Limit concurrent trials to avoid overwhelming resources
        if search_alg is not None:
            search_alg = ConcurrencyLimiter(
                search_alg,
                max_concurrent=self.tuning_config.max_concurrent_trials,
            )

        return search_alg

    def _create_scheduler(self):
        """
        Create trial scheduler for early stopping.

        Schedulers decide when to:
        - Stop unpromising trials early (save compute)
        - Promote promising trials (allocate more resources)
        - Perturb hyperparameters (PBT only)

        ASHA (Asynchronous Successive Halving):
        - Runs all trials for grace_period epochs
        - Then periodically stops worst-performing trials
        - At each "rung", keeps top 1/reduction_factor trials
        - Example: reduction_factor=2 keeps top 50%

        Rung Schedule Example (grace_period=1, reduction_factor=2):
        - Rung 1 (epoch 1): All 20 trials
        - Rung 2 (epoch 2): Best 10 trials
        - Rung 3 (epoch 4): Best 5 trials
        - Rung 4 (epoch 8): Best 2-3 trials
        - Final: Best 1 trial
        """
        scheduler_name = self.tuning_config.scheduler.lower()

        if scheduler_name == "asha":
            scheduler = ASHAScheduler(
                time_attr="training_iteration",  # Usually = epoch
                metric=self.tuning_config.metric,
                mode=self.tuning_config.mode,
                max_t=self.base_config.num_epochs,
                grace_period=self.tuning_config.grace_period,
                reduction_factor=self.tuning_config.reduction_factor,
            )

        elif scheduler_name == "pbt":
            # Population Based Training
            # Explores hyperparameters during training by:
            # 1. Periodically copying weights from best trials
            # 2. Perturbing hyperparameters of copied trials
            scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                metric=self.tuning_config.metric,
                mode=self.tuning_config.mode,
                perturbation_interval=self.tuning_config.grace_period,
                hyperparam_mutations=self.search_space,
            )

        elif scheduler_name == "fifo":
            # No early stopping
            scheduler = FIFOScheduler()

        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        return scheduler

    def _training_function(self, config: Dict[str, Any]):
        """
        Training function executed for each trial.

        This function:
        1. Merges trial config with base config
        2. Creates a Trainer
        3. Runs training loop
        4. Reports metrics to Ray Tune for optimization

        Args:
            config: Hyperparameters for this trial (from search space)
        """
        # Merge trial config with base config
        trial_config = TrainingConfig.from_dict({
            **self.base_config.to_dict(),
            **config,
        })

        # Create trainer
        trainer = Trainer(trial_config)

        # Load model and data
        trainer.load_model_and_tokenizer()
        trainer.prepare_datasets()
        trainer.setup_optimizer_and_scheduler()

        # Initialize MLflow logging if enabled
        if self.tuning_config.use_mlflow:
            mlflow.set_tracking_uri(self.tuning_config.mlflow_tracking_uri)
            mlflow.set_experiment(self.tuning_config.mlflow_experiment_name)
            mlflow.start_run(
                run_name=f"trial-{tune.get_trial_id()}",
                nested=True,  # Nested under parent tuning run
            )
            mlflow.log_params(config)

        # Training loop with periodic reporting
        trainer.model.train()

        for epoch in range(trial_config.num_epochs):
            # Train for one epoch
            epoch_loss = 0.0
            num_steps = 0

            for step, batch in enumerate(trainer.train_dataloader):
                batch = {k: v.to(trainer.device) for k, v in batch.items()}

                # Forward pass
                outputs = trainer.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"],
                )
                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Optimizer step
                if (step + 1) % trial_config.gradient_accumulation_steps == 0:
                    trainer.optimizer.step()
                    trainer.scheduler.step()
                    trainer.optimizer.zero_grad()

                epoch_loss += loss.item()
                num_steps += 1

            avg_loss = epoch_loss / num_steps if num_steps > 0 else 0

            # Evaluate
            eval_metrics = trainer.evaluate()

            # Report metrics to Ray Tune
            # This allows the scheduler to make decisions
            metrics = {
                "loss": eval_metrics["loss"],
                "perplexity": eval_metrics["perplexity"],
                "training_iteration": epoch + 1,
            }

            # Report to Ray Tune
            tune.report(**metrics)

            # Log to MLflow
            if self.tuning_config.use_mlflow:
                mlflow.log_metrics(metrics, step=epoch)

        # End MLflow run
        if self.tuning_config.use_mlflow:
            mlflow.end_run()

    def tune(self):
        """
        Run hyperparameter tuning.

        This method:
        1. Initializes Ray
        2. Creates search algorithm and scheduler
        3. Launches parallel trials
        4. Tracks results
        5. Returns best configuration

        Returns:
            Ray Tune ResultGrid with all trial results
        """
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init()

        # Create search algorithm and scheduler
        search_alg = self._create_search_algorithm()
        scheduler = self._create_scheduler()

        # Configure progress reporter
        reporter = CLIReporter(
            metric_columns=[
                self.tuning_config.metric,
                "perplexity",
                "training_iteration",
            ],
            max_report_frequency=30,  # Update every 30 seconds
        )

        logger.info("=" * 80)
        logger.info("Starting hyperparameter tuning")
        logger.info("=" * 80)
        logger.info(f"Number of trials: {self.tuning_config.num_samples}")
        logger.info(f"Concurrent trials: {self.tuning_config.max_concurrent_trials}")
        logger.info(f"Search space: {self.search_space}")

        # Start parent MLflow run for tuning
        if self.tuning_config.use_mlflow:
            mlflow.set_tracking_uri(self.tuning_config.mlflow_tracking_uri)
            mlflow.set_experiment(self.tuning_config.mlflow_experiment_name)
            mlflow.start_run(run_name="hyperparameter-tuning")

        # Run tuning
        tuner = tune.Tuner(
            tune.with_resources(
                self._training_function,
                resources=self.tuning_config.resources_per_trial,
            ),
            param_space=self.search_space,
            tune_config=tune.TuneConfig(
                metric=self.tuning_config.metric,
                mode=self.tuning_config.mode,
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=self.tuning_config.num_samples,
            ),
            run_config=ray.train.RunConfig(
                name="llm-tuning",
                storage_path=self.tuning_config.storage_path,
                progress_reporter=reporter,
                verbose=self.tuning_config.verbose,
                checkpoint_config=ray.train.CheckpointConfig(
                    checkpoint_frequency=self.tuning_config.checkpoint_frequency,
                    checkpoint_at_end=self.tuning_config.checkpoint_at_end,
                ),
            ),
        )

        results = tuner.fit()

        # Get best result
        best_result = results.get_best_result()

        logger.info("=" * 80)
        logger.info("Hyperparameter tuning completed!")
        logger.info("=" * 80)
        logger.info(f"Best trial: {best_result.config}")
        logger.info(f"Best {self.tuning_config.metric}: {best_result.metrics[self.tuning_config.metric]:.4f}")

        # Log best config to MLflow
        if self.tuning_config.use_mlflow:
            mlflow.log_params({f"best_{k}": v for k, v in best_result.config.items()})
            mlflow.log_metric(f"best_{self.tuning_config.metric}", best_result.metrics[self.tuning_config.metric])
            mlflow.end_run()

        return results

    def get_best_config(self, results) -> TrainingConfig:
        """
        Extract best training configuration from tuning results.

        Args:
            results: Ray Tune ResultGrid from tune()

        Returns:
            TrainingConfig with best hyperparameters
        """
        best_result = results.get_best_result()
        best_config = TrainingConfig.from_dict({
            **self.base_config.to_dict(),
            **best_result.config,
        })

        return best_config
