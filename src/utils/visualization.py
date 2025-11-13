"""
Visualization Utilities

Tools for visualizing training progress, model comparisons, and results.

Features:
- Training history plots (loss, learning rate, metrics)
- Model comparison charts
- Resource utilization graphs
- Cost analysis visualizations
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class TrainingVisualizer:
    """
    Visualizer for training metrics and results.

    This class creates comprehensive visualizations for:
    - Training and validation loss curves
    - Learning rate schedules
    - Performance metrics over time
    - Model comparisons
    - Resource utilization

    Usage:
        >>> visualizer = TrainingVisualizer()
        >>> visualizer.plot_training_history(history)
        >>> visualizer.save_figure("training_plot.png")
    """

    def __init__(self, style: str = "seaborn"):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style ('seaborn', 'ggplot', 'bmh')
        """
        if style:
            try:
                plt.style.use(style)
            except:
                pass

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[str] = None,
    ):
        """
        Plot training history with loss and metrics.

        Args:
            history: Dictionary with metrics over time
                     {
                         "train_loss": [1.2, 1.0, 0.9, ...],
                         "val_loss": [1.3, 1.1, 1.0, ...],
                         "learning_rate": [5e-5, 4.8e-5, ...],
                     }
            title: Plot title
            save_path: Optional path to save figure
        """
        num_plots = len(history)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

        if num_plots == 1:
            axes = [axes]

        for idx, (metric_name, values) in enumerate(history.items()):
            ax = axes[idx]
            steps = list(range(len(values)))

            ax.plot(steps, values, linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('Step', fontsize=12)
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f"{metric_name.replace('_', ' ').title()} over Time", fontsize=14)
            ax.grid(True, alpha=0.3)

            # Add trend line
            if len(values) > 3:
                z = np.polyfit(steps, values, 2)
                p = np.poly1d(z)
                ax.plot(steps, p(steps), "--", alpha=0.5, color='red', label='Trend')
                ax.legend()

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        return fig

    def plot_loss_comparison(
        self,
        losses: Dict[str, List[float]],
        title: str = "Loss Comparison",
        save_path: Optional[str] = None,
    ):
        """
        Compare loss curves for different models or configurations.

        Args:
            losses: Dictionary mapping names to loss values
                    {
                        "Baseline": [1.5, 1.3, 1.2, ...],
                        "Fine-tuned": [1.2, 1.0, 0.9, ...],
                    }
            title: Plot title
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(12, 6))

        for name, values in losses.items():
            steps = list(range(len(values)))
            plt.plot(steps, values, label=name, linewidth=2, marker='o', markersize=4)

        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.tight_layout()
        return plt.gcf()

    def plot_model_comparison(
        self,
        benchmark_results: List[Dict[str, Any]],
        metrics: List[str] = ["perplexity", "inference_tokens_per_second"],
        save_path: Optional[str] = None,
    ):
        """
        Create comparison bar charts for model benchmarks.

        Args:
            benchmark_results: List of benchmark result dictionaries
            metrics: Metrics to plot
            save_path: Optional path to save figure
        """
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6))

        if num_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            names = [r['model_name'] for r in benchmark_results if metric in r]
            values = [r[metric] for r in benchmark_results if metric in r]

            bars = ax.bar(range(len(names)), values, color=sns.color_palette("husl", len(names)))
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f"{metric.replace('_', ' ').title()} Comparison", fontsize=14)

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{value:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        return fig

    def plot_cost_analysis(
        self,
        cost_breakdown: Dict[str, float],
        title: str = "Cost Breakdown",
        save_path: Optional[str] = None,
    ):
        """
        Create pie chart for cost analysis.

        Args:
            cost_breakdown: Dictionary of cost categories
                           {"Compute": 45.0, "Storage": 10.0, "Network": 5.0}
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        colors = sns.color_palette("pastel", len(cost_breakdown))
        wedges, texts, autotexts = ax1.pie(
            cost_breakdown.values(),
            labels=cost_breakdown.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax1.set_title(title, fontsize=14, fontweight='bold')

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # Bar chart
        names = list(cost_breakdown.keys())
        values = list(cost_breakdown.values())
        bars = ax2.bar(names, values, color=colors)
        ax2.set_ylabel('Cost ($)', fontsize=12)
        ax2.set_title('Cost by Category', fontsize=14)
        ax2.set_xticklabels(names, rotation=45, ha='right')

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'${value:.2f}',
                ha='center',
                va='bottom'
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        return fig

    def plot_hyperparameter_importance(
        self,
        param_importance: Dict[str, float],
        title: str = "Hyperparameter Importance",
        save_path: Optional[str] = None,
    ):
        """
        Visualize hyperparameter importance from tuning results.

        Args:
            param_importance: Dictionary mapping params to importance scores
            title: Plot title
            save_path: Optional path to save figure
        """
        # Sort by importance
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        params, scores = zip(*sorted_params)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(params)), scores, color=sns.color_palette("viridis", len(params)))
        plt.yticks(range(len(params)), [p.replace('_', ' ').title() for p in params])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(score, i, f' {score:.3f}', va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        return plt.gcf()


def plot_training_history(
    history_path: str,
    save_path: Optional[str] = None,
):
    """
    Load and plot training history from file.

    Args:
        history_path: Path to history JSON/CSV file
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    import json

    # Load history
    path = Path(history_path)

    if path.suffix == '.json':
        with open(path, 'r') as f:
            history = json.load(f)
    elif path.suffix == '.csv':
        df = pd.read_csv(path)
        history = df.to_dict('list')
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Create visualizer
    visualizer = TrainingVisualizer()
    fig = visualizer.plot_training_history(history, save_path=save_path)

    return fig


def create_dashboard(
    training_history: Dict[str, List[float]],
    benchmark_results: List[Dict[str, Any]],
    cost_breakdown: Dict[str, float],
    output_dir: str = "./outputs/visualizations",
):
    """
    Create a comprehensive dashboard with all visualizations.

    Args:
        training_history: Training metrics over time
        benchmark_results: Model benchmark results
        cost_breakdown: Cost analysis data
        output_dir: Directory to save visualizations

    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = TrainingVisualizer()
    saved_plots = {}

    # Training history
    logger.info("Creating training history plot...")
    fig = visualizer.plot_training_history(
        training_history,
        save_path=str(output_dir / "training_history.png")
    )
    saved_plots["training_history"] = str(output_dir / "training_history.png")
    plt.close(fig)

    # Model comparison
    if benchmark_results:
        logger.info("Creating model comparison plot...")
        fig = visualizer.plot_model_comparison(
            benchmark_results,
            save_path=str(output_dir / "model_comparison.png")
        )
        saved_plots["model_comparison"] = str(output_dir / "model_comparison.png")
        plt.close(fig)

    # Cost analysis
    if cost_breakdown:
        logger.info("Creating cost analysis plot...")
        fig = visualizer.plot_cost_analysis(
            cost_breakdown,
            save_path=str(output_dir / "cost_analysis.png")
        )
        saved_plots["cost_analysis"] = str(output_dir / "cost_analysis.png")
        plt.close(fig)

    logger.info(f"Dashboard created in {output_dir}")
    return saved_plots
