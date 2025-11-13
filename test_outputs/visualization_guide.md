# Visualization Guide

## Available Visualizations

### 1. Training History
```python
from src.utils.visualization import TrainingVisualizer
viz = TrainingVisualizer()
viz.plot_training_history(history, save_path='training.png')
```

**Shows**:
- Training and validation loss curves
- Learning rate schedule
- Trend lines for loss prediction

**Example Data**:
```json
{
  "epochs": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10
  ],
  "train_loss": [
    2.5,
    2.2,
    2.0,
    1.8,
    1.6,
    1.5,
    1.4,
    1.3,
    1.25,
    1.2
  ],
  "val_loss": [
    2.6,
    2.3,
    2.1,
    1.9,
    1.7,
    1.6,
    1.5,
    1.45,
    1.4,
    1.35
  ],
  "learning_rate": [
    5e-05,
    4.8e-05,
    4.5e-05,
    4.2e-05,
    3.9e-05,
    3.6e-05,
    3.3e-05,
    3e-05,
    2.7e-05,
    2.5e-05
  ]
}
```

### 2. Model Comparison
```python
viz.plot_model_comparison(benchmark_results, metrics=['perplexity', 'speed'])
```

**Shows**:
- Side-by-side performance comparison
- Bar charts for multiple metrics
- Value labels on bars

### 3. Cost Analysis
```python
viz.plot_cost_analysis(cost_breakdown, save_path='costs.png')
```

**Shows**:
- Pie chart of cost categories
- Bar chart for comparison
- Percentage breakdown

### 4. Hyperparameter Importance
```python
viz.plot_hyperparameter_importance(param_importance)
```

**Shows**:
- Horizontal bar chart of parameter importance
- Sorted by impact on performance
- Helps prioritize tuning efforts

## Creating a Dashboard

```python
from src.utils.visualization import create_dashboard

create_dashboard(
    training_history=history,
    benchmark_results=benchmarks,
    cost_breakdown=costs,
    output_dir='./dashboard'
)
```

This generates all visualizations at once in a single directory.

## Customization

```python
# Use different style
viz = TrainingVisualizer(style='ggplot')  # or 'bmh', 'seaborn'

# Customize colors
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Change figure size
plt.rcParams['figure.figsize'] = (16, 8)
```

---

*Install dependencies: `pip install matplotlib seaborn pandas`*