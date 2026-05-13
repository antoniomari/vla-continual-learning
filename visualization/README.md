# Result Visualization

This directory contains notebooks for exploring current evaluation artifacts.

## Data source

Current notebooks are wired to:

- `results/eval_results.csv`

## Notebooks

- `01_results_overview.ipynb`: quick dataset overview and core metrics plots.
- `02_task_success_breakdown.ipynb`: per-task success analysis and heatmap/bar plots.

## Usage

From repository root:

```bash
jupyter lab visualization/
```

If plotting dependencies are missing in your environment:

```bash
uv pip install matplotlib pandas seaborn jupyter
```
