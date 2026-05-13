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
python3 -m venv .venv-viz
source .venv-viz/bin/activate
python -m pip install --upgrade pip
python -m pip install -r visualization/requirements.txt
python -m ipykernel install --user --name vla-viz --display-name "Python (vla-viz)"
jupyter lab visualization/
```
