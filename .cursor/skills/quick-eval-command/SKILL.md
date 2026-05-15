---
name: quick-eval-command
description: Provides quick copy-paste commands to run embodied eval jobs for one or many checkpoints/steps. Use when the user asks for an eval command example, full eval command, eval sweep command, or how to launch checkpoint eval in parallel.
---
# Quick Eval Command

## Purpose

Return short, copy-paste commands for the embodied eval scripts in this repo:

- `examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh`
- `examples/crl_experiment/jobs/embodiment_slurm_eval_sweep.sh`

Prefer commands with explicit values so the user can run immediately.

## Canonical Examples

### 1) Full eval, one checkpoint step

```bash
bash examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh \
  /users/anmari/vla-continual-learning/logs_spatial/sequential/task_1_seed184 \
  50 \
  crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial \
  184
```

### 2) Full eval, multiple checkpoint steps (parallel jobs)

```bash
bash examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh \
  /users/anmari/vla-continual-learning/logs_spatial/sequential/task_1_seed184 \
  10,20,30,40,50 \
  crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial \
  184
```

### 3) Eval sweep script with env overrides

```bash
EVAL_TARGET=/users/anmari/vla-continual-learning/logs_spatial/sequential/task_1_seed184 \
EVAL_STEPS=10,20,30,40,50 \
EVAL_SEED=184 \
bash examples/crl_experiment/jobs/embodiment_slurm_eval_sweep.sh
```

### 4) Dry run (print sbatch job scripts without submitting)

```bash
DRY_RUN=1 bash examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh \
  /users/anmari/vla-continual-learning/logs_spatial/sequential/task_1_seed184 \
  50 \
  crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial \
  184
```

## Seed Guidance

- Seed override is optional; default is `1234`.
- Keep one fixed eval seed when comparing checkpoints.
- Change seed only for robustness checks.

## Response Style

When the user asks for a quick eval command:

1. Return one command first (most likely the multi-step full eval command).
2. Add one short line explaining which values to replace (`run path`, `steps`, `seed`).
3. Keep the answer concise unless the user asks for more variants.
