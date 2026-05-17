#!/bin/bash
# Submit/preview full embodied eval jobs for many run directories and checkpoint steps.
#
# This is a thin wrapper around:
#   examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh
#
# Defaults target the rps8/200 runs produced by scripts/run_opd_loss_variants.sh:
#   - GRPO baseline
#   - OPD group-normalized REINFORCE
#   - OPD REINFORCE without normalization
#   - OPD with GRPO clipping loss
# for tasks 1 and 4.
#
# Preview:
#   DRY_RUN=1 bash scripts/run_full_eval_sweep.sh
#
# Submit:
#   bash scripts/run_full_eval_sweep.sh
#
# Override targets or steps:
#   EVAL_TARGETS="logs_spatial/sequential/run_a logs_spatial/sequential/run_b" \
#   EVAL_STEPS="20,40,60" \
#   bash scripts/run_full_eval_sweep.sh
#
# Base model eval too:
#   INCLUDE_BASE=1 DRY_RUN=1 bash scripts/run_full_eval_sweep.sh

set -euo pipefail

FULL_EVAL="examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh"

EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial}"
EVAL_SEED="${EVAL_SEED:-184}"
EVAL_STEPS="${EVAL_STEPS:-25,50,75}"
EVAL_ROLLOUTS_PER_TASK="${EVAL_ROLLOUTS_PER_TASK:-320}"
INCLUDE_BASE="${INCLUDE_BASE:-0}"



DEFAULT_TARGETS=(
  "logs_spatial/sequential/grpo_rps32_gb2048_gs8_steps100_si25_task_1_seed4096_spatial"
  "logs_spatial/sequential/grpo_rps32_gb2048_gs8_steps100_si25_task_4_seed4096_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_rps32_steps100_si25_task_1_seed184_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_rps32_steps100_si25_task_4_seed184_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv1_grpo_loss_rps32_steps100_si25_task_1_seed184_spatial_norm_group_zscore"
  "logs_spatial/sequential/opd_sftteacher_adv1_grpo_loss_rps32_steps100_si25_task_4_seed184_spatial_norm_group_zscore"
  "logs_spatial/sequential/opd_sftteacher_adv1_rps32_steps100_si25_task_1_seed184_spatial_norm_group_zscore"
  "logs_spatial/sequential/opd_sftteacher_adv1_rps32_steps100_si25_task_4_seed184_spatial_norm_group_zscore"
)

TARGETS=()
if [[ -n "${EVAL_TARGETS:-}" ]]; then
  read -r -a TARGETS <<< "${EVAL_TARGETS}"
else
  TARGETS=("${DEFAULT_TARGETS[@]}")
fi

if [[ "${INCLUDE_BASE}" == "1" ]]; then
  TARGETS=("base" "${TARGETS[@]}")
fi

echo "Full eval sweep wrapper"
echo "  eval helper: ${FULL_EVAL}"
echo "  config: ${EVAL_CONFIG_NAME}"
echo "  seed: ${EVAL_SEED}"
echo "  steps: ${EVAL_STEPS}"
echo "  rollouts/task: ${EVAL_ROLLOUTS_PER_TASK}"
echo "  targets: ${#TARGETS[@]}"
echo "=================================="

job_group_count=0
for TARGET in "${TARGETS[@]}"; do
  if [[ "${TARGET}" == "base" ]]; then
    TARGET_STEPS="0"
  else
    TARGET_STEPS="${EVAL_STEPS}"
  fi

  echo "Submit full eval target=${TARGET} steps=${TARGET_STEPS}"
  env \
    "EVAL_ROLLOUTS_PER_TASK=${EVAL_ROLLOUTS_PER_TASK}" \
    bash "${FULL_EVAL}" "${TARGET}" "${TARGET_STEPS}" "${EVAL_CONFIG_NAME}" "${EVAL_SEED}"
  job_group_count=$((job_group_count + 1))
done

echo "=================================="
echo "Submitted/previewed ${job_group_count} target group(s)."
