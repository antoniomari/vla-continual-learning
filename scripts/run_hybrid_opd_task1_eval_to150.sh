#!/bin/bash
# Eval the six Task-1 Hybrid OPD runs launched by the previous sweep script.
#
# Targets:
#   - group z-score, lambda=1.0, seeds 1 and 3
#   - group z-score, lambda=2.0 and 5.0, seed 2
#   - no reward normalization, lambda=0.1 and 1.0, seed 2
#
# Default checkpoints: 25,50,75,100,125,150.
#
# Preview:
#   DRY_RUN=1 bash scripts/run_hybrid_opd_task1_eval_to150.sh
#
# Submit:
#   bash scripts/run_hybrid_opd_task1_eval_to150.sh

set -euo pipefail

FULL_EVAL="examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh"
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial}"
EVAL_SEED="${EVAL_SEED:-184}"
EVAL_STEPS="${EVAL_STEPS:-25,50,75,100,125,150}"
EVAL_ROLLOUTS_PER_TASK="${EVAL_ROLLOUTS_PER_TASK:-320}"

TARGETS=(
  "logs_spatial/sequential/opd_sftteacher_adv1_group_zscore_success_gate_lam1p0_thr0p0_rps32_task_1_seed1_spatial_norm_group_zscore"
  "logs_spatial/sequential/opd_sftteacher_adv1_group_zscore_success_gate_lam1p0_thr0p0_rps32_task_1_seed3_spatial_norm_group_zscore"
  "logs_spatial/sequential/opd_sftteacher_adv1_group_zscore_success_gate_lam2p0_thr0p0_rps32_task_1_seed2_spatial_norm_group_zscore"
  "logs_spatial/sequential/opd_sftteacher_adv1_group_zscore_success_gate_lam5p0_thr0p0_rps32_task_1_seed2_spatial_norm_group_zscore"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam0p1_thr0p0_rps32_task_1_seed2_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam1p0_thr0p0_rps32_task_1_seed2_spatial"
)

echo "Task-1 Hybrid OPD eval-to-150 wrapper"
echo "  eval helper: ${FULL_EVAL}"
echo "  config: ${EVAL_CONFIG_NAME}"
echo "  seed: ${EVAL_SEED}"
echo "  steps: ${EVAL_STEPS}"
echo "  rollouts/task: ${EVAL_ROLLOUTS_PER_TASK}"
echo "  targets: ${#TARGETS[@]}"
echo "=================================="

for TARGET in "${TARGETS[@]}"; do
  echo "Submit full eval target=${TARGET} steps=${EVAL_STEPS}"
  env \
    "EVAL_ROLLOUTS_PER_TASK=${EVAL_ROLLOUTS_PER_TASK}" \
    bash "${FULL_EVAL}" "${TARGET}" "${EVAL_STEPS}" "${EVAL_CONFIG_NAME}" "${EVAL_SEED}"
done

echo "=================================="
echo "Submitted/previewed ${#TARGETS[@]} target group(s)."
