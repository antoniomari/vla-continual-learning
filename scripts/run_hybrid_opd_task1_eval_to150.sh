#!/bin/bash
# Eval Task-1 Hybrid OPD runs from the recent sweep scripts.
#
# Existing targets, already evaluated through 150:
#   - group z-score, lambda=1.0, seeds 1 and 3
#   - group z-score, lambda=2.0 and 5.0, seed 2
#   - no reward normalization, lambda=0.1 and 1.0, seed 2
#   Default checkpoints: 175,200.
#
# New delta targets:
#   - no reward normalization, lambda=10.0, seed 2
#   - no reward normalization, lambda=1.0, seeds 1 and 3
#   Default checkpoints: 25,50,75,100,125,150.
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
EXISTING_EVAL_STEPS="${EXISTING_EVAL_STEPS:-175,200}"
NEW_EVAL_STEPS="${NEW_EVAL_STEPS:-25,50,75,100,125,150}"
EVAL_ROLLOUTS_PER_TASK="${EVAL_ROLLOUTS_PER_TASK:-320}"

EXISTING_TARGETS=(
  "logs_spatial/sequential/opd_sftteacher_adv1_group_zscore_success_gate_lam1p0_thr0p0_rps32_task_1_seed1_spatial_norm_group_zscore"
  "logs_spatial/sequential/opd_sftteacher_adv1_group_zscore_success_gate_lam1p0_thr0p0_rps32_task_1_seed3_spatial_norm_group_zscore"
  "logs_spatial/sequential/opd_sftteacher_adv1_group_zscore_success_gate_lam2p0_thr0p0_rps32_task_1_seed2_spatial_norm_group_zscore"
  "logs_spatial/sequential/opd_sftteacher_adv1_group_zscore_success_gate_lam5p0_thr0p0_rps32_task_1_seed2_spatial_norm_group_zscore"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam0p1_thr0p0_rps32_task_1_seed2_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam1p0_thr0p0_rps32_task_1_seed2_spatial"
)

NEW_TARGETS=(
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam10p0_thr0p0_rps32_task_1_seed2_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam1p0_thr0p0_rps32_task_1_seed1_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam1p0_thr0p0_rps32_task_1_seed3_spatial"
)

submit_eval_batch() {
  local label="$1"
  local steps="$2"
  shift 2
  local targets=("$@")

  echo "----------------------------------"
  echo "${label}"
  echo "  steps: ${steps}"
  echo "  targets: ${#targets[@]}"

  for TARGET in "${targets[@]}"; do
    echo "Submit full eval target=${TARGET} steps=${steps}"
    env \
      "EVAL_ROLLOUTS_PER_TASK=${EVAL_ROLLOUTS_PER_TASK}" \
      bash "${FULL_EVAL}" "${TARGET}" "${steps}" "${EVAL_CONFIG_NAME}" "${EVAL_SEED}"
  done
}

echo "Task-1 Hybrid OPD eval wrapper"
echo "  eval helper: ${FULL_EVAL}"
echo "  config: ${EVAL_CONFIG_NAME}"
echo "  seed: ${EVAL_SEED}"
echo "  existing steps: ${EXISTING_EVAL_STEPS}"
echo "  new steps: ${NEW_EVAL_STEPS}"
echo "  rollouts/task: ${EVAL_ROLLOUTS_PER_TASK}"
echo "  existing targets: ${#EXISTING_TARGETS[@]}"
echo "  new targets: ${#NEW_TARGETS[@]}"
echo "=================================="

submit_eval_batch "Existing six runs, finish checkpoints" "${EXISTING_EVAL_STEPS}" "${EXISTING_TARGETS[@]}"
submit_eval_batch "New three runs, evaluate through 150" "${NEW_EVAL_STEPS}" "${NEW_TARGETS[@]}"

echo "=================================="
echo "Submitted/previewed $((${#EXISTING_TARGETS[@]} + ${#NEW_TARGETS[@]})) target group(s)."
