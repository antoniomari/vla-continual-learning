#!/bin/bash
# Eval latest no-normalization Hybrid OPD runs through checkpoint 200.
#
# Excludes:
#   - task 1, seed 1, lambda=0.1
#
# Includes:
#   - task 1, seed 3, lambda=0.1
#   - task 4, seeds 1/2/3, lambda=0.1
#   - task 4, seeds 1/2/3, lambda=1.0
#
# Preview:
#   DRY_RUN=1 bash scripts/run_latest_hybrid_opd_nonorm_eval_to200.sh
#
# Submit:
#   bash scripts/run_latest_hybrid_opd_nonorm_eval_to200.sh

set -euo pipefail

FULL_EVAL="examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh"
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial}"
EVAL_SEED="${EVAL_SEED:-184}"
EVAL_STEPS="${EVAL_STEPS:-25,50,75,100,125,150,175,200}"
EVAL_ROLLOUTS_PER_TASK="${EVAL_ROLLOUTS_PER_TASK:-320}"

TARGETS=(
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam0p1_thr0p0_rps32_task_1_seed3_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam0p1_thr0p0_rps32_task_4_seed1_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam0p1_thr0p0_rps32_task_4_seed2_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam0p1_thr0p0_rps32_task_4_seed3_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam1p0_thr0p0_rps32_task_4_seed1_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam1p0_thr0p0_rps32_task_4_seed2_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_success_gate_lam1p0_thr0p0_rps32_task_4_seed3_spatial"
)

echo "Latest no-norm Hybrid OPD eval wrapper"
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
