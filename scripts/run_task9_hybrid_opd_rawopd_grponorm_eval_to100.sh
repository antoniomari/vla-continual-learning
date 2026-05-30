#!/bin/bash
# Eval the six task-9 SFT-teacher Hybrid OPD runs through checkpoint 100.
#
# Targets:
#   - lambdas: 0.1, 1.0
#   - seeds: 1, 2, 3
#   - checkpoints: 25, 50, 75, 100
#
# Preview:
#   DRY_RUN=1 bash scripts/run_task9_hybrid_opd_rawopd_grponorm_eval_to100.sh
#
# Submit:
#   bash scripts/run_task9_hybrid_opd_rawopd_grponorm_eval_to100.sh

set -euo pipefail

FULL_EVAL="examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh"
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial}"
EVAL_SEED="${EVAL_SEED:-184}"
EVAL_STEPS="${EVAL_STEPS:-25,50,75,100}"
EVAL_ROLLOUTS_PER_TASK="${EVAL_ROLLOUTS_PER_TASK:-320}"

TARGETS=(
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_grpo_plus_success_gate_lam0p1_thr0p0_rps32_rawopd_grponorm_task_9_seed1_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_grpo_plus_success_gate_lam0p1_thr0p0_rps32_rawopd_grponorm_task_9_seed2_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_grpo_plus_success_gate_lam0p1_thr0p0_rps32_rawopd_grponorm_task_9_seed3_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_grpo_plus_success_gate_lam1p0_thr0p0_rps32_rawopd_grponorm_task_9_seed1_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_grpo_plus_success_gate_lam1p0_thr0p0_rps32_rawopd_grponorm_task_9_seed2_spatial"
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_grpo_plus_success_gate_lam1p0_thr0p0_rps32_rawopd_grponorm_task_9_seed3_spatial"
)

echo "Task 9 Hybrid OPD raw-OPD/GRPO-norm eval wrapper"
echo "  eval helper: ${FULL_EVAL}"
echo "  config: ${EVAL_CONFIG_NAME}"
echo "  eval seed: ${EVAL_SEED}"
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
