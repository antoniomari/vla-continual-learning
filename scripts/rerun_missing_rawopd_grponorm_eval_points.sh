#!/bin/bash
# Rerun only the missing/crashed eval points for raw OPD + normalized GRPO Hybrid OPD.
#
# Preview:
#   DRY_RUN=1 bash scripts/rerun_missing_rawopd_grponorm_eval_points.sh
#
# Submit:
#   bash scripts/rerun_missing_rawopd_grponorm_eval_points.sh

set -euo pipefail

FULL_EVAL="examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh"
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial}"
EVAL_SEED="${EVAL_SEED:-184}"
EVAL_ROLLOUTS_PER_TASK="${EVAL_ROLLOUTS_PER_TASK:-320}"

submit_eval() {
  local target="$1"
  local step="$2"

  echo "Submit missing eval target=${target} step=${step}"
  env \
    "EVAL_ROLLOUTS_PER_TASK=${EVAL_ROLLOUTS_PER_TASK}" \
    bash "${FULL_EVAL}" "${target}" "${step}" "${EVAL_CONFIG_NAME}" "${EVAL_SEED}"
}

echo "Missing raw OPD + normalized GRPO Hybrid OPD eval points"
echo "  eval helper: ${FULL_EVAL}"
echo "  config: ${EVAL_CONFIG_NAME}"
echo "  seed: ${EVAL_SEED}"
echo "  rollouts/task: ${EVAL_ROLLOUTS_PER_TASK}"
echo "  points: 4"
echo "=================================="

submit_eval \
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_grpo_plus_success_gate_lam1p0_thr0p0_rps32_rawopd_grponorm_task_1_seed2_spatial" \
  "175"

submit_eval \
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_grpo_plus_success_gate_lam0p1_thr0p0_rps32_rawopd_grponorm_task_4_seed1_spatial" \
  "150"

submit_eval \
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_grpo_plus_success_gate_lam1p0_thr0p0_rps32_rawopd_grponorm_task_4_seed2_spatial" \
  "75"

submit_eval \
  "logs_spatial/sequential/opd_sftteacher_adv0_nonorm_grpo_plus_success_gate_lam1p0_thr0p0_rps32_rawopd_grponorm_task_4_seed3_spatial" \
  "125"

echo "=================================="
echo "Submitted/previewed 4 missing eval point(s)."
