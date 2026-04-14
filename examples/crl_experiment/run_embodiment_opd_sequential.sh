#!/bin/bash
### OPD (on-policy distillation): same as run_embodiment_sequential.sh but defaults CONFIG_NAME to
### crl_experiment/libero_spatial_opd_openvlaoft_spatial when arg 4 is omitted or empty.
###
### Usage: bash examples/crl_experiment/run_embodiment_opd_sequential.sh TASK_ID_OR_RANGE [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
### Example (single task): bash examples/crl_experiment/run_embodiment_opd_sequential.sh 0
### Example (task range): bash examples/crl_experiment/run_embodiment_opd_sequential.sh "0,3"
### Example (max_epoch):  bash examples/crl_experiment/run_embodiment_opd_sequential.sh 0 "" 15
### Example (custom cfg): bash examples/crl_experiment/run_embodiment_opd_sequential.sh 0 "" "" crl_experiment/my_opd_variant 1234
###
### Override default OPD config without editing this file:
###   OPD_CONFIG=crl_experiment/libero_spatial_opd_openvlaoft_spatial bash .../run_embodiment_opd_sequential.sh 0
###
### WandB / Hydra run name: prepends EXPERIMENT_NAME_PREFIX (default opd_) to the sequential run name.
###   EXPERIMENT_NAME_PREFIX=myexp_ bash .../run_embodiment_opd_sequential.sh 0
###
### Post-train eval: delegates to run_embodiment_sequential.sh, which calls eval_embodiment.sh with
###   env.fixed_task_ids=null (all LIBERO suite tasks, e.g. 10 for spatial). Eval config is derived
###   as the GRPO eval yaml for the same suite (see derive_eval_config_name in common_functions.sh).
###
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OPD_CONFIG="${OPD_CONFIG:-crl_experiment/libero_spatial_opd_openvlaoft_spatial}"
export EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-opd_}"

TASK_INPUT="${1:-0}"
MANUAL_CHECKPOINT_PATH="${2:-}"
MAX_EPOCH="${3:-}"
CONFIG_NAME="${4:-${DEFAULT_OPD_CONFIG}}"
SEED="${5:-1234}"

exec bash "${SCRIPT_DIR}/run_embodiment_sequential.sh" \
  "${TASK_INPUT}" \
  "${MANUAL_CHECKPOINT_PATH}" \
  "${MAX_EPOCH}" \
  "${CONFIG_NAME}" \
  "${SEED}"
