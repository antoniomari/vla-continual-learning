#!/bin/bash
# Launch SFT-teacher OPD with success-gated GRPO/teacher advantages.
#
# Successful rollouts use the environment/GRPO advantage.
# Failed rollouts use lambda_teacher * SFT-teacher OPD advantage.
#
# W&B names include:
#   opd_sftteacher_adv1_group_zscore_success_gate_lam{...}_thr0p0_rps8_

set -euo pipefail

RUN_MODE=train \
BASE_MODEL=1 \
GRPO_HP_FROM_SWEEP=1 \
TRAIN_TASK_INPUTS_OVERRIDE="1 4" \
TRAIN_MAX_EPOCHS_OVERRIDE=200 \
TRAIN_SEEDS_OVERRIDE=2 \
TRAIN_GROUP_SIZES_OVERRIDE=8 \
TRAIN_NUM_GROUP_ENVS_OVERRIDE=1 \
TRAIN_ROLLOUT_EPOCHS_OVERRIDE=1 \
TRAIN_OPD_LOSS_TYPES_OVERRIDE="embodied_opd_success_gate" \
TRAIN_OPD_REWARD_NORMALIZATIONS_OVERRIDE="group_zscore" \
TRAIN_OPD_SUCCESS_GATE_TEACHER_LAMBDAS_OVERRIDE="0.1 0.3 1.0" \
TRAIN_OPD_SUCCESS_GATE_REWARD_THRESHOLDS_OVERRIDE="0.0" \
SWEEP_SAVE_INTERVAL=25 \
bash examples/crl_experiment/jobs/embodiment_slurm_opd_sweep.sh
