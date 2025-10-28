#!/bin/bash

# Usage:
#   bash scripts/run_eval_repeats.sh <experiment_name> <yaml_config>
# Example:
#   bash scripts/run_eval_repeats.sh libero_10_ppo_openvlaoft_eval libero_10_ppo_openvlaoft_eval.yaml

if [ $# -lt 2 ]; then
  echo "Usage: $0 <experiment_name> <yaml_config>"
  exit 1
fi

EXPERIMENT_NAME="$1"
YAML_CONFIG="$2"

# Get absolute path of the directory this script is in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"   # one level up from 'scripts/'

# Log file location
LOG_FILE="$SCRIPT_DIR/${EXPERIMENT_NAME}.txt"

# Command to run (relative to project root)
CMD="bash examples/embodiment/eval_embodiment.sh ${YAML_CONFIG}"

echo "Running experiment: ${EXPERIMENT_NAME}"
echo "Using config file: ${YAML_CONFIG}"
echo "Logging output to: ${LOG_FILE}"
echo "----------------------------------------" > "$LOG_FILE"

# Run 8 times and append output
for i in {1..8}; do
  echo "Starting run #$i..." | tee -a "$LOG_FILE"
  (cd "$ROOT_DIR" && $CMD >> "$LOG_FILE" 2>&1)
  echo "Finished run #$i" | tee -a "$LOG_FILE"
  echo "----------------------------------------" >> "$LOG_FILE"
done

echo "✅ All runs completed. Logs saved to ${LOG_FILE}"