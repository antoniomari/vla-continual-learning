#!/usr/bin/env bash
# Upload the configured LIBERO dataset folder to a Hugging Face dataset repo.
#
# Cluster usage:
#   cd /users/anmari/vla-continual-learning
#   source .venv/bin/activate
#   hf auth login   # or export HF_TOKEN=...
#   HF_DATASET_REPO_ID="your-user-or-org/libero-spatial-256-from-rlds-reverted" \
#     bash scripts/upload_libero_dataset_to_hf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/configs/libero_dataset_paths.env"

if [[ -z "${HF_DATASET_REPO_ID}" ]]; then
  echo "ERROR: HF_DATASET_REPO_ID is unset."
  echo "Example:"
  echo "  HF_DATASET_REPO_ID=\"your-user-or-org/libero-spatial-256-from-rlds-reverted\" bash scripts/upload_libero_dataset_to_hf.sh"
  exit 2
fi

if [[ ! -d "${LIBERO_DATASET_DIR}" ]]; then
  echo "ERROR: LIBERO_DATASET_DIR does not exist: ${LIBERO_DATASET_DIR}"
  exit 2
fi

if ! python - <<'PY' >/dev/null 2>&1
import huggingface_hub
PY
then
  echo "ERROR: huggingface_hub is not installed in the active Python environment."
  echo "Install with:"
  echo "  python -m pip install -U huggingface_hub hf_transfer"
  exit 2
fi

echo "LIBERO dataset upload"
echo "  repo root:       ${REPO_ROOT}"
echo "  dataset folder:  ${LIBERO_DATASET_DIR}"
echo "  HF repo:         ${HF_DATASET_REPO_ID}"
echo "  path in repo:    ${HF_PATH_IN_REPO}"
echo "  private:         ${HF_PRIVATE:-1}"
echo ""

PRIVATE_FLAG=()
if [[ "${HF_PRIVATE:-1}" == "1" || "${HF_PRIVATE:-}" == "true" ]]; then
  PRIVATE_FLAG+=(--private)
fi

DRY_RUN_FLAG=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_FLAG+=(--dry-run)
fi

python "${REPO_ROOT}/scripts/libero_hf_dataset_transfer.py" upload \
  --repo-id "${HF_DATASET_REPO_ID}" \
  --repo-type "${HF_REPO_TYPE}" \
  --path-in-repo "${HF_PATH_IN_REPO}" \
  --dataset-dir "${LIBERO_DATASET_DIR}" \
  --commit-message "Upload ${LIBERO_DATASET_NAME}" \
  "${PRIVATE_FLAG[@]}" \
  "${DRY_RUN_FLAG[@]}"
