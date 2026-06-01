#!/usr/bin/env bash
# Download the configured LIBERO dataset folder from a Hugging Face dataset repo.
#
# Local/cluster usage:
#   cd /path/to/vla-continual-learning
#   source .venv/bin/activate
#   hf auth login   # for private repos, or export HF_TOKEN=...
#   HF_DATASET_REPO_ID="your-user-or-org/libero-spatial-256-from-rlds-reverted" \
#     bash scripts/download_libero_dataset_from_hf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/configs/libero_dataset_paths.env"

if [[ -z "${HF_DATASET_REPO_ID}" ]]; then
  echo "ERROR: HF_DATASET_REPO_ID is unset."
  echo "Example:"
  echo "  HF_DATASET_REPO_ID=\"your-user-or-org/libero-spatial-256-from-rlds-reverted\" bash scripts/download_libero_dataset_from_hf.sh"
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

echo "LIBERO dataset download"
echo "  repo root:       ${REPO_ROOT}"
echo "  target folder:   ${LIBERO_DATASET_DIR}"
echo "  staging folder:  ${HF_DATASET_STAGING_DIR}"
echo "  HF repo:         ${HF_DATASET_REPO_ID}"
echo "  path in repo:    ${HF_PATH_IN_REPO}"
echo "  revision:        ${HF_REVISION:-main}"
echo "  overwrite:       ${OVERWRITE:-0}"
echo ""

OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG+=(--overwrite)
fi

DRY_RUN_FLAG=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_FLAG+=(--dry-run)
fi

python "${REPO_ROOT}/scripts/libero_hf_dataset_transfer.py" download \
  --repo-id "${HF_DATASET_REPO_ID}" \
  --repo-type "${HF_REPO_TYPE}" \
  --path-in-repo "${HF_PATH_IN_REPO}" \
  --dataset-dir "${LIBERO_DATASET_DIR}" \
  --staging-dir "${HF_DATASET_STAGING_DIR}" \
  --revision "${HF_REVISION:-main}" \
  "${OVERWRITE_FLAG[@]}" \
  "${DRY_RUN_FLAG[@]}"
