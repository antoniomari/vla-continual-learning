#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

TARGET_DIR="${TARGET_DIR:-LIBERO/libero/datasets/libero_spatial_256_from_rlds_reverted}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
HF_REPO_ID="${HF_REPO_ID:-openvla/modified_libero_rlds}"
CLEAN_FIRST="${CLEAN_FIRST:-1}"
export TARGET_DIR

echo "[rebuild] repo root: ${REPO_ROOT}"
echo "[rebuild] target dir: ${TARGET_DIR}"
echo "[rebuild] task suite: ${TASK_SUITE}"
echo "[rebuild] hf repo: ${HF_REPO_ID}"

mkdir -p "${TARGET_DIR}"

if [[ "${CLEAN_FIRST}" == "1" ]]; then
  echo "[rebuild] cleaning old demo files in ${TARGET_DIR}"
  rm -f "${TARGET_DIR}"/*_demo.hdf5
fi

python LIBERO/benchmark_scripts/convert_modified_libero_rlds_to_libero_hdf5.py \
  --output-dir "${TARGET_DIR}" \
  --task-suite-name "${TASK_SUITE}" \
  --download-first \
  --hf-repo-id "${HF_REPO_ID}" \
  --revert-libero-actions \
  --validate-written-files

python - <<'PY'
import os
import sys

target = os.environ.get("TARGET_DIR", "LIBERO/libero/datasets/libero_spatial_256_from_rlds_reverted")
expected_task4 = "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5"

files = sorted([f for f in os.listdir(target) if f.endswith("_demo.hdf5")])
print(f"[verify] wrote {len(files)} demo files to {target}")

if not files:
    raise SystemExit("[verify] ERROR: no *_demo.hdf5 files produced")

if expected_task4 not in files:
    raise SystemExit(
        "[verify] ERROR: missing expected task-4 file: "
        f"{expected_task4}"
    )

print("[verify] found expected task-4 file")
PY

echo "[rebuild] done"
