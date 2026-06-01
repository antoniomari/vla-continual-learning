# LIBERO Dataset Transfer Through Hugging Face

This repo expects LIBERO datasets at:

```text
${LIBERO_REPO_PATH}/libero/datasets/${LIBERO_DATASET_NAME}
```

The shared defaults live in:

```text
configs/libero_dataset_paths.env
```

Current defaults:

```text
LIBERO_REPO_PATH=${REPO_ROOT}/LIBERO
LIBERO_DATASET_NAME=libero_spatial_256_from_rlds_reverted
LIBERO_DATASET_DIR=${LIBERO_REPO_PATH}/libero/datasets/${LIBERO_DATASET_NAME}
HF_PATH_IN_REPO=${LIBERO_DATASET_NAME}
```

## Upload From Cluster

Log into the cluster and run from repo root:

```bash
cd /users/anmari/vla-continual-learning
source .venv/bin/activate
python -m pip install -U huggingface_hub hf_transfer
hf auth login

HF_DATASET_REPO_ID="your-user-or-org/libero-spatial-256-from-rlds-reverted" \
HF_PRIVATE=1 \
bash scripts/upload_libero_dataset_to_hf.sh
```

Use `DRY_RUN=1` to print the resolved paths without uploading:

```bash
HF_DATASET_REPO_ID="your-user-or-org/libero-spatial-256-from-rlds-reverted" \
DRY_RUN=1 \
bash scripts/upload_libero_dataset_to_hf.sh
```

## Download Into This Repo

After the upload completes:

```bash
cd /path/to/vla-continual-learning
source .venv/bin/activate
python -m pip install -U huggingface_hub hf_transfer
hf auth login

HF_DATASET_REPO_ID="your-user-or-org/libero-spatial-256-from-rlds-reverted" \
bash scripts/download_libero_dataset_from_hf.sh
```

If the target folder already exists and should be replaced:

```bash
HF_DATASET_REPO_ID="your-user-or-org/libero-spatial-256-from-rlds-reverted" \
OVERWRITE=1 \
bash scripts/download_libero_dataset_from_hf.sh
```

## Path Notes

Training and evaluation wrappers set:

```bash
LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-${REPO_PATH}/LIBERO}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_REPO_PATH}}"
```

So the downloaded folder must end up at:

```text
LIBERO/libero/datasets/libero_spatial_256_from_rlds_reverted
```

relative to the repo root unless `LIBERO_REPO_PATH` is explicitly overridden.
