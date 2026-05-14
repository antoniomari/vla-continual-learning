---
name: regenerate-libero-spatial-rlds-reverted
description: Regenerate the LIBERO OPD BC dataset `libero_spatial_256_from_rlds_reverted` from `openvla/modified_libero_rlds` using the in-repo converter. Use when the user asks to rebuild, recreate, or refresh this dataset, or when OPD fails with missing `*_demo.hdf5` task files.
disable-model-invocation: true
---

# Regenerate LIBERO Spatial Reverted Dataset

Rebuild:
- `LIBERO/libero/datasets/libero_spatial_256_from_rlds_reverted`

using:
- `LIBERO/benchmark_scripts/convert_modified_libero_rlds_to_libero_hdf5.py`

## When To Use

- OPD/BC crashes with `LiberoSFTDataset: task filter removed all files`.
- Missing task file errors such as `..._demo.hdf5` not found.
- User explicitly asks to regenerate `libero_spatial_256_from_rlds_reverted`.

## Quick Command

From repo root:

```bash
bash .cursor/skills/regenerate-libero-spatial-rlds-reverted/scripts/rebuild_dataset.sh
```

## What This Does

1. Uses `--download-first` to pull `openvla/modified_libero_rlds` into `LIBERO/libero/datasets`.
2. Converts RLDS TFRecords for `libero_spatial` into LIBERO-style HDF5 demos.
3. Writes output to `LIBERO/libero/datasets/libero_spatial_256_from_rlds_reverted`.
4. Validates written files with converter-side checks.
5. Verifies expected task demo file names exist (including the task-4 bowl file).

## Manual Equivalent

```bash
python LIBERO/benchmark_scripts/convert_modified_libero_rlds_to_libero_hdf5.py \
  --output-dir LIBERO/libero/datasets/libero_spatial_256_from_rlds_reverted \
  --task-suite-name libero_spatial \
  --download-first \
  --revert-libero-actions \
  --validate-written-files
```

## Notes

- Keep default image/action transforms from the converter (rotation and last-dim flip are enabled by default).
- By default, the helper script deletes existing `*_demo.hdf5` files in the target folder before rebuild. Set `CLEAN_FIRST=0` to keep existing files.
