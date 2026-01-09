import os

import h5py
import numpy as np

EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.dirname(EMBODIED_PATH)
LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")

dataset_a = "libero_spatial4"
dataset_b = "libero_spatial2"

path_a = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset_a}"
path_b = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset_b}"

task_files_a = sorted([f for f in os.listdir(path_a) if f.endswith(".hdf5")])
task_files_b = sorted([f for f in os.listdir(path_b) if f.endswith(".hdf5")])

print(f"Found {len(task_files_b)} matching tasks\n")

for task_file in task_files_b:
    print(f"Checking {task_file}")

    file_a = os.path.join(path_a, task_file)
    file_b = os.path.join(path_b, task_file)

    with h5py.File(file_a, "r") as fa, h5py.File(file_b, "r") as fb:
        demo_a = fa["data"]["demo_0"]
        demo_b = fb["data"]["demo_0"]

        # ---- Load actions ----
        logits_a = demo_a["processed_action_logits"][:]
        logits_b = demo_b["processed_action_logits"][:]

        if logits_a.shape != logits_b.shape:
            print("  ❌ Logit shape mismatch:", logits_a.shape, logits_b.shape)
            continue

        # ---- Compare logits ----
        if np.array_equal(logits_a, logits_b):
            print("  ✅ Logits match exactly")
        else:
            diff = np.abs(logits_a - logits_b)

            # indices where they differ
            idxs = np.argwhere(diff != 0)

            print(f"  ❌ Logits differ at {len(idxs)} positions")

            # limit output so you don’t spam the terminal
            MAX_PRINT = 10
            for i, idx in enumerate(idxs[:MAX_PRINT]):
                idx = tuple(idx)
                print(
                    f"    at {idx}: "
                    f"logits_a={logits_a[idx]}, "
                    f"logits_b={logits_b[idx]}, "
                    f"|Δ|={diff[idx]}"
                )

            if len(idxs) > MAX_PRINT:
                print(f"    ... and {len(idxs) - MAX_PRINT} more differences")

    print("--------------------------------------")
