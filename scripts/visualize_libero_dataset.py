import os

import h5py

EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.dirname(EMBODIED_PATH)
LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")

dataset = "libero_spatial"
path = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset}"

task_files = [f for f in os.listdir(path) if f.endswith(".hdf5")]
print(f"Found {len(task_files)} tasks:\n")
for f in task_files:
    print(" •", f)

print("\n--------------------------------------\n")

for task_file in task_files:
    file_path = os.path.join(path, task_file)
    print(f"Inspecting {task_file}")

    with h5py.File(file_path, "r") as f:
        # extract first trajectory
        demo = f["data"]["demo_0"]
        for key in demo.keys():
            data = demo[key]
            if key == "obs":
                data = data["agentview_rgb"]

            print(f"    {key}: shape={data.shape}, dtype={data.dtype}")

    print("\n--------------------------------------\n")
