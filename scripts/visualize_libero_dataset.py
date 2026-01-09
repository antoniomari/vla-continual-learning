import os

import h5py
import imageio
import numpy as np

EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.dirname(EMBODIED_PATH)
LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")

dataset = "libero_spatial_simplevla"
path = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset}"

video_out_dir = os.path.join(path, "videos_demo0")
os.makedirs(video_out_dir, exist_ok=True)

task_files = [f for f in os.listdir(path) if f.endswith(".hdf5")]
print(f"Found {len(task_files)} tasks:\n")
for f in task_files:
    print(" •", f)

print("\n--------------------------------------\n")

for task_file in task_files:
    file_path = os.path.join(path, task_file)
    task_name = task_file.replace(".hdf5", "")
    video_path = os.path.join(video_out_dir, f"{task_name}_demo0.mp4")

    with h5py.File(file_path, "r") as f:
        print(f"Inspecting {task_file} with {len(f['data'])} demos")
        demo = f["data"]["demo_1"]

        for key in demo.keys():
            data = demo[key]
            if key == "obs":
                data = data["agentview_rgb"]

            print(f"    {key}: shape={data.shape}, dtype={data.dtype}")

            if key == "actions":
                print(f"        Actions: {data[:][:5, :]}")

        # --------- VIDEO EXPORT ----------
        frames = demo["obs"]["agentview_rgb"][:]  # (T, H, W, 3)
        frames = frames.astype(np.uint8)

        writer = imageio.get_writer(video_path, fps=20, codec="libx264", quality=8)

        for frame in frames:
            writer.append_data(frame)

        writer.close()

        print(f"    🎥 Saved video to {video_path}")

    print("\n--------------------------------------\n")
