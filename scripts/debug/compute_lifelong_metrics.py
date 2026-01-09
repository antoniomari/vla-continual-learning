import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# based off of metric calculations from original LIBERO paper: https://arxiv.org/pdf/2306.03310

def compute_fwt(results, current_key, train_tasks=range(5), heldout_tasks=range(5, 10)):
    """
    Forward Transfer (FWT):
    Measures generalization performance on held-out tasks after training i tasks.
    """
    if current_key == "baseline":
        return 0.0  # for baseline

    perf = results[current_key]
    fwt = np.mean([perf[f"task_{j}"] for j in heldout_tasks])
    return fwt

# bwt computed on all 10 tasks since base model is SFT on all tasks
def compute_bwt(results, current_idx, keys, train_tasks=range(10)):
    """
    Backward Transfer (BWT):
    Measures average forgetting on previous tasks up to this stage.
    """
    if current_idx == 0:
        return 0.0  # for baseline

    s = np.zeros((len(keys), len(train_tasks)))
    for i, key in enumerate(keys):
        for j in train_tasks:
            if f"task_{j}" in results[key]:
                s[i][j] = results[key][f"task_{j}"]

    bwt_values = []
    k = current_idx
    for prev in range(k):
        diffs = [s[prev, prev] - s[k, prev]]
        bwt_values.extend(diffs)

    if len(bwt_values) == 0:
        return 0.0
    return np.mean(bwt_values)


def compute_avg_train_success(results, current_key, train_tasks=range(5)):
    """Average performance on all train tasks so far."""
    perf = results[current_key]
    avg = np.mean([perf[f"task_{j}"] for j in train_tasks if f"task_{j}" in perf])
    return avg


def plot_metrics(metrics_over_time, keys, save_path="metrics_plot.png"):
    """Plots the change in metrics over tasks."""
    tasks = np.arange(len(keys))

    plt.figure(figsize=(8, 5))
    plt.plot(tasks, metrics_over_time["FWT"], label="FWT (Forward Transfer)", marker='o')
    plt.plot(tasks, metrics_over_time["BWT"], label="BWT (Backward Transfer)", marker='o')
    plt.plot(tasks, metrics_over_time["Train Success"], label="Avg Train Success", marker='o')
    plt.xticks(tasks, keys, rotation=45)
    plt.xlabel("Training Stage")
    plt.ylabel("Metric Unit")
    plt.title("Lifelong Learning Metrics Progression")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ Metrics plot saved to: {os.path.abspath(save_path)}")
    plt.show()


def main(json_path):
    with open(json_path, "r") as f:
        results = json.load(f)

    task_keys = list(results.keys())
    metrics_over_time = {"FWT": [], "BWT": [], "Train Success": []}

    print("\n===== Lifelong Learning Metrics Progression =====")

    for i, key in enumerate(task_keys):
        fwt = compute_fwt(results, key)
        bwt = compute_bwt(results, i, task_keys)
        avg_train = compute_avg_train_success(results, key)

        metrics_over_time["FWT"].append(fwt)
        metrics_over_time["BWT"].append(bwt)
        metrics_over_time["Train Success"].append(avg_train)

        print(f"\nAfter training on {key}:")
        print(f"\tAvg Train Success: {avg_train:.3f}")
        print(f"\tFWT:               {fwt:.3f}")
        print(f"\tBWT:               {bwt:.3f}")

    print("\n===============================================\n")

    # Plot and save results
    plot_metrics(metrics_over_time, task_keys, save_path="metrics_progression.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_metrics.py <path_to_results.json>")
        sys.exit(1)
    main(sys.argv[1])