<h1 align="center">
  Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2603.11653"><img src="https://img.shields.io/badge/arXiv-Paper-red" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

Official implementation for continual reinforcement learning (CRL) with Vision-Language-Action (VLA) models. Built on top of [RLinf](https://arxiv.org/abs/2509.15965).

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Downloading Models](#downloading-models)
- [Quick Start](#quick-start)
- [CRL Experiment Scripts](#crl-experiment-scripts)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Precomputing Base Logits](#precomputing-base-logits)
- [Architecture & Code Structure](#architecture--code-structure)
- [Citation](#citation)

---

## Overview

We study continual reinforcement learning for large Vision-Language-Action models and find that **simple sequential fine-tuning with LoRA** consistently avoids catastrophic forgetting, maintains plasticity, and preserves zero-shot generalization, often matching or surpassing dedicated continual learning methods.

This codebase provides:

- **Sequential fine-tuning** (Seq. FT)
- **CRL baselines** — EWC, Experience Replay, Dark Experience Replay, Weight Merge, SLCA
- **Multitask oracle** — joint training upper bound
- **Non-VLA baseline** — Simple CNN policy
- **Evaluation tools** — per-task success, LoRA scaling analysis

**Supported models:** [OpenVLA](https://github.com/openvla/openvla), [OpenVLA-OFT](https://github.com/moojink/openvla-oft)

**Supported simulators:** [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (Spatial, Object, Goal, Long suites)

**Supported algorithms:** PPO, GRPO

---

## Installation

### Prerequisites

- Linux (tested on Ubuntu 22.04/24.04)
- NVIDIA GPU(s) with CUDA 12.x
- Conda

### Setup

```bash
# 1. Clone the repository (includes bundled dependencies)
git clone git@github.com:UT-Austin-RobIn/continual-vla-rl.git
cd continual-vla-rl

# 2. Create conda environment
conda create -n vlacrl python=3.11.10 -y
conda activate vlacrl

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install bundled packages (included in this repo)
cd transformers-openvla-oft && pip install -e . && cd ..
cd openvla-oft && pip install -e . && cd ..
cd LIBERO && pip install -e . && cd ..
```
> **Optional:** For Flash Attention support (recommended for speed), install separately:
> ```bash
> pip install flash-attn --no-build-isolation
> ```
> **Note:** The repository includes `transformers-openvla-oft`, `openvla-oft`, and `LIBERO` as bundled directories. Steps 4–5 install them as editable packages — no separate cloning is needed.

---

## Downloading Models

Pre-trained SFT checkpoints are available on Hugging Face. Download them into the `model/` directory at the repository root:

| Model | Suite | Command |
|-------|-------|---------|
| OpenVLA-OFT SFT <br>(Spatial, 1 traj) | LIBERO-Spatial | `hf download Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 --local-dir ./model/Openvla-oft-SFT-libero-spatial-traj1` |
| OpenVLA-OFT SFT <br>(Object, 1 traj) | LIBERO-Object | `hf download Haozhan72/Openvla-oft-SFT-libero-object-traj1 --local-dir ./model/Openvla-oft-SFT-libero-object-traj1` |
| OpenVLA-OFT SFT <br>(10-task, all traj) | LIBERO-10 | `hf download Haozhan72/Openvla-oft-SFT-libero10-trajall --local-dir ./model/Openvla-oft-SFT-libero10-trajall` |

The default configs expect models at `./model/<hf-repo-name>`. To use a custom path, override these in the YAML config:

```yaml
rollout:
  model_dir: /your/custom/path
actor:
  checkpoint_load_path: /your/custom/path
  tokenizer:
    tokenizer_model: /your/custom/path
```

### LIBERO Path

The LIBERO environment path defaults to `./LIBERO` (the bundled copy). To use a different installation:

```bash
export LIBERO_REPO_PATH=/path/to/your/LIBERO
```

---

## Quick Start

After installation and model download, run a single-task sequential fine-tuning experiment:

```bash
# Train on task 0 of LIBERO-Spatial
bash examples/crl_experiment/run_embodiment_sequential.sh 0
```

Or train sequentially on tasks 0 through 4:

```bash
bash examples/crl_experiment/run_embodiment_sequential.sh "0,4"
```

---

## CRL Experiment Scripts

All scripts are in `examples/crl_experiment/` and source `common_functions.sh` for shared utilities.

### Training

| Script | Method | Example |
|--------|--------|---------|
| `run_embodiment_sequential.sh` | Sequential Fine-Tuning (Seq. FT) | `bash ... 0` or `bash ... "0,4"` |
| `run_embodiment_ewc.sh` | EWC (Elastic Weight Consolidation) | `bash ... "0,4"` |
| `run_embodiment_er.sh` | Experience Replay | `bash ... 0 0.03` |
| `run_embodiment_der.sh` | Dark Experience Replay | `bash ... 0 0.03` |
| `run_embodiment_weight_merge.sh` | Weight Merge | `bash ... "0,3" 0.8` |
| `run_embodiment_slca.sh` | SLCA (learning-rate schedules) | `bash ... "1,4" "2e-6,2e-6,1e-5"` |
| `run_embodiment_multitask.sh` | Multitask (joint training) | `bash ... "0,2,4"` |
| `run_embodiment_simple_cnn.sh` | Simple CNN baseline | `bash ... 0` |
| `run_embodiment_sequential_reorder.sh` | Seq. FT with custom task order | `bash ... "0,4"` |

### Script Arguments

Each script has its own argument pattern. Below are the signatures for each:

**Sequential Fine-Tuning:**
```
bash run_embodiment_sequential.sh TASK_ID_OR_RANGE [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
# Example: bash ... "0,4"
# Example: bash ... 0 "" 15 "" 42
```

**EWC:**
```
bash run_embodiment_ewc.sh TASK_ID_OR_RANGE [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
# Example: bash ... "0,4"
```

**Experience Replay:**
```
bash run_embodiment_er.sh TASK_ID_OR_RANGE [ER_COEFF=0.03] [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
# Example: bash ... "0,4" 0.03
```

**Dark Experience Replay:**
```
bash run_embodiment_der.sh TASK_ID_OR_RANGE [DER_COEFF=0.03] [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
# Example: bash ... "0,4" 0.03
```

**Weight Merge:**
```
bash run_embodiment_weight_merge.sh TASK_ID_OR_RANGE MERGE_COEFF [CONFIG_NAME] [SEED]
# Example: bash ... "0,4" 0.8
```

**SLCA:**
```
bash run_embodiment_slca.sh TASK_ID_OR_RANGE LR_STRING [CONFIG_NAME] [SEED]
# Example: bash ... "1,4" "2e-6,2e-6,1e-5"
```

**Multitask:**
```
bash run_embodiment_multitask.sh TASK_IDS [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
# Example: bash ... "0,2,4"
```

**Sequential (custom order):**
```
bash run_embodiment_sequential_reorder.sh TASK_IDS RUN_ID [MAX_EPOCH] [CONFIG_NAME] [SEED] [INIT_CHECKPOINT]
# Example: bash ... "4,3,2,1,0" reorder_v1
```

**Simple CNN:**
```
bash run_embodiment_simple_cnn.sh TASK_ID_OR_RANGE [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
# Example: bash ... 0
```

**LoRA Scale Evaluation:**
```
bash eval_embodiment_lora_scale.sh CHECKPOINT_LOCATION CURRENT_LORA_SCALE [PREVIOUS_LORA_COEFF] [STEP_NUMBER] [CONFIG_NAME]
# Example: bash ... logs/sequential/task_0 0.5
```

Common defaults across scripts:
- **SEED**: `1234`
- **CONFIG_NAME**: `crl_experiment/libero_spatial_grpo_openvlaoft_spatial` (varies for Simple CNN)
- **TASK_ID_OR_RANGE**: Single task (`0`) or comma-separated range (`"0,4"` trains tasks 0 through 4 sequentially)

### Evaluation

```bash
# Evaluate a checkpoint (default: global_step_10)
bash examples/crl_experiment/eval_embodiment.sh logs/sequential/task_0_seed1234

# Evaluate at a specific step
bash examples/crl_experiment/eval_embodiment.sh logs/sequential/task_0_seed1234 20

# Evaluate Simple CNN
bash examples/crl_experiment/eval_embodiment.sh logs/simple_cnn/task_0_seed1234 10 crl_experiment/libero_spatial_grpo_simple_cnn_eval

# LoRA scale evaluation
bash examples/crl_experiment/eval_embodiment_lora_scale.sh logs/sequential/task_0 0.5
```

---

## Configuration

Configs are YAML files in `examples/embodiment/config/`. For general RLinf configuration options (batch sizes, learning rates, FSDP settings, logging, etc.), see the [RLinf documentation](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/user/yaml.html).

Below are the CRL-specific parameters used in this work.

### CRL Method Parameters

**Experience Replay / Dark Experience Replay:**
```yaml
algorithm:
  use_experience_replay: True  # enable replay buffer
  bc_coeff: 0.03               # replay loss coefficient

  # DER-specific (logit-based replay):
  +algorithm.use_reference_logits_bc: True
  +algorithm.use_cached_bc_logits: True
```

**EWC (Elastic Weight Consolidation):**
```yaml
algorithm:
  +algorithm.use_ewc=True
```

**Weight Merge:**
```yaml
# Controlled via script argument (merge coefficient)
# e.g., bash run_embodiment_weight_merge.sh "0,4" 0.8
```

**SLCA (per-module learning rates):**
```yaml
# Controlled via script argument (comma-separated LRs for vision, LLM, head)
# e.g., bash run_embodiment_slca.sh "0,4" "2e-6,2e-6,1e-5"
```

### Task Configuration

```yaml
env:
  fixed_task_ids: [0,1,2]              # which task(s) to train on (null indicates all)
```

---

## Evaluation

Evaluation runs the trained policy across all tasks (train + held-out) and reports per-task success rates. Results are printed as a dictionary:

```python
{
    'eval/env_info/task_0_success': 1.0,       # success rate for task 0
    'eval/env_info/task_0_success_total': 8.0,  # number of eval episodes
    'eval/env_info/task_1_success': 0.75,
    ...
    'eval/env_info/success_once': 0.775,       # overall success (any point in episode)
    'eval/env_info/success_at_end': 0.625,     # success at final timestep
    'eval/env_info/return': 3.125,             # cumulative return
    'eval/env_info/episode_len': 512.0,        # episode length
}
```

Key metrics:
- **`task_X_success`**: Success rate for task X across evaluation episodes.
- **`success_once`**: Fraction of episodes where the task was completed at least once.
- **`success_at_end`**: Fraction of episodes where the task was completed at the final timestep.

Results are logged to WandB and/or TensorBoard (configurable via `runner.logger.logger_backends`).

---

## Precomputing Base Logits

For Dark Experience Replay (DER), base model logits can be precomputed and cached to disk to avoid recomputation during training.

### Prerequisites

Download the modified RLDS dataset into your LIBERO datasets directory:
```bash
hf download openvla/modified_libero_rlds --repo-type dataset --local-dir ./LIBERO/libero/datasets/
```

This places the dataset at `./LIBERO/libero/datasets/`.

### Running
```bash
bash examples/embodiment/compute_base_logits_embodiment.sh [CONFIG_NAME]
```

This runs `compute_base_logits_embodied_agent.py`, which generates logits for each task's demonstration trajectories and saves them alongside the dataset. These cached logits are loaded automatically when `use_cached_bc_logits: True` is set in the config.

---

## Architecture & Code Structure

### Training Flow

1. **Initialization** — Cluster setup, create actor (FSDP), rollout worker, and environment workers from config
2. **Rollout** — Environment interaction + action generation (pipelined across workers)
3. **Advantage computation** — GAE (PPO) or group-relative advantages (GRPO)
4. **Policy update** — LoRA parameter updates via the chosen algorithm
5. **Repeat** — Sync weights to rollout worker, loop

Entry point: `examples/embodiment/train_embodied_agent.py`

### Key Directories

```
examples/
  embodiment/
    config/                                 # YAML configs
      crl_experiment/                       # CRL-specific configs
    train_embodied_agent.py                 # Training entry point
    eval_embodied_agent.py                  # Evaluation entry point
    compute_base_logits_embodied_agent.py   # Logit precomputation
    run_embodiment.sh                       # Core training launcher
    compute_base_logits_embodiment.sh       # Logit precomputation launcher
  crl_experiment/
    run_embodiment_sequential.sh            # Sequential fine-tuning
    run_embodiment_ewc.sh                   # EWC
    run_embodiment_er.sh                    # Experience Replay
    run_embodiment_der.sh                   # Dark Experience Replay
    run_embodiment_weight_merge.sh          # Weight Merge
    run_embodiment_slca.sh                  # SLCA
    run_embodiment_multitask.sh             # Multitask training
    run_embodiment_simple_cnn.sh            # Simple CNN baseline
    run_embodiment_sequential_reorder.sh    # Custom task order
    eval_embodiment.sh                      # Checkpoint evaluation
    eval_embodiment_lora_scale.sh           # LoRA scale evaluation
    common_functions.sh                     # Shared utilities

rlinf/custom/                               # Custom modules
  libero_trajectory_dataset.py              # LIBERO dataset loader
  logits_precompute_worker.py               # Logit caching worker
  loss.py                                   # CRL loss functions (EWC, ER, DER)
  random_action_rollout_worker.py           # Random baseline rollout
  simple_cnn_utils.py                       # CNN policy utilities
```

---

## Citation

If you use this codebase, please cite our paper:

```bibtex
@article{hu2026vlacrl,
  title={Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning},
  author={Hu, Jiaheng and Shim, Jay and Tang, Chen and Sung, Yoonchang and Liu, Bo and Stone, Peter and Martin-Martin, Roberto},
  journal={arXiv preprint arXiv:2603.11653},
  year={2026},
  url={https://arxiv.org/abs/2603.11653}
}
```

Since this codebase is built on RLinf, we recommend additionally citing:

```bibtex
@article{yu2025rlinf,
  title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation},
  author={Yu, Chao and Wang, Yuanqing and Guo, Zhen and Lin, Hao and Xu, Si and Zang, Hongzhi and Zhang, Quanlu and Wu, Yongji and Zhu, Chunyang and Hu, Junhao and Huang, Zixiao and Wei, Mingjie and Xie, Yuqing and Yang, Ke and Dai, Bo and Xu, Zhexuan and Wang, Xiangyuan and Fu, Xu and Liu, Zhihao and Chen, Kang and Liu, Weilin and Liu, Gang and Li, Boxun and Yang, Jianlei and Yang, Zhi and Dai, Guohao and Wang, Yu},
  journal={arXiv preprint arXiv:2509.15965},
  year={2025}
}
```

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
