# Embodied VLA Training - RLinf Documentation Summary

This document consolidates important information for training Vision-Language-Action (VLA) models in embodied environments (ManiSkill3, LIBERO) using RLinf. Adapt and merge sections into your main README as needed.

---

## 1. Overview

RLinf supports training VLAs for robotic manipulation via reinforcement learning. The framework uses FSDP + Hugging Face for embodied experiments.

**Supported Models:**
- [OpenVLA](https://github.com/openvla/openvla)
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
- [Pi0](https://github.com/Physical-Intelligence/openpi)

**Supported Simulators:**
- **ManiSkill3** - GPU-accelerated, contact-rich manipulation
- **LIBERO** - Large-scale household manipulation (built on robosuite/MuJoCo)

**Supported Algorithms:**
- PPO (Proximal Policy Optimization)
- GRPO (Group Relative Policy Optimization)

---

## 2. Installation (Embodied)

### Custom environment (Debian/Ubuntu)

Maybe add conda installation method...

1. **Common deps first:**
   ```bash
   uv sync
   UV_TORCH_BACKEND=auto uv sync
   ```

2. **Embodied-specific:**
   ```bash
   uv sync --extra embodied
   bash requirements/install_embodied_deps.sh
   ```

3. **Model-specific:**
   ```bash
   # OpenVLA
   UV_TORCH_BACKEND=auto uv pip install -r requirements/openvla.txt --no-build-isolation

   # OpenVLA-OFT
   UV_TORCH_BACKEND=auto uv pip install -r requirements/openvla_oft.txt --no-build-isolation
   ```

### Docker

```bash
docker pull rlinf/rlinf:agentic-openvla-rlinf0.1-torch2.5.1         # OpenVLA
docker pull rlinf/rlinf:agentic-openvlaoft-rlinf0.1-torch2.5.1      # OpenVLA-OFT

docker run -it --gpus all \
   --shm-size 100g \
   --net=host \
   --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
   --name rlinf \
   rlinf/rlinf:CHOSEN_IMAGE /bin/bash

git clone https://github.com/RLinf/RLinf.git
cd RLinf
```

---

## 3. Quick Start

### 3.1 PPO Training on ManiSkill3

**Step 1: Download pre-trained model (OpenVLA)**

```bash
hf download gen-robot/openvla-7b-rlvla-warmup \
  --local-dir /path/to/model/openvla-7b-rlvla-warmup/
```

**Step 2: Edit config** (`examples/embodiment/config/maniskill_ppo_openvla_quickstart.yaml`)

- `cluster.component_placement`: set to `"0-3"` or `"0-7"` for 4/8 GPUs
- `rollout.model_dir`: path to downloaded model
- `actor.checkpoint_load_path`: path to downloaded model
- `actor.tokenizer.tokenizer_model`: path to tokenizer

**Step 3: Launch training**

```bash
bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart
```

**Step 4: Monitor**

- Checkpoints & metrics: `../results`
- TensorBoard: `tensorboard --logdir ../results/tensorboard/ --port 6006`
- Key metrics: `rollout/env_info/return`, `rollout/env_info/success_once`

### 3.2 CRL / Lifelong Experiments (LIBERO)

For Continual Reinforcement Learning (CRL) experiments on LIBERO, use the scripts in `examples/crl_experiment/`. These support sequential task training, weight merge, EWC, multitask training, and more.

**Prerequisites:** Same as above; ensure LIBERO and model paths are set in the config files under `examples/embodiment/config/crl_experiment/`.

**Run from the repository root.**

---

## 4. CRL Experiment Scripts (examples/crl_experiment/)

All scripts source `common_functions.sh` and use configs from `examples/embodiment/config/crl_experiment/`.

### Training Scripts

| Script | Description |
|--------|-------------|
| `run_embodiment_sequential.sh` | Sequential tasks (no merge between tasks) |
| `run_embodiment_ewc.sh` | Sequential + EWC (Elastic Weight Consolidation) |
| `run_embodiment_multitask.sh` | Joint multitask training (multiple tasks in one run) |
| `run_embodiment_sequential_reorder.sh` | Sequential training with custom task order |
| `run_embodiment_weight_merge.sh` | Weight merge: merge previous adapters, add new LoRA for each task |
| `run_embodiment_slca.sh` | SLCA: learning-rate experiments (vision/llm/head LoRA LRs) |
| `run_embodiment_simple_cnn.sh` | Simple CNN policy (non-VLA baseline) |
| `run_embodiment_er.sh` | Sequential + ER (Experience Replay) |
| `run_embodiment_der.sh` | Sequential + DER (Dark Experience Replay) |

### Evaluation Scripts

| Script | Description |
|--------|-------------|
| `eval_embodiment.sh` | Direct evaluation of a checkpoint |
| `eval_embodiment_lora_scale.sh` | Evaluation with LoRA scaling (single or multi-LoRA) |

### Usage Examples

**Sequential (single task):**
```bash
bash examples/crl_experiment/run_embodiment_sequential.sh 0
```

**Sequential (task range 0–3):**
```bash
bash examples/crl_experiment/run_embodiment_sequential.sh "0,3"
```

**Sequential with custom max_epoch and seed:**
```bash
bash examples/crl_experiment/run_embodiment_sequential.sh 0 "" 15 "" 42
```

**Sequential with ER (ER coefficient 0.03):**
```bash
bash examples/crl_experiment/run_embodiment_er.sh 0 0.03
```

**Sequential with DER (DER coefficient 0.03):**
```bash
bash examples/crl_experiment/run_embodiment_der.sh 0 0.03
```

**Weight merge (merge coefficient 0.8):**
```bash
bash examples/crl_experiment/run_embodiment_weight_merge.sh "0,3" 0.8
```

**SLCA (learning-rate experiment, task range with custom LRs):**
```bash
bash examples/crl_experiment/run_embodiment_slca.sh "1,4" "2e-6,2e-6,1e-5"
```

**Multitask (joint training on tasks 0, 2, 4):**
```bash
bash examples/crl_experiment/run_embodiment_multitask.sh "0,2,4"
```

**Simple CNN (single task or range):**
```bash
bash examples/crl_experiment/run_embodiment_simple_cnn.sh 0
bash examples/crl_experiment/run_embodiment_simple_cnn.sh "0,3"
```

**Precomputing Logits to Disk:**
```bash
bash examples/embodiment/compute_base_logits_embodiment.sh
```

**Evaluation:**
```bash
# Standard LoRA checkpoint (default step 10)
bash examples/crl_experiment/eval_embodiment.sh logs/sequential/task_0_seed1234

# With specific step number
bash examples/crl_experiment/eval_embodiment.sh logs/sequential/task_0_seed1234 20

# Simple CNN
bash examples/crl_experiment/eval_embodiment.sh logs/simple_cnn/task_0_seed1234 10 crl_experiment/libero_spatial_grpo_simple_cnn_eval

# LoRA scale evaluation (single LoRA)
bash examples/crl_experiment/eval_embodiment_lora_scale.sh logs/bcrl_logit/0.3/task_0 0.5
```

### Config Names

Default configs vary by script. Override with the 4th (or 3rd) positional argument, for example:

- `crl_experiment/libero_spatial_grpo_openvlaoft_spatial` (sequential, weight_merge, EWC, etc.)
- `crl_experiment/libero_spatial_grpo_simple_cnn` (simple CNN)
- `crl_experiment/libero_10_grpo_openvlaoft_long` (10-task suite)
- `crl_experiment/libero_object_grpo_openvlaoft_object` (object suite)

---

## 5. Configuration Files & Key Parameters

TODO: LLM generated need to change

**Config directory:** `examples/embodiment/config/`

### ManiSkill3
- OpenVLA + PPO: `maniskill_ppo_openvla.yaml`
- OpenVLA-OFT + PPO: `maniskill_ppo_openvlaoft.yaml`
- OpenVLA + GRPO: `maniskill_grpo_openvla.yaml`
- OpenVLA-OFT + GRPO: `maniskill_grpo_openvlaoft.yaml`

### LIBERO
- OpenVLA-OFT + PPO: `libero_10_ppo_openvlaoft.yaml`
- OpenVLA-OFT + GRPO: `libero_10_grpo_openvlaoft.yaml`

### Key YAML fields
- `actor.model.lora_path`: path to LoRA weights for finetuning
- `runner.task_type`: `"embodied"`
- `cluster.component_placement`: GPU allocation for actor, rollout, env
- `rollout.model_dir`: path to HF model
- `actor.checkpoint_load_path`: initial checkpoint
- `actor.checkpoint_save_path`: where to save
- `actor.tokenizer.tokenizer_model`: tokenizer path
- `runner.logger.log_path`: log directory
- `runner.max_epochs`, `runner.max_steps`: training duration
- `actor.global_batch_size`, `actor.micro_batch_size`: batch sizes
- `Other YAML fields`: https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/user/yaml.html

---

## 6. Algorithms (Embodied)

### PPO (embodied only)

```yaml
algorithm:
  require_values: True
  normalize_advantages: True
  group_size: 1
  adv_type: embodied_gae
  loss_type: embodied_ppo
  loss_agg_func: "token-mean"
  micro_batch_size: 20
  global_batch_size: 80
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  value_clip: 0.2
  gamma: 0.99
  gae_lambda: 0.95
```

### GRPO (embodied)

```yaml
algorithm:
  adv_type: embodied_grpo
  loss_type: embodied_grpo
  loss_agg_func: "token-mean"
  group_size: 8
  normalize_advantages: True
```

---

## 7. Environment Details

### ManiSkill3
- **Task:** robotic arm grasping, Put-on-Plate
- **Observation:** RGB 224×224
- **Action:** 7D (x, y, z, roll, pitch, yaw, gripper)

### LIBERO
- **Task:** household manipulation (pick-place, stacking, drawers, spatial)
- **Observation:** RGB 224×224
- **Action:** 7D continuous (x, y, z, roll, pitch, yaw, gripper)
- **Suites:** Spatial, Goal, Object, Long

### Data structure
- Images: `[batch_size, 3, 224, 224]`
- Actions: discrete tokens → normalized continuous
- Rewards: step-level from task completion

---

## 8. Evaluation

Scripts live in `examples/embodiment/`. Entry point: `eval_embodied_agent.py`

### ManiSkill3 OOD evaluation
- Twelve OOD tasks (texture/object/position variants)
- Config: `env.eval` must be `maniskill_ood_template`
- Set `actor.model.ckpt_path` to RL checkpoint

### LIBERO evaluation
- `CONFIG_NAME`: e.g. `libero_goal_grpo_openvlaoft.eval`
- Set `LIBERO_REPO_PATH`, `rollout.model_dir`, `actor.checkpoint_load_path`
- Set `actor.model.ckpt_path` for RL checkpoint
- Env vars: `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`

### Metrics output
- `eval/env_info/success_once` - success rate (at least once per episode)
- `eval/env_info/return`
- `eval/env_info/episode_len`
- `eval/env_info/success_at_end`
- `eval/env_info/task_*_success` - success rate for each task

---

## 9. Visualization & Monitoring

### TensorBoard

```bash
tensorboard --logdir ./logs --port 6006
```

### Video logging

```yaml
video_cfg:
  save_video: True
  info_on_video: True
  video_base_dir: ./logs/video/train
```

### WandB

```yaml
runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "test_openvla"
    logger_backends: ["wandb"] # tensorboard, wandb, swanlab
```

---

## 10. Programming Flow (Embodied)

**Entry point:** `examples/embodiment/train_embodied_agent.py`

**High-level flow:**
1. Cluster + HybridComponentPlacement from config
2. Create actor (EmbodiedFSDPActor), rollout (MultiStepRolloutWorker), env (EnvWorker)
3. `EmbodiedRunner.init_workers()` then `runner.run()`

**Training loop:**
- `update_rollout_weights()` - sync actor ↔ rollout
- `generate_rollouts()` - env.interact + rollout.generate (pipelined)
- `actor.compute_advantages_and_returns()`
- `actor.run_training()`

---

## 11. Benchmark Results (Reference)

### ManiSkill3 (Put-on-Plate OOD)
- PPO-OpenVLA: ~82% average (Vision 82%, Semantic 80.6%, Position 89.3%)
- PPO-OpenVLA-OFT: ~64.5% (faster convergence, 24h vs 48h)
- GRPO-OpenVLA: ~75.5%
- GRPO-OpenVLA-OFT: ~61.5%
- rl4vla baseline: 76.1%

### LIBERO (OpenVLA-OFT + GRPO)
- SFT one-shot: ~34.4% avg
- RLinf fine-tuned: ~97.9% (Spatial 99%, Goal 99%, Object 99%, Long 94.4%)

---

## 12. Files & Directories

```
examples/embodiment/
  config/                 - YAML configs (incl. crl_experiment/)
  train_embodied_agent.py
  run_embodiment.sh
  eval_embodied_agent.py

examples/crl_experiment/
  run_embodiment_er.sh
  run_embodiment_der.sh
  run_embodiment_sequential.sh
  run_embodiment_ewc.sh
  run_embodiment_multitask.sh
  run_embodiment_sequential_reorder.sh
  run_embodiment_weight_merge.sh
  run_embodiment_slca.sh
  run_embodiment_simple_cnn.sh
  eval_embodiment.sh
  eval_embodiment_lora_scale.sh
  common_functions.sh

rlinf/custom/
  libero_trajectory_dataset.py
  logits_precompute_worker.py
  loss.py
  random_action_rollout_worker.py
  rlds_logits_precompute_worker.py
  simple_cnn_utils.py
```
