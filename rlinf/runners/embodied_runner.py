# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import torch
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.model_load_info import log_embodied_driver_inventory
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import cfg_show_progress_bar, check_progress
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _format_rollout_env_success(metrics: dict) -> str:
    """One-line summary of env success rates from compute_rollout_metrics."""
    parts = []
    for k, v in sorted(metrics.items()):
        if not k.startswith("env_info/"):
            continue
        if "success" not in k.lower():
            continue
        try:
            parts.append(f"{k.replace('env_info/', '')}={float(v):.3f}")
        except (TypeError, ValueError):
            continue
    return ", ".join(parts)


def _format_eval_success(metrics: dict) -> str:
    """Format task success keys from compute_evaluate_metrics (env_info/task_*_success)."""
    parts = []
    for k, v in sorted(metrics.items()):
        if "success" in k and "task_" in k:
            try:
                parts.append(f"{k.split('/')[-1]}={float(v):.3f}")
            except (TypeError, ValueError):
                continue
    return ", ".join(parts)


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: EmbodiedFSDPActor,
        rollout: MultiStepRolloutWorker,
        env: EnvWorker,
        critic=None,
        reward=None,
        run_timer=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        print(f"Runner max_steps: {self.max_steps}")

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)
        # OPD: BC warmup runs in init_workers after actor init, before rollout/env (no sim needed).
        self._opd_bc_warmup_completed = False
        # W&B requires monotonically increasing step; teacher eval must not reuse step=0 after BC
        # logged steps 0..N-1 or W&B drops those metrics.
        self._opd_bc_wandb_step_cursor = 0
        # Added to RL step index for W&B after OPD BC (+1 if teacher eval logged at cursor).
        self._wandb_rl_step_offset = 0

    def _run_teacher_like_training_steps(self, num_steps: int, metric_prefix: str) -> None:
        """Run rollout->advantage->actor update loop for teacher warmup."""
        if num_steps <= 0:
            return
        _show_pbar = cfg_show_progress_bar(self.cfg)
        for _step in tqdm(
            range(num_steps),
            ncols=120,
            disable=not _show_pbar,
            desc=f"{metric_prefix} training",
        ):
            with self.timer("teacher_step"):
                with self.timer("teacher_rollout"):
                    self.update_rollout_weights()
                    self.generate_rollouts()

                with self.timer("teacher_cal_adv_and_returns"):
                    actor_futures = self.actor.compute_advantages_and_returns()
                    actor_rollout_metrics = actor_futures.wait()

                with self.timer("teacher_actor_training"):
                    actor_training_futures = self.actor.run_training(
                        is_last_step=(_step == num_steps - 1)
                    )
                    actor_training_metrics = actor_training_futures.wait()

            time_metrics = self.timer.consume_durations()
            rollout_metrics = {
                f"{metric_prefix}/rollout/{k}": v
                for k, v in actor_rollout_metrics[0].items()
            }
            time_metrics = {f"{metric_prefix}/time/{k}": v for k, v in time_metrics.items()}
            training_metrics = {
                f"{metric_prefix}/train/{k}": v
                for k, v in actor_training_metrics[0].items()
            }
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

    def _run_opd_rl_teacher_warmup_and_save_teacher(self) -> None:
        """Teacher warmup with pure GRPO rollout training, then freeze as OPD teacher."""
        teacher_steps = int(self.max_steps)
        if teacher_steps <= 0:
            raise ValueError("[OPD] rl_teacher enabled but max_steps <= 0.")

        orig_adv = self.cfg.algorithm.adv_type
        orig_loss = self.cfg.algorithm.loss_type
        try:
            print(
                "[OPD] rl_teacher=True: pretraining teacher with pure GRPO before OPD student RL "
                f"for {teacher_steps} step(s).",
                flush=True,
            )
            self.actor.set_algorithm_mode("embodied_grpo", "embodied_grpo").wait()
            self.rollout.set_algorithm_adv_type("embodied_grpo").wait()
            with open_dict(self.cfg.algorithm):
                self.cfg.algorithm.adv_type = "embodied_grpo"
                self.cfg.algorithm.loss_type = "embodied_grpo"

            with self.timer("opd_teacher_rl_warmup"):
                self._run_teacher_like_training_steps(
                    num_steps=teacher_steps, metric_prefix="opd_teacher_rl"
                )
            self._opd_bc_wandb_step_cursor = teacher_steps
        finally:
            self.actor.set_algorithm_mode(orig_adv, orig_loss).wait()
            self.rollout.set_algorithm_adv_type(orig_adv).wait()
            with open_dict(self.cfg.algorithm):
                self.cfg.algorithm.adv_type = orig_adv
                self.cfg.algorithm.loss_type = orig_loss

        teacher_root = os.path.join(
            self.cfg.runner.logger.log_path,
            "opd_bc_teacher",
        )
        os.makedirs(teacher_root, exist_ok=True)
        teacher_actor_path = os.path.join(teacher_root, "actor")
        self.actor.save_checkpoint(teacher_actor_path, 0).wait()
        self.actor.set_opd_teacher_model_path(teacher_actor_path).wait()
        self.rollout.set_opd_teacher_model_path(teacher_actor_path).wait()
        with open_dict(self.cfg.algorithm):
            self.cfg.algorithm.opd_teacher_model_path = teacher_actor_path
        self._opd_bc_warmup_completed = True

    def _refresh_wandb_rl_step_offset(self) -> None:
        """RL metrics must log at steps after BC (0..N-1) and optional teacher eval (at cursor)."""
        if not self._opd_bc_warmup_completed:
            self._wandb_rl_step_offset = 0
            return
        off = self._opd_bc_wandb_step_cursor
        if self.cfg.algorithm.get("opd_eval_teacher_after_bc", True):
            off += 1  # teacher eval used step == cursor
        self._wandb_rl_step_offset = off

    def init_workers(self):
        # Actor first, then rollout+env.
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()
        opd_bc_steps = self.cfg.algorithm.get("opd_bc_steps", 0)
        rl_teacher = bool(self.cfg.algorithm.get("rl_teacher", False))
        if rl_teacher:
            if opd_bc_steps > 0:
                print(
                    "[OPD] rl_teacher=True: ignoring opd_bc_steps and using GRPO teacher warmup.",
                    flush=True,
                )
            self._run_opd_rl_teacher_warmup_and_save_teacher()
        elif opd_bc_steps > 0:
            self._run_opd_bc_warmup_and_save_teacher(opd_bc_steps)
        if self._opd_bc_warmup_completed and self.cfg.algorithm.get(
            "opd_eval_teacher_after_bc", True
        ):
            with self.timer("opd_teacher_eval"):
                print(
                    "[OPD] Teacher eval: pushing post-BC teacher weights to rollout "
                    "(student restored to pre-BC base right after eval), then Libero eval.",
                    flush=True,
                )
                self.update_rollout_weights()
                teacher_eval_metrics = self.evaluate()
                te_line = _format_eval_success(teacher_eval_metrics)
                if te_line:
                    print(f"[OPD] Teacher eval (sim success): {te_line}", flush=True)
                teacher_eval_metrics = {
                    f"eval_teacher/{k}": v for k, v in teacher_eval_metrics.items()
                }
                te_step = self._opd_bc_wandb_step_cursor
                print(
                    f"[OPD] Logging eval_teacher to metrics ({len(teacher_eval_metrics)} keys) "
                    f"at wandb step={te_step} (after opd_bc steps 0..{te_step - 1}).",
                    flush=True,
                )
                self.metric_logger.log(data=teacher_eval_metrics, step=te_step)
            self._restore_student_after_bc()
        log_embodied_driver_inventory(self.cfg)

    def _run_opd_bc_warmup_and_save_teacher(self, opd_bc_steps: int) -> None:
        if self.cfg.algorithm.adv_type != "embodied_opd":
            print(
                "[OPD] Warning: opd_bc_steps > 0 but adv_type is not embodied_opd; "
                "BC still runs and opd_teacher_model_path will be set, but RL will not use OPD rewards.",
                flush=True,
            )
        # Optional: initialize teacher warmup from an explicit checkpoint path.
        # This path is expected to be a save_checkpoint-style directory
        # (adapter_model.bin for LoRA or model.pt for full-model).
        teacher_init_path = self.cfg.algorithm.get("opd_teacher_init_model_path", None)
        if teacher_init_path:
            print(
                f"[OPD] Initializing BC teacher from opd_teacher_init_model_path={teacher_init_path}",
                flush=True,
            )
            self.actor.restore_student_from_checkpoint(teacher_init_path).wait()
        else:
            print(
                "[OPD] BC teacher warmup starts from actor initialization "
                f"(actor.checkpoint_load_path={self.cfg.actor.checkpoint_load_path}).",
                flush=True,
            )

        with self.timer("opd_bc_warmup"):
            bc_futures = self.actor.run_opd_bc_warmup(opd_bc_steps)
            bc_metrics_list = bc_futures.wait()
            bc_pack = bc_metrics_list[0]
            if isinstance(bc_pack, dict) and "per_step" in bc_pack:
                for step_i, step_m in enumerate(bc_pack["per_step"]):
                    self.metric_logger.log(
                        {f"opd_bc/{k}": v for k, v in step_m.items()},
                        step=step_i,
                    )
                n_bc = len(bc_pack["per_step"])
                self._opd_bc_wandb_step_cursor = n_bc if n_bc > 0 else 1
            else:
                self.metric_logger.log(
                    {f"opd_bc/{k}": v for k, v in bc_pack.items()},
                    step=0,
                )
                self._opd_bc_wandb_step_cursor = 1
        teacher_root = os.path.join(
            self.cfg.runner.logger.log_path,
            "opd_bc_teacher",
        )
        os.makedirs(teacher_root, exist_ok=True)
        teacher_actor_path = os.path.join(teacher_root, "actor")
        save_futures = self.actor.save_checkpoint(teacher_actor_path, 0)
        save_futures.wait()
        self.actor.set_opd_teacher_model_path(teacher_actor_path).wait()
        self.rollout.set_opd_teacher_model_path(teacher_actor_path).wait()
        with open_dict(self.cfg.algorithm):
            self.cfg.algorithm.opd_teacher_model_path = teacher_actor_path
        self._opd_bc_warmup_completed = True
        if isinstance(bc_pack, dict) and bc_pack.get("mean"):
            mean_m = bc_pack["mean"]
            loss = mean_m.get("bc/loss")
            tok_acc = mean_m.get("bc/token_accuracy")
            if loss is not None and tok_acc is not None:
                print(
                    f"[OPD] BC warmup summary: mean bc/loss={float(loss):.4f}, "
                    f"mean bc/token_accuracy={float(tok_acc):.4f} "
                    f"(expert action tokens; sim success follows in teacher eval)",
                    flush=True,
                )
            elif loss is not None:
                print(
                    f"[OPD] BC warmup summary: mean bc/loss={float(loss):.4f} "
                    f"(cross-entropy on expert action tokens; not sim task success)",
                    flush=True,
                )

    def _restore_student_after_bc(self) -> None:
        print(
            "[OPD] Restoring RL student from base model initialization "
            f"(actor.checkpoint_load_path={self.cfg.actor.checkpoint_load_path}).",
            flush=True,
        )
        self.actor.restore_student_to_base_model().wait()
        self.update_rollout_weights()

    def update_rollout_weights(self):
        rollout_futures = self.rollout.sync_model_from_actor()
        actor_futures = self.actor.sync_model_to_rollout()
        actor_futures.wait()
        rollout_futures.wait()
        self.actor.preallocate_memory()

    def generate_rollouts(self):
        env_futures = self.env.interact()
        rollout_futures = self.rollout.generate()
        actor_futures = self.actor.recv_rollout_batch()
        env_futures.wait()
        actor_results = actor_futures.wait()
        rollout_futures.wait()
        if actor_results and self.cfg.runner.get("log_rollout_collection_summary", True):
            stats = next((r for r in actor_results if isinstance(r, dict)), None)
            if stats:
                nt = stats.get("n_rollout_trajectories")
                nc = stats.get("n_chunk_steps")
                nx = stats.get("n_chunk_times_traj")
                if nt is not None and nc is not None and nx is not None:
                    print(
                        "[EmbodiedRunner] Rollout phase: "
                        f"{int(nt)} trajectory slots in batch "
                        f"({int(nc)} policy chunk steps each, "
                        f"{int(nx)} chunk×trajectory cells before shuffle/split). "
                        "Same layout per actor rank if batch is replicated.",
                        flush=True,
                    )

    def evaluate(self):
        env_futures = self.env.evaluate()
        rollout_futures = self.rollout.evaluate()
        env_futures.wait()
        rollout_results = rollout_futures.wait()
        eval_metrics_list = [
            results for results in rollout_results if results is not None
        ]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        opd_bc_steps = self.cfg.algorithm.get("opd_bc_steps", 0)
        rl_teacher = bool(self.cfg.algorithm.get("rl_teacher", False))
        # OPD BC normally finishes in init_workers (before env). Fallback if run() is entered alone.
        if not self._opd_bc_warmup_completed and (opd_bc_steps > 0 or rl_teacher):
            if rl_teacher:
                self._run_opd_rl_teacher_warmup_and_save_teacher()
            else:
                self._run_opd_bc_warmup_and_save_teacher(opd_bc_steps)
            if self.cfg.algorithm.get("opd_eval_teacher_after_bc", True):
                with self.timer("opd_teacher_eval"):
                    print(
                        "[OPD] Teacher eval: pushing post-BC teacher weights to rollout "
                        "(student restored to pre-BC base right after eval), then Libero eval.",
                        flush=True,
                    )
                    self.update_rollout_weights()
                    teacher_eval_metrics = self.evaluate()
                    te_line = _format_eval_success(teacher_eval_metrics)
                    if te_line:
                        print(f"[OPD] Teacher eval (sim success): {te_line}", flush=True)
                    teacher_eval_metrics = {
                        f"eval_teacher/{k}": v
                        for k, v in teacher_eval_metrics.items()
                    }
                    te_step = self._opd_bc_wandb_step_cursor
                    print(
                        f"[OPD] Logging eval_teacher to metrics ({len(teacher_eval_metrics)} keys) "
                        f"at wandb step={te_step} (after opd_bc steps 0..{te_step - 1}).",
                        flush=True,
                    )
                    self.metric_logger.log(data=teacher_eval_metrics, step=te_step)
            self._restore_student_after_bc()

        self._refresh_wandb_rl_step_offset()

        start_step = self.global_step
        _show_pbar = cfg_show_progress_bar(self.cfg)
        for _step in tqdm(
            range(start_step, self.max_steps),
            ncols=120,
            disable=not _show_pbar,
        ):
            _wb_step = _step + self._wandb_rl_step_offset
            if (
                _step % self.cfg.runner.val_check_interval == 0
                and self.cfg.runner.val_check_interval > 0
            ):
                with self.timer("eval"):
                    self.update_rollout_weights()
                    eval_metrics = self.evaluate()
                    eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    self.metric_logger.log(data=eval_metrics, step=_wb_step)

            with self.timer("step"):
                _log_wall = self.cfg.runner.get("log_step_phase_timings", True)
                with self.timer("rollout"):
                    # Syncs model weights from actor to rollout
                    _tw0 = time.perf_counter()
                    self.update_rollout_weights()
                    _tw1 = time.perf_counter()
                    # Generates rollouts using the updated rollout model
                    self.generate_rollouts()
                    _tw2 = time.perf_counter()
                    if _log_wall:
                        print(
                            f"[train step {_step}] driver_wall_s: "
                            f"sync_actor_to_rollout={_tw1 - _tw0:.2f} "
                            f"generate_rollouts={_tw2 - _tw1:.2f} "
                            f"rollout_total={_tw2 - _tw0:.2f}",
                            flush=True,
                        )

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    _ta0 = time.perf_counter()
                    actor_futures = self.actor.compute_advantages_and_returns()
                    actor_rollout_metrics = actor_futures.wait()
                    if _log_wall:
                        print(
                            f"[train step {_step}] driver_wall_s: "
                            f"compute_advantages_and_returns={time.perf_counter() - _ta0:.2f}",
                            flush=True,
                        )

                rm0 = actor_rollout_metrics[0]
                succ_line = _format_rollout_env_success(rm0)
                rew = rm0.get("rewards")
                rew_s = ""
                if rew is not None:
                    try:
                        rew_s = f" mean_reward={float(rew):.4f}"
                    except (TypeError, ValueError):
                        rew_s = ""
                if succ_line or rew_s:
                    print(
                        f"[train step {_step}] rollout:{rew_s} env_success: {succ_line or '(none)'}",
                        flush=True,
                    )

                # actor training.
                with self.timer("actor_training"):
                    is_last_step = (_step == self.max_steps - 1)
                    _tt0 = time.perf_counter()
                    actor_training_futures = self.actor.run_training(
                        is_last_step=is_last_step
                    )
                    actor_training_metrics = actor_training_futures.wait()
                    if _log_wall:
                        print(
                            f"[train step {_step}] driver_wall_s: "
                            f"run_training={time.perf_counter() - _tt0:.2f}",
                            flush=True,
                        )

                if self.cfg.runner.get("log_training_step_summary", True):
                    tm0 = actor_training_metrics[0]
                    pref = [k for k in sorted(tm0.keys()) if k.startswith("actor/")][:5]
                    tparts = []
                    for k in pref:
                        try:
                            tparts.append(f"{k}={float(tm0[k]):.4f}")
                        except (TypeError, ValueError):
                            continue
                    if tparts:
                        print(
                            f"[train step {_step}] train: " + " ".join(tparts),
                            flush=True,
                        )

                self.global_step += 1

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            if self.cfg.runner.get("log_step_phase_timings", True) and time_metrics:
                parts = [f"{k}={v:.2f}s" for k, v in sorted(time_metrics.items())]
                print(
                    f"[train step {_step}] ScopedTimer: " + " ".join(parts),
                    flush=True,
                )

            rollout_metrics = {
                f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
            }
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            training_metrics = {
                f"train/{k}": v for k, v in actor_training_metrics[0].items()
            }
            self.metric_logger.log(rollout_metrics, _wb_step)
            self.metric_logger.log(time_metrics, _wb_step)
            self.metric_logger.log(training_metrics, _wb_step)

        self.metric_logger.finish()

        # Compute and save EWC data if enabled
        if self.cfg.algorithm.get("use_ewc", False):
            ewc_save_path = os.path.join(
                self.cfg.runner.logger.log_path,
                "ewc_data.pt"
            )

            # Delegate EWC saving to the actor worker group (runs on training ranks)
            if hasattr(self.actor, "compute_and_save_ewc_data"):
                futures = self.actor.compute_and_save_ewc_data(ewc_save_path)
                futures.wait()

    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        save_futures = self.actor.save_checkpoint(actor_save_path, self.global_step)
        save_futures.wait()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        # Will be 10 for the current configs
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
