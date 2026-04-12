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

import torch
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.model_load_info import log_embodied_driver_inventory
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


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

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()
        log_embodied_driver_inventory(self.cfg)

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
        actor_futures.wait()
        rollout_futures.wait()

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
        if opd_bc_steps > 0 and self.cfg.algorithm.adv_type != "embodied_opd":
            print(
                "[OPD] Warning: opd_bc_steps > 0 but adv_type is not embodied_opd; "
                "BC still runs and opd_teacher_model_path will be set, but RL will not use OPD rewards.",
                flush=True,
            )
        if opd_bc_steps > 0:
            with self.timer("opd_bc_warmup"):
                bc_futures = self.actor.run_opd_bc_warmup(opd_bc_steps)
                bc_metrics_list = bc_futures.wait()
                bc_metrics = {f"opd_bc/{k}": v for k, v in bc_metrics_list[0].items()}
                self.metric_logger.log(bc_metrics, step=0)
            teacher_root = os.path.join(
                self.cfg.runner.logger.log_path,
                "opd_bc_teacher",
            )
            os.makedirs(teacher_root, exist_ok=True)
            teacher_actor_path = os.path.join(teacher_root, "actor")
            save_futures = self.actor.save_checkpoint(teacher_actor_path, 0)
            save_futures.wait()
            self.actor.set_opd_teacher_model_path(teacher_actor_path).wait()
            with open_dict(self.cfg.algorithm):
                self.cfg.algorithm.opd_teacher_model_path = teacher_actor_path

            # Same eval protocol as the student: sync post-BC weights to rollout, then env.evaluate + rollout.evaluate.
            if self.cfg.algorithm.get("opd_eval_teacher_after_bc", True):
                with self.timer("opd_teacher_eval"):
                    self.update_rollout_weights()
                    teacher_eval_metrics = self.evaluate()
                    teacher_eval_metrics = {
                        f"eval_teacher/{k}": v
                        for k, v in teacher_eval_metrics.items()
                    }
                    self.metric_logger.log(data=teacher_eval_metrics, step=0)

        start_step = self.global_step
        for _step in tqdm(range(start_step, self.max_steps), ncols=120):
            if (
                _step % self.cfg.runner.val_check_interval == 0
                and self.cfg.runner.val_check_interval > 0
            ):
                with self.timer("eval"):
                    self.update_rollout_weights()
                    eval_metrics = self.evaluate()
                    eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    self.metric_logger.log(data=eval_metrics, step=_step)

            with self.timer("step"):
                with self.timer("rollout"):
                    # Syncs model weights from actor to rollout
                    self.update_rollout_weights()
                    # Generates rollouts using the updated rollout model
                    self.generate_rollouts()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_futures = self.actor.compute_advantages_and_returns()
                    actor_rollout_metrics = actor_futures.wait()

                # actor training.
                with self.timer("actor_training"):
                    is_last_step = (_step == self.max_steps - 1)
                    actor_training_futures = self.actor.run_training(is_last_step=is_last_step)
                    actor_training_metrics = actor_training_futures.wait()

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

            rollout_metrics = {
                f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
            }
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            training_metrics = {
                f"train/{k}": v for k, v in actor_training_metrics[0].items()
            }
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

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
