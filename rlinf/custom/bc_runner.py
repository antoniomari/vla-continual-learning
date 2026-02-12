import os

from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.custom.bc_only_fsdp_actor_worker import BCOnlyFSDPActor
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger


class BCOnlyRunner:
    """Runner for BC-only training, matching the structure of EmbodiedRunner."""

    def __init__(self, cfg: DictConfig, actor: BCOnlyFSDPActor):
        self.cfg = cfg
        self.actor = actor

        self.global_step = 0
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

    def init_workers(self):
        """Initialize the actor worker."""
        self.actor.init_worker().wait()

    def run(self):
        """Main training loop."""
        start_step = self.global_step

        for epoch in tqdm(range(start_step, self.max_steps), ncols=120, desc="Epochs"):
            with self.timer("step"):
                # Run one epoch of training
                with self.timer("bc_training"):
                    training_futures = self.actor.run_training()
                    training_metrics = training_futures.wait()

                self.global_step += 1

                # Check if we should save checkpoint
                if (epoch + 1) % self.cfg.runner.save_interval == 0:
                    self._save_checkpoint()

            # Collect and log metrics
            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            training_metrics = {f"train/{k}": v for k, v in training_metrics[0].items()}

            self.metric_logger.log(time_metrics, epoch)
            self.metric_logger.log(training_metrics, epoch)

        self.metric_logger.finish()

    def _save_checkpoint(self):
        """Save model checkpoint."""
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        save_futures = self.actor.save_checkpoint(actor_save_path, self.global_step)
        save_futures.wait()

    def set_max_steps(self):
        """Set maximum training steps (epochs in this case)."""
        self.num_steps_per_epoch = 1  # One epoch = one step
        self.max_steps = self.cfg.runner.get("max_epochs", 10)

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step
