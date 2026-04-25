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

from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Worker
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.eval_results_csv import append_eval_results_row
from rlinf.utils.metric_utils import compute_evaluate_metrics


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: DictConfig,
        rollout: Worker,
        env: Worker,
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_loger = MetricLogger(cfg)

    def evaluate(self):
        aggregated_eval_metrics = []

        for _ in range(self.cfg.algorithm.eval_rollout_epoch):
            env_futures = self.env.evaluate()
            rollout_futures = self.rollout.evaluate()
            env_futures.wait()
            rollout_results = rollout_futures.wait()
            eval_metrics_list = [
                results for results in rollout_results if results is not None
            ]
            eval_metrics = compute_evaluate_metrics(eval_metrics_list)
            aggregated_eval_metrics.append(eval_metrics)

        final_metrics = {}
        for k in aggregated_eval_metrics[0].keys():
            vals = [m[k] for m in aggregated_eval_metrics]
            if k.endswith("_success_total"):
                # Per-epoch value is an episode count; sum across eval_rollout_epoch (mean was wrong).
                final_metrics[k] = float(sum(vals))
            elif k.endswith("_success") and "/task_" in k:
                # Weighted success rate: sum(rate_i * n_i) / sum(n_i), not mean(rate_i).
                total_key = f"{k}_total"
                totals = [m[total_key] for m in aggregated_eval_metrics]
                den = float(sum(totals))
                if den > 0:
                    final_metrics[k] = sum(
                        r * t for r, t in zip(vals, totals, strict=True)
                    ) / den
                else:
                    final_metrics[k] = -1.0
            else:
                final_metrics[k] = sum(vals) / len(vals)
        return final_metrics

    def run(self):
        eval_metrics = self.evaluate()
        try:
            append_eval_results_row(eval_metrics, self.cfg)
        except Exception as e:  # noqa: BLE001
            print(
                f"[Eval] append_eval_results_row failed (continuing without CSV): {e}",
                flush=True,
            )
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.metric_loger.log(step=0, data=eval_metrics)
        print(f"{eval_metrics=}")

        self.metric_loger.finish()
