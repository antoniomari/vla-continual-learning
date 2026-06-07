# Working Paper Draft And Plot Plan

## Working Title

Hybrid OPD for Sample-Efficient Continual Adaptation of Vision-Language-Action Models

## One-Line Thesis

SFT-teacher OPD can improve sample efficiency over pure GRPO, but it is unstable unless combined with normalized environment-reward learning; the current best formulation uses a normalized GRPO branch plus a raw, failure-gated SFT-teacher OPD branch.

## Current Story

We study continual adaptation on LIBERO-Spatial. The main comparison is between:

- **GRPO**: environment reward only, normalized advantages.
- **Hybrid SFT-OPD + GRPO**: normalized GRPO advantage plus raw SFT-teacher OPD advantage, gated by task failure.
- **RL-teacher OPD**: stronger-teacher variant to use as an upper-bound / diagnostic when coverage is complete. The paper-facing version should be plain OPD REINFORCE: raw OPD reward, no OPD normalization, and no GRPO/environment reward branch.

Current core evidence is based on RPS32 runs, checkpointed every 25 steps up to 200, using same-task success and held-out success.

> **Task-set note:** the current aggregate plot available in the repo is over tasks **1, 4, 9**. The requested paper note says **1, 3, 9**; before finalizing the paper, confirm whether task 3 is intended or whether this was meant to be task 4.

## Method Formalization

Let \(r_{\mathrm{env}}\) be the sparse environment success reward and let \(m_{\mathrm{opd}}\) be the teacher-student log-probability margin:

\[
m_{\mathrm{opd}}(a \mid s) = \log \pi_{\mathrm{teacher}}(a \mid s) - \log \pi_{\theta}(a \mid s).
\]

The current hybrid objective combines:

\[
A_{\mathrm{env}} = \operatorname{Normalize}_{\mathrm{GRPO}}(r_{\mathrm{env}})
\]

and a gated raw OPD term:

\[
A_{\mathrm{opd}} =
\lambda \cdot \mathbf{1}[r_{\mathrm{env}} \le \tau] \cdot m_{\mathrm{opd}}.
\]

The current setting uses:

- \(\tau = 0\): teacher feedback is active on failed rollouts.
- \(\lambda \in \{0.1, 1.0\}\): teacher strength.
- GRPO branch: normalized.
- OPD branch: raw, not z-scored.
- No replay.

The combined advantage used by the hybrid loss is:

\[
A_{\mathrm{hybrid}} = A_{\mathrm{env}} + A_{\mathrm{opd}}.
\]

## Required Paper Figure Registry

### Figure 1: Aggregate RPS32 Results Across Tasks

![Aggregate RPS32 tasks 1, 4, 9](../visualization/results/plots/rps32_tasks_1_4_9_avg_tasks_then_seed_ci_COMPLETE_GRPO_MATCHED_CLEAN_20260606_105607.png)

This is the main paper plot to keep track of and edit when we refer to "the paper plot".

Source CSV:

`visualization/results/plots/rps32_tasks_1_4_9_avg_tasks_then_seed_ci_COMPLETE_GRPO_MATCHED_20260606_105541.csv`

Plot construction:

- First average raw success values over tasks for each seed.
- Then average over seeds.
- Error band is Gaussian 95% CI across seed-level task averages.
- Current task set is 1, 4, 9; confirm whether the final should instead be 1, 3, 9.

Expected interpretation:

- Hybrid OPD + GRPO with \(\lambda=0.1\) is the most conservative teacher-mixing setting.
- Hybrid OPD + GRPO with \(\lambda=1.0\) can be more aggressive, often improving faster but risking more forgetting.
- GRPO remains the stable reference but is less sample-efficient.

## Candidate Supporting Figures

- Task-specific curves for tasks 1, 4, and 9.
- Wall-clock-time version of the aggregate plot.
- Task 5 teacher diagnostics after the 2-image/no-proprio evaluation finishes.
- Ablation plot comparing raw OPD, group-zscore OPD, and hybrid normalized-GRPO/raw-OPD.
- RL-teacher OPD comparison once coverage is complete.

## RL-Teacher OPD Coverage Audit

For expanding Figure 1 with RL-teacher OPD, current local result tables do **not** have full coverage for three seeds and three tasks.

Available RL-teacher OPD data:

| Variant | Task | Seeds | Checkpoints |
| --- | ---: | --- | --- |
| RL-teacher OPD, raw REINFORCE, no normalization | 1 | 2 only | 0 to 200 |
| RL-teacher OPD, raw REINFORCE, no normalization | 4 | 2 only | 0 to 200 |

There are also older RL-teacher runs with GRPO-style clipped loss and group z-score normalization, but those are diagnostic only and should not be used for the clean RL-teacher comparison unless explicitly separated.

Missing for a three-task, three-seed aggregate:

- Missing task 9 RL-teacher OPD.
- Missing seeds 1 and 3 for tasks 1 and 4.
- If the intended task set is truly 1, 3, 9, then we only have task 1 seed 2, and task 3/task 9 are missing.

Conclusion: RL-teacher OPD should not be added to the aggregate paper plot yet unless we either run the missing jobs or clearly mark it as incomplete / diagnostic-only.

## Current Paper Outline

### Abstract

Briefly state the problem of continual VLA adaptation, the sample inefficiency of sparse-reward RL, and the instability of naive teacher distillation. Summarize Hybrid OPD as normalized environment-gradient learning plus failure-gated teacher-gradient learning.

### Introduction

Motivate continual adaptation in embodied agents. Explain the tension: GRPO learns from true task reward but is sample inefficient; SFT teachers provide dense signal but may forget or be imperfect. The paper asks whether dense teacher feedback can be multiplexed with sparse environment reward.

### Background

Cover GRPO, OPD-style teacher-student log-probability margins, and why normalization matters. Include the observation that chunk/group z-score normalization can help but is not fully principled for action-chunk credit assignment.

### Method

Define Hybrid OPD + GRPO. Emphasize that the environment branch is normalized while the OPD branch remains raw and is scaled by \(\lambda\). The teacher branch is gated by failure using \(\tau=0\).

### Experiments

Use LIBERO-Spatial continual adaptation. Primary tasks: currently 1, 4, 9; confirm whether task 3 replaces task 4. Main metric is same-task success and held-out success over checkpoints. Primary comparison is GRPO vs Hybrid OPD + GRPO.

### Results

Present Figure 1 as the main result. Discuss the sample-efficiency/stability trade-off between \(\lambda=0.1\) and \(\lambda=1.0\). Mention RL-teacher OPD as promising but incomplete for full aggregate comparison.

### Discussion

Interpret the hybrid signal as a way to use teacher feedback when reward is sparse while letting environment reward dominate successful rollouts. Discuss task-5 teacher/eval mismatch diagnostics separately if included.

### Limitations

Current results are limited by incomplete RL-teacher coverage, pending task-set confirmation, and sensitivity to evaluation/preprocessing details.

### Next Steps Before Final Paper

- Confirm the final task set: 1, 4, 9 or 1, 3, 9.
- Regenerate Figure 1 with final paper styling after task-set confirmation.
- Decide whether to run missing RL-teacher OPD seeds/tasks.
- Add task 5 only after the teacher/eval issue is resolved.
- Convert this Markdown plan into the final TeX paper draft.
