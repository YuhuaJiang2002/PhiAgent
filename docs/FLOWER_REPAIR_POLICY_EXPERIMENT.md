# Learned flower-repair routing experiment

Evidence date: 2026-08-11. Overall status: **PARTIAL**.

## Question and scope

This experiment tests whether the agentic flower-replacement workflow can learn
from prior real-video candidate evaluations. It does not fine-tune MiniMax-H3 or
claim real-robot manipulation. The learned component is a small ridge utility
ranker that maps the raw candidate's seven diagnostic scores and one of four
bounded repair recipes to predicted post-repair utility.

The evaluator now applies a capability non-regression contract to both the
default five-round audit and the optional `--repair-policy` route. The learned
route evaluates the raw candidate, ranks the four bounded recipes, and renders
candidates until one passes the measured contract. Failed predictions cannot
enter the final pool. The inference module is standard-library-only; NumPy is
loaded only by the training script.

## Current v2 correction: non-regression is a hard constraint

The v1 aggregate-utility conclusion below is superseded. It selected a candidate
whose background and flower proxies rose while explicit motion fell from
0.57002 to 0.48194 and EPL minimum fell from 0.51675 to 0.44953. Averaging those
metrics allowed safety gains to hide action degradation.

The current immutable run is
`outputs/flower-repair-policy/20260810T205055Z-e65420b0`. Its training target is
now a non-regression-constrained utility. Motion preservation, EPL minimum,
temporal consistency, and robot identity may each regress by at most 0.01;
subject replacement may regress by at most 0.02. Any larger regression receives
a hard training penalty, and measured online evaluation independently rejects
the candidate even if the learned ranker predicts it incorrectly.

Leave-one-action-out v2 results cover the same nine action/campaign groups. The
constrained oracle is selected in 8/9 cases; the one miss has only 0.000053
utility regret. The learned first choice passes non-regression in 9/9 groups,
the guarded final pass rate is 9/9, and mean candidate count remains two rather
than five. Maximum selected regressions are:

| Protected capability | Maximum regression | Allowed |
| --- | ---: | ---: |
| Explicit motion | 0.004713 | 0.010000 |
| EPL phase minimum | 0.005250 | 0.010000 |
| Temporal consistency | 0.000463 | 0.010000 |
| Robot identity | 0.000350 | 0.010000 |
| Subject replacement | 0.000004 | 0.020000 |

The matched `inspect-flower` replay uses a checkpoint trained only on insert and
handover. It selects `face-safe-background-lock` in one attempt:

| Metric | Raw | v1 aggregate winner | v2 non-regression |
| --- | ---: | ---: | ---: |
| Mean proxy utility | 0.53572 | 0.78422 | 0.67760 |
| Background lock | 0.00061 | 1.00000 | 1.00000 |
| Flower/object lock | 0.00045 | 1.00000 | 0.00052 |
| Robot identity | 0.70824 | 0.71250 | 0.71183 |
| Explicit control motion | 0.57002 | 0.48194 | 0.56575 |
| Temporal consistency | 0.95407 | 0.94592 | 0.95363 |
| EPL minimum | 0.51675 | 0.44953 | 0.51159 |

Relative to v1, v2 recovers 0.08381 motion and 0.06206 EPL. It deliberately
does not restore source flowers because both flower-restoration recipes violate
the action/EPL/subject contract. The strict result remains rejected and records
`REGENERATE_WORLD_MODEL_CANDIDATE`; no available post-processing recipe solves
both flower preservation and action fidelity.

A stress replay intentionally loads the old v1 checkpoint. The hard guard
rejects its first choice for 0.07808 excess motion regression, rejects its
second choice for 0.16042 excess motion regression, and only then admits the
non-regressing background-lock candidate. This establishes that the online
guard remains effective when the learned ranker is wrong.

The current 124-frame, 1280x262, 24 FPS comparison is
`demo-nonregression-v2c/heldout-inspect-nonregression-guard.mp4`, SHA-256
`d89a0cee9c130b52e1093bea5a07936df324e65f9c09e9b0386f3c5b4a0c7f1f`.
Its four columns show the real source, raw world model, superseded aggregate
winner, and current non-regression result. Poster and six-time storyboard review
passed after encoding.

## Historical v1 data and result (superseded)

## Data and leakage control

The immutable run is
`outputs/flower-repair-policy/20260810T204236Z-8885faf5`. It collects 45 cached
candidates from nine real-video groups: three flower actions (`insert-flower`,
`handover-flower`, and `inspect-flower`) across three MiniMax-H3 campaigns, with
five fixed repair recipes per group. Every row keeps its source evolution JSON,
content hash, raw scorecard, repair parameters, target utility, and encoded
features. The dataset JSONL SHA-256 is
`c2bd03f419cb8f5971975c7b53ff013dd1eaf836c6537044b7165eae47b55ad4`.

Evaluation uses leave-one-action-out folds. Each fold removes all 15 candidates
for one action before training and tests recipe selection on the excluded
action. This prevents direct action-label leakage. It does not prevent
same-scene or same-repair-family leakage; all folds use one source scene and the
same four available repair implementations.

## Historical v1 measured result

Across the nine held-action action-by-campaign groups, the learned policy
selected the measured best repair in 9/9 cases. Mean oracle regret is zero and
candidate-utility RMSE is 0.01161. Relative to the existing generic first repair
(`tracked-mask-background-lock`), selected utility improves by 0.14119 on
average; relative to the raw world-model candidate it improves by 0.28423.
Inference evaluates two candidates rather than five, a 60% reduction in
candidate render/evaluation count.

These are narrow offline routing results. The utility target is a deterministic
mean of preservation, identity, motion, and temporal proxies; it is not human
preference, causal contact, or physics.

## Historical v1 held-action replay and demo

The `inspect-flower` fold checkpoint was trained only on `insert-flower` and
`handover-flower`. The evaluator replay at
`outputs/flower-repair-policy/20260810T204236Z-8885faf5/heldout-inspect-v1`
first measured the raw real-scene candidate, predicted
`face-safe-plus-flower-restore`, and rendered only that repair. Its result
exactly matches the previously measured oracle candidate.

| Metric | Raw | Learned selection |
| --- | ---: | ---: |
| Mean proxy utility | 0.53572 | 0.78422 |
| Background lock | 0.00061 | 1.00000 |
| Flower/object lock | 0.00045 | 1.00000 |
| Robot identity | 0.70824 | 0.71250 |
| Explicit control motion | 0.57002 | 0.48194 |
| Temporal consistency | 0.95407 | 0.94592 |
| EPL minimum | 0.51675 | 0.44953 |

The labelled 124-frame, 1248x316, 24 FPS, 5.167-second comparison is
`demo-heldout-inspect-v1/heldout-inspect-before-after.mp4`, SHA-256
`94c6e5221324762db9df98e5b893f113d7324d7acdb5aa541e601fb076cb9065`.
The poster and six-sample storyboard were visually inspected after encoding.

## Current honest conclusion and next gate

The current experiment establishes a learned and hard-guarded workflow that
does not allow aggregate proxy gains to hide catastrophic action loss. It does
not establish a better video generator or successful flower inspection. The
v2 candidate preserves the raw action within tolerance and locks the background,
but flower/object lock remains 0.00052 and absolute motion/EPL/identity remain
below their task thresholds. It is evidence that the repair set is infeasible,
not evidence of task success.

The next experiment should collect multiple camera scenes, robot embodiments,
stem-instance tracks, and human semantic reviews, then use scene-grouped rather
than action-only held-out evaluation. Learned routing should fall back to the
full audit whenever predicted uncertainty is high or a required hard gate fails.

## Reproduction

```bash
.venv/bin/python scripts/train_flower_repair_policy.py \
  --experiment-root outputs/flower-repair-policy \
  --alpha 0.01 --seed 20260811 --regression-penalty 2.0 \
  --motion-regression-tolerance 0.01 --epl-regression-tolerance 0.01 \
  --temporal-regression-tolerance 0.01 --identity-regression-tolerance 0.01 \
  --subject-regression-tolerance 0.02

.venv/bin/python scripts/evaluate_minimax_h3_flower_validation.py \
  --source SOURCE.mp4 --raw-h3 RAW.mp4 --motion-reference CONTROL.mp4 \
  --robot-reference ROBOT.png --anchor-mask MASK.png \
  --backend-metadata METADATA.json --output-dir NEW_UNIQUE_RUN \
  --action-override --repair-policy HOLDOUT_POLICY.json

.venv/bin/python scripts/build_flower_repair_policy_demo.py \
  --evolution NEW_UNIQUE_RUN/evolution.json \
  --regressed-evolution SUPERSEDED_V1_RUN/evolution.json \
  --output-dir NEW_UNIQUE_DEMO_DIR
```
