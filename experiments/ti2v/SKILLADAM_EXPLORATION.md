# Exploring SkillAdam with PhiAgent for TI2V

We integrated SkillAdam as a reusable-skill optimizer for robot video generation, then tested whether PhiAgent's contact-event sampling improved the videos selected from its outputs. The full optimization loop ran successfully, but the combined sampling variant reduced three trajectory metrics. The current research direction moves the intervention into task-constrained prompt repair while keeping video selection fixed.

## What the integration preserves

[SkillAdam](https://arxiv.org/abs/2609.08944v1) uses memory of problems and previous solution attempts to stabilize skill updates, and a budget driven by case-level improvement volatility to control revision size. Our TI2V integration uses the authors' `SkillAdamRunner` at commit [`72ef9ba48bbadc059c8a2d099f1b7d531fc8288f`](https://github.com/ruc-datalab/SkillAdam/tree/72ef9ba48bbadc059c8a2d099f1b7d531fc8288f), with its core source unchanged.

The runner receives an explicit shared initial skill through its supported interface, so Stage0 skill generation is bypassed. Strict diffs, Momentum tools, case-level improvement variation, the EMA edit budget, validation decisions, and atomic checkpoints remain active. Checkpoint writing was observed; interruption/resume was not independently tested.

Our adapter registers `ewm_ti2v`, translates remote video rollouts into case-matched official metrics, and supplies the domain acceptance rule. The pinned registry has no public extension API, so registration adds an explicit private-registry entry and package search path. These changes are an adaptation boundary, not a modification of the upstream optimizer.

| Component | Implementation and responsibility |
|---|---|
| SkillAdam optimizer | `scripts/run_ti2v_skill_optimization.py`: calls `SkillAdamRunner.run(cases=..., initial_skill=...)` |
| TI2V adapter | `integrations/skilladam_ti2v/adapter.py`: requires all five finite metrics and matching official score evidence |
| Execution backend | `integrations/skilladam_ti2v/backend.py`: executes skills through Qwen and MiniMax, records trajectories, and transports Momentum tool calls |
| Update acceptance | No decrease in BLEU, CLIP, hsd, dyn, or ndtw, with strict improvement in at least one metric on the same validation batch |
| Video selection | Original five visible gates; a candidate must pass every gate and repair an explicit parent failure |

## Frozen experiment

Run `20260916T032105Z` used two optimization rounds, each with two training and two validation cases. The 20 development cases were grouped into 12 training, four validation, and four optimizer-isolated audit cases using split seed 20260916. All three seeds for a case stayed together. These cases had already been used for development, including the audit group.

Optimization used generation seed 20260910; final evaluation covered all 20 cases with seeds 20260910, 20260911, and 20260912. SkillAdam and the separate PhiAgent bounded optimizer shared the initial skill, data split, number of rounds, and generation caps. Each method allowed at most 108 new video requests, 1,200 auxiliary requests, and eight hours. Actual calls, cache reuse, failures, and warmup were recorded separately from logical rollout usage.

The edit budget used ndtw with `base=4`, `minimum=1`, `beta=0.9`, and `v_max=0.01`, with at most two patch-format attempts. This scalar controlled edit size only; acceptance still used all five metrics. Qwen3.8-27B executed and revised skills, and MiniMax-H3 generated eight-second videos. Exact revisions and settings are in [model-pins.json](configs/model-pins.json).

Training and validation scores were available to the optimizer. Reference futures stayed at the scoring service; generation and per-video selection did not read official scores. Final skills were frozen before the 60-output evaluation, and final scores were not fed back into that optimization run.

## What SkillAdam learned

SkillAdam accepted both updates. The first improved all five validation means; the second improved four while dyn stayed unchanged. The resulting [frozen skill](evidence/skilladam-final-skill.md) explicitly prohibits unrequested motion after the intended endpoint: pickup ends in a stable hold, placement ends with the object resting, and neither implies a later arm retraction. It also avoids asserting an invisible grasp or support state.

The final skill SHA-256 is `1b9426cd4089ca58995ca59330fb9a352219dfd587fe614f81e43a146bc8d1a4`. This makes the explored artifact inspectable rather than reducing the integration to a baseline name. The [evidence export](evidence/skilladam-exploration.json) preserves both update decisions and the server-produced metrics, with hashes of the collected sources.

The separate PhiAgent optimizer proposed at most two whole-line changes per round. Both updates were rejected: the first regressed four protected metrics, and the second produced no strict improvement. Its final skill therefore remained the common initial skill. It shares the execution and evaluation infrastructure but uses a distinct update algorithm.

## Completed SkillAdam + PhiAgent experiment

We crossed the optimizer with two frame-sampling policies for the existing video gates. Uniform sampling used 16 frames. Event sampling used 12 uniform positions plus four positions near visible contact changes, falling back to uniform sampling when an event could not be located. Each sampling pair reused exactly the same generated candidates; additional visual calls were recorded separately.

| Optimizer and sampling | BLEU ↑ | CLIP ↑ | hsd ↑ | dyn ↑ | ndtw ↑ |
|---|---:|---:|---:|---:|---:|
| PhiAgent bounded updates + uniform | 0.211015 | 89.192610 | 0.325833 | 0.245550 | 0.331750 |
| PhiAgent bounded updates + event (prespecified primary) | 0.217046 | 89.317654 | 0.320183 | 0.236467 | 0.327150 |
| SkillAdam + uniform | 0.199855 | 89.133862 | 0.335967 | 0.259583 | 0.329717 |
| SkillAdam + PhiAgent event sampling | 0.205104 | 88.906126 | 0.319867 | 0.249867 | 0.312417 |

All rows contain 60/60 scored outputs under the same per-video caption-seed protocol. Combining SkillAdam with event sampling increased BLEU but reduced CLIP, hsd, dyn, and ndtw relative to SkillAdam with uniform sampling. Event sampling also reduced all three trajectory means under the bounded optimizer. This completed combination provides negative evidence; it is not the currently promoted candidate. Its historical record does not authorize further evaluator work.

## How this informs the relation-repair direction

SkillAdam's accepted endpoint constraint and the later repair diagnosis identify a concrete issue to investigate: a generic motion template can introduce an action absent from the task. In the original relation vocabulary, some proposed edits added withdrawal even though the task only requested placement. The current compiler therefore requires the relation's literal task precondition, and rejects unsupported actions before generation.

The complementary roles are clear: SkillAdam updates a reusable skill across training cases; relation repair specializes a bounded part of that skill using the current task and the parent video's failure evidence. A prospective composition would freeze the SkillAdam skill, pass it as `skill` to `method.propose`, generate under matched settings, and retain the original selector. The interface accepts a caller-supplied skill, but this particular composition has not completed an end-to-end experiment. Run `20260918T095225Z` must retain its own frozen parent identity and cannot be relabeled as SkillAdam plus repair.

A direct next comparison would hold the selector fixed and cross parent skill (common initial / frozen SkillAdam) with repair (off / on). That separates gains from reusable skill learning, task-specific repair, and their interaction. It would require a new remote run, a compatible declared repair slot, and a frozen protocol. No such result is included here.

## Evidence boundary

Small-batch update acceptance did not establish full-set superiority. In the complete comparison, SkillAdam exceeded the VideoWeaver adaptation only in hsd, while historical Ours-v2 exceeded it in BLEU and hsd means. No completed method leads on all five metrics. Full results and the historical paired uncertainty analysis are in [RESULTS.md](RESULTS.md) and [METHOD.md](METHOD.md). The completed [paired repair experiment](PAIRED_RESULTS.md) does not establish a benefit over its matched parent. Combining repair with a frozen SkillAdam skill remains untested; its result cannot be inferred from either experiment.
