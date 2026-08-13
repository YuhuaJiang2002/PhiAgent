# Quantitative agentic-workflow benchmarks

Evidence date: 2026-08-11.

## Bottom line

The current workflow shows large, reproducible gains on retained real-video
proxy tasks, but the conclusion differs by branch:

- **Robot embodiment replacement:** the agent greatly improves object lock,
  image-space motion, EPL phase minimum, and the aggregate proxy score while
  preserving background, replacement coverage, and robot identity. The best
  same-run candidate still passes only 5/7 hard gates, so learned end-to-end
  replacement remains `PARTIAL`. A separate fail-closed 2-D renderer passes all
  39 of its declared structural and temporal gates across 660 frames, but it is
  a deterministic fallback rather than learned or physical robot execution.
- **AC-WM:** final human-gated action success is 2/3, compared with 0/3 for the
  retained MiniMax-H3 baseline and 1/3 for the original single-pass OSCAR
  batch. This is a +66.7 percentage-point gain over H3 and +33.3 points over
  single-pass OSCAR on this one real scene. It remains `PARTIAL`, not a
  general action-control claim.

Scores from different evaluators are never averaged. Every gain below is
computed only within a matched evidence bundle, and later user review
supersedes an earlier automatic or visual pass.

## 1. Replacing the human embodiment with a robot

### Same 660-frame input: single pass versus agent-selected safe round

This is the cleanest within-run ablation. Both candidates use the same real
27.5-second source, robot anchor, masks, evaluator, and thresholds. The
traditional baseline is round 0: one render with no object protection or
feedback-driven repair. The workflow candidate is the safety-valid round
selected after diagnosing failures and changing only the bounded repair
parameters.

| Metric | Single-pass baseline | Agent-selected round | Absolute gain |
| --- | ---: | ---: | ---: |
| Background lock | 1.0000 | 1.0000 | +0.0000 |
| Object lock | 0.0282 | 1.0000 | **+0.9718** |
| Subject replacement | 0.9985 | 0.9984 | -0.0001 |
| Robot identity | 0.9995 | 0.9995 | -0.0000 |
| Motion preservation | 0.0351 | 0.5873 | **+0.5522** |
| Temporal consistency | 0.9029 | 0.9173 | +0.0145 |
| EPL phase minimum | 0.0235 | 0.5455 | **+0.5219** |
| Mean proxy score | 0.5697 | 0.8640 | **+0.2943 (+51.7% relative)** |
| Hard gates passed | 4/7 | 5/7 | +1 gate |

The high aggregate score is useful diagnostic evidence, but it does not erase
failed gates. Motion preservation is 0.5873 against a 0.72 requirement and EPL
minimum is 0.5455 against 0.62, so the candidate is rejected for full
acceptance.

### Matched 124-frame renderer comparison

The second ablation holds the real source window, robot reference, flower
protection, background relock, frame count, and evaluator fixed. It compares
the traditional explicit 2-D pose-rig renderer with a MiniMax-H3 generation
followed by the same bounded agentic repair family.

| Metric | 2-D pose-rig baseline | H3 + agentic repair | Absolute gain | Relative gain |
| --- | ---: | ---: | ---: | ---: |
| Motion preservation | 0.3346 | 0.6185 | **+0.2840** | +84.9% |
| EPL phase minimum | 0.3337 | 0.6093 | **+0.2756** | +82.6% |
| Temporal consistency | 0.8596 | 0.9311 | +0.0714 | +8.3% |
| Mean proxy score | 0.7182 | 0.8084 | **+0.0903** | +12.6% |

This result is also `PARTIAL`: robot identity is 0.6005 versus the 0.72 gate,
motion is 0.6185 versus 0.72, and EPL minimum is 0.6093 versus 0.62. It covers
only 124 frames (5.17 seconds), not the full 660-frame video.

### Current fail-closed delivery state

After the later structure/morphology evolution, the accepted v19 deterministic
fallback passes **39/39 declared 2-D gates** across **660/660 decoded frames**.
The accepted checks include connected limb chains, exact segment endpoints and
hand roots, bouquet presence and contact overlap, background/person clearing,
action direction and magnitude correspondence, temporal indexing, transition
outliers, and explicit human review. Its maximum encoded transition ratios are
2.6067 full-frame and 2.7375 in the person ROI against a 4.0 threshold.

This is strong workflow-safety evidence, not a learned-model gain. Fixed 2-D
morphology, hidden depth, finger contact, force, physics, generalization, and
real-robot execution are outside its accepted scope.

## 2. AC-WM

### Human-gated action success

Automatic bowl-position proxies once marked the three MiniMax-H3 outputs as
successful, but the user rejected all three for robot quality and insufficiently
convincing action execution. The human gate is therefore the authoritative
success metric.

| Method | Accepted / selected action types | Success rate | Gain vs H3 baseline |
| --- | ---: | ---: | ---: |
| MiniMax-H3 negative baseline | 0/3 | 0.0% | — |
| Single-pass OSCAR after posthoc review | 1/3 | 33.3% | +33.3 points |
| Agentic OSCAR condition evolution | **2/3** | **66.7%** | **+66.7 points** |

The agentic row retains native `lift-up`, replaces the rejected direct-right
slide with a separately declared lift-then-carry-right action, and leaves the
failed left action rejected. The +33.3-point gain over single-pass OSCAR is one
additional accepted action type, not evidence that the original tabletop slide
was repaired.

### Accepted native OSCAR results

Every numeric gate is 0.75 and human review is mandatory.

| Accepted case | Action | Embodiment | Object | Temporal | Background | Human gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Lift up | 0.9937 | 0.9680 | 0.9753 | 0.8659 | 0.9233 | pass |
| Lift then carry right | 0.9492 | 0.8557 | 0.9259 | 1.0000 | 0.9131 | pass |
| **Mean of accepted cases** | **0.9715** | **0.9118** | **0.9506** | **0.9329** | **0.9182** | 2/2 pass |

These metrics demonstrate controlled camera-image futures on one fixed real
Hand2Dex-2 scene. They do not validate robot-base trajectories, metric 3-D
motion, contact force, collision safety, or physical execution.

### Learned repair-router efficiency

The current non-regression repair router was evaluated on 45 hashed candidates
from nine same-scene action/campaign groups:

| Metric | Exhaustive/traditional schedule | Learned guarded route | Change |
| --- | ---: | ---: | ---: |
| Candidate evaluations per group | 5.0 | 2.0 | **-60%** |
| Held-action oracle selection | — | 8/9 (88.9%) | — |
| Guarded non-regression | — | 9/9 (100%) | — |
| Mean utility gain over raw | 0.0000 | 0.1433 | **+0.1433** |
| Mean oracle regret | — | 0.000006 | — |

This is routing/post-processing evidence from one source scene, not a world-model
fine-tuning or cross-scene generalization result.

## Interpretation and reporting rules

1. A difference of 0.10 on a `[0,1]` score is reported as a 0.10 absolute gain
   or 10 percentage points. Relative percentages are shown only when they add
   useful context.
2. No confidence interval or significance claim is made: the main robot
   comparison is one retained video, and AC-WM contains three selected action
   types in one scene.
3. The aggregate proxy score is diagnostic. A failed hard gate or user veto
   always wins.
4. The current evidence supports high **workflow** gains and safe fallback
   behavior. It does not support a claim of solved embodiment transfer, general
   AC-WM control, or real-robot task success.

## Reproduce the aggregation

The standard-library-only compiler reads the immutable evidence, verifies the
required user-rejection ledger record, hashes every source JSON, and writes a
new report directory with Git state, host, Python version, JSON, and Markdown:

```bash
python scripts/summarize_agentic_workflow_benchmarks.py \
  --output-dir outputs/agentic-workflow-benchmark/NEW_UNIQUE_RUN_ID
```

The evidence snapshot used for this document is
`outputs/agentic-workflow-benchmark/20260811T041800Z-evidence-v3`. Canonical
source evidence remains in the experiment directories referenced by the
generated `benchmark.json`, [`EXPERIMENTS.md`](EXPERIMENTS.md),
[`ACWM_WORKFLOW.md`](ACWM_WORKFLOW.md), and the append-only experience ledger.
