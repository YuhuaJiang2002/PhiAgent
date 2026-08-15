# PhiAgent physical-transfer evaluation protocol

## Objective

PhiAgent's primary evaluation unit is a complete attempted transfer of one
scene, action, object, embodiment, and seed. A result is valid only when the
action, EPL phase/contact state, object interaction, target embodiment,
temporal motion, and protected background all pass their own gates. A high
average score cannot compensate for a failed gate.

## Metric hierarchy

| Role | Metric family | Implementation |
| --- | --- | --- |
| Primary | Full-horizon action trajectory, direction, terminal state, and coverage | `phiagent/evaluation/task_motion.py` |
| Primary | EPL phase overlap/boundaries and contact event agreement | `phiagent/evaluation/task_motion.py` |
| Primary | Object visibility/identity, contact, coupling, causality, and terminal state | `phiagent/evaluation/interaction.py` |
| Required | Robot topology, landmark tracking, geometry, identity, and articulation | `phiagent/evaluation/embodiment.py` |
| Required | Global, late, ROI, and worst-window temporal consistency and motion smoothness | `phiagent/evaluation/video_quality.py` |
| Guardrail | Foreground-excluded, small-camera-motion-compensated background preservation | `phiagent/evaluation/video_quality.py` |
| Diagnostic | Deterministic sharpness proxy | `phiagent/evaluation/video_quality.py` |

The sharpness score is not a perceptual-realism metric and is diagnostic by
default. It becomes a hard gate only when an experiment predeclares
`minimum_visual_proxy_quality`.

Image-plane velocity, acceleration, and jerk use an explicitly declared
`timestamp_unit_seconds`. Jerk is divided by the predeclared
`jerk_normalization` before it enters the smoothness exponent, so changing the
timestamp representation from seconds to milliseconds cannot silently change
the result.

## Strict acceptance

`phiagent/evaluation/physical_transfer.py` maps the specialized scorecards into
the following default hard gates:

- action adherence >= 0.75;
- EPL phase agreement >= 0.75;
- contact agreement >= 0.75;
- target-embodiment consistency >= 0.75;
- object-interaction quality >= 0.75 and the interaction evaluator's strict
  contract passes;
- temporal consistency >= 0.75;
- motion physicality >= 0.75;
- background consistency >= 0.75;
- explicit human review is `true`.

Acceptance is the conjunction of these gates. Mean and weighted scores are
retained only for ranking or diagnosis.

Each assessment stores the exact acceptance contract used to make its decision.
An experiment summary rejects mixed or mismatched contracts instead of
silently recomputing acceptance under different thresholds.

## Experiment-level reporting

Use `IndependentEvaluationUnit` to identify the intervention unit. Repeated
frames, camera views, candidates, or reviews from the same unit must not
increase the sample count. `summarize_physical_transfers` reports:

- valid transfer rate (VTR), as exact accepted/attempted counts;
- a two-sided Wilson 95% interval;
- per-group VTR for a predeclared field such as action or embodiment;
- worst-group VTR;
- gate failure counts and human-review pending/rejected counts.

## Evidence boundary

The new modules are dependency-free evaluators over structured observations.
They do not create robot masks, landmarks, calibrated 3-D object tracks, EPL
phase labels, or contact evidence from raw video. Those observations must come
from a pinned perception/annotation pipeline and retain explicit coordinate
frames. Camera, world, and robot-base coordinates are never relabelled
implicitly.

The current engineering validation uses deterministic synthetic CPU fixtures,
including wrong-direction, wrong-path, missing-contact, rigid-translation,
late-fragmentation, object-teleportation, pre-contact motion, frozen-video,
late-flicker, blur, and background-corruption failures. It is not yet a
real-input acceptance result and must not be reported as a working PhiAgent
milestone until the complete contract is run on frozen real evaluation units.
