# PhiAgent-Bench (v0.1)

PhiAgent-Bench is a source-conditioned, cross-embodiment benchmark contract:

```text
human / ego video
  -> target-robot video
  -> metric geometry and structured action
  -> simulation execution gate
  -> recorded real-robot audit
  -> optional downstream policy-utility evaluation
```

Status: **PARTIAL**. The dependency-free schema, clean-room H2R aggregation,
L1--L5 aggregation, pinned RoboWM command adapter, recorded-hardware evidence
adapter, smoke suite, CLI, and CPU tests work. Isaac Lab execution, official H2R
annotations, model-based video-quality inference, and new hardware trials are not
claimed to have run in this checkout.

## What the five levels mean

| Level | Primary question | Main outputs |
|---|---|---|
| L1 Visual transfer | Did the requested robot visibly preserve the source task? | H2R M1--M5 and H2RCore |
| L2 Geometry | Are camera, scene, hand, and object estimates metrically accurate? | ATE/RPE, depth, reprojection, pose errors |
| L3 Action | Does the recovered action agree with synchronized action truth? | EEF, joint, gripper, contact, timing errors |
| L4 Simulation | Does the action pass kinematic/physics gates and finish the task? | IK, limits, collisions, singularity, contact, task success |
| L5 Real | Is the exact accepted request safe and successful on recorded hardware? | coverage, real precision, E2E valid success, safety, goodput |

L1 follows the public equations in H2R-Bench arXiv:2608.13049v1. It is a
clean-room reproduction, not the unpublished official evaluator. H2R's 25-frame
evidence budget, three independent 0--4 judges, hard source-grounding/contact
rule, hard embodiment-failure rule, and published weights are preserved.

The reproduced visual score is:

```text
H2RCore = 100 * (0.15 M1 + 0.15 M2 + 0.30 M3 + 0.30 M4 + 0.10 M5)
```

M1 and M2 are weighted goal-predicate and action-event completion. M3 averages
the applicable contact dimensions but becomes zero for a judge that cannot
ground the generated interaction in the source. M4 uses robot presence, human
absence, category, end-effector, and temporal morphology weights of
`0.20/0.20/0.20/0.25/0.15`, with the public hard failure. M5 averages imaging,
aesthetic, adjacent-frame stability, and interpolation smoothness components.
Task-family-stratified bootstrap confidence intervals are reported across a
suite. The public paper's 120 clips, private case annotations, and judge backend
are not redistributed here.

L4 treats a RoboWM task-success log as `task_outcome_only`. It does **not**
fabricate collision, singularity, or velocity evidence. Full L4 passage requires
`physical_gate_complete=true` from a backend that measured every declared gate.

L5 never commands a robot. It imports an existing PhiAgent
`RealRobotTrialEvidence` bundle containing action, calibration, videos, telemetry,
safety log, outcome review, timestamps, and hashes. A hardware SDK may be added
later behind a separately reviewed adapter.

## Quick start

From the repository root:

```bash
python -m phiagent.benchmark.cli validate \
  --suite benchmark/suites/smoke-v0.1/suite.json \
  --submission benchmark/suites/smoke-v0.1/submission-reference.json

python -m phiagent.benchmark.cli evaluate \
  --suite benchmark/suites/smoke-v0.1/suite.json \
  --submission benchmark/suites/smoke-v0.1/submission-reference.json \
  --output /tmp/phiagent-bench-smoke.json

python -m phiagent.benchmark.cli run \
  --suite benchmark/suites/smoke-v0.1/suite.json \
  --submission benchmark/suites/smoke-v0.1/submission-reference.json \
  --output-dir runs/phiagent-bench/NEW-RUN-ID

python -m phiagent.benchmark.cli h2r-score \
  --annotation benchmark/suites/smoke-v0.1/h2r-annotation.json \
  --judge benchmark/suites/smoke-v0.1/h2r-judge-a.json \
          benchmark/suites/smoke-v0.1/h2r-judge-b.json \
          benchmark/suites/smoke-v0.1/h2r-judge-c.json \
  --video-quality benchmark/suites/smoke-v0.1/video-quality.json

python -m phiagent.benchmark.cli adapter-check \
  --manifest benchmark/adapters/rm65-ag2f90c-recorded.json \
  --suite benchmark/suites/smoke-v0.1/suite.json
```

The smoke submission is an acceptance fixture for equations and aggregation,
not a PhiAgent model result. The repository's real public artifact is imported
separately and fail-closed:

```bash
python -m phiagent.benchmark.cli evaluate \
  --suite benchmark/suites/public-preview-v0.1/suite.json \
  --submission benchmark/suites/public-preview-v0.1/submission-current.json
```

It reports `ik_success_rate=1.0`, while L1/L2/L3 stay missing and L4 stays false
because the artifact has no calibrated source action truth, cloth contact
dynamics, or full physical gate. This is the intended behavior.

To emit a pinned RoboWM-Bench one-episode replay command:

```bash
python -m phiagent.benchmark.cli robowm-command \
  --checkout external/RoboWM-Bench \
  --revision 0a8b0eab3ebfb7993f6ab895f12eac41dfefa1c1 \
  --task pick \
  --trajectory-root /path/to/actions \
  --output-root /path/to/new/run \
  --episode-index 0
```

The command is emitted, not executed. Isaac Sim 5.1 and the bundled Isaac Lab
environment remain optional heavyweight dependencies.

## Non-cherry-picked real reporting

For all cases that require L5, the benchmark reports:

```text
Coverage = N(sim pass) / N(all requests)
RealPrecisionLowerBound = N(valid real success) / N(sim pass)
E2E-VSR = N(valid real success) / N(all requests)
ValidDataGoodput = valid real trajectory seconds / total GPU-hours
```

Missing hardware executions after a simulation pass lower the precision bound
and are separately visible through `real_audit_completion`. Simulation rejection
also remains in the E2E denominator.

## Recommended public tracks

1. **H2R transfer**: human video to requested embodiment, with L1 plus action and
   execution where available.
2. **Action reconstruction**: robot RGB to the same robot's hidden telemetry;
   exact EEF/joint/gripper errors are meaningful here.
3. **Cross embodiment**: human or robot A to robot B; final success, contact,
   object state, and safety are primary because a unique joint-space truth does
   not exist.
4. **Policy utility**: matched training budget, comparing real-only against
   real-plus-PhiAgent data on held-out real trials.

The core leaderboard should initially fix two or three reference embodiments.
Additional robots belong in an adapter track rather than weakening comparability
by claiming immediate support for every commercial arm and gripper.

Every hardware adapter declares its embodiment, ordered joints, control rate,
telemetry channels, safety channels, and whether it is permitted to execute or
only import recorded evidence. The checked-in RM65/AG2F90-C manifest is
`evidence_only=true` and `execution_enabled=false`; benchmark evaluation cannot
move the robot.

## Submission contract and leaderboard

JSON Schemas live in [`schemas/`](schemas/); the Python parser remains the
authoritative semantic validator for named frames, exact suite coverage, and
conditional gates. `phiagent-bench leaderboard` sorts by end-to-end valid real
success first, then simulation and visual performance; incomplete submissions
remain visible but have `eligible=false` and no rank. Public reports should
always include the five-dimensional vector and real-audit denominators, not one
scalar that lets appearance compensate for failed physics.

## Upstream boundaries

Pinned revisions and license observations are in
[`third_party.lock.json`](third_party.lock.json). Neither upstream repository is
vendored. H2R-Bench currently publishes the paper and website but not evaluator
code or benchmark annotations. RoboWM-Bench publishes tasks and scripts, but its
top-level checkout does not currently contain a repository-wide license file;
PhiAgent therefore invokes it as a user-provided external checkout and copies no
source.
