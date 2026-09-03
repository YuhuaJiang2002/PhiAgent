# PhiAgent-Bench (v0.2 alpha)

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
L1--L5 aggregation, policy-bound repeated trials, resumable batch controller,
content-addressed artifacts, pinned RoboWM and HarnessEval-W adapters,
recorded-hardware evidence adapter, smoke suite, CLI, and CPU tests work. A frozen RoboWM `Franka-pick`
episode has run successfully in Isaac Lab 5.1 (`1/1` upstream task success) on
an RTX PRO 5000; it is deliberately not promoted to a full L4 physical pass
because the upstream replay does not export every required safety/physics gate.
Official H2R annotations, model-based video-quality inference, and new hardware
trials are not claimed to have run.

The v0.1 fixtures remain readable for compatibility. New formal runs use the
v0.2 schemas and must name a frozen policy. The development policy is useful for
pipeline qualification; the real-pilot policy requires five complete simulation
episodes plus ten recorded trials distributed over at least three sessions.

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

For a full L4 candidate, each simulator writes a per-step trace matching
[`physical-gate-trace-v0.2.schema.json`](schemas/physical-gate-trace-v0.2.schema.json).
The normalizer derives rates from all samples and keeps intended contacts
separate from forbidden collisions:

```bash
python -m phiagent.benchmark.cli physical-gate \
  --trace benchmark/examples/physical-gate-trace-v0.2.example.json \
  --output /tmp/physical-evidence.json
```

No caller-supplied aggregate can override a bad sample. A complete trace is
still simulator evidence, not hardware evidence.

L5 never commands a robot. It imports an existing PhiAgent
`RealRobotTrialEvidence` bundle containing action, calibration, videos, telemetry,
safety log, outcome review, timestamps, and hashes. A hardware SDK may be added
later behind a separately reviewed adapter.

A valid L5 success must also name the frozen blind protocol, be pre-registered,
carry a privacy-preserving reviewer SHA-256, and bind the complete action,
calibration, initial/predicted/execution video, telemetry, safety, and outcome
artifact set by SHA-256. Merely setting `blind_review=true` is insufficient.

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
  --manifest benchmark/adapters/rm65-ag2f90d-recorded.json \
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
  --episode-index 0 \
  --episode-sha256 0d7c012eae5381346374f8b13c6785eb5f539404df6e12d2854495a2fcb4e631 \
  --pose-sha256 7bde01322ce53bde15712d1a3b2271f1cabc4223a11dfdeb6ccb846ccea5c2e5
```

The command is emitted, not executed. Isaac Sim 5.1 and the bundled Isaac Lab
environment remain optional heavyweight dependencies.

## Batch generation and evaluation

The v0.2 controller expands `suite × candidate × ordered stage` into immutable
jobs. Commands are argv arrays, never shell strings. It records seeds, source
hashes, git state, package inventory, hostname, stdout/stderr hashes, physical
GPU inventory for GPU stages, and expected outputs in a local SHA-256 CAS. A
completed job is verified again on resume, so changed or missing artifacts fail
closed.

Start from [`method-v0.2.example.json`](examples/method-v0.2.example.json) and
replace the two worker paths with a model generator and evaluator that emit
declared `record-patch.json` files:

```bash
python -m phiagent.benchmark.cli batch-plan \
  --suite /path/to/frozen-suite-v0.2.json \
  --method /path/to/method-v0.2.json \
  --output-dir runs/phiagent-bench/BATCH-ID

python -m phiagent.benchmark.cli batch-run \
  --run-dir runs/phiagent-bench/BATCH-ID --max-workers 8 \
  --gpu-device 0 --gpu-device 1 --gpu-device 2 --gpu-device 3

python -m phiagent.benchmark.cli batch-status \
  --run-dir runs/phiagent-bench/BATCH-ID

python -m phiagent.benchmark.cli batch-compile \
  --run-dir runs/phiagent-bench/BATCH-ID \
  --selection /path/to/frozen-selection.json \
  --output /tmp/submission-v0.2.json
```

GPU stages must declare `resources.gpus` and explicitly bind the same number of
devices. `--gpu-device` enables collision-free local leasing and overrides the
template's development binding; without a supplied pool the method must set
`CUDA_VISIBLE_DEVICES` itself. The controller refuses an implicit GPU selection.
The built-in executor is single-host. A cluster scheduler may claim
the same immutable job manifests, but a Slurm/Kubernetes/Ray backend is not yet
implemented here.

## HarnessEval-W integration

[HarnessEval-W](https://github.com/mirros-lab/harnesseval-w) is pinned as an
external Apache-2.0 dependency. Its evidence-tree evaluation over observation
quality, transition correctness, and world persistence is useful for replacing
one-shot visual impressions with case-specific, auditable questions. For
PhiAgent, the most useful skills are embodiment identity, gripper state,
contact/action phase, protected background/object state, and occlusion/revisit
consistency.

It is an **L1 supplementary panel**, not a physical oracle. A visually inferred
physics score cannot supply metric camera geometry, EEF/joint truth, collision
telemetry, simulator success, or a real-robot result. H2RCore and HarnessEval-W
scores therefore remain separate until a frozen robot-specific calibration set
establishes how to combine them.

```bash
python -m phiagent.benchmark.cli harnesseval-preflight \
  --checkout /path/to/harnesseval-w \
  --revision ed4ccc6486b8271723ee8baea60d89b32d0a7518
```

The paper reports stronger human alignment than the compared visual-world
baseline, but those self-reported results do not validate PhiAgent's action or
hardware layers. See [arXiv:2608.16859](https://arxiv.org/abs/2608.16859).

## Repeated blind real-robot protocol

Only L4-passing cases enter real planning. The planner randomizes case order,
hides method identity from the operator/reviewer schedule, and retains the
coordinator mapping separately. It never commands hardware:

```bash
python -m phiagent.benchmark.cli real-plan \
  --suite /path/to/frozen-suite-v0.2.json \
  --submission /tmp/submission-v0.2.json \
  --policy benchmark/policies/real-pilot-v0.2.json \
  --protocol benchmark/protocols/real-robot-blind-v0.2.json \
  --adapter-manifest benchmark/adapters/rm65-ag2f90d-recorded.json \
  --session-id session-a --session-id session-b --session-id session-c \
  --output-dir /tmp/real-plan
```

The checked-in RM65/AG2F90-D adapter remains evidence-only and execution is
disabled, so this command correctly returns `blocked_pending_site_authorization`.
An approved site executor, calibrated hardware, and genuinely recorded trials
are required before any L5 claim.

The exact tested environment, frozen episode hashes, compatibility patches,
warnings, and claim boundary are recorded in
[`execution/`](execution/README.md). The adapter emits both `--headless` and
`--enable_cameras`; omitting the latter prevents the RoboWM tiled cameras from
initializing.

Validate the three-case frozen public L1 pilot and the embodiment catalog with:

```bash
python -m phiagent.benchmark.cli freeze-check \
  --manifest benchmark/suites/public-visual-pilot-v0.1/freeze.json \
  --repository-root .

python -m phiagent.benchmark.cli registry-check \
  --registry benchmark/embodiments/registry-v0.1.json
```

The public pilot freezes rigid rearrangement, deformable configuration, and
surface transformation sources by SHA-256. It is deliberately L1-only because
those public display videos have no calibrated metric/action truth. A future
core L2--L5 split must use synchronized calibration, object state, robot state,
and outcome annotations rather than inventing labels for these clips.

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
telemetry channels, safety channels, end-effector mass/stroke/force/speed limits,
and whether it is permitted to execute or only import recorded evidence. The
primary company-hardware manifest is RM65/AG2F90-D and remains
`evidence_only=true` and `execution_enabled=false`; benchmark evaluation cannot
move the robot. The older C-labelled manifest remains only for reproducing the
published visual asset.

The D manifest records 100 and 250 mm/s modes from the manufacturer page, but
keeps `high_speed_requires_confirmation=true`: the page marks 250 mm/s with a
double asterisk and does not establish the installed firmware/load/duty-cycle
conditions. The default evaluation limit therefore remains 100 mm/s.

Manufacturer source: [CTAG2F90-D specifications](https://www.changingtek.com/diandong/147).

The pinned source catalog is
[`embodiments/registry-v0.1.json`](embodiments/registry-v0.1.json). It stores
official URLs, revisions, asset paths, licenses, validation tiers, and caveats;
it does not vendor a pile of third-party URDFs or equate parseability with safe
execution.

The initial catalog intentionally has six representative arm sources rather
than hundreds of copied files: RM65-B, UR5e, FR3, Kinova Gen3 7DoF, xArm7, and
ViperX 300 S. Promotion proceeds from `metadata_only` to `source_pinned`,
`kinematic_validated`, `simulation_validated`, and finally
`hardware_validated`. Only the latter two belong in physical leaderboard claims.

## Submission contract and leaderboard

JSON Schemas live in [`schemas/`](schemas/); the Python parser remains the
authoritative semantic validator for named frames, exact suite coverage, and
conditional gates. `phiagent-bench leaderboard` sorts by end-to-end valid real
success first, then simulation and visual performance; incomplete submissions
remain visible but have `eligible=false` and no rank. Public reports should
always include the five-dimensional vector and real-audit denominators, not one
scalar that lets appearance compensate for failed physics.

The implementation boundary and remaining path to a public frozen leaderboard
are tracked in [`ROADMAP.md`](ROADMAP.md).

## Upstream boundaries

Pinned revisions and license observations are in
[`third_party.lock.json`](third_party.lock.json). Neither upstream repository is
vendored. H2R-Bench currently publishes the paper and website but not evaluator
code or benchmark annotations. RoboWM-Bench publishes tasks and scripts, but its
top-level checkout does not currently contain a repository-wide license file;
PhiAgent therefore invokes it as a user-provided external checkout and copies no
source.
