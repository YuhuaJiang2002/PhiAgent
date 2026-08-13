# Foundation-model-assisted metric contact pipeline

Evidence date: 2026-08-12. The full pipeline is `PARTIAL`. Two interleaved
learned-metric camera runs and calibrated-pose virtual views have run on the
27.5-second candidate, but independent scale, full robot
generalized coordinates, accepted per-stem 3-D rods, and physical contact-force
evidence have not all passed. This document defines what the pretrained models
may estimate and where exact assets, calibration, or physics must take over.

## First-principles decomposition

The reconstructed state at source frame `t` is

```text
x[t] = {
  K[t], world_from_camera[t], depth[t], scale_covariance[t],
  exact_robot_asset_hashes, robot_base_from_world[t], q[t], qdot[t],
  stem_id -> {centerline_world[t], velocity[t], material[t], root[t]},
  contact_id -> {bodies, point_world[t], normal_world[t], mode[t], force_world[t]}
}
```

A foundation model is a proposal distribution over observable geometry. It is
not allowed to change an exact URDF/MJCF asset, relabel learned confidence as
calibration, or emit a contact force without a sensor or a named physics
solver. `phiagent/perception/foundation_contact.py` makes those evidence classes
machine-readable and computes one fail-closed end-to-end decision.

The architecture follows four ordered physical stages:

1. **Metric camera.** DA3 Nested proposes depth, intrinsics, and camera motion
   over a sparse full-span sample. DA3 is appropriate because it predicts
   spatially consistent geometry from an arbitrary number of inputs, with or
   without known camera poses ([Depth Anything 3](https://arxiv.org/abs/2511.10647)).
   UniDepth is the independent single-frame fallback. Learned metric scale is
   still tied to a fiducial, RGB-D observation, or known-length hash-bound robot
   link before any absolute-force claim. MOMA likewise treats monocular depth as
   affine-ambiguous and aligns it with sparse ground-truth depths rather than
   trusting a metric label ([MOMA](https://arxiv.org/abs/2506.17110)); recent
   relative-pose work reports that affine correction helps even nominally metric
   depth priors ([MADPose](https://arxiv.org/abs/2501.05446)).
2. **Exact robot and full `q`.** The model may propose keypoints, silhouettes,
   and an initial asset ID. RoboPEPP's joint masking and confidence-filtered
   keypoint proposals are useful under truncation and occlusion
   ([RoboPEPP, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Goswami_RoboPEPP_Vision-Based_Robot_Pose_and_Joint_Angle_Estimation_through_Embedding_CVPR_2025_paper.html)).
   The selected Unitree G1 and bilateral Sharpa MJCFs remain exact hash-bound
   assets. Every joint is then estimated by full-asset render-and-compare, the
   articulated analogue of RoboPose's joint-angle and camera optimization
   ([RoboPose, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Labbe_Single-View_Robot_Pose_and_Joint_Angle_Estimation_via_Render__CVPR_2021_paper.html)).
   Recent differentiable robot rendering directly optimizes control parameters
   from appearance, including G1 experiments
   ([Differentiable Robot Rendering](https://arxiv.org/abs/2410.13851)); here it
   is only an optimization mechanism. Data-Jacobian rank, posterior uncertainty,
   held-out reprojection, and alternative-asset comparisons decide acceptance.
   Pixel wrist traces and prior-filled invisible joints are never treated as a
   complete trajectory.
3. **Persistent per-stem rod state.** A mask only identifies the image support.
   V-DPM supplies dynamic point identities and full 3-D point motion across a
   video ([V-DPM](https://arxiv.org/abs/2601.09499)); SpaTracker can provide an
   additional long track proposal. Points are associated with one immutable
   stem ID and jointly optimized as a rooted, inextensible Cosserat rod across
   visible and occluded frames. This replaces independent per-frame skeleton
   lifting.
4. **Contact force.** A calibrated tactile or force/torque stream is preferred.
   Without one, force is the residual required by the accepted metric robot and
   rod trajectories under gravity, inertia, elasticity, damping, support
   reactions, friction, and non-penetration. PhysTwin supports the overall
   inverse-physics decomposition for deformable objects
   ([PhysTwin](https://arxiv.org/abs/2503.17973)); the local implementation uses
   a smaller explicit rod residual. Visual likelihood is never a force sensor.

Long videos carry this explicit state at absolute source frames. Generated
boundary pixels do not become physical memory. Geometry may be evaluated at a
sparse rate for efficiency, but 24-FPS contact intervals are reconstructed from
persistent tracked state and checked against every source frame before
rendering.

## Implemented pipeline

| Stage | Implementation | Current evidence | Status |
| --- | --- | --- | --- |
| Camera | `scripts/run_da3_metric_video.py`, `scripts/generate_model_derived_rgbd_views.py`, `scripts/calibrate_foundation_metric_camera.py`, direct metric-RGB-D input in `scripts/compile_foundation_contact_pipeline.py` | 110 interleaved DA3 RGB-D samples at 4 Hz plus 55 exact-pose virtual views and a fail-closed sparse-metric bridge; separately, one complete simulator-depth sequence | proposal `WORKING`; direct simulator camera passes; real Pexels physical stage `PARTIAL` with 0 independent anchors |
| Robot | `phiagent/perception/exact_asset_trajectory.py` and `scripts/fit_foundation_exact_asset_trajectory.py` | exact 29+22+22 joint schema and asset hashes pass; real single-view observability audit rejects missing dense full-q/render evidence | `PARTIAL`; no trajectory artifact emitted |
| Stems | `scripts/run_vdpm_video.py`, `phiagent/perception/flower_track_routing.py`, `phiagent/perception/multistem_rod_optimizer.py`, `scripts/optimize_multistem_flower_state.py` | one 17-frame SAM2/DA3 active-stem probe plus a persistent multi-rod compatibility run | old proposal rejected by observation-consistency gate; V-DPM/SpatialTracker input still required |
| Forces | `infer_stem_contact_forces` in `phiagent/rendering/contact_dynamics.py` | tested synthetic inverse-rod solver with covariance | no real force result; `PARTIAL` |
| Compiler | `scripts/compile_foundation_contact_pipeline.py` | all four stages and exact assets compiled without imputing missing data | overall `PARTIAL` |
| Evolution | `scripts/evolve_foundation_contact_pipeline.py` | four structural experiments plus five adversarial attack classes | no architecture promoted |

The pinned model revisions, robot hashes, evidence thresholds, frame names, and
promotion rule are frozen in `configs/foundation_contact_pipeline_v1.json`.

## Measured 27.5-second run

The input is 660 frames at 624x352 and 24 FPS, SHA-256
`bf15763ff5ceeb26bd2a744045d76579bad2b758ffa787176e4e8693d5a5ff91`.
DA3 Nested ran on physical A800 GPU 1 at 504x280 over 55 uniformly sampled
frames (2 Hz):

- frame extraction: 0.3614 seconds;
- one-time model load: 21.8081 seconds;
- inference: 5.1189 seconds;
- throughput: 10.7445 sampled geometry frames/second;
- full-span coverage: 5.2827 source-video seconds per inference second;
- cold start including extraction, model load, and inference: 27.2883 seconds,
  or 1.0078 times real time for this 27.5-second clip;
- warm inference-only execution: 5.2827 times real time;
- peak GPU allocation: 13,537 MiB.

The two-context audit shares 12 source frames. Its scale-ratio variation p95 is
0.003107 (0.31%) and the worst per-pixel relative-depth residual p95 is 0.027975
(2.80%). This establishes bounded sensitivity to the tested temporal context,
not calibrated absolute scale; both contexts can share a common metric bias.

### Model-generated new RGB-D observation

A second pinned DA3 run on physical A800 GPU 0 evaluates 55 previously unseen
source frames `6,18,...,654`, offset by six frames from the first lattice. It
uses the same source SHA, repository revision, and checkpoint SHA. Inference
takes 3.6071 seconds at 15.2476 sampled FPS and 7.4967 source-video real time;
model load takes 13.6995 seconds and peak allocation is 13,537 MiB. Combining
the two disjoint lattices produces 110 synchronized RGB-D proposals at 4 Hz
with a maximum six-frame gap, 100% finite positive depth, and 0.1881% difference
between the two run-level median depths.

`scripts/generate_model_derived_rgbd_views.py` additionally splats the 55 new
RGB-D frames into alternating 4 cm virtual camera baselines. The transform
`target_camera_from_source_camera` is exact by construction and every DA3 run
retains its own named learned world frame. Mean visible-surface coverage is
96.3955% and p05 coverage is 96.2721%. High-resolution review shows the expected
disocclusion cracks around fingers, robot contours, and flower stems; the new
occluded-surface fraction is explicitly zero. The warning-free v2 run at
`outputs/foundation-contact/20260812T132000Z-model-derived-rgbd-virtual-views-v2`
therefore passes all six proposal diagnostics but records zero independent
physical groups and `physical_calibration_passed=false`.

The v9 compiler binds the source, model report, and every RGB-D/view artifact
hash at
`outputs/foundation-contact/20260812T136000Z-compiled-model-rgbd-v9`. All four
physical gates remain false. The checked supervisor at
`outputs/foundation-contact/20260812T138000Z-continual-supervisor-model-rgbd-v9`
rejects all 5/5 spoof attacks, does not promote, and keeps external metric
calibration as the next dependency-ready experiment. Thus model-generated
views increase temporal supervision and expose geometric failure modes, but do
not create information behind an occluder or an independent metre.

The new metric-camera bridge fits the physically identifiable model
`1 / z_metric = a / z_proposal + b` with uncertainty-weighted Huber IRLS. It
requires at least 20 registered observations from at least two independent
groups, group-held-out p95 error at most 6%, bootstrap scale standard-deviation
fraction at most 2%, and at least 80% robust inliers. Sensor/calibrated geometry
is admissible. A robot-link anchor is admissible only when its SHA is in the
frozen G1/Sharpa registry, the complete `q` is present, and held-frame render
reprojection is at most 8 pixels. Learned depth, language/object-size priors,
same-model context agreement, and partial-`q` silhouettes cannot establish
scale. This explicit scale/shift treatment follows the ambiguity identified by
MADPose and MOMA. If held-group residuals expose depth-dependent or spatial
distortion, the bridge rejects instead of overfitting a global affine map;
PolyRad motivates a monotone nonlinear sensor-guided successor
([PolyRad](https://arxiv.org/abs/2503.17182)), while UniDAC's relative-depth plus
spatial-scale decomposition motivates a spatial successor
([UniDAC](https://arxiv.org/abs/2603.27105)).

The synthetic acceptance fixture passes end to end: a calibrated NPZ is emitted,
all source/sample/output hashes bind, and the compiled `metric_camera` physical
gate passes. The real 27.5-second run at
`outputs/foundation-contact/20260812T122000Z-metric-camera-bridge-real-v2`
has a matching source SHA but zero independent metric observations, so it takes
0.00712 seconds, returns `PARTIAL`, and deliberately emits no calibrated sample.
The hash-bound recompile at
`outputs/foundation-contact/20260812T122100Z-compiled-pipeline-camera-bridge-v5`
therefore retains the learned proposal and all four physical gates remain false.

The active-stem probe lifts 17 tracked samples into 12-node world-frame
centerlines at 55.166 stem-frames/second. Visibility is 100%, but maximum
temporal segment-length CV is 1.857 against an immutable 0.12 bound. The result
is correctly rejected: mask truncation and foreground/background depth jumps
change the inferred arc length. Loosening this bound would hide the
representation error.

The compiled report records `proposal_passed=true` for learned camera geometry
and passes all three exact robot asset hashes. It deliberately keeps the
physical metric-camera stage false because independent absolute scale is
missing, alongside rejected full-`q` reprojection, per-stem rigidity, and
sensor/solver forces. Overall status remains `PARTIAL`.

The exact-asset trajectory bridge now derives all 73 named joint coordinates
and limits directly from the hashed G1 and bilateral Sharpa MJCFs. It accepts a
foundation model only as a source of 2-D keypoints, confidence, masks, and an
initial state. A bounded robust analysis-by-synthesis solver fits
`camera_from_robot_base` and complete `q`; its observability calculation uses
the data Jacobian only, excluding the stabilizing prior. Promotion additionally
requires a dense 660-frame timeline, at least two held-out temporal groups,
reprojection p95 at most 8 pixels, silhouette IoU p05 at least 0.65, joint
standard deviation at most 0.08 rad, base-translation standard deviation at
most 0.02 m, and at least a 4-pixel advantage over alternative assets. The v2
contract also requires joint-state evidence with sensor, calibrated-view, or
physics authority plus named, revision-pinned proposal and exact-render stacks;
an asset file merely being present cannot prove the pixels depict that asset.

Ten synthetic and adversarial tests recover identifiable camera/full-q state
and reject rank-deficient joints, wrong hashes, ambiguous robot identity,
missing held-out groups, and partial q arrays without numerical warnings. On
the real 27.5-second candidate, manual high-resolution review finds the torso
and partial arms visible, all 12 leg joints outside the frame, most 44 finger
joints occluded by flowers, and late color/structure corruption. Thus the 73
joint plus 6 camera parameters are not identifiable from the current view. The
latest real run at
`outputs/foundation-contact/20260812T124800Z-exact-asset-full-q-real-v2`
finishes validation in 0.01075 seconds, verifies all three asset hashes and the
29+22+22 schema, returns `PARTIAL`, and deliberately emits no
`robot-trajectory.npz`. The bound recompile at
`outputs/foundation-contact/20260812T124900Z-compiled-pipeline-exact-asset-v7`
preserves the exact missing-evidence reason instead of substituting wrist tracks
or a mean pose.

## 2026-08-13 tracker and multi-rod update

The latest public-method audit is in
`docs/FLOWER_LATEST_OPEN_MODELS_20260813.md`. The selected runnable proposal
stack is V-DPM plus SpatialTrackerV2, with MultiDLO as an RGB-D topology critic.
`phiagent/perception/flower_track_routing.py` enforces the information boundary:
monocular tracks remain learned relative geometry; only independently
calibrated RGB-D can open the metric route.

`phiagent/perception/multistem_rod_optimizer.py` adds one joint state over all
frames and stems with immutable instance IDs, fixed/free material roots, exact
arc-length projection, temporal coupling, factorial small-set ID-swap audit and
occlusion-inflated covariance. Structural acceptance additionally requires
observation-residual p95 no larger than 0.10 of full material length; the
optimizer cannot pass by moving a bad proposal arbitrarily far.

The real compatibility run at
`outputs/foundation-contact/20260813T030500Z-multistem-active-pink-v3`
uses the existing `active-pink-stem-01` DA3/SAM2 observations:

- segment CV falls from 1.85723 to `1.26e-14`;
- ID continuity passes for the single available stem;
- observation-residual p95 is `1.5965` times full stem length, versus the
  frozen 0.10 bound;
- no independent calibration is supplied.

The result remains `PARTIAL` and demonstrates that stronger optimization cannot
repair the old mask skeleton. New V-DPM/SpatialTracker point identities are the
dependency-ready experiment. The official V-DPM source is pinned and imports
with a measured CUDA smoke on A800, but its public 6.65 GB checkpoint is absent
from every authorized cache; HF CLI and direct official/mirror range requests
time out. No untrained or randomly initialized substitute is run.

## Calibrated 660-frame simulation unblocker

The compiler now accepts exactly one of two camera bundles: the existing DA3
proposal plus calibration path, or a hash-bound direct metric RGB-D report and
sample file. It never treats direct sensor/simulator depth as DA3 evidence.
Exact-asset registry matching is also part of the robot-stage decision rather
than report-only metadata.

The first direct-input run was superseded after red-team review. Its advertised
six-pad closure state was constructed independently of the exact hand, and its
stage files had no shared producer bundle. The corrected compiler rerun at
`outputs/foundation-contact/20260812T141600Z-recompile-superseded-full660-v1-lineage-v11`
therefore returns `PARTIAL`; the old 300/300 claim is revoked.

The accepted successor is
`outputs/foundation-contact/20260812T201000Z-metric-flower-coupled-force-full660-v7`.
It uses the frozen 29-DOF G1 and bilateral 22-DOF Sharpa assets, a fixed MuJoCo
metric camera/depth buffer, one persistent 12-node stem, exact transformed
Sharpa elastomer meshes, and the
`exact-pad-friction-cone-coupled-inverse-rod-v3` solver. Unlike v6, contact
forces solve the exact pad friction cones against the inverse-rod-required
wrench; the external wrench is not copied from invented pad forces. The output contains 660
frames at 24 FPS over 27.5 seconds. Measured results are:

- 1.0 valid metric-depth coverage over all 660 frames;
- a complete 73-coordinate robot trajectory with 2.31093 rad/s maximum joint
  velocity and zero simulator-ground-truth render reprojection error;
- 0.04265 maximum stem segment-length CV against the frozen 0.12 limit;
- 0.000654 N p95 combined inverse-rod plus coupled pad-wrench residual against
  the 0.08 N limit;
- causal flower response on 298/298 driven contact frames;
- a fixed 18-anchor right-hand tree with zero collapsed bones and
  `5.17e-15` maximum bone-length CV;
- exact distributed-pad 6-D force-closure certificates on all 300/300 contact
  frames, with at least two distinct real fingertip pads, rank six, and maximum
  2.999999 mm surface gap;
- four pre-projection failures at frames 401, 405, 455, and 466 repaired by the
  first bounded candidate, exactly `[-2 mm, 0, 0]` in `robot_base:g1`; the
  remaining 296 frames are unchanged;
- complete video decode and an independent `WORKING` compile at
  `outputs/foundation-contact/20260812T202000Z-compiled-coupled-force-full660-v7`,
  including common bundle ID, source hash, 660 frame indices, FPS, timeline,
  coordinate frame, stem IDs, and all artifact hashes.

Uniform and contact-window full-resolution review show the exact robot
approaching, closing around, bending, releasing, and retracting from the same
flower. The result remains `PARTIAL` for the original request: its physical
`WORKING` status is simulation-only and cannot calibrate or reconstruct the
Pexels camera.

`scripts/export_metric_flower_vace_dataset.py` converts the accepted rollout
into 12 training clips and four source-frame-disjoint validation clips at
`outputs/flower-task-adaptation/20260812T203000Z-metric-physical-vace-data-v3`.
Every 17-frame control is rendered only from calibrated depth, the persistent
stem centerline, and contact state; target RGB is excluded. One separately
rendered robot-only image on a neutral dark background supplies identity. It
matches zero target frames exactly and its minimum 16x16 corner MAD to any
target is 186.83. Train/validation source-frame overlap is exactly zero.

A rank-4, 12-step VACE smoke on physical A800 GPU 1 completes and writes the
5,503,040-byte adapter with SHA-256
`5e76632ee41c996107a17310ec03c99ec5eaf49bb3062789c0b1e19c6052381e`.
Training metadata binds the nonempty adapter and log hashes. A strict evaluator
then binds the frozen validation record, manifest, control, identity, target,
trajectory, both inference metadata files, model files, prompt, seed, Git state,
and adapter.

On unseen contact clip 001, the adapted output improves contact ROI by 0.000608
and contact-motion similarity by 0.003968, so the narrow proxy gates pass.
However, absolute contact/global similarities are only 0.05898/0.00776, global
similarity is slightly below zero-shot, and full storyboard review rejects both
generated rows: they replace the pink flower and beige workspace with a green
rounded mass, dark field, and color band. The explicit human veto stops the
remaining held-out clips and rejects promotion. These clips and the adapter
remain development-only evidence, not independent physical scenes or a
real-video solution.

## Architecture-level self-evolution

`derive_foundation_pipeline_experiments` reads the stage report and emits only
observability or representation changes:

1. bridge learned camera geometry to an independent metric observation;
2. full-asset, full-`q` analysis-by-synthesis with held-frame reprojection;
3. V-DPM/SpaTracker identities plus joint rooted-rod optimization, keeping the
   failed segment-length threshold unchanged;
4. sensor fusion or inverse dynamics only after camera, robot, and stem stages
   pass.

The generated plan attacks common-mode scale spoofing, wrong asset hashes,
partial `q`, mask truncation, stem identity swaps, occlusion, frozen flower
response, false 2-D contact, missing force covariance, and inflated solver
residuals. No mean score can override one failed physical gate.

The continual supervisor is implemented in
`phiagent/agent/foundation_contact_supervisor.py` and
`scripts/supervise_foundation_contact_evolution.py`. It watches immutable
pipeline reports, separates diagnostic progress from physical gates, rejects
five evidence-spoof classes, and ranks only dependency-ready architecture
experiments. The latest model-RGB-D-bound audit completes in 0.05405 seconds, rejects
promotion with 0/4 physical gates, passes all 5/5 adversarial rejection checks,
and selects `calibrated-metric-camera-bridge-v1`. The latest v9 manifest at
`outputs/foundation-contact/20260812T138000Z-continual-supervisor-model-rgbd-v9`
repeats the result with 0/4 physical gates, 5/5 semantic spoof attacks, and a
valid supervision check; `WORKING` here describes the supervisor, not the
video/contact model.

Local SkillHone is now configured through mode-0600
`~/.skillhone/settings.json`, Ollama, and a loopback Forgejo 16.0.2 service. The
public `skillhone/evolve-foundation-contact` skill and private evaluation repo
are versioned independently. The first Claude-Agent/Ollama bridge produced no
answer files and scored 0/2 with 44.07-second recorded mean item latency. A
provider-aware JSON route removes the incompatible Write-tool round trip. The
current campaign scores 4/4 probe, 3/3 test, and 7/7 adversarial after adding
sealed prior-filled-cropped-joint and asset-presence identity attacks; the
private evaluation revision is `6e18538`. An intentionally misrouted rerun
eventually completes at 0/7 with seven missing answers because Qwen cannot write
the Anthropic answer artifact, while the provider-aware run scores 7/7. The
provenance-complete aggregate campaign scores 14/14 at 0.3367 items/second over
41.58 seconds and explicitly records `physical_model_promoted=false`. The
earlier five-item native rerun remains a historical throughput measurement at
0.6030 items/second and 16.0059 generated tokens/second. The same-model,
same-seed, same-split strict
probe improves from 0/4 to 4/4 without a prior-pass regression. A non-thinking
instruct candidate passed ordinary 7/7 but only 1/4 adversarial checks, so it was
rejected and the stronger `qwen3:4b` remained the configured champion.

`phiagent/agent/foundation_contact_skill_eval.py` and
`scripts/run_foundation_contact_skill_eval.py` preserve exact decisions,
latency, token throughput, hashes, Git state, package versions, seed, and raw
outputs. `scripts/compare_foundation_contact_skill_evals.py` can promote only a
comparable behavioral skill with a strict score gain and no regression; it
always records `promote_physical_model=false`.

## Remaining acceptance work

The shortest defensible path to `WORKING` is:

1. collect at least 20 registered RGB-D/fiducial depths across two independent
   groups, or complete-`q`/accepted-reprojection observations of a hash-bound
   G1/Sharpa link, then bind DA3 scale and retain calibration covariance;
2. add synchronized robot telemetry or a calibrated additional view that
   observes cropped legs and occluded fingers; then run full-`q`
   analysis-by-synthesis on all 660 frames and pass exact-asset, posterior,
   joint-limit, velocity, alternative-asset, and held-frame reprojection gates;
3. replace the rejected mask skeleton with dynamic point identities and solve
   one occlusion-aware rod trajectory for every manipulated stem;
4. ingest synchronized tactile/force-torque data, or run inverse dynamics with
   identified rod material and explicit support reactions;
5. render a new 660-frame candidate, then run force closure, causal flower
   response, high-resolution finger/contact review, and every adversarial
   attack.

Until those steps pass on real inputs, the pipeline is useful for generating
structured supervision and rejecting physically impossible candidates, but it
does not prove 3-D grasp force or solve the original video.
