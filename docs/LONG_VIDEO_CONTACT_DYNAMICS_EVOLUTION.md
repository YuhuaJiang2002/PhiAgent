# Long-video articulated contact and flower dynamics

Evidence date: 2026-08-12. The current 27.5-second candidate remains
`PARTIAL`. This document separates the implemented first-principles algorithm
and evaluator from the real-video evidence that is still missing.

## Measured diagnosis

The audited artifact is
`outputs/wan-long-robot-contract/20260812T073750Z-first-principles-v1/hand-union-repair-v1/robot-hand-union-lossless.mkv`,
660 frames at 624x352 and 24 FPS. The immutable force-closure audit is under
`outputs/wan-long-contact-dynamics/20260812T082000Z-first-principles-evolution-v1/audit-current-v2-force-closure`.

- The complete visual audit processes all 660 frames in 13.6573 seconds, or
  48.326 FPS on the local CPU.
- The motion floors are derived from the candidate's static-background optical
  flow rather than fitted to failed interactions: 0.65193 pixels/frame for the
  visible hand and 0.32596 pixels/frame for the local flower support.
- There are 329 frames in which the visible hand is moving above its floor and
  is projected near a valid flower mask. Nine lack a visible flower response
  within two frames. The failed runs are frames 60, 88--91, 126, 136--137, and
  219. The longest run is four frames, approximately 167 ms.
- The only named active-stem track covers 17 sparse samples over source frames
  272--320. It is useful visual evidence but cannot establish every stem's
  identity, depth, contact, or force over 660 frames.
- Full-resolution human review rejects intermittent finger morph/smear and
  exact grasp contact. This veto is preserved even when aggregate pixel metrics
  look favorable.

The existing source-state projection correctly protects known flowers and the
background, but its state partition treats the flower layer as immutable scene
content. That removes appearance drift by construction and can also overwrite
the response of a flower being manipulated. Conversely, the robot hand is a
generated raster field without an immutable joint tree. A temporal pixel model
therefore has no conservation law preventing finger count, bone length, or
contact topology from changing between frames. These are representation errors,
not settings that can be repaired by another guidance-scale or denoising-step
sweep.

## First-principles replacement

The long-horizon state must be explicit:

`state[t] = {camera calibration, robot joint state, one centerline and velocity
field per named stem, material state, contact mode, contact point, normal and
force}`.

The implemented `metric-articulated-rod-residual-v1` core enforces the following
factorization:

1. A fixed articulated hand contract owns joint names, parents, limits,
   fingertips, metric frame, and bone lengths. The renderer may change
   appearance but cannot invent or delete joints.
2. Every flower stem has a persistent instance ID and rooted deformable-rod
   state. Contact force, not image adjacency, drives its motion.
3. A deterministic spring/rod transition supplies the conservative dynamics
   backbone. A learned residual may model unobserved material effects, following
   the physics-plus-residual pattern, but cannot move the rooted endpoint or
   bypass finite-energy checks.
4. Metric contact requires calibrated intrinsics and scale, surface gap,
   occlusion order, contact normals, observed or simulated contact forces, and
   external/support wrench. Two objects touching in the image is never upgraded
   to contact.
5. Force closure is stronger than equilibrium. The evaluator discretizes each
   3-D friction cone, builds a normalized 6-D grasp-wrench matrix, requires rank
   six, and requires a strictly positive null-vector whose wrench residual is
   below the frozen tolerance. A balanced two-finger example that cannot span
   all 6-D wrenches is deliberately rejected.
6. Long-video windows exchange the explicit state at an absolute source frame;
   they do not recursively use a generated boundary image as physical memory.
   Diffusion is restricted to rendering an appearance residual conditioned on
   this state.

The lightweight implementation is in `phiagent/rendering/contact_dynamics.py`.
It deliberately accepts a NumPy-like module from the caller, so importing
`phiagent` still does not require NumPy, Torch, CUDA, a simulator, or a
checkpoint. Architecture selection and structural repair generation are in
`phiagent/agent/contact_dynamics_evolution.py`.

## Adversarial acceptance contract

`scripts/audit_contact_dynamics_long_video.py` rejects the candidate unless all
of these gates pass:

- fixed articulated metric hand sequence with finite coordinates, stable bone
  lengths, no collapsed bones, and respected joint limits;
- millimetre-scale 3-D contact plus a force-closure certificate;
- a causal flower response whenever the grasping hand moves;
- persistent named-stem identity over every interaction;
- high-resolution human review.

It also attacks the evaluator by erasing flower response, removing depth/force
evidence while preserving 2-D overlap, and corrupting the hand parent tree. All
three attacks are detected in the current run. The 2-D flow result remains a
visual coupling proxy, not physical-contact evidence.

## Self-evolution result

The SkillHone-guided loop uses a complete architecture tournament, immutable
hard gates, explicit cost, and no average-score override. The final tournament
is under
`outputs/wan-long-contact-dynamics/20260812T082000Z-first-principles-evolution-v1/evolution-tournament-v3-force-closure`.
It compares the current `pixel-source-lock-v6` family and the new
`metric-articulated-rod-residual-v1` family over both chronological halves of
the real video. These halves give complete same-video coverage but are not
misreported as independent-scene generalization.

No architecture is promoted. The current pixel family fails articulated hand,
metric force closure, persistent identity, and human review in both halves; it
also fails causal stem motion in frames 0--329. The new family has an executable
state/evaluator core and synthetic adversarial coverage, but it has no real
metric reconstruction or rendered candidate for this clip, so all unobserved
real-video gates remain false. The loop automatically emits five structural
mutations: articulated hand state, metric contact state, rooted deformable
object dynamics, persistent object memory, and non-overridable high-resolution
review. It emits no generator hyperparameter sweep.

The repository skill passes SkillHone's strict offline static check with no
errors or warnings. A model-graded SkillHone `optim.py` run did not start because
the required Forgejo URL/token and local settings profile are absent. The local
immutable tournament is therefore evidence for the algorithmic loop, not a
claim that an official remote SkillHone optimization completed.

## Literature-to-architecture mapping

- [C2Dex (2026-08-07)](https://arxiv.org/abs/2608.07045) motivates stable
  object-side contacts aggregated in canonical object space, trajectory-level
  contact constraints, and local interaction geometry during retargeting. Its
  public repository revision `a23a16dcb1d172f3b4d6136619c5b8f8123ef2b1`
  currently contains the project website and media, not runnable method code.
- [CHOIR (2026-05-20)](https://arxiv.org/abs/2605.20992) motivates ray-depth
  correction followed by joint 4-D hand/object/contact optimization. This is
  the direct answer to the unobservable depth-order problem in 2-D adjacency.
- [CARI4D (CVPR 2026)](https://arxiv.org/abs/2512.11988) provides a released
  metric-scale monocular 4-D reconstruction route. Its official custom-video
  pipeline says it is not designed for partially visible bodies or long-term
  occlusion, and it expects object masks, a reconstructed object mesh, human
  body keypoints, UniDepth, SMPL-H, FoundationPose, and additional licensed
  checkpoints. A thin deformable flower and generated robot hand are outside
  its direct rigid human-object assumptions, so the code is a reconstruction
  reference, not a drop-in fix.
- [DeformMaster (2026-05-10)](https://arxiv.org/abs/2605.09586) motivates a
  structured physics rollout, sparse hand motion as a compliant distributed
  actuator, and a learned residual for appearance and unmodeled dynamics.
- [PGRD (2026-07-15)](https://arxiv.org/abs/2607.13451) supports an optimizable
  spring-mass backbone, velocity-state formulation, sliding temporal memory,
  and learned residual dynamics instead of a purely neural rollout.
- [Transferring Contact, Not Just Motion (2026-06-14)](https://arxiv.org/abs/2606.15516)
  motivates calibrated physical torque/force observations and a shared
  force-position interface across hand embodiments.
- [DeforM (2026-07-21)](https://arxiv.org/abs/2607.18664) supports focusing
  generator capacity on physics-critical spatial-temporal regions. In this
  design those masks are downstream of explicit contact state, never a
  substitute for it.

## What is still required for a claim-eligible fixed video

The current monocular RGB artifact cannot by itself prove depth or force. A real
acceptance run requires either synchronized RGB-D/multiview calibration or a
reviewed metric reconstruction, the actual robot hand URDF/joint trajectory,
one tracked 3-D centerline per stem, contact forces from a calibrated sensor or
named physics solver including support reaction, and a newly rendered 660-frame
candidate. That candidate must pass both independent scene/object groups, all
hard machine gates, full-resolution review, and the response-erasure,
depth/force-spoof, broken-topology, occlusion, and long-freeze attacks. Until
those inputs and that run exist, the correct status is `PARTIAL`, not a visually
plausible but physically unproved success.

## Simulation result and real-video boundary

A full-length simulator run at
`outputs/foundation-contact/20260812T201000Z-metric-flower-coupled-force-full660-v7`
now supplies the state that was previously absent: calibrated metric RGB-D,
complete 73-coordinate exact-asset robot state, one persistent 12-node stem,
solver force with covariance, fixed hand topology, and explicit contact mode.
All four physical compiler stages and the cross-stage bundle lineage pass, as
do 300/300 exact Sharpa-pad force-closure frames and 298/298 causal-response
frames. Four failed raw contact poses are repaired by the first bounded
`[-2 mm, 0, 0]` robot-base candidate; no other frame changes. The 660-frame
video also passes complete decode, uniform/contact review, and consecutive
review around all four repairs.

The v7 force path supersedes v6 after a second red-team audit. Exact pad
proximity alone is not a force measurement: v7 solves nonnegative friction-cone
forces so their net force and moment match the inverse-rod-required wrench, and
the force gate uses the combined rod plus contact coupling residual. Its p95 is
0.000654 N against the fixed 0.08 N limit.

This result supersedes the earlier v1 simulation claim. Red-team review found
that v1 used a constructed six-pad fixture rather than actual hand geometry and
could combine unrelated stage files. The six-pad helper was removed, contacts
now come from transformed exact elastomer vertices, and `WORKING` additionally
requires one hash-bound source/bundle/frame/FPS/frame/instance lineage.

This resolves the executable representation and data-generation blocker, not
the observability blocker in the original clip. The simulator camera is not an
independent observation of Pexels 5893642, and its solver forces are not
measurements from that scene. The accepted rollout is valid for
physics-grounded adapter training and adversarial evaluator tests only. The
original replacement remains `PARTIAL` until newly captured RGB-D/multiview or
another admissible metric observation is bound to the real timeline.
