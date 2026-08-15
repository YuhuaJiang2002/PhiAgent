# Architecture

## Embodied Data Engine control plane

The Data Engine wraps PhiAgent's heterogeneous research routes in a small,
standard-library-only orchestration contract:

    authorized sources + pinned target assets + campaign contract
      -> manager: deterministic source x target x seed jobs and rolling windows
      -> executor plugin: retarget and generate one bounded job
      -> hash-bound artifact manifest
      -> independent read-only auditor: all required hard gates
      -> accepted shard or retained rejection evidence

Source frames must be named `camera:*`; target assets must be named
`robot_base:*`. Generator capability and per-target retargeter capability are
validated before a plan is admitted. Visual-training-data and
physically-grounded claims are separate contracts: the latter cannot omit
metric camera, exact trajectory, persistent geometry, or contact-force gates.

The manager owns immutable campaign state. Its file-backed controller atomically
leases jobs, binds submissions to artifact-manifest SHA-256, revalidates the
plan hash on every load, increments state revisions, and appends transition
events under a POSIX file lock. Stranded jobs require an explicit reasoned
requeue before a new worker can claim them. Executors cannot accept their own
work, auditors cannot replace hard-gate conjunction with a mean score, and a
rejected job can be reclaimed without deleting its audit history. Heavyweight
adapters are discovered only through the `phiagent.data_engine.plugins` entry
point so importing the package remains CPU-only. See
[`docs/DATA_ENGINE.md`](DATA_ENGINE.md) for the schema, CLI, evidence boundaries,
and measured capacity projection.

## PhiZero reproduction mainline

The target is Figure 8(b), Human Hand to Sharpa Dexterous Hand Transfer. It uses
the paper's learned video-tokenization route, not explicit hand tracking,
kinematic retargeting, or a two-arm object handover:

    HumanHandVideo
      -> adapted PhiZero spatiotemporal encoder
      -> transition-level Q-Former, 32 queries per adjacent latent pair
      -> FSQ(8, 5, 5, 5, 5, 5)
      -> unchanged 25K-vocabulary PhysicalLanguageTokenSequence

    first(HumanHandVideo)
      -> first-frame appearance edit: human hand to Sharpa dexterous hand
      -> TargetFirstFrame

    PhysicalLanguageTokenSequence + TargetFirstFrame
      -> PhiZero physical-language-conditioned Wan2.2-5B diffusion decoder
      -> TransferredSharpaVideo

For the paper's standard 33-frame representation, nine latent temporal states
produce `(9 - 1) * 32 = 256` discrete symbols. The Physical Language Reasoner is
not required for this transfer demo because the tokens are encoded from the
source video rather than predicted from text.

`PhysicalLanguageTokenSequence` and PhiAgent's existing EPL are different.
PhiZero tokens are learned FSQ indices and are not directly grounded in
interpretable physical variables. EPL remains an auxiliary robotics
representation and must not be reported as a reproduction of the tokenizer.

## Auxiliary PhiAgent pipelines

The explicit robotics route remains useful for downstream analysis:

    HumanVideo -> perception -> EPL -> retargeting -> simulation/repair
      -> trajectory-conditioned Cosmos3-Nano rendering

The trajectory-rendering contract binds accepted robot/object trajectories,
camera calibration, scene assets, control video, prompt, and verification
evidence. This is a separate method and cannot satisfy Figure 8(b).
The MuJoCo control producer resamples the verified joint path at video FPS,
reruns verification, exports measured object poses in a named robot-base frame,
and hashes the complete bundle. Cosmos output is compared frame-by-frame with
the control using edge SSIM; this is a diagnostic and never substitutes for the
pending pose-level alignment gate.

Native Wan2.2-Animate is also an independent motion-transfer baseline. It does
not implement PhiZero's transition Q-Former, FSQ bottleneck, source-domain
adaptation, or physical-language-conditioned Wan2.2-5B decoder.

The agentic proxy wraps released visual-transfer backends in a bounded loop:

    source + Sharpa first-frame candidates
      -> multi-backend/seed candidate generation
      -> local evaluator: motion, identity, object, temporal scores
      -> accept all thresholds or feed diagnoses to the repair agent
      -> revised prompt/seed and another bounded round

All traces use the method label `agentic_proxy_not_official_phizero`. This path
targets a similar visual result without claiming the unreleased representation.

Wan-Animate-2 is an additional direct-driving proxy:

    human driving clip + pose-matched clean Sharpa reference
      -> two-GPU Wan-Animate-2 distilled inference
      -> local evaluation and bounded temporal repair

It removes the original Animate pose/face-control preprocessing dependency. A
pose-matched case-1 run visibly replaces the human hand with one Sharpa hand, but
object and temporal gates remain failed. This is not learned physical language or
unchanged-token decoding.

The accepted long flower visualization applies that same short-window model
without changing its learned weights or inference recipe:

    660-frame source + one fixed robot reference + one fixed prompt/seed
      -> ten overlapping 81-frame Wan-Animate-2 driving jobs
      -> nine identical half-window-offset bridge jobs
      -> per-window physical-GPU preflight and immutable provenance
      -> bounded stable-background color alignment
      -> stable-range minimum-cost seam search over all 19 windows
      -> four-frame-radius cosine blend on corresponding global-time frames
      -> unchanged central core from the reviewed three-second anchor
      -> bounded cosine repair at six diagnosed anchor-external transitions
      -> encoded-metric, dense, before/after, and residual-peak review

The windows share source time, reference, prompt, seed, model, and settings but
do not share diffusion memory. The staggered jobs avoid a repeatable unstable
local-frame position, while explicit protected ranges prevent temporal repair
from rewriting the reviewed three-second core. Overlap and local processing
reduce visible resets; they do not create long-horizon state or prove contact
consistency.

The early-continuity hybrid uses the rejected-as-pixels H3 run only as a motion
teacher:

    stitched Wan pixels + full-length H3 guide + real source
      -> subject-region transition energy and rolling local ratios
      -> retain events where Wan is an outlier against H3/source consensus
      -> convert H3/source motion weights into monotonic bridge timing
      -> interpolate only between nearby Wan endpoints
      -> protect the reviewed three-second Wan anchor
      -> encoded rescoring plus consecutive-frame and dense review

This transfers temporal information without transferring H3 appearance or
identity. It is a bounded repair stage, not shared diffusion state or a learned
long-horizon controller.

Contact-rich, same-scene manipulation needs a stricter route than the gesture
showcase or a whole-frame character generator:

    real RGB video + camera calibration
      -> per-frame person / left hand / right hand / flower / tool tracks
      -> named camera-frame hand, wrist, object, and contact trajectories
      -> target-robot IK and grasp/contact feasibility checks in robot-base frame
      -> pose-conditioned robot rendering or video replacement inside tracked support
      -> source flower/tool layers composited by per-frame depth and contact order
      -> visual gates + simulation gates + real-robot acceptance

The visual branch can produce a presentation proxy; it cannot establish that the
robot can execute the task. Promotion therefore requires independent full-body
replacement, object/contact preservation, temporal, kinematic, collision, and
real-input review gates. No camera-to-robot-base transform is inferred implicitly.

For long-duration and low-latency output, a deterministic hybrid baseline avoids
full-frame generation:

    source video + separately rendered robot layer + source object track
      -> object-relative screen-space robot placement
      -> robot-only alpha composition over unchanged source pixels
      -> exact source-object pixel restoration
      -> optional downstream robot-region enhancement

The current baseline names both source and robot image-pixel frames and does not
infer a camera/world transform. It is intentionally PARTIAL: object-relative
placement is not a calibrated 3D wrist pose, exposed human-hand pixels are not
yet removed, and no generative region enhancement has run. This path targets
stable long-video engineering rather than exact PhiZero reproduction.

The accepted jump-audited flower visualization uses a related but stricter
global trajectory compositor:

    real source + per-frame 2D shoulder/elbow/wrist/hand landmarks
      -> centered temporal-median outlier rejection
      -> zero-phase trajectory smoothing with same-frame index map
      -> reviewed arm-removed robot torso base in camera pixels
      -> generated robot-only alpha cutout aligned by ORB/RANSAC
      -> six alpha-masked rigid robot limb layers
      -> exact-endpoint per-part similarity transforms in camera pixels
      -> hard source-person clear + unchanged source background
      -> full-frame and person-ROI transition gates + consecutive-frame review

It is accepted only for zero detected artificial discontinuities and same-frame
2D action correspondence in the reviewed same-scene visualization. The torso is
not a moving 3D body, the limb transforms are not robot joint commands, and
finger articulation, flower contact, depth, force, and collision safety are not
inferred. Those exclusions prevent a visually clean result from being used as
evidence of task transfer or robot capability.

The CPU-only scene replacement planner adds an explicit contract above local
compositors:

    ordered shots + per-shot normalized entity tracks
      -> identity-stable subject routes with explicit left/right side
      -> hand, hand-and-forearm, or full-body replacement operations
      -> protected-object operations composited after robot layers
      -> structured missing-track and low-confidence diagnostics

Tracks never carry across a declared camera cut. Multiple subjects and protected
objects are routed independently, and replacement pixels are clipped to their
declared image boxes. This solves deterministic scene routing and z-order policy;
it does not infer tracks, generate photorealistic robot pixels, or verify 3D
contact physics.

ArtiCraft remains an independent asset-generation research path:

    text description / reference image
      -> pinned mini-ArtiCraft subprocess
      -> isolated USDZ candidate + upstream run evidence
      -> explicit target-simulator conversion and physical validation
      -> calibrated simulation asset

Generated assets do not bypass simulation gates, but they are not inputs
required by the paper's Sharpa video-transfer protocol.

## Native action-conditioned world-model branch

The AC-WM branch is independent of the PhiZero reproduction mainline and the
generic human-to-robot visual replacement proxies:

    language instruction or exact numeric action + real-scene source
      -> typed frame-explicit ACWMActionCondition
      -> representation/frame compatibility router
      -> isolated OSCAR, BWM, Kinema4D, or FlowWAM adapter
      -> matched-seed candidate batch in a new experiment directory
      -> action, embodiment, object, temporal, and background gates
      -> mandatory human review
      -> accept, native-condition repair, or compatible-backend reroute

The router preserves the training-time action interface of every backend.
OSCAR consumes `camera:*` 2D robot skeleton videos. Boundless World Model
consumes exactly 14 `robot_base:*` EEF or joint channels. Kinema4D consumes a
robot pointmap condition and additionally requires an explicit robot URDF and
camera calibration. FlowWAM consumes robot-only optical-flow videos and requires
the URDF, camera calibration, and flow-producer provenance that produced them.
A camera-pixel wrist trace is never silently promoted to a metric robot-base
action, 3D pointmap, or geometry-grounded robot flow.

The interactive numeric branch specializes the BWM route to its released
57-action-frame, dual-arm absolute-EEF contract. Action sample rate and output
video FPS are separate. Its standard-library compiler either preserves every
supplied 14D frame or interpolates declared keyframes with linear translation
and profile-appropriate Euler interpolation or quaternion SLERP. It validates
the named frame and channel profile against the exact normalization statistics
and can lock frame 0 to a measured first-frame state. A durable single-worker
job manager invokes the same pinned GPU adapter, records provenance and
failures, and serves generated MP4 byte ranges to the demo. The output remains
pending evaluation and human review; the API does not equate successful
decoding with action adherence or execution.

The current Hand2Dex-2 source has only the native OSCAR action condition, so the
OSCAR path ran while BWM and Kinema4D stopped at the input-contract gate. Raw
OSCAR accepts upward lifting; posthoc user review rejects the rightward result
because its hand topology fragments, and leftward movement fails to move the
object. The bounded morphology-repair branch is:

    shared real/robot first frame
      -> pinned SAM2 prompted robot-hand mask
      -> select one bowl-excluding connected component
      -> fixed-scale rigid projection from camera skeleton contact/elbow points
      -> inpaint only declared hand support
      -> restore protected yellow-object pixels exactly before encoding
      -> topology, area, edit-support, object, visual, and AC-WM gates

The repaired rightward result keeps one connected hand and passes the complete
gate, so the workflow is 2/3 accepted after one labelled repair while raw OSCAR
is 1/3. Finger articulation is deliberately frozen; the repair cannot be
reported as raw model morphology, 3-D hand kinematics, calibrated contact
physics, or physical robot execution. See `docs/ACWM_WORKFLOW.md` for revisions,
commands, metrics, and evidence paths.

## PhiZero acceptance gates

1. Pin the paper, project-page revision, official code/model revisions, and all
   three public `hand2dex` source/transferred reference pairs.
2. Run the released tokenizer and decoder rather than approximating their outputs
   with EPL, Wan2.2-Animate, Cosmos, or kinematic retargeting.
3. Confirm source-domain adaptation uses HRDexDB human videos without paired
   robot data or cross-embodiment correspondence.
4. Reuse the source physical-language sequence unchanged and change only the
   first-frame embodiment condition.
5. Persist source video, edited first frame, tokens, configs, model/source
   revisions, seed, logs, and transferred video.
6. Evaluate motion preservation, Sharpa identity/geometry, object interaction,
   and temporal consistency against the three pinned 3-second, 896x512 pairs.

## Coordinate and unit contract

Every 3D point carries a `FrameRef`; every pose is named `target_T_source`.
Transforms reject mismatched composition, and projection only accepts camera
frames. EPL v0.1 uses metres, seconds, and XYZW unit quaternions. Robot joint
ordering is name-based and checked against declared limits. The current robot
trajectory schema supports revolute joints in radians; arbitrary prismatic URDF
support is not claimed.

HaMeR outputs are converted from root-relative MANO/OpenPose-ordered joints plus
the official full-frame camera translation into metric camera-frame keypoints.
FoundationPose text matrices are interpreted exactly as `camera_T_object`.
Absolute dex-retargeting POSITION mode is rejected until a named
camera-to-robot-base calibration transform is supplied; vector modes are
translation invariant.

## Simulation acceptance

`SimulationRequest` distinguishes required contact pairs from forbidden contact
pairs and can specify terminal object-position goals. MuJoCo returns all contacts
but counts a collision only when its geom pair is explicitly forbidden. Physical
validity, task success, collision, contact, reachability, joint limits, object
poses, and rollout paths remain separate fields.

The agent loop invokes explicit simulate, collision, contact, and reachability
checks. Its current safe repair is declared joint-limit clamping; unsupported
collision/contact repairs fail rather than fabricate success. These checks apply
to the auxiliary robotics pipeline and are not claimed as the PhiZero method.
