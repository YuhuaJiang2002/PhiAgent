# Agentic AC-WM workflow

Evidence date: 2026-08-12.

## Result

PhiAgent now has a native action-conditioned world-model branch for real-scene
counterfactual video generation. The branch compiles language into a typed,
frame-explicit action contract, routes only to a backend that natively accepts
that representation, runs the model in an isolated GPU environment, evaluates
each candidate, and requires explicit human review before acceptance.

The real OSCAR-2B result is `PARTIAL`. `lift-up` passes as a native model result.
The original direct `slide-right` passes coarse numeric proxies but is rejected
by user review because its hand fragments in later frames. A SAM2-based
fixed-topology repair removed the fragmentation, but the user correctly rejected
it too because the whole hand moves like a rigid pasted layer. The accepted
rightward result instead reuses the successful lift's reviewed vertical motion,
adds a rightward arc, and asks OSCAR to regenerate the full shoulder, elbow,
wrist, palm, and finger motion. `slide-left` and its stronger action-condition
retry still fail. Two of the three selected action types now pass as native
OSCAR generations; the direct-slide attempts remain negative evidence.

| Case | Action adherence | Embodiment | Object | Temporal | Background | Human | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `slide-left` | 0.2848 | 0.9175 | 1.0000 | 0.6409 | 0.8926 | fail | REJECTED |
| `slide-right` raw | 0.8401 | 0.9515 | 0.9506 | 1.0000 | 0.9203 | user fail | USER-REJECTED STRUCTURE |
| `slide-right` structure-locked | 0.8349 | 0.9428 | 0.9259 | 1.0000 | 0.8734 | user fail | USER-REJECTED RIGID SHIFT |
| `carry-right` lift-arc native | 0.9492 | 0.8557 | 0.9259 | 1.0000 | 0.9131 | pass | ACCEPTED |
| `lift-up` | 0.9937 | 0.9680 | 0.9753 | 0.8659 | 0.9233 | pass | ACCEPTED |

All numeric gates are 0.75 and human review is mandatory. The evaluator uses
yellow-object and image-edge proxies, so these results demonstrate controlled
image-space futures, not calibrated 3D action, contact force, or real-robot
execution.

## EPIC Ego ten-second visual recovery

The first EPIC-KITCHENS `P03_28` bottle comparison is user-rejected: all three
actions show source-human hand or sleeve ghosts and the repaired output is too
blurred. The failure came from restoring source pixels after H3 generation, not
from a missing decode check. Its ledger record is retained and the old videos
remain negative evidence.

The recovery route follows the accepted cabbage demo's stronger invariant: one
joint subject replacement is the candidate, and no source-person alpha repair or
temporal blur runs afterward. H3 supplies an action-specific 240-frame robot
driver. Wan2.2-Animate receives one robot reference, the full driver pose, five
history frames, and the real Ego image only outside a conservative replacement
support. `factored_guard` combines the driver subject mask with a lower-frame
human-risk guard, so an uncertain human pixel is generated rather than copied.
The source face control is blacked out. Every run records its physical GPU,
`CUDA_VISIBLE_DEVICES`, seed, revisions, commands, masks, logs, and hashes.

This engineering route is informed by, but does not claim to reproduce, recent
training papers:

- [Robot-Factored World Models](https://arxiv.org/abs/2607.22535) motivates
  factoring visible robot geometry from environment prediction.
- [OSCAR](https://arxiv.org/abs/2606.04463) supports explicit kinematic pose
  control and domain-curated egocentric/robot training data.
- [LongVie](https://arxiv.org/abs/2508.03694) and
  [LongVie2](https://arxiv.org/abs/2512.13604) motivate global control
  normalization, history context, and degradation-aware training/routing.
- [Cosmos Policy](https://arxiv.org/abs/2601.16163) and the official
  [Cosmos Predict2.5 repository](https://github.com/nvidia-cosmos/cosmos-predict2.5)
  define the future path for action-conditioned stateful world-model training.

The completed recovery produces three 880x512, 240-frame, 24 FPS videos. Dense
60-frame/action review plus full-resolution boundary frames finds no visible
human residual or destructive blur. Foreground p10 sharpness is 1.2345--1.6177x
the user-rejected version, safe-background/source sharpness is 0.8781--0.9287,
and all pairwise action MAD means are 22.4176--26.6564. This is WORKING visual
recovery, but action control remains PARTIAL and image-space only.

The deployed recovery uses the pinned Wan and H3 checkpoints without claiming
that those papers were retrained locally. The repository contains an
explicit BWM action-adapter training entry point and a real-robot WorldArena
compiler. The older RoboTwin conversion module is historical and excluded from
active training, evaluation, and claims. BWM action promotion remains gated on
balanced real-data training, frozen real-video evaluation, and physically
executed correct/counterfactual branches.

## Branch structure

```text
language instruction + real scene
  -> ACWMActionCondition
       representation + named frame + timestamps + channels + values
  -> native-capability router
       camera:skeleton         -> OSCAR
       robot_base:EEF/joints   -> Boundless World Model
       camera:pointmap + URDF  -> Kinema4D
       camera:robot_flow + provenance -> FlowWAM
  -> pinned backend batch on one selected physical GPU
  -> action / embodiment / object / temporal / background evaluator
  -> mandatory human review
  -> accept, repair the native condition/prompt and resample, or reroute
```

`phiagent/acwm/schema.py` rejects implicit camera-to-robot-base relabeling.
Screen-pixel skeleton paths cannot be passed to a model that expects metric EEF
or joint actions. Kinema4D additionally requires an explicit robot URDF and
camera calibration. These checks prevent a visually plausible condition from
being reported as a physically grounded robot action.

## Exact numeric action-to-video path

`phiagent/acwm/numeric.py` and `phiagent/agent/numeric_action.py` add an
executable numeric-control skill above the native BWM adapter:

```text
exact JSON samples or numeric keyframes
  -> strict 14D validation
       robot_base:* frame
       dual-arm XYZ meters + Euler/gripper
       or dual-arm XYZ meters + quaternion XYZW
       exact coordinate-frame/channel match to action statistics
  -> preserve 57 exact samples
       or linear-position + Euler-linear / quaternion-SLERP interpolation
  -> optional measured frame-0 lock against the configured first-frame state
  -> immutable action.json + request.json + job.json
  -> BWM physical-GPU preflight and isolated inference
  -> prediction.mp4 + backend metadata + experiment manifest
  -> browser Range playback
  -> GENERATED / PENDING REVIEW
```

The fixed public-checkpoint contract is 57 action frames. Action sample rate and
generated-video FPS are separate fields: for example, the selected WorldArena2
case has synchronized 30 Hz observations/actions while BWM writes a 24 FPS
prediction. Exact-sample mode does not interpolate, resample, transform, or
relabel values. Keyframe mode requires frames 0 and 56 and records its
profile-specific interpolation in the action timeline. The semantic instruction
describes the scene, but the 14D values are the motion condition.

WorldArena2 `observations/end_pose` contains two `XYZ + quaternion XYZW`
poses. The previous compiler preserved these values but mislabeled the final
four-value blocks as Euler plus gripper. The corrected compiler and
`scripts/prepare_worldarena_numeric_action_case.py` prove unit quaternion norms
before emitting derived channel metadata; historical numeric arrays and hashes
are not rewritten.

Start the loopback service with a real scene, the matching named robot frame,
the pinned BWM artifacts, and preferably a measured default condition:

```bash
PYTHONPATH=. python scripts/serve_numeric_acwm_demo.py \
  --first-frame /absolute/path/to/first-frame.png \
  --source-video /absolute/path/to/source-video.mp4 \
  --coordinate-frame robot_base:your-robot \
  --default-condition /absolute/path/to/measured-57-frame-action.json \
  --bwm-repo external/boundless-world-model \
  --bwm-base-model checkpoints/Wan2.2-TI2V-5B \
  --bwm-checkpoint checkpoints/BWM/step-12000.safetensors \
  --bwm-action-stats /absolute/path/to/action-stat.json \
  --gpu 0
```

The service exposes:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/numeric-capabilities` | action frame/rate, output FPS, named frame, channel profile, unit, and training bounds |
| `POST /api/numeric-jobs` | validate an explicitly matching `coordinate_frame`, compile, persist, and queue one exact action |
| `GET /api/numeric-jobs/<id>` | durable status and generated-video URL |
| `GET /api/numeric-jobs/<id>/action` | canonical per-frame action actually sent to BWM |
| `GET /api/numeric-jobs/<id>/video` | byte-range MP4 playback after generation |

Every completed or failed job is appended to the experience ledger. Failures do
not return a preset or synthetic success-shaped fallback. Generation alone
remains `PARTIAL`: action adherence, embodiment consistency, causal object
interaction, and human review must still run before acceptance.

### Measured WorldArena2 run

The selected real input is held-out `wipe_table/episode_0`, source/action frames
296--352. It is preferred over the other 19 test episodes because the right EEF
travels 0.484006 m with 0.126686 m terminal displacement while all 798 values
remain inside both the matching training min/max and p01/p99 bounds. The source
is 640x480 at 30 Hz and the two four-value orientation blocks have norms within
`[0.9999999727, 1.0000000271]`.

Official BWM revision `738a8d3c` ran on `a800-4` physical A800 GPU 0, selected
with 81,225 MiB free and no compute process. Seed 20260812 and 20 inference steps
produced 57 frames at 896x672/24 FPS in 52.9085 seconds. The generated MP4 hash
is `5f62ef4c53139d62456347ad706242597f1b35b3d7f2c8fc8e0abffb36b27cae`.
Future SSIM is 0.85843 and motion-amplitude error is 0.08721, but flow-direction
cosine is only 0.14303. Dense storyboard review also finds a large later
shadow-like duplicate. The result is therefore published as
`GENERATED / REJECTED / PARTIAL`, not accepted control or execution evidence.

## Model router

| Backend | Native action input | Current route | Reason |
| --- | --- | --- | --- |
| OSCAR-2B | first frame, `camera:*` 2D skeleton video, prompt | RAN | all native inputs are present |
| SAM2 morphology lock | reviewed canonical robot-hand mask + existing camera skeleton | USER-REJECTED | stable silhouette but rigid whole-hand translation |
| Boundless World Model | 14-channel `robot_base:*` EEF or joint sequence | RAN / USER-REJECTED | synchronized WorldArena quaternion state is now supported; direction and duplicate-robot gates fail |
| Kinema4D | robot RGB+pointmap condition, URDF, camera calibration | GATED | calibrated geometry preprocessing is absent |
| FlowWAM | robot-only optical-flow video, URDF, camera calibration, flow provenance | INTEGRATED / GATED ON THIS SCENE | released backend is wired; Cobot-Magic action-to-flow calibration is not verified |
| MiniMax-H3 | reference images/video and text/control proxy | RAN / PARTIAL | 10 s macro-action distinctness passes, but strict window and seam gates fail |

The BWM, Kinema4D, and FlowWAM adapters preflight native artifacts, revisions,
environment, and GPU. BWM has run on synchronized WorldArena state. Kinema4D and
FlowWAM remain gated for that Cobot-Magic scene because manufacturing pointmaps
or robot-only flow without a verified robot/camera producer would invalidate
the comparison.

## Pinned upstream inputs

| Component | Revision |
| --- | --- |
| OSCAR source | `4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb` |
| OSCAR-2B | `c9781ffa7dd8556d862d7d9f338a2ea008a58ca6` |
| Cosmos-Reason1-7B runtime | `3210bec0495fdc7a8d3dbb8d58da5711eab4b423` |
| Wan2.1 VAE runtime | `37ec512624d61f7aa208f7ea8140a131f93afc9a` |
| SAM2 source | `0e78a118995e66bb27d78518c4bd9a3e95b4e266` |
| Boundless World Model source | `44acfd1b06f35f365f02f7bb2fc5da6beafcd6bc` |
| Boundless World Model weights | `738a8d3c008e637b8b1b18d5e98a82f6de9c04aa` |
| Kinema4D source | `716e80249376cb2843af41188a832d56a2d8d78d` |
| Kinema4D weights | `0c52ee34ee464e9a568e84945e431f62106c4270` |

Official project pages: [OSCAR](https://github.com/wuzy2115/oscar-public),
[Boundless World Model](https://github.com/boundless-large-model/boundless-world-model),
and [Kinema4D](https://github.com/mutianxu/Kinema4D).

## Reproduction

Prepare the exact OSCAR checkout, isolated environment, primary checkpoint, and
its pinned text-encoder/VAE runtime assets:

```bash
python scripts/prepare_acwm_models.py \
  --backend oscar \
  --install \
  --download-model
```

Compile the three camera-pixel skeleton conditions from the retained real-scene
source and trajectories:

```bash
PYTHONPATH=. python scripts/build_oscar_bowl_skeleton_conditions.py \
  --control-run outputs/acwm-bowl-action-controls/20260810T121000Z-hand2dex2-v1 \
  --action-manifest demo/acwm_bowl_actions.json \
  --experiment-dir outputs/acwm-oscar-conditions/NEW_UNIQUE_RUN_ID
```

The accepted rightward action is a separate immutable condition run. It uses
the reviewed `lift-up` y trajectory as a vertical template while retaining the
rightward x target, and it carries a prompt that explicitly requires continuous
native joint articulation:

```bash
PYTHONPATH=. python scripts/build_oscar_bowl_skeleton_conditions.py \
  --control-run outputs/acwm-bowl-action-controls/20260810T121000Z-hand2dex2-v1 \
  --action-manifest demo/acwm_oscar_reference_actions.json \
  --experiment-dir outputs/acwm-oscar-conditions/NEW_UNIQUE_LIFT_ARC_RUN_ID \
  --vertical-motion-template slide-right=lift-up

PYTHONPATH=. python scripts/run_agentic_acwm.py \
  --condition-manifest outputs/acwm-oscar-conditions/NEW_UNIQUE_LIFT_ARC_RUN_ID/manifest.json \
  --case slide-right \
  --backend oscar \
  --oscar-repo external/oscar \
  --oscar-checkpoint checkpoints/OSCAR-2B \
  --oscar-cosmos-reason checkpoints/OSCAR-runtime/Cosmos-Reason1-7B \
  --oscar-wan-vae checkpoints/OSCAR-runtime/Wan2.1_VAE.pth \
  --offline --seed 20260810 --maximum-rounds 1
```

Run the matched batch. The adapter inspects physical GPUs, selects or validates
one with sufficient free memory, sets `CUDA_VISIBLE_DEVICES`, loads OSCAR once,
and records the complete run:

```bash
PYTHONPATH=. python scripts/run_agentic_acwm.py \
  --condition-manifest outputs/acwm-oscar-conditions/20260810T131648Z-hand2dex2-v2/manifest.json \
  --backend oscar \
  --oscar-repo external/oscar \
  --oscar-checkpoint checkpoints/OSCAR-2B \
  --oscar-cosmos-reason checkpoints/OSCAR-runtime/Cosmos-Reason1-7B \
  --oscar-wan-vae checkpoints/OSCAR-runtime/Wan2.1_VAE.pth \
  --offline \
  --seed 20260810 \
  --maximum-rounds 1 \
  --human-review-dir outputs/acwm-open-models/20260810T143008Z-ae32011f/human-review
```

The evaluator can also be run independently. A candidate cannot be accepted
without a human-review JSON:

```bash
PYTHONPATH=. python scripts/evaluate_acwm_candidate.py \
  --candidate candidate.mp4 \
  --condition action.json \
  --first-frame first-frame.png \
  --source real-scene-source.mp4 \
  --metadata candidate.metadata.json \
  --human-review human-review.json
```

Build the portable browser evidence package:

```bash
PYTHONPATH=. python scripts/build_oscar_acwm_showcase.py
```

The historical morphology guard is reproducible in two stages. SAM2 runs as an optional
GPU tool and records its physical-GPU selection, checkpoint hash, prompts,
candidate scores, source revision, packages, and reviewed single-component
mask. The second stage is deterministic CPU compositing:

```bash
PYTHONPATH=. python scripts/segment_acwm_canonical_hand.py \
  --image first-frame.png \
  --sam2-repo external/sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_large.pt \
  --positive-point 371,267 --positive-point 388,258 \
  --positive-point 404,265 --positive-point 425,307 \
  --positive-point 385,339 --positive-point 455,347 \
  --negative-point 303,267 --negative-point 350,215 \
  --negative-point 520,270 --negative-point 510,410 \
  --box 330,225,488,375 \
  --output-dir outputs/acwm-hand-canonical/NEW_UNIQUE_RUN_ID

PYTHONPATH=. python scripts/repair_acwm_hand_structure.py \
  --candidate raw-slide-right.mp4 \
  --condition slide-right/action-condition.json \
  --canonical-image first-frame.png \
  --canonical-mask canonical-hand-mask.png \
  --scale 0.75 \
  --output-root outputs/acwm-hand-structure
```

The user-rejected repair has one projected connected component in all 81 frames, a
1.0034 maximum/minimum mask-area ratio, fixed 0.75 scale, zero pre-encode changes
outside declared edit support, and zero changed protected-object channels. It
intentionally freezes finger articulation; that exact property caused the
visible rigid-translation failure. It is retained as negative evidence, not as
an accepted 3-D hand controller.

## Evidence locations

- Main OSCAR experiment:
  `outputs/acwm-open-models/20260810T143008Z-ae32011f`
- Condition-repair experiment:
  `outputs/acwm-open-models/20260810T144421Z-efdb8fea`
- Reviewed SAM2 canonical hand used by the rejected rigid repair:
  `outputs/acwm-hand-canonical/20260810T161000Z-oscar-slide-right-sam2-v5`
- User-rejected fixed-topology rightward repair:
  `outputs/acwm-hand-structure/20260810T162000Z-oscar-slide-right-lock-v1`
- Accepted native articulated lift-arc carry:
  `outputs/acwm-open-models/20260810T155518Z-06311bc4`
- Accepted lift-arc condition bundle:
  `outputs/acwm-oscar-conditions/20260810T155400Z-hand2dex2-right-lift-arc-v2`
- Portable comparison and manifest: `demo/showcase/oscar-acwm-*`
- UI: `demo/index.html`
- CPU regression tests: `tests/test_acwm_workflow.py`

The main experiment records command, Git state, hostname, package freeze, source
hashes, selected physical GPU, backend preflight, requests, model outputs,
candidate metadata, evaluator evidence, and the final trace. Failed setup and
inference attempts remain in their own immutable experiment directories and in
the experience ledger.
