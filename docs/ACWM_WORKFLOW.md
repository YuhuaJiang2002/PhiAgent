# Agentic AC-WM workflow

Evidence date: 2026-08-10.

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

## Branch structure

```text
language instruction + real scene
  -> ACWMActionCondition
       representation + named frame + timestamps + channels + values
  -> native-capability router
       camera:skeleton         -> OSCAR
       robot_base:EEF/joints   -> Boundless World Model
       camera:pointmap + URDF  -> Kinema4D
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

## Model router

| Backend | Native action input | Current Hand2Dex-2 route | Reason |
| --- | --- | --- | --- |
| OSCAR-2B | first frame, `camera:*` 2D skeleton video, prompt | RAN | all native inputs are present |
| SAM2 morphology lock | reviewed canonical robot-hand mask + existing camera skeleton | USER-REJECTED | stable silhouette but rigid whole-hand translation |
| Boundless World Model | 14-channel `robot_base:*` EEF or joint sequence | GATED | synchronized robot state and calibration are absent |
| Kinema4D | robot RGB+pointmap condition, URDF, camera calibration | GATED | calibrated geometry preprocessing is absent |
| MiniMax-H3 | reference images/video and text/control proxy | NEGATIVE BASELINE | prior three-action result failed user visual review |

The BWM and Kinema4D adapters are implemented and preflight their native
artifacts, model revisions, environment, and GPU. They have not been run on this
scene, because manufacturing their required 3D inputs from 2D wrist traces
would invalidate the comparison.

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
