# Grounded action-conditioned world models

Evidence date: 2026-08-13.

## Thesis

Vector-conditioned RGB generators fail at precise robot control because a
robot-base action identifies a physical state but does not identify the pixels,
depth, links, contacts, or background region that must realize that state.
PhiAgent therefore treats dense, frame-aligned geometry as the primary control
object and RGB generation as a residual rendering problem.

## Measured failure that motivates the redesign

The official Boundless World Model run at
`outputs/numeric-bwm-real-inference/20260812T135323.974031Z-worldarena-wipe-episode0-official-seed20260812`
uses:

- one synchronized WorldArena2 real-robot input;
- 57 measured dual-arm `XYZ + quaternion XYZW` states at 30 Hz;
- a named `robot_base:worldarena2-cobot-magic-max-end-pose` frame;
- matching action statistics with zero values outside the training p01/p99
  interval;
- official BWM revision `738a8d3c`, seed 20260812, and 20 denoising steps.

The video reaches future SSIM 0.85843 but flow-direction cosine only 0.14303 and
develops a large shadow-like duplicate. SSIM is therefore a scene-appearance
guard, not action-causality evidence.

Source inspection explains the result. BWM maps each 14-D state through MLPs and
injects the embeddings through cross-attention and timestep modulation. It does
not provide a robot occupancy mask, projected links, depth, pointmap, contact
state, or fixed-background constraint. The public `cfg_scale` argument was also
unused by the released denoising loop.

## First-principles requirements

### Action causality

The required chain is:

```text
robot-base action
  -> controller / FK
  -> camera-aligned robot geometry or flow
  -> rendered future
```

A matched counterfactual action must change the generated robot trajectory while
the first frame, seed, model, and scene remain fixed.

### Robot topology

One URDF and one joint trajectory should define link count, connectivity, joint
limits, and handedness. A diffusion model should not be responsible for
inventing robot kinematics.

### Contact and object response

Object motion must come from a contact/physics state trajectory or explicit 3-D
object tracks. Pixel overlap alone is not contact evidence.

### Background invariance

For a fixed camera, pixels outside the temporally dilated
robot/object/contact/disocclusion support should be copied from observed scene
state. A generator should synthesize residual appearance, not redraw the entire
laboratory.

## Verified released models

| Model | Primary spatial action representation | Open release | Native limitation |
| --- | --- | --- | --- |
| [FlowWAM](https://arxiv.org/abs/2607.13017) | robot-only optical-flow video held fixed while RGB is denoised | [Apache-2.0 code](https://github.com/YixiangChen515/FlowWAM_WorldArena) at `f06fa46042e97738c6619c868f1097be6749d48d`; [ungated Apache-2.0 weights](https://huggingface.co/YixiangChen/FlowWAM) at `1e68f76cecfb2caa973abfb24fca92cbc5312a6e` | Released WorldArena path expects RoboTwin-compatible robot-only rendering or an equivalent precomputed flow producer |
| [Kinema4D](https://arxiv.org/abs/2603.16669) | robot RGB plus per-pixel XYZ pointmaps, or pointmaps only | [Apache-2.0 code](https://github.com/mutianxu/Kinema4D) at `716e80249376cb2843af41188a832d56a2d8d78d`; [released checkpoints](https://huggingface.co/Minoday/Kinema4D) | 49-frame Wan2.1-I2V-14B path; requires URDF, camera calibration, and prepared pointmaps |
| [OSCAR-2B](https://arxiv.org/abs/2606.04463) | projected 2-D kinematic skeleton video | [Apache-2.0 code](https://github.com/wuzy2115/oscar-public) at `4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb`; [released checkpoint](https://huggingface.co/zywu2115/OSCAR-2B) | Skeletons constrain joints in image space but not surface depth or occupancy |
| [RynnWorld-Teleop](https://arxiv.org/abs/2607.06558) | depth-aware hand/skeleton control video | [Apache-2.0 code](https://github.com/alibaba-damo-academy/RynnWorld-Teleop) and released SFT/causal checkpoints | Primarily egocentric dexterous teleoperation |
| [Ctrl-World](https://arxiv.org/abs/2510.10125) | frame-aligned DROID actions and multiview history | [MIT code](https://github.com/Robert-gyj/Ctrl-World) and [weights](https://huggingface.co/yjguo/Ctrl-World) | DROID/Franka-specific, low-resolution, short native prediction window |
| [IRASim](https://arxiv.org/abs/2406.14540) | frame-level numeric robot actions | [Apache-2.0 code and checkpoints](https://github.com/bytedance/IRASim) | Older 16-frame, low-resolution, dataset-specific baseline |

FlowWAM is the highest-priority executable backend because its dense flow
condition directly supplies motion direction and support while fitting one
A800-80GB GPU. Kinema4D is the strongest released 3-D mechanism when exact
calibration and a robot model are present. OSCAR is the cheapest geometry probe.

Newer methods support the same factorization but are not currently executable:
[RoFacto](https://arxiv.org/abs/2607.22535),
[GeniWorld](https://arxiv.org/abs/2608.06332),
[Mask2Real-WM](https://arxiv.org/abs/2607.04546), and
[ContactFlow](https://arxiv.org/abs/2607.26579) do not release complete
inference weights/code as of the evidence date.

## Experiments

### Target-versus-hold action guidance

PhiAgent implements an auditable BWM ablation:

```text
epsilon = epsilon_hold
        + scale * (epsilon_target - epsilon_hold)
```

The hold condition repeats the normalized initial state. Scale 1 executes the
released single-pass target branch and produces a bit-identical MP4.

On the frozen WorldArena input and seed 20260813:

| Scale | Future SSIM | Flow cosine | Flow EPE | Motion-amplitude error | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 unpatched | 0.86796 | -0.05725 | 2.37988 | 0.25991 | baseline |
| 1.0 patched | 0.86796 | -0.05725 | 2.37988 | 0.25991 | exact SHA match |
| 1.5 | 0.86144 | -0.07644 | 2.48897 | 0.04394 | reject |
| 2.0 | 0.85794 | -0.02965 | 2.52264 | 0.08258 | reject |
| 3.0 | 0.84906 | -0.04094 | 2.62816 | 0.40344 | reject |

Guidance changes motion magnitude but does not recover the correct direction and
regresses fidelity/flow EPE. Dense review retains the shadow duplicate. This
negative result rules out “make the vector condition stronger” as the primary
solution.

### Correct counterfactual geometry

WorldArena `observations/end_pose` is dual-arm
`XYZ + quaternion XYZW`. Earlier action swaps treated quaternion components as
Euler angles and grippers. The corrected counterfactual compiler now:

1. preserves the observed history exactly;
2. rebases donor XYZ displacement at the last history state;
3. computes `q_delta = inverse(q_donor_anchor) * q_donor_future`;
4. applies `q_source_future = q_source_anchor * q_delta`;
5. validates unit quaternion outputs.

Historical causal conclusions based on the old Euler/gripper swap require
replacement rather than reinterpretation.

## Execution plan

1. Treat vector BWM and its rejected guidance sweep as baselines.
2. Add FlowWAM as a dense-flow backend with a strict producer contract:
   robot-only flow, source action, URDF/model revision, camera frame, and hashes.
3. Run FlowWAM first on its native ALOHA/RoboTwin contract to reproduce the
   released checkpoint.
4. Do not apply it to the real WorldArena Cobot-Magic scene until a held-out
   action-to-flow producer passes camera/embodiment calibration. Future RGB or
   optical flow from the evaluation video may be used only as an explicitly
   labelled oracle-renderer ablation.
5. When calibration exists, compare BWM vector, OSCAR skeleton, FlowWAM flow,
   and Kinema4D pointmap with identical first state and matched factual/swapped
   actions.
6. Require positive flow direction, positive correct-vs-wrong action margin,
   one robot component, endpoint pose alignment, background non-regression, and
   human review. SSIM alone cannot promote a model.

## Claim boundary

The released-model survey and BWM guidance ablation identify a defensible
architecture and eliminate one plausible vector-only repair. They do not yet
establish that FlowWAM or Kinema4D solves the current real Cobot-Magic scene,
because that scene lacks a verified URDF/camera-to-flow or pointmap producer.
