# Flower-arranging transfer: first-principles diagnosis and evolution plan

Status: **PARTIAL**. The new hybrid candidate is materially better than the old
2D/generative candidates, but it has not passed the complete semantic contract
or a blind full-video preference review and is not described as “very good.”

## Why the hand replacement works and flower arranging does not

The 20.7-second hand result is a low-entropy, strongly constrained problem:
21 observed hand landmarks are retargeted to one known 24-DoF hand, the edit is
local, there is no independently moving object, and unaffected pixels can be
copied from the source. A deterministic mapping plus temporal filtering is
enough to keep topology and identity stable.

Flower arranging is a coupled long-horizon interaction problem. The output must
simultaneously preserve one humanoid morphology, two arm and hand trajectories,
multiple thin deformable stems, grasp/release state, correct front/back order,
the source scene, and identity for 660 frames. A video generator conditioned by
one reference image and optical flow is under-specified: many visually plausible
videos satisfy those inputs without performing the demonstrated task.

The historical evaluator compounded the problem. It accepted continuity,
connected components, mean flow, or a stable torso as substitutes for task
success. That produced three distinct false-positive families:

- Wan: coherent robot appearance, but measured motion preservation was about
  0.293 and the robot was nearly static.
- H3 layered composition: clearer robot pixels, but missing limbs, missing held
  flowers, and no persistent contact state.
- 2D pose rig: exact image-space limb endpoints and continuity, but paper-like
  morphology, no 3D depth, and no contact physics.

## New hard contract

There is no aggregate acceptance score. Every gate must pass on all 660 frames:
background lock, absence of human remnants, robot morphology, robot identity,
embodied motion, flower-instance integrity, hand-flower contact, occlusion order,
temporal consistency, and blind full-video human preference. Missing evidence is
a failure, and user review can veto every proxy score.

The implementation is in `phiagent/agent/flower_evolution.py`. It changes the
pipeline family when a semantic gate repeatedly fails. Prompt, flow, blur, and
transition repair are forbidden responses to morphology or contact failures.

## Evidence from the current evolution

1. The constraint-first audit rejected the old H3 v34 baseline on nine hard
   gates and selected an explicit 3D layered representation.
2. A MuJoCo G1 plus articulated Sharpa hands removed generative limb melting and
   identity drift.
3. The first contact calibration used an unreachable wrist gain and failed at
   0.1363 m maximum IK error. The bounded v3 rerun reduced this to 0.1162 m and
   passed the declared 0.12 m structural gate on all 660 source frames.
4. Camera alignment reduced median wrist error from 80.8 to 29.7 pixels. On 574
   source-contact observations, conditional flower proximity reached 0.9094.
5. The same candidate still scores only 0.625 for wrist motion within 35 pixels,
   with a 0.353/0.897 left/right split. Its flower mask is a union, not a held
   instance, and every flower is placed in front of the robot. It therefore
   remains rejected by motion, contact-identity, and occlusion gates.

## Paper-informed next representation

- [X-Humanoid](https://arxiv.org/abs/2512.04537) shows that third-person
  human-to-humanoid conversion benefits from a task-specific video-to-video
  model trained on paired synthetic human/humanoid clips, rather than zero-shot
  full-frame prompting.
- [SPIRAL](https://arxiv.org/abs/2603.08403) motivates the think-act-reflect loop:
  decompose approach/grasp/manipulate/release, generate phase-local candidates,
  critique each segment, and carry failure memory forward.
- [VISTA](https://arxiv.org/abs/2510.15831) motivates pairwise candidate
  tournaments and specialized critics instead of trusting a single mean score.
- [ObjRetarget](https://arxiv.org/abs/2607.03828) and
  [HOWTransfer](https://arxiv.org/abs/2606.10743) motivate object-aware arm
  trajectories, explicit contact onset, and grasp-preserving hand constraints.
- [Human2Humanoid](https://arxiv.org/abs/2606.03476) motivates morphology-aware
  end-effector consistency plus physical feasibility rather than raw human-pose
  copying.
- [Dream2Act](https://arxiv.org/abs/2603.19709) motivates planning in a
  robot-native video/motion space when direct cross-morphology retargeting cannot
  satisfy interaction constraints.

The next selected family is `robot_centric_adapted`: build paired synthetic
flower-arranging clips, train a task-specific adapter on held-out phases, keep
explicit robot and object trajectories, and use local generation only for
appearance residuals. A candidate can be promoted only after it beats the v5
baseline in blind full-video review and passes every semantic hard gate.
