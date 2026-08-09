# Known issues

1. The official PhiZero repository at pinned revision
   `6bc7428f2ad5282e0c1a7b122465957b6abb1edc` contains only a release notice:
   code and pretrained models are still being prepared. Exact Figure 8(b)
   inference is therefore blocked.
2. PhiZero's learned 25K-symbol FSQ physical language is not PhiAgent's typed EPL.
   Replacing the unavailable tokenizer with EPL would be a different method, not
   a reproduction.
3. The paper briefly adapts the tokenizer on the human-hand portion of HRDexDB,
   but does not publish the exact subset, adaptation checkpoint, or first-frame
   GPT-Image 2.0 edit prompts/assets needed to duplicate the displayed outputs.
4. The public project-page repository exposes three 3-second, 896x512 source and
   transferred pairs, but no license file was found at the pinned site revision.
   PhiAgent downloads them on demand into an ignored directory and does not
   redistribute them.
5. The agentic proxy can search released generators but cannot recover or verify
   PhiZero's unavailable discrete tokens. Its four score thresholds are local
   engineering gates, not metrics reported by the paper.
6. Full-frame temporal averaging can hide severe late hand/object flicker. The
   first case-1 run used two Wan diffusion clips for 89 frames with
   `frame_num=49` and one-frame overlap. A true `frame_num=97` single-clip run
   removed that boundary but still scored only 0.279 regionally, showing that
   model-level localized instability remains.
7. The EPL repair-policy result is measured on deterministic synthetic
   diagnostics whose ambiguous labels were constructed to require phase/contact
   context. Its 1.000 accuracy and 10.8-point ablation gain do not establish
   performance on simulator-generated failures, real videos, or a robot.
8. Wan2.2-Animate does not condition generation on its prompt, so feedback text
   only affects future prompt-aware backends; seed and target-image changes are
   the effective Wan repair actions.
9. The Cosmos3-Nano backend implements the frame-explicit
   `TrajectoryConditionedVideoRenderer` contract, but no real GPU inference has
   run. The deterministic MuJoCo control producer and edge-SSIM diagnostic are
   implemented but have not run in the pinned Python 3.10 simulation
   environment. Edge control preserves 2D structure, not exact 3D pose, and the
   pose-level generated-video alignment evaluator is still missing.
10. RoboMaster is a plausible 2D collaborative-trajectory video teacher, but its
   official code repository has no published license. Its released base
   CogVideoX checkpoint also has separate terms. PhiAgent therefore does not
   execute or vendor it.
11. Wan2.2-Animate is a separate character-animation model, not PhiZero's
   physical-language tokenizer and decoder.
12. Native Wan animate accepts a prompt argument but does not pass it into its
   animation generator. PhiAgent records the prompt as provenance only.
13. Basic upstream pose retargeting assumes a front-facing stretched reference.
   FLUX retargeting remains opt-in under separate model terms.
14. HaMeR requires MANO assets obtained directly under MANO's terms. Its
   detector stack is best isolated in a dedicated environment; it has not yet
   run on a real project video.
15. FoundationPose is license-restricted and needs RGB-D, a mesh, intrinsics, and
   an initial mask. The core package only consumes its explicit pose matrices.
16. EPL v0.1 has conservative centre-distance contact inference and no learned
   slip estimator. MuJoCo slip events are not yet implemented.
17. The linear retargeter is an auditable smoke baseline, not inverse kinematics.
   dex-retargeting is limited to EPL wrist/fingertip vector configurations until
   calibrated camera-to-base transforms are added.
18. The current `RobotTrajectory` field is revolute-joint/radian specific;
   general prismatic and mixed-unit URDF action support remains future work.
19. The supported mini-ArtiCraft release exports USDZ candidates. PhiAgent does
   not yet convert these assets to its MuJoCo scene format or verify generated
   scale, mass, contacts, collision geometry, and grasp suitability.
20. Legacy full-frame object-consistency scoring accepted a visibly dropped
   spoon with score 0.8698. Evaluator v4 now takes the minimum of instance
   contour, color, deformation, coverage, relative trajectory, and lift recall;
   it scores that failure 0.0. The current color tracker is designed for the
   cyan case-1 object and still requires an explicit first-frame ROI.
21. Replacement mode can leave exposed human pixels when its character mask is
   smaller than the human silhouette. Duplicate-object cleanup fixes the spoon
   instance but does not fix those residual human regions; the repaired case-1
   candidate therefore remains PARTIAL.
