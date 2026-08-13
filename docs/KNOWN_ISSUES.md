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
22. The five-finger Shadow showcase is a fixed-camera, single-hand gesture
   retargeting problem with no manipulated object. Flower arranging adds a full
   body, two hands, thin/deformable flowers, tools, self-occlusion, changing
   hand-object depth order, and contact timing. Success on the former is not a
   valid acceptance test for the latter.
23. Wan-Animate-2 substantially improves same-scene appearance on the flower
   contact window, but the measured motion and robot-identity proxies still fail.
   It is a direct character-driving generator, not a calibrated 3D contact or
   dynamics estimator; prompt text cannot guarantee that a stem remains in the
   same gripper at the same time.
24. A single person mask propagated from one frame is not a reliable occlusion
   track for a 660-frame bimanual manipulation. It can either expose the human or
   restore human pixels during object protection. Per-frame multi-object tracks
   are required before background/object locking can be accepted.
25. The old compositor's exact-pixel `subject_change` and support-coverage
   `human_leakage_risk` metrics admitted a false positive: tiny alpha changes
   counted as replacement, while person support was tautologically covered by
   construction. The gate now uses thresholded visible change, mean absolute
   subject change, and mandatory human review; earlier automatic acceptance is
   superseded by the run's `final/review.json`.
26. Independent Wan-Animate-2 windows can reset fine robot/flower details even
   with identical reference, prompt, and seed. The first overlap-only result
   retained periodic spikes because local frame 65 was unstable in multiple
   81-frame windows. Nine half-window-offset bridge jobs, stable-range seam
   search, and bounded local crossfades reduce the encoded maximum transition
   ratio from 3.9027 to 2.6564 while preserving the reviewed three-second core.
   Bidirectional optical-flow repair was rejected because it moved the spikes to
   repair endpoints and raised the maximum ratio above 5.3. The accepted method
   still has no shared diffusion memory; exact stem identity and hand-object
   contact can drift despite improved visual continuity, and any new crossfade
   site requires consecutive-frame review for ghosting.
27. A lower whole-video maximum transition ratio did not guarantee that the
   early sequence looked continuous. Subject-region analysis found 19 early Wan
   spikes that were 2x or more above the motion supported jointly by MiniMax-H3
   and the real source. H3 cannot be substituted directly because its full-run
   robot-identity gate failed. The accepted hybrid therefore copies no H3
   pixels: it uses H3/source motion only to detect and time bounded interpolation
   between nearby Wan endpoints. This reduces the diagnosed early spikes but
   remains local post-processing; new inputs require fresh detection and visual
   ghost review.
28. Pose tracking can report all frames as present while distal wrist/hand
   landmarks repeatedly attach to flowers or occluders. On the 660-frame flower
   source, raw landmark velocity reached 0.185 image diagonals per frame and a
   plain Gaussian either retained jumps or erased too much action. The accepted
   v11 used centered temporal-median outlier rejection before zero-phase
   smoothing, then gated inlier position error and eight-frame action direction
   and magnitude. A second independent failure came from moving flower/wall/
   ribbon pixels embedded in the robot hand texture; a robot-only alpha cutout
   removes that source. Posthoc user review then exposed a third failure: exact
   mapping of a 7.32-pixel right wrist-index anchor made the entire robot hand
   vary from 0.027x to 4.922x scale even though transition gates passed. v12
   decoupled morphology from that noisy endpoint, kept the hand root exact,
   locked hand scale, bridged unreliable directions, bounded angular steps, and
   independently gated transform scale and rendered mask area. It fixed scale
   breathing, but later user review showed that morphology gates alone do not
   establish coherent limb topology.
29. Hard-inserting the short MiniMax-H3 flower result into the recursive long
   timeline creates visible arm/hand and head discontinuities at its entry and
   exit, including residual human skin/hair.  The accepted visual repair is to
   use that run only for EPL phase evidence, recursively condition each long
   window on the prior overlap pose, select hard seams without cross-dissolves,
   relock protected source pixels after lossy intermediate decoding, and apply
   bounded low-motion optical-flow stabilization.  This passes the visual seam
   gate but does not make the result physically valid: identity, source-motion,
   and EPL proxy gates still fail, and the successful local physics evidence is
   a separate five-second pre-grasped authored MuJoCo trajectory rather than a
   successful three-second generated insertion.
30. v12's robot calibration texture had both hands crossed in front of the
   chest. Six overlapping capsule masks therefore copied some finger,
   opposite-arm, and torso alpha pixels into multiple independently transformed
   layers, producing split hands and floating fragments even though scale and
   transition gates passed. v14 uses a non-overlapping calibration pose,
   assigns every opaque source pixel to exactly one nearest normalized limb
   segment, permits overlap only at same-side elbow/wrist joints, and rejects
   any frame whose shoulder-to-hand union has more than one meaningful
   component. The accepted clip has zero cross-side source-mask pixels and zero
   disconnected left/right frames across all 660 frames. Posthoc user review
   still rejected v14 because its broad person clear erased the manipulated
   flowers and its fixed 0.62x hands were too small. v14 therefore verifies only
   reusable 2D limb topology; it is not an accepted flower-arranging result.
31. Manipulated-object preservation cannot be inferred from a person mask or a
   color/instance mask with low semantic recall. v15 demonstrated that tracked
   source masks, GrabCut, and SLIC could pass local pixel checks while retaining
   only flower fragments and mixing ribbons, plants, face, or hands. A complete
   bouquet-only RGBA layer solved semantic completeness, but v16-v18 exposed
   three independent failure modes: wrong depth order put flowers behind whole
   forearms; binding to both wrists made the bouquet float when hands separated;
   and binding 100 pixels along the holding-hand axis amplified a bounded hand
   rotation into a 34.17-pixel object jump. v19 composes upper/lower arms,
   bouquet, then hands; binds the bouquet inside the holding hand; and applies a
   2.5-frame zero-phase filter plus symmetric 12-pixel 2D step bound. Its
   accepted clip has 660/660 nonempty bouquet frames, minimum 537-pixel
   hand-bouquet overlap, 1.002964 bouquet-area p99/p01 ratio, and zero encoded
   full-frame/ROI transition outliers. This remains fixed 2D object composition,
   not flower deformation, articulated grasp/contact force, 3D depth, or
   physical execution.
32. Real-background diversity and analytically correct 2-D training renders do
   not by themselves make a qkv/out-projection Ref2VA LoRA control limb
   topology. The r4b curriculum passes subject, scene, source, and action-tag
   coverage after one residual-human ROI was rejected, yet both its step-12
   full-strength and step-36 half-scale held-out candidates reproduce the
   baseline's 100-frame non-unique left-shoulder/head-clearance failure. The
   route is stopped; future learned work needs an explicit structural loss or a
   backbone/control interface that exposes pose topology, not another capacity
   or LoRA-scale increase.
