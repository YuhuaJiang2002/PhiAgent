# Status

Evidence date: 2026-08-09. Status labels describe acceptance evidence, not code
presence. Measured runs span `a800-1` through `a800-4` and `zhaoli`; artifact
locations are recorded per experiment below.

## WORKING

- Phase A remote environment: Python 3.10.12, PyTorch 2.6.0+cu124,
  flash-attn 2.7.4.post1, MuJoCo 3.3.7, CUDA visible, and all dependencies in
  `.venv-gpu` on `a800-1`.
- RTX PRO 5000 Blackwell environment on `pro5000`: Python 3.12.3,
  PyTorch 2.7.1+cu128, FlashAttention 2.8.3, and the pinned SAM2 CUDA
  extension execute on compute capability 12.0. Both Wan animation and
  replacement preflights pass against the complete pinned 68 GiB checkpoint;
  12 targeted Wan adapter tests and Ruff pass.
- Native Wan2.2-Animate execution on the pinned official upstream sample:
  physical GPU 4 produced a decodable 1280x720, 30 FPS, 105-frame H.264 video
  with SHA-256 `49c6c0c4440eba79e08b65ff78e51312e94bde91880eaed9e5d61b91d7a61188`.
  This validates model execution, not the requested human-to-robot evaluation.
- Remote regression suite: 26 tests passed and Ruff passed.
- Local regression after background-safe Sunday hand repair:
  127 tests passed and
  Ruff passed; the optional MuJoCo test module was skipped because MuJoCo is not
  installed in the local lightweight environment.
- PhiZero Milestone 0: arXiv:2607.28624v1 Figure 8(b), Appendix C.2, official
  site/code revisions, and all three public `hand2dex` reference pairs are pinned.
  `external/PhiZero-reference/manifest.json` verifies all six files by size and
  SHA-256.
- Wan Animate clip length is explicit through validated `frame_num`; upstream's
  `infer_frames` applies to S2V and is ignored by Animate. The evaluator takes
  the minimum of global, final-third, and high-jerk regional temporal scores so
  localized late flicker cannot be hidden by full-frame averaging.
- Lightweight EPL repair-action policy: nine matched 20k-example, 60-epoch runs
  used physical GPUs across `a800-2`, `a800-3`, `a800-4`, and `zhaoli` for seeds
  42-50. EPL conditioning achieved 1.000 test accuracy for every seed versus
  0.884-0.896 with EPL features masked. Mean masked accuracy was 0.8906; mean
  matched gain was 0.1094 with population standard deviation 0.0039. Majority
  baselines were 0.172-0.184. This is accepted evidence only for the deterministic
  synthetic classification task, not simulator or real-robot repair.
- EPL ablation comparison video: seed 42's matched checkpoints were replayed on
  the same 3,000 held-out examples. The 14.17-second, 1280x720, 30 FPS,
  425-frame H.264 artifact shows eight deterministically selected masked failures
  corrected by EPL and the full-set result, 89.3% without EPL versus 100.0% with
  EPL. SHA-256 is
  `ee516409f537cc11591224a9e7304bc3236a3e0f3533dfece330ec16013c6ed9`.
  It visualizes synthetic repair classification, not robot motion.
- Intuitive apple-grasp EPL comparison: pinned official `hand2dex_3` source and
  transferred reference were packaged as a 3-second, 1280x720, 30 FPS,
  90-frame side-by-side video. It shows a human hand and dexterous robot hand
  grasping/lifting the same red apple-like object with a coarse
  `APPROACH -> GRASP -> LIFT/HOLD` EPL overlay. SHA-256 is
  `d8b1f9c641ce138563d9be6c37adbe1bce383e6d351a50be9e9daeea11c511dc`.
  The robot side is the official PhiZero reference and the EPL phases are manual
  visual annotations, not a new PhiAgent generation or released PhiZero tokens.
- PhiAgent-generated apple-grasp proxy: the case-3 Wan replacement path produced
  a real generated 896x512, 89-frame robot-hand video, packaged beside the human
  source as a 3-second, 1280x720 comparison with SHA-256
  `6477209341109f07891d333617b6e9e82ee31fd723769d5c78e527c27c1e5b3f`.
  Motion preservation is 0.854 and target identity 0.945, but object consistency
  is 0.004 and regional temporal consistency 0.444. The artifact is therefore
  `PARTIAL`; its EPL phase display is an annotation only and did not condition
  the Wan generator.
- Multi-hand apple comparison: an official Apache-2.0 Linker Hand L20 preview
  (revision `075cc7d42cc1e756bdcbece0fc069a0779fc5237`) was composited into a
  same-scene condition and used for a real PhiAgent Animate run on `a800-2`.
  The 3-second, 1280x720 three-column artifact compares the human source,
  PhiAgent Sharpa output, and PhiAgent Linker L20 output; SHA-256 is
  `819e6e9af35ac2537f1a283ee4ce29e0f96615db1d5d73aff61c84748a5d2656`.
  The Linker candidate scored motion 0.668, identity 0.908, object 0.000, and
  regional temporal 0.002, so it is a failed/partial visual example, not an
  accepted embodiment transfer.
- Learned Linker replacement route: analysis of the stronger teacher artifact
  identified the effective method as compiled-SAM2 replacement plus relighting
  LoRA and object-confidence routing, not animation-mode generation or stronger
  temporal filtering. The route preserves the raw generated candidate when its
  object track is complete and stable while the source track is unreliable,
  disabling source-object overwrite, duplicate cleanup, deghosting, and temporal
  smoothing. Reusing the measured replacement candidate produced a new
  3-second three-column comparison with SHA-256
  `5afe243f6e2564209c667f1ce058ef8b4cf4e072a863962f611143e6855d999f`.
  Visual quality is substantially better than the animation fallback, but strict
  object and regional temporal gates still fail, so status remains `PARTIAL`.
  Replacement preflight now rejects environments missing the compiled SAM2
  `_C` extension before GPU inference.
- Lightweight Sharpa adaptation manifest: the CPU-only schema and CLI freeze
  zero-shot, appearance-LoRA, and Animate-LoRA arms by content hash. They
  reject reference-video training leakage, duplicate content, incompatible
  supervision, and overwrite. Five targeted manifest tests pass. This validates
  experiment preparation, not adapter training or a visual result.
- DiffSynth Wan2.2-Animate trainer intake: the Apache-2.0 source is pinned and
  locally verified at `b1c02ce76aabc989f6bf534756b2da84532249e5`. The training
  preflight enforces its exact target/pose/face metadata contract, the pinned
  Wan checkpoint revision, eight distinct physical GPUs, memory thresholds,
  `CUDA_VISIBLE_DEVICES`, immutable run metadata, and local checkpoint paths.
  Three targeted trainer tests pass. No training job has run.
- Synthetic Sharpa appearance smoke data: eight 640x480 identity frames were
  extracted from the accepted Apache-2.0 Sharpa MuJoCo rollout and frozen by
  content hash under
  `outputs/sharpa-adaptation-data/20260808T122500Z-synthetic-appearance`.
  This validates data preparation only; the fixed-camera frames are insufficient
  for adapter training and contain no Animate pose/face controls.
- Typed EPL v0.1, frame-safe SE(3), tokenizer, aligned perception schema,
  conservative phase/contact extraction, and JSON round trips.
- EPL overlay renderer: hand skeleton, object axes, contact points, wrist/EEF
  path, and phase label. The 640x480/30 FPS synthetic verification artifact has
  60 frames; this is explicitly not a learned-model result.
- Deterministic EPL-to-simulation fixture: EPL produced a robot trajectory and
  MuJoCo replayed it without reachability or joint-limit failure.
- Tabletop push component: 1000 MuJoCo steps, 133 measured contact-begin
  events, 0 forbidden collisions, 0.254 m object displacement, and 3.7 mm
  terminal object-goal error.
- Deterministic repair component: one declared joint-limit corruption was
  detected, clamped, re-simulated, verified, and accepted in a 10-action trace.
  Two versioned repair examples were generated from measured simulator feedback.
- One-EPL-to-many orchestration: the same EPL produced masked 1-DOF and 2-DOF
  trajectories; both replayed, and the 1-DOF embodiment completed the push task.
- ArtiCraft authored-SDK lane: the pinned mini-ArtiCraft 0.1.0 runtime compiled
  its canonical hinged-box model through the PhiAgent adapter and exported a
  5,166-byte USDZ. The compile report recorded 0 failures and 0 warnings in a
  new experiment directory. This validates asset compilation, not handover use.
- First ArtiCraft asset demo: the authored 140 mm handover case has two parts,
  rubber grip bands, one 0-90 degree lid joint, a successful USDZ export, and a
  decodable 1280x720/30 FPS/120-frame geometric motion preview. Its compile
  report has 0 failures and 0 warnings. It is explicitly not a robot handover or
  physics-transfer result.
- Frame-explicit trajectory-rendering contract: accepted physical verification,
  exactly aligned robot/object trajectories, named robot-base-to-camera
  calibration, deterministic control video, action prompt, scene assets, and an
  alignment report are mandatory.
- First auxiliary rendering demo: the deterministic MuJoCo control bundle at
  `/data0/jiangyuhua/PhiAgent-0/outputs/control/20260808T091723Z-b82a5ee9`
  produced a decodable 640x480, 30 FPS, 61-frame video. Verification accepted
  the rollout with task success, 131 contact events, 0 forbidden collisions,
  and 4.38 mm terminal object-goal error. All five persisted artifact hashes
  match the manifest. This is a physics/control artifact, not a Cosmos or
  PhiZero model result.
- Cosmos3-Nano GPU smoke: the 68-file checkpoint snapshot was verified against
  the pinned Hugging Face revision, then a vision-only eager run completed on
  physical GPU 4. The immutable experiment
  `/data0/jiangyuhua/PhiAgent-0/outputs/trajectory_render/20260808T114055Z-b9685eaa`
  produced a decodable 736x544, 30 FPS, 61-frame H.264 video with SHA-256
  `bf69af1f0d33f6703060105344721d46b488e244d8bd9c640b0ee4f380f10203`.
  Mean/minimum edge SSIM against the processed control were 0.9626/0.9553.
- First real-camera-scene proxy demo: Wan2.2-Animate ran on the official
  `hand2dex_1` real human-hand source video and a same-scene robot target frame.
  The immutable remote experiment is
  `/data0/jiangyuhua/PhiAgent-0/outputs/real-world-demo/20260808T092850Z-3f146ae9`.
  It produced an 896x512, 30 FPS, 89-frame output and passed the deterministic
  proxy thresholds with mean 0.8994: motion 0.7786, target identity 0.9493,
  object consistency 0.8698, and temporal consistency 1.0000. Manual checks at
  0.2, 1.2, and 2.2 seconds confirmed a stable real captured tabletop scene,
  robot appearance, and object presence. This is generated imagery in a real
  camera scene, not footage of a physical robot and not official PhiZero.

## PARTIAL

- Robotiq two-finger gripper attempt: the pinned MuJoCo Menagerie 2F-85 asset at
  revision `c1a4eeb85694ae1dffe33ff1797d4e528928a133` was composited into the
  same case-3 apple scene and run through replacement, compiled SAM2, relighting
  LoRA, and confidence routing on `a800-2`. The generated 896x512 candidate
  scored motion 0.848, identity 0.919, object 0.004, and regional temporal 0.350.
  It remained human-hand-like rather than preserving two-finger geometry. The
  new 3-second, 1280x720 four-column comparison preserves the existing Sharpa
  and Linker results and has SHA-256
  `0bf9146f9494861d5e3f8baf03a8fc79e15f2a47a2a7c710bad4bc704b0dda62`.
- Linker-style confidence-routed vendor comparison: Allegro and Shadow were rerun
  using same-scene full-arm condition images, Wan replacement mode, compiled
  SAM2, relighting LoRA, `frame_num=89`, seed 42, object ROI
  `(0.39, 0.60, 0.18, 0.32)`, and one routing round, matching the
  `three-hand-apple-comparison/20260809T1100-confidence-routed` method. Both
  routes selected `preserve_raw_candidate_source_track_unreliable`; candidate
  object tracks cover all frames with area ratios 1.788 (Allegro) and 1.801
  (Shadow). The 89-frame comparison is
  `/data0/jiangyuhua/PhiAgent-0/outputs/vendor-confidence-routed/20260809T1205Z/comparison/vendor-hand-apple-comparison.mp4`
  with SHA-256
  `348c6e2e920c35a1eda13d06a3157935027f2ef1ebd66f0559ff0a2d4265e92f`.
  Background and full-arm consistency improve substantially, but Wan visibly
  normalizes both vendor hands toward a Sharpa-like white hand. Allegro scores
  motion/identity/object/temporal 0.837/0.919/0.004/0.326 and Shadow
  0.850/0.896/0.004/0.359; both remain rejected.
- Wan-Animate-2 pose-matched Sharpa proxy: official Apache-2.0 source commit
  `3ad2fef7d61d6200c9c653e0fe47be7616b323f3` and ModelScope checkpoint
  revision `7053fd05166cdd99a49896364d01c06c281a9d69` ran on physical A800
  GPUs 0-1. A clean Sharpa reference was rendered from the pinned Apache-2.0
  MJCF, composited after source-hand inpainting, and paired with the synchronized
  1.0-3.0 second source segment. The immutable experiment
  `/data0/jiangyuhua/PhiAgent-0/outputs/wan-animate2-pose-matched/20260808T200738Z-5689827b`
  produced a decodable 616x352, 24 FPS, 48-frame video with SHA-256
  `7277d2b671b9474ca3abbc4fd6b807f35f0c6ddf1df510ea168a439542509519`.
  Fixed-time review at 0.2, 1.0, and 1.7 seconds confirms a single clean Sharpa
  hand grasps, moves, and lowers the teal object without the original human hand.
  Raw motion/identity/object/temporal scores are
  0.7384/0.9306/0.0039/0.0853. Temporal-only denoising yields SHA-256
  `da587990765170df4f808f702752341b4eeef975f1dd622e67fdb56bf97e43c7`
  and scores 0.7443/0.9305/0.0039/0.0994. The visual replacement is demonstrated,
  but object and temporal gates fail, so this is an approximate proxy rather than
  accepted PhiZero reproduction or physical manipulation evidence.
- Wan-Animate-2 negative controls: an official real Sharpa photo preserved its
  laboratory scene and white cylinder rather than the hand2dex scene; an
  unaligned clean-render reference produced a duplicate hand at the initial pose;
  restoring source-object pixels also restored adjacent human-hand pixels. These
  variants were rejected rather than selected by appearance alone.

- Stable-background cross-vendor visualization: deterministic screen-space
  overlays remove the complete visible human arm while retaining the original
  `hand2dex_3` pixels elsewhere and restoring the tracked apple. Shadow uses its
  modelled forearm; Allegro uses an explicitly procedural mechanical forearm
  because its pinned hand model has no arm. The 90-frame,
  1280x720 comparison is
  `/data0/jiangyuhua/PhiAgent-0/outputs/vendor-hand-overlay/20260809T1120Z/comparison/vendor-hand-apple-comparison.mp4`
  with SHA-256
  `cf6fb573fc8212e4d63f8ffd06a3a926aa2bdf1f616e164ee8f6e0879a05b2a3`.
  Allegro and Shadow panels are explicitly labelled `(full-arm overlay)`. This fixes
  background drift for presentation, but the rigid screen-space hands are not
  model inference, articulated retargeting, or physics-valid grasps.
- Cross-vendor apple-grasp comparison: the same official `hand2dex_3` source,
  seed 42, Wan revision, checkpoint, resolution, and evaluator were used for
  Wonik Allegro and Shadow Robot Hand targets alongside the pinned Sharpa
  reference. The 89-frame 2x2 artifact is
  `/data0/jiangyuhua/PhiAgent-0/outputs/vendor-hand-apple/20260809T0355Z/comparison/vendor-hand-apple-comparison.mp4`
  with SHA-256
  `6b0b070fa7615097b1d64b0a6642e9775a320a4b73fef50c5c66b293b9c19c9b`.
  Allegro scored motion/identity/object/temporal
  0.3302/0.6525/0.0039/0.0007; Shadow scored
  0.4005/0.7813/0.0039/0.0007. Both are rejected: manual keyframe review agrees
  that the generated hands and apple deform severely. This establishes a
  multi-vendor failure comparison, not successful transfer or physical grasping.
- Long-input human-to-robot hand conversion: the pinned dex-retargeting example
  supplies one uncut 20.70-second, 621-frame human-hand gesture video. The
  experiment `outputs/long-human-retarget/20260809T0410Z-allegro-20p7s` detected
  all 621 frames with MediaPipe, applied Dexpilot retargeting to the 16-DOF
  Allegro hand, and rendered a synchronized 20.70-second comparison on physical
  A800 GPU 0. Both output videos decode fully. Manual review at frames 30, 150,
  270, 390, 510, and 600 confirms distinct open, pinch, fist, side-view, and
  reopened poses with no duplicated hand or chunk transition. This geometric
  hand-gesture conversion is WORKING, but it is not object manipulation and not
  official PhiZero inference.
- Background-locked long-input comparison:
  `outputs/long-human-composite/20260809T1055Z-allegro-background-locked-v2`
  reuses each of the 621 decoded source frames and changes pixels only inside the
  union of the detected human-hand mask and aligned Allegro-hand mask. The
  lossless 1280x720 output is 20.70 seconds at 30 FPS. A post-encode, post-decode
  audit measured exactly zero changed RGB channel values outside the replacement
  mask on every frame. Manual review at frames 30, 150, 270, 390, 510, and 600
  confirms that the original hand is removed while the wall, tabletop objects,
  person, clothing, and forearm remain source-identical. This composited
  geometric result is WORKING; the four-finger Allegro morphology and absence of
  object manipulation remain explicit limitations.
- Five-finger hand-and-forearm replacement:
  `outputs/long-human-shadow-arm/20260809T1910Z-shadow-smoothed-stable-scale-v2`
  retargets all 621 frames to the 24-DOF, five-finger Shadow Dexterous Hand and
  uses its rendered wrist/forearm model to replace the visible human forearm
  through the lower frame edge. The uncut output and comparison are 20.70
  seconds at 1280x720 and 30 FPS. A post-encode decode audit measured zero RGB
  channel differences outside the hand-and-forearm replacement mask on every
  frame. Manual review at frames 30, 150, 270, 390, 510, and 600 confirms five
  distinct digits, continuous wrist-to-forearm attachment, no residual human
  skin, and preservation of the thermos, yellow object, clothing, and wall.
  A zero-phase Gaussian filter with sigma 4 frames reduces the 8-9 second
  maximum 24-DOF frame delta from 0.4634 to 0.2607, while a fixed 148-pixel hand
  scale removes the fist-induced apparent size collapse. Frame-by-frame review
  from 7.5 through 9.5 seconds confirms continuous robot morphology with no
  human-hand pixels.
  This geometric gesture conversion is WORKING; it does not demonstrate object
  manipulation or official PhiZero inference.
- Rejected long flower-arranging visualization:
  `outputs/full-robot-flower-demo/20260809T074000Z-pexels5893642/render-v6`
  uses one continuous 27.5-second Pexels clip. A temporal union of all person
  masks replaces the full visible florist with one static clean plate before a
  pinned Unitree G1 body and Sharpa Wave or Wonik Allegro hands are composited.
  All 660 frames and all three outputs decode fully on physical A800 GPU 7.
  The full-body centroid has zero frame-to-frame movement; 201 hand-detection
  gaps hold the previous robot pose instead of restoring human pixels. Maximum
  exact source-RGB retention in the erased non-flower person region is 0.38%.
  Manual review at frames 24, 120, 216, 312, 408, 504, and 600 found no remaining
  face, clothing, human hand, or previous-frame human trails in sparse keyframes,
  but full-video review found conspicuous clean-plate smearing, flower-mask
  remnants, and distorted procedural screen-space arm links. The visualization
  is rejected and removed from the README. A replacement must use articulated
  3D joints without source-person pixel restoration.
- Clean articulated flower-arranging replacement:
  `outputs/articulated-flower-demo/20260809T120000Z-full660-v3` discards the
  rejected screen-space compositor and renders independent MuJoCo target scenes.
  Human pose drives the real G1 shoulder and elbow joints through bounded IK;
  left/right Sharpa or Allegro models are scaled once and attached directly to
  the corresponding G1 wrist frames. The 27.5-second, 660-frame outputs decode
  fully. Maximum wrist-target error is 0.0613 m, maximum frame-to-frame arm-joint
  step is capped at 0.1200 rad, all joint positions remain in range, and all four
  hand-attachment errors are exactly 0 m. Target outputs contain no source-person
  pixels and no procedural screen-space joints, so the previous ghosting and
  stretched-joint failure modes are absent. The geometry pipeline is WORKING, but
  the output is rejected for the README because its synthetic MuJoCo scene differs
  too much from the real-scene Wan replacement demos. Flower contact physics,
  photorealistic replacement, and real robot execution remain NOT STARTED.
- Rejected edited long-demo attempt: the local CPU composition experiment
  `outputs/phizero-demo/20260809T0125-two-case-continuous-10s` combines the
  existing case-1 confidence-routed and case-2 deghosted real-input Wan proxy
  results with 0.25-second crossfades and an explicitly labelled case-1 hero
  replay. Full decoding verified 315 H.264 frames at 1344x256 and 30 FPS
  (10.50 seconds), and manual review at frames 45, 105, 165, 225, and 285
  confirmed the three-panel source/reference/agent layout and robot-hand
  presence. It was rejected for the long-input requirement because it is an
  edited sequence, not a single model run on one source longer than 10 seconds.
- `pro5000` real-input Wan batch: eight official `hand2dex` source/reference
  runs completed on physical GPUs 0-7 with immutable traces and evaluations.
  None passed all four proxy gates. The best candidate was replacement case 1,
  seed 42 under
  `/data0/jiangyuhua/PhiAgent-0/outputs/pro5000-singleclip100/replacement/case1/seed42`:
  motion 0.8617, identity 0.9162, object consistency 0.8466, and temporal
  consistency 0.6609. Its SHA-256 is
  `a6e5601f123c7f88f1d54343c4cb5c8bd4fd1ae0d7123d9b51911e084db35460`.
  Despite the experiment-root name and `infer_frames=100`, upstream Animate
  produced two diffusion windows for `target_frames=177`; this is not
  single-window evidence and remains a rejected Wan proxy, not PhiZero.
- The `pro5000` case-1 best run was re-routed with the clean-apex object
  confidence policy. Its raw candidate is tracked in all 89 frames, has area
  ratio 2.204 (below 3), and lift recall 1.0, so inpainting, source-object
  compositing, and temporal filtering are disabled. Start/middle/end review of
  `outputs/pro5000-showcase/case1-confidence-routed-clean-comparison.mp4`
  confirms that the compositor-induced hand and bowl trails are absent. The
  clean raw candidate SHA-256 is
  `a62fcc7c16ca7dfa7e92c042fb4441efa0ad9b3b6cda4f18bd42aa8979f31378`.
  It remains rejected: object consistency is 0.0247 and object-trajectory
  similarity is 0.3335, so this is a visually cleaner demo rather than improved
  physical-parity evidence.
- Paper `hand2dex_2` human-to-robot-hand proxy: the best measured Wan replacement
  run used the official real-camera human source, the 1.2-second same-scene
  robot-hand anchor, corrected object ROI `(0.25, 0.42, 0.35, 0.42)`, T5 CPU
  offload, seed 42, and no relighting LoRA. The immutable experiment is
  `/data0/jiangyuhua/PhiAgent-0/outputs/hand2dex-2-midframe-norelight-proxy/20260808T141126Z-4ac3ef3d`.
  It produced a decodable 896x512, 30 FPS, 77-frame video with SHA-256
  `5ade5a75c809202d22e51aa986040e7cdb395cf938faf9d0cafbe342f33728e8`.
  Robot identity (0.9239) and object consistency (0.8574) pass; motion (0.6971)
  and temporal consistency (0.4527) do not, so the run remains rejected. Manual
  checks at 0.2, 1.2, and 2.2 seconds confirm a robot hand grasps and lifts the
  yellow container in the real captured scene. This is a Wan proxy, not official
  PhiZero inference.
- Targeted `hand2dex_2` deghost refinement: the reproducible masked filter applies
  `hqdn3d=7` only through the recorded robot-character and object masks, leaving
  the rest of the real scene unfiltered. The immutable remote output is
  `/data0/jiangyuhua/PhiAgent-0/outputs/hand2dex-2-deghost-final/hand2dex-2-deghost.mp4`
  with SHA-256
  `c0e610fda478ffd52daf39cd0bd1745e4fbaf6f87ec36d8215c71a1ba0996611`.
  Against the unfiltered candidate, motion improved from 0.6971 to 0.8745,
  temporal consistency from 0.4527 to 0.6295, target identity from 0.9239 to
  0.9342, while object consistency remains above threshold at 0.7698. Temporal
  consistency still fails the 0.75 gate, so this is an improved PARTIAL result,
  not an accepted reproduction.
- Stronger `hand2dex_2` object-ghost repair: residual analysis localized complete
  duplicate-container instances to frames 44-58 and 70-76. The v4 tool removes
  the duplicate object layer, restores a tracked clean source-object instance,
  restores the robot foreground, and uses three-frame cosine transitions. The
  output is
  `/data0/jiangyuhua/PhiAgent-0/outputs/hand2dex-2-deghost-v4/hand2dex-2-deghost-v4.mp4`
  with SHA-256
  `49276067cfce8dcb45d5c17dd6ab328c9e307d97c0adf507dcb202c8089aee5d`.
  Manual inspection at the previous worst frame (1.7 seconds) confirms the
  second container outline is removed. Object consistency is 0.8494 and target
  identity 0.9227; motion 0.6950 and temporal consistency 0.5316 remain below
  gate because the local layer reconstruction changes the measured trajectory.
  This is a visual deghost variant, not an accepted PhiZero reproduction.
- Clean-apex method audit and alternate-hand version: the confidence-routed
  policy does not denoise or composite when the generated object track is
  reliable; it preserves the raw candidate and disables destructive repair.
  Direct Wan conditioning on an isolated case-2 hand and a regional VACE edit
  were rejected because they changed the person/scene or lost the spoon track.
  The accepted alternative uses a SAM2 hand-instance track over the clean-apex
  raw candidate and applies a material-only deep-graphite edit inside that mask.
  The final video is
  `/data0/jiangyuhua/PhiAgent-0/outputs/alt-hand-confidence-routed-final/graphite-hand.mp4`
  with SHA-256
  `228dcc18fa874ad502e632ef6f7446c56be6bd1f8df609344607f566d2683537`.
  The local VS Code delivery copy is H.264/AVC (`avc1`, yuv420p, faststart)
  with SHA-256
  `fa64a16b7194a092c333a72d37a9d4ccb746604af49b61d60bf695fa641b25a2`;
  the original mp4v artifact is retained beside it.
  Confidence routing preserves the raw edited result: the spoon track exists in
  every frame, area ratio is 1.613, lift recall is 1.0, and trajectory
  similarity is 0.9839. Start/middle/end review found no hand/spoon composite
  ghosting. This changes hand/forearm material appearance, not mechanical hand
  geometry, and remains a proxy rather than official PhiZero.
- Three-row synchronized comparison: the local H.264 video
  `outputs/three-way-arm-comparison/human-silver-graphite-vertical.mp4`
  places the real human input on top, the confidence-routed silver robot arm in
  the middle, and the graphite-arm variant on the bottom. It contains 89 frames
  at strict 30 FPS, 672x1152, with SHA-256
  `4ba1bc61a52eebe789d9de2663ae644772c8047215cbaf528e074d5c604f8ddc`.
- Sudo R1-inspired appearance and four-row comparison: a separate SAM2 track
  covers the complete robot (head, torso, and both arms), which is restyled with
  a white shell, black joints/chest cavity, and dual-camera face. Object
  confidence routing remains enabled (all frames tracked, lift recall 1.0,
  trajectory similarity 0.9680), with no object repair. This is an
  appearance-inspired proxy and does not claim exact Sudo R1 geometry. The
  four-row human/silver/graphite/full-Sudo comparison is
  `outputs/four-way-arm-comparison/human-silver-graphite-sudo-vertical.mp4`,
  89 frames at 30 FPS, 672x1536 H.264, SHA-256
  `24553e8c9aadf9db72411dd16f64d91666391717bd990ab0916a6792eca9ccfa`.
- Real-world scene demo rerun on GPU 6: the official `hand2dex_1` captured human
  video and same-scene Sharpa frame produced a decodable candidate with SHA-256
  `40342d688c7f5817849b45d5896a44a871323e96ed7ba7c84886b23689e3c5a4`.
  Proxy scores passed (motion 0.7786, identity 0.9493, object 0.8698, temporal
  1.0000), but manual inspection at 1.2 seconds found the spoon on the table
  instead of in the robot hand. This is a real captured scene with a generated
  robot, but it fails object/contact fidelity and is not accepted as a physical
  handover. The side-by-side comparison is retained with the experiment.
- First agentic proxy demo, official `hand2dex` case 1: GPU 4 produced a
  decodable 896x512, 30 FPS, 89-frame candidate with SHA-256
  `3489ee501c42ff25f7d95a459ff09313c62b71fc4f81251902946934a4872100`.
  Human review identified visible flicker that evaluator v1 underweighted.
  Evaluator v2 increased excess-jerk sensitivity and measured baseline temporal
  consistency as 0.694. Temporal-only `hqdn3d` strength 12 produced an improved
  candidate with SHA-256
  `823eabca2efa5b3c2afd1f525c5893222fe8a88ed49eb89ce884a568b998477b`:
  motion 0.625, target identity 0.942, reference-edge object proxy 0.847, and
  temporal consistency 0.753. A second human review still found severe late
  flicker, proving that the full-frame v2 temporal metric is insufficient; visual
  acceptance remains failed independently of the numeric gate. The generation
  log shows two diffusion clips for 89 real frames because `frame_num=49`,
  with only one reference frame carried across the boundary.
  The experiment is `outputs/phizero-agentic-proxy/20260808T092301Z-d0d14730`.
- True single-clip Animate test: `frame_num=97` covered the padded 97-frame
  condition in one diffusion pass and produced a decodable 89-frame output with
  SHA-256 `5a29d00bb53708e6bc9a728a7cbc3a183826acf142c091fc0c69b7b531d9e9ea`.
  Motion improved to 0.829 and identity to 0.965, but object color was 0.719 and
  regional temporal consistency was only 0.279, so the candidate was correctly
  rejected. Removing the chunk boundary did not solve localized flicker.
- Agentic PhiZero proxy: a backend-independent generate/evaluate/repair
  controller, multi-image/seed Wan CLI, bounded feedback repair, immutable trace,
  and executable local ffmpeg evaluator are implemented. The evaluator measures
  block-motion agreement, Sharpa/reference structure, reference-edge similarity,
  and excess temporal jerk while recording all evidence. A recovered case-2
  candidate now passes every deterministic proxy gate: motion 0.9049, target
  identity 0.9813, object consistency 0.7700, and temporal consistency 0.7597.
  Its SHA-256 is
  `4f45aca5baff0b7040e714bc64ca22c66e55654a28a9ab00ed3e56a6948e4b64`.
  Human review of the source/PhiZero/candidate keyframes still finds blurred and
  deformed fingers at contact, so visual parity remains rejected despite the
  numeric pass.
- Object-action correction is now an internal hard gate rather than a visual
  similarity average: tracked trajectory, lift recall, mask coverage, and
  instance deformation can each reject a candidate. A case-1 SAM2 repair removed
  the stale generated tool, restored one source-trajectory tool, and visibly
  lifted it at the middle keyframe. The candidate remains rejected because its
  temporal-consistency proxy is below threshold; this is not an accepted
  PhiZero reproduction.
- Adaptive object recovery now learns arbitrary chromatic object appearance
  from the declared first-frame ROI in both tracking and duplicate cleanup.
  This fixed the previous cyan-only cleanup bug: case-2 object consistency
  improved from 0.6616 to 0.8854 before temporal filtering.
- Case-2 object-confidence routing now mirrors the clean-apex policy: because
  the raw candidate object is tracked in every frame, has area ratio 2.265
  (below 3), and lift recall 1.0, destructive inpainting, source-object
  compositing, and temporal filtering are disabled. The resulting
  `agent-clean-apex.mp4` has SHA-256
  `e225e61000b233d6794d009ad0a05a506e381c155a3f0d334f46759e846584d7`;
  manual start/middle/end review confirms that compositor-induced hand/bowl
  trails are absent. Its object trajectory proxy is 0.735, so it is retained as
  the visually cleaner demo rather than relabeled as accepted physical parity.
- Linker Hand L20 appearance-transfer demo: a same-scene first-frame condition
  replaced the human hand over the red apple with a white L20-style hand. The
  raw 896x512, 30 FPS, 89-frame output has SHA-256
  `3d7091e25dacb0a7e5a850f652ac0442a28d8e25ebe53bc7d58a3ec0e409776b`.
  The source color track was unreliable (area ratio 373.7), while the candidate
  track was complete with area ratio 1.83; fail-safe confidence routing
  therefore preserved the raw candidate instead of restoring source pixels.
  Manual start/middle/end review shows the L20-style hand approaching, lifting,
  and releasing the apple without compositor-induced human-hand overlay. This
  is an appearance proxy, not measured Linker Hand dynamics or official
  PhiZero inference.
- Same-case vertical embodiment comparison: the original bare human-hand video
  is the top row, PhiAgent Sharpa is the middle row, and PhiAgent Linker L20 is
  the bottom row. All rows are normalized to 896x512, 30 FPS, 89 frames and
  stacked into a decodable 896x1536 video with SHA-256
  `7641b53a675092c7478da37a78b5fd68e2c7d2d26a596a2657dced1de03f8c10`.
- Targeted case-1 ghost removal tested four bounded mask/inpainting settings.
  The selected temporal-radius-2, 9x9-dilation, radius-5 repair improved the
  temporal proxy from 0.313 to 0.371 while retaining 0.839 motion preservation
  and 0.972 target identity. A stronger temporal-radius-3 repair regressed to
  0.304 and was rejected. The selected result is improved but remains below the
  0.75 temporal acceptance threshold.
- A second case-1 pass addressed the remaining object-edge halo rather than the
  global temporal score. It replaces H.264-compressed masks with lossless PNG
  masks, uses binary source-object alpha with zero support outside the instance,
  and encodes the deliverable as H.264 High 4:4:4 at CRF 12. Measured pre-encode
  source leakage in the five-pixel exterior ring fell from 5.84 mean absolute
  RGB levels to 0. The full CPU regression has 109 passed and one optional
  MuJoCo skip.
- Apex-window diagnosis showed that the visually dominant "ghost" was the
  inpaint scar left on the robot torso after deleting an already valid generated
  object. SAM2 measured a stable candidate mask (1.53x area range) and 1.0 lift
  recall, so the confidence router now disables destructive object repair for
  this case and preserves the raw replacement candidate. This removes the apex
  inpaint scar, but its relative object-trajectory similarity is only 0.492 and
  its global temporal proxy remains below threshold; it is a targeted visual
  correction, not final acceptance.
- A matched robot-model contrast used the same case-1 source, seed 42, 89-frame
  single clip, resolution, and Wan revision while changing the first-frame hand
  from Sharpa to a pinned Wonik Allegro render from MuJoCo Menagerie
  `c1a4eeb85694ae1dffe33ff1797d4e528928a133`. The Allegro run preserved coarse
  motion (0.812) and target-image structure (0.973) but failed object consistency
  and temporal gates; manual apex review found duplicated hands and lost object
  contact. The side-by-side artifact is explicitly labelled `Allegro-FAILED`,
  establishing a model-sensitivity failure rather than a successful transfer.
- A follow-up replaced the entire reference character with the pinned Unitree G1
  full-body render, eliminating the mixed Sharpa-body/Allegro-hand condition.
  Pure G1 and G1-with-tool controls used the same source, seed, frame budget, and
  Wan revision. Both were rejected: pure G1 motion preservation was 0.138 and
  G1-with-tool was 0.180; manual apex review found detached limbs, while the tool
  control produced multiple object trails. This isolates a second failure:
  coherent robot identity is insufficient when the reference pose/camera differs
  strongly from the source first frame. FLUX pose-retarget assets are absent, so
  pose-aligned G1 generation remains pending rather than silently approximated.
- The requested Sunday Robotics control now uses the exact Sharpa generation
  route rather than the earlier animation-only controls: official case-1 source,
  seed 42, 89-frame single clip, 896x512, Wan replacement mode, identical ROI,
  and the official Sunday Memo hero frame as the only changed conditioning
  input. Memo identity and arm motion remain visually coherent through the
  lift-return cycle. The generated object becomes a white box instead of the
  turquoise tool, so object preservation is rejected and the raw result is
  retained only as an internal robot-model contrast.
- Sunday correction follow-up sampled the official company video and selected
  the t=6 s frame with visible black Memo grippers. A reference containing humans
  caused Wan to animate the human; a SAM2-isolated Memo still decoded as a human,
  showing that reference cleanup alone does not enforce Memo hand topology. A
  separate source-scene compositor now changes only the union of SAM2 source-human
  and Memo masks (27.99% mean frame area), with source table/background pixels
  unchanged outside that region and the source tool restored. The table requirement
  is satisfied, but the generated hands remain mitten-like. Attempts to replace
  them with official static gripper and forearm cutouts produced duplicate hands
  or rigid compositing artifacts and were rejected. Correct Memo hands are BLOCKED
  on a Memo-specific identity/LoRA adapter or a generator that supports multiple
  gripper references; no post-hoc result is presented as solved.
- Residual-ghost diagnosis showed the blur was already present in the original
  Wan2.2-Animate Memo hand frames, not introduced by the source-table mask. A
  complete Wan-Animate-2 distilled backend at commit
  `3ad2fef7d61d6200c9c653e0fe47be7616b323f3` was run on two A800 GPUs with the
  isolated Memo reference. Its 48-frame result has stable SAM2 robot masks
  (1.022 area ratio), crisp two-finger grippers, and no old hand trails. The
  selected table blend preserves source-table pixels below row 375 exactly
  before encoding (MAE 0). Memo color/identity drifts toward a generic humanoid
  and the restored tool is not in gripper contact, so the result is labelled
  PARTIAL rather than accepted.
- Sunday hand-morphology v5 narrows that failure without claiming full robot
  success. Following the Unitree full-body route, it conditions Wan on the
  official Sunday t=6 s full-body frame and selects seed 1051 for stable identity.
  SAM2 tracks only the active generated hand; a 0.90-width distal crop of the
  official t=3 s Sunday gripper replaces that region and emits its exact rendered
  mask. DINOv2 comparison against the official gripper raises mean native-hand
  similarity from 0.491 to 0.841, worst-frame similarity from 0.321 to 0.769,
  mean temporal similarity from 0.826 to 0.948, and worst-frame temporal
  similarity from 0.558 to 0.898; all five hand gates pass. The decodable result
  is `outputs/robot-model-contrast/sunday-memo-improved-20260809/sunday-memo-improved.mp4`
  with SHA-256
  `5a81456956cecc64f93ba00aa505113f7ecaa782bee2d8b2595555b347ac1abc`.
  Manual review still sees reference-pose arm duplication in the underlying Wan
  body, so this is a PARTIAL hand-shape improvement, not an accepted full-body
  Sunday manipulation result.
- Background/action-preserving Sunday v2 supersedes v5 for the stricter user
  constraint. It uses the original replacement-mode `sunday-memo.mp4` as the
  sole base, retains its 89 frames and 30 FPS, and edits only the union of the
  SAM2-tracked original hand and rendered native-gripper mask. The allowed edit
  region averages 10.73% of the frame. Outside that mask, decoded-pixel MAE is
  1.46 (worst frame 1.56), changed-pixel fraction is 0.18% on average (worst
  0.46%), and motion error is 0.00199 on average (worst 0.00372); all eight
  background/action preservation gates pass. Using the exact rendered hand mask,
  DINOv2 native-hand similarity is 0.892 mean and 0.844 worst-frame, while
  temporal similarity is 0.968 mean and 0.909 worst-frame; all five hand gates
  pass. The result is
  `outputs/robot-model-contrast/sunday-memo-background-safe-20260809/sunday-memo-background-action-preserved.mp4`
  with SHA-256
  `0293ce7788ab4857894277810124679ebbd9ade6a90a87ef68690a9121a046a7`.
  This satisfies measured background and action preservation but remains a
  PARTIAL image-space hand replacement rather than Memo-specific learned
  kinematics.
- Wan replacement-mode support is implemented to preserve source background and
  object pixels outside the estimated character mask, with strict SAM2 and
  relighting-LoRA preflight checks. GPU 7 produced a real replacement candidate
  under `outputs/object-preserving-demo-20260808T1200Z/20260808T114215Z-489d137c`.
  The raw replacement duplicated the spoon and v4 correctly rejected it with
  object consistency 0.0. Deterministic corridor cleanup removed the duplicate;
  the repaired video has SHA-256
  `32fabb083c4e5a58fdbdc2c479d2bda4228128a5065b0b487649b67b3b297853`,
  object consistency 0.9178, trajectory similarity 0.9327, and lift recall 1.0.
  Overall acceptance still fails because motion is 0.6395, temporal consistency
  is 0.4178, and manual review finds exposed human pixels.
- Full-robot object-lock route: replacement output is no longer used as the
  final image because its character mask retained large human regions. The
  full-frame animation candidate is instead cleaned only at the tracked
  duplicate spoon instance and receives the calibrated source-spoon trajectory.
  Version 7 is fully robotic with no exposed human body, has SHA-256
  `64206a1970a03da748f47a5edd3ed2b4871f5eb5ddb576d7858cbfc44c9fdaed`,
  object consistency 0.8105, and identity 0.9574. It remains PARTIAL because
  motion is 0.6276, temporal consistency is 0.2602, and contact alignment is
  only image-space rather than physics verified.
- Complete-spoon restoration v10: SAM2 was prompted with positive points on the
  spoon handle/bowl and negative points on the hand/table, then propagated a
  77-frame object-only mask on physical GPU 7. Applying that mask to the
  full-robot candidate removes the low-saturation holes left by color masks
  without restoring human hand or clothing. The decodable result has SHA-256
  `819e82e88bd95bca255107d6e32da88589acb949f9db1594f8df2fcb37521f66`,
  object consistency 0.8062, and identity 0.9575. It remains PARTIAL because
  motion is 0.6238, temporal consistency is 0.2519, and hand-object contact is
  image-space rather than physically verified.
- Confidence-routed bowl extension v4: starting from the unmodified clean-apex
  video, SAM2 image masks isolate the case-2 yellow bowl plus robot hand and the
  case-1 terminal spoon. The 179-frame, 896x512, 30 FPS output extends duration
  from 2.97 to 5.97 seconds: the mirrored left robot hand brings the bowl in,
  the bowl stops below the right hand, and the single spoon instance moves into
  the bowl. A same-camera clean table frame replaces the terminal spoon region,
  avoiding the inpaint smear and double-spoon trail seen in v1-v3. The output is
  `outputs/phizero-demo/20260809T0215-case1-bowl-extension/agent-clean-apex-extended-bowl-v4.mp4`
  with SHA-256
  `28a13936018b8a1ca7be7e02da09578d7bf0f68d8a115dd9d82e75175427cabb`.
  This is a deterministic 2D instance composition using a bowl captured from a
  different camera, not a learned long-horizon generation or physical execution,
  so it remains PARTIAL.
- Native-hand bowl extension v9 supersedes v4. Case-2 hand pixels are removed
  from the bowl mask, and no mirrored or additional hand layer is rendered.
  Instead, the bowl enters from the left and stops in the grasp opening of the
  robot hand already present in `agent-clean-apex`; the spoon target is bound to
  the final bowl coordinates and ends behind the bowl rim. The 179-frame,
  896x512, 30 FPS video is
  `outputs/phizero-demo/20260809T0215-case1-bowl-extension/agent-clean-apex-extended-bowl-v9-native-hand.mp4`
  with SHA-256
  `20f62bce80cf8ecfa394496832e019d519fd665b27971472fc1783a63d54fc57`.
  Manual review at 3.3, 4.4, and 5.5 seconds confirms two robot hands total,
  same-robot hand appearance, bowl-rim contact, and spoon insertion. It remains
  PARTIAL because the extension is deterministic 2D composition rather than
  learned generation or physical execution.
- Deterministic hybrid compositor: a real 89-frame case-1 smoke overlays the
  separately rendered Sharpa 22-DOF layer relative to the tracked source object,
  leaves 97.9% of source pixels unchanged, and restores 680,105 tracked object
  pixels exactly (`object_exact_fraction=1.0`) in 27.7 seconds on CPU. The run is
  `outputs/hybrid-compositor/20260808T120557Z-4db075ad`. This is PARTIAL because
  placement is screen-space rather than camera-calibrated and exposed human-hand
  pixels remain outside the robot silhouette.
- Lightweight regional training infrastructure: a pinned DiffSynth VACE-1.3B
  single-GPU LoRA path, leakage-safe `vace_lora` manifest, immutable ModelScope
  revision marker, GPU preflight, and 17-frame smoke configuration are
  implemented. A two-example `development_only` pseudo-target dataset has passed
  DiffSynth loading with aligned 448x256 target/control/reference inputs. This
  validates only the training pipeline; pseudo-targets cannot support visual
  quality claims, and no finite-loss GPU step or adapter checkpoint exists yet.
- Authorized VACE adaptation evidence: an Apache-2.0 Sharpa-only procedural
  dataset contains 12 training and 4 held-out 448x256x17 clips with fixed seed
  and disjoint clip IDs. A matched rank-4, 5-epoch, 300-step LoRA ablation on
  `a800-2` completed for geometry control and neutral-black control. On held-out
  clip 012, geometry control reached SSIM 0.6920, edge SSIM 0.8582, and PSNR
  29.56 dB versus 0.6826/0.8399/28.54 for neutral control and
  0.4647/0.5102/12.73 zero-shot. This supports only synthetic-domain
  trainability; it does not establish real-video or PhiZero-reference quality.
  Checkpoints and logs are persisted under
  `outputs/sharpa-vace-authorized-training/20260809T020500Z`, with the held-out
  comparison under `outputs/sharpa-vace-authorized-eval/20260809T020000Z-heldout-clip012`.
- Trained-model long-action stress test: the confidence-routed case-1 video was
  extended from 2.97 to 6.86 seconds with a 33-frame control for left-hand bowl
  entry, right-hand spoon reacquisition, and spoon placement. The trained VACE
  LoRA preserved the terminal scene better than zero-shot (static-reference SSIM
  0.728 vs 0.522) and generated a bowl near the left hand, but the spoon track
  existed in only 16/33 frames, reached aspect ratio 8.33, and failed final
  containment/contact review. The run under
  `outputs/spoon-bowl-extension/20260809T022100Z-result` is REJECTED and is not a
  successful manipulation result.
- Continuous-grasp revision: branching at 2.25 seconds, before the original
  right hand releases the spoon, removes the release/regrasp phase. A separate
  arm enters from the left carrying the bowl while the original robot hand stays
  connected to the spoon handle and moves it into the bowl region. The 6.13 s
  artifact is
  `outputs/spoon-bowl-extension/20260809T103500Z-continuous-grasp-result/agent-clean-apex-continuous-spoon-bowl-extended.mp4`.
  It remains REJECTED because a duplicate spoon part appears mid-sequence,
  spoon geometry and terminal robot identity drift, and contact is not
  physically credible.
- Verifier-selected learned extension: the trained rank-4 Sharpa VACE-1.3B LoRA
  naturalized the dense 25-frame v9 control instead of generating from a sparse
  outline. Seed 42 preserves the same robot identity, produces a continuous bowl
  entry held by the robot hand, and moves one cyan spoon behind the bowl rim.
  The phase-aware verifier separately passes bowl presence, single-spoon,
  persistence, pre-insertion shape stability, hand-bowl contact, hand-spoon
  contact, final containment, robot identity, temporal stability, and control
  alignment. Measured values include identity 0.9620, hand-bowl contact 0.76,
  hand-spoon contact 1.0, final containment 1.0, spoon area ratio 1.94, temporal
  jerk 0.0115, and edge error 0.0468. The 25-frame extension has SHA-256
  `1d82db860732fef38e8fcd7ae9c66f3f3661c99a64ebedc8244e7dd526816387`;
  the 5.90-second joined result is
  `outputs/phizero-demo/20260809T1130-vace-naturalized/final/agent-clean-apex-vace-bowl-extended.mp4`
  with SHA-256
  `346cd55b93efd3293e56acb940b9f14072fe34c320d4d6eb8e593de6f72283b3`.
  This is the strongest current visual extension, but verifier contact remains
  image-space and does not establish physical execution.
- Robot table-slide prompt sweep: a 49-frame/12 FPS control keeps the bowl
  bottom at image y=241 px with zero support error, moves it monotonically from
  viewer-right with the in-frame robot free hand, locks the other hand to one
  spoon, and starts spoon motion only after the bowl stops. Four fixed-seed
  prompts (`roles`, `physics`, `strict`, `temporal`) were compared. `roles`
  ranked first at 0.8208 automatic score with scene SSIM 0.8606, 49/49 spoon
  tracking, area ratio 1.86, and temporal jerk 0.00882. Motion interpolation to
  30 FPS reduced measured jerk to 0.00289. The 5.90-second result is
  `outputs/spoon-bowl-extension/20260809T113500Z-optimized-table-slide/agent-clean-apex-robot-slides-bowl-smooth.mp4`.
  It is PARTIAL: the intended support/contact/object-count gates pass in
  screen-space keyframe review, but arm proportions remain visibly elongated
  and no 3D dynamics simulation validates contact forces.
- Original-arm-only revision: masked VACE local-edit sweeps at denoising 0.45,
  0.60, and 0.75 failed respectively by insufficient arm motion, block
  artifacts, and literal control rendering. The selected fallback therefore
  uses no generated arm pixels: both robot arms and the spoon are extracted
  from the 2.25-second reference frame and animated as shoulder-anchored
  source-pixel layers; only the bowl is procedural. Outside these layers,
  original pixels remain unchanged. The 5.90-second result is
  `outputs/spoon-bowl-extension/20260809T121000Z-original-arms-final/agent-clean-apex-original-arms-push-bowl.mp4`.
  Bowl support, monotonic slide, hand-bowl contact, continuous spoon grasp, and
  final image-space containment pass by construction; 30 FPS interpolation
  reduces jerk from 0.00107 to 0.00052. It remains PARTIAL because the free arm
  reaches via a 2D similarity transform up to 1.36x scale rather than physical
  joint kinematics.
- Frame-continuity correction: the former
  `20260809T150500Z-zero-jump-original-arms-v3` passed a scalar delta test but
  human review found severe pose-morph double exposure, so that result is
  REJECTED and the metric-only decision is invalid. The replacement removes all
  pose morphing and cross-video continuation: it begins directly at the first
  pushing pose, holds 16 effectively identical frames, then runs the native
  30-FPS C2 trajectory. Hold delta is at most 0.00290, hold-to-motion delta is
  0.00680, moving maximum is 0.524 below the 0.845 robust limit, isolated jumps
  are zero, and keyframe review finds no double exposure. The 4.53-second
  standalone clip is
  `outputs/spoon-bowl-extension/20260809T151000Z-stable-start-original-arms/stable-start-original-arms-push-bowl.mp4`.
  It is PARTIAL and explicitly not claimed as a seamless continuation; arm
  motion remains a 2D transform rather than physical joint motion.
- Auxiliary Cosmos 3 robotics renderer: the pinned Cosmos3-Nano adapter, trajectory and
  verification persistence, control-video frame/FPS checks, GPU selection,
  official inference command, deterministic MuJoCo control-bundle producer,
  structural edge-SSIM evaluator, failure provenance, checkpoint preparation,
  and GPU inference work. The smoke output passed structural alignment, but
  pose-level robot/object alignment and PhiZero-reference visual acceptance have
  not run.
- Visual-transfer baseline: the official native Wan adapter, strict
  preflight, provenance, source checkout, checkpoint, preprocessing, and
  inference work on the official upstream sample. The immutable experiment is
  `/data0/jiangyuhua/PhiAgent-0/outputs/visual_transfer/20260808T081713Z-4ece1657`.
  A real human manipulation video and robot reference image have not yet been
  evaluated.
- Real-perception foundation: the HaMeR right-hand adapter, FoundationPose
  matrix importer, physical-state extraction, and overlay are implemented. Only
  a synthetic teacher-observation fixture has passed end to end; a real video
  has not met the temporal-stability acceptance test.
- dex-retargeting 0.4.6 adapter: VECTOR/DEXPILOT EPL landmark mapping is
  implemented; the optional dependency installation and a real Allegro URDF run
  are still pending.
- Sharpa Wave right-hand support: a geometry baseline maps 21 named human hand
  points to the official 22-joint order and reads limits from the pinned
  Apache-2.0 MJCF. Six relevant remote tests passed and the official model
  produced an accepted rendered MuJoCo rollout from an explicitly synthetic
  hand fixture. No real human video or physical hand has been accepted.
- Multi-embodiment orchestration has evidence for two small test embodiments,
  not yet for production robot embodiments.
- ArtiCraft parallel asset route: a pinned, subprocess-isolated mini-ArtiCraft
  adapter, source preparation script, USDZ validation, failure recording, and
  six offline adapter tests are implemented. The pinned source checkout and
  isolated runtime are prepared at commit
  `7d43e25b26e9459aabf53d77d1d9325805bc1ea3`. No provider-backed asset has been
  generated because no provider credential is configured, and no generated
  asset has been converted into the target simulator yet.
- Trajectory-conditioned rendering has a tested Cosmos3-Nano backend, but its
  control producer and structural evaluator have not run in the pinned remote
  environment; pose-level robot/object alignment remains unimplemented.

## BLOCKED

- Exact PhiZero Figure 8(b) inference is blocked because the official repository
  has not released implementation code, pretrained tokenizer/decoder weights, or
  the human-hand adaptation checkpoint. EPL, Cosmos, and Wan2.2-Animate are not
  silently substituted for those artifacts.
- Visual-transfer evaluation needs a real human manipulation video and robot
  reference image. Public upstream assets can validate model execution but do
  not replace the requested human-to-robot evaluation pair.
- RoboMaster integration is blocked because its official code repository has no
  published license. Its released CogVideoX base checkpoint also carries
  separate use terms; neither is silently treated as Apache-2.0 code.
- HaMeR needs the separately licensed `MANO_RIGHT.pkl`; the project deliberately
  does not download or redistribute it.
- FoundationPose execution requires its NVIDIA license, CAD mesh, RGB-D,
  intrinsics, and initial mask. PhiAgent only imports explicit `object-in-camera`
  matrices and never labels 2D tracking as 6D pose.

## NOT STARTED

- Primary project goal: reproduce PhiZero Figure 8(b), transferring an encoded
  human-hand transition to a Sharpa dexterous hand by decoding the unchanged
  physical-language token sequence from an edited first frame.
- Target-specific implementation: official tokenizer execution, HRDexDB
  human-only source-domain adaptation, Sharpa first-frame conditioning,
  unchanged-token decoding, and three-case reference evaluation.
- Claim-eligible Sharpa adapter training remains blocked pending licensed,
  non-evaluation real or photorealistic synthetic Sharpa manipulation triplets.
  Development-only VACE pseudo-target training is staged separately and cannot
  substitute for this evidence.
- SPIDER teacher execution, production Franka+Allegro retargeting, Genesis and
  Isaac backends, learned PhiAgent heads, training stages, quantitative learned
  evaluation, and polished upload UI.
- ArtiCraft USDZ-to-target-simulator conversion and generated-asset scale, mass,
  collision, contact, and grasp acceptance tests.
- No learned component has replaced a teacher module.
