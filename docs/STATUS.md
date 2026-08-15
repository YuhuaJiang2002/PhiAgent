# Status

Evidence date: 2026-08-15. Status labels describe acceptance evidence, not code
presence. Measured runs span `a800-1` through `a800-4` and `zhaoli`; artifact
locations are recorded per experiment below.

## WORKING

- Official-model Wuji Hand video-retargeting and source-scene-lock audit
  infrastructure: `scripts/build_wuji_hand_comparison.py` and
  `scripts/build_wuji_scene_locked_comparison.py` consume an MP4 through the pinned
  official Wuji retargeter at commit
  `2918c60643cca3482ffa2d14d1f7fece1d9d7db9` and the official Wuji description
  submodule at `7d547ad50ca8cff92d999ae2cc01fc69bcb7c2b6`. The retained 20.7-second run
  produces a complete 621x20 `q`/`qdot` artifact from 621/621 direct MediaPipe
  detections with no held observations, no frozen frame transitions, zero joint
  position violations, and zero URDF velocity-limit violations; the largest
  velocity is 59.535% of its joint limit. The 2560x768 offscreen render takes
  17.8757 seconds (**34.7399 FPS**) and the encoded result decodes to exactly
  621 frames. The original standalone simulation builder completes the full
  audit in 30.5977 seconds (**20.2956 effective FPS**). The retained v7
  Shadow-style same-scene compositor uses a mask-excluded 21-sample temporal
  mosaic for the occluded background, obtains 92.323% direct background
  coverage, and spatially fills only the 8.197% never observed. Its 47.378--55.275
  pixel rigid forearm has zero width step, and the explicit wrist connector has
  at least 64 overlap pixels in every frame. Encoded replacement-side background
  second-difference falls 62.8% relative to v4. The compositor preserves every
  pixel outside its declared mask exactly before H.264 encoding; the static
  clean-plate interior deviates by at most one channel value, and the fixed hand
  scale remains 0.819879. The exact final reproducibility run takes 181.6535
  seconds (**3.4186 effective FPS**, 5.3487 render FPS). Earlier v1-v3 candidates
  were rejected for forearm-axis error, scene-mask leakage, or residual source-
  skin triangles; v5 was rejected by high-resolution review for visible blockwise
  clean-plate boundaries. Exact source, config, URDF, MJCF, trajectory, video,
  clean plate, mask, and poster hashes are recorded. The builder and
  identity/trajectory/scene-lock/decode gates are WORKING. The displayed result
  remains PARTIAL because it is monocular, vision-derived official-model
  simulation with a procedural forearm—not footage of physical Wuji hardware,
  metric depth/contact evidence, or real execution.
- Fail-closed long-video self-evolution and SAM2-observed flower/contact state:
  the v45 challenger at
  `outputs/joyai-self-evolution/20260814T201500Z-flower-harness-v45` is the first
  candidate promoted after v1-v44 were retained as rejected evidence. The loop
  keeps specialty limits immutable, runs a 14.4284-second dense grasp preflight
  before the 993.0692-second native audit, and requires a separate
  high-resolution human veto. The accepted repair takes 68.1048 seconds for 660
  frames (9.6909 FPS), 3.264x the rejected v6 baseline; its separately measured
  pinned SAM2 Hiera-L observation takes 199.0271 seconds (3.3161 FPS) on physical
  A800 GPU 4. The corrected symmetric decoder passes all 14 frozen full-video
  gates, including 138/138 projected-contact frames at or after 20 seconds and
  147/147 persistent-grasp frames; it rejects 147/147 grasp-erasure attacks and
  detects color, topology, contact-removal, and structure-ghost attacks. The
  harness, observation adapter, repair path, early-rejection gate, and audit
  infrastructure are WORKING. The displayed one-scene video remains PARTIAL:
  its contact is 2-D visual evidence, inherited grey torso texture remains, and
  metric depth, force closure, cross-scene generalization, and real execution
  are not established.
- Right-arm / flower ownership audit and repair infrastructure: the final
  660-frame challenger at
  `outputs/joyai-temporal-state/20260814T120000Z-right-arm-contact-lock-v61`
  combines dual-flow anatomical-right-arm state, source-grounded flower
  ownership over each complete reconstruction footprint, and an immutable
  projected-contact evidence layer. The independent v62 audit passes all 14
  frozen gates: right-arm self-flow mean improves 2.21746 to 2.18105, p95
  improves 4.43694 to 4.20351, and high-flicker count falls 29 to 18; wrong
  flower occlusion improves from 0.791397 mean / 1.0 p95 to 0.004413 / 0.0,
  while flower-owner flips improve from 0.082400 mean / 0.5 p95 to 0.006728 /
  0.0. The v63/v64 source- and candidate-motion audits both pass. The v65
  full native audit retains anchor contact 16/16, pre-20-second contact 33/35,
  late contact 11/11, persistent grasp 147/147, erasure rejection 147/147, and
  all four adversarial detectors. The ownership/contact guard runs at 7.9002
  FPS; with its dual-flow arm intermediate the added pipeline runs at 1.4573
  effective FPS. This audit and repair infrastructure is WORKING. The display
  candidate remains PARTIAL because the unchanged spatial-chroma-TV gate keeps
  the image-space result at 13/14, and all contact evidence remains 2-D rather
  than metric depth or force evidence.
- Dual-observer temporal audit infrastructure: source-motion and candidate-self
  optical-flow observers independently evaluate the same 161 frozen risk
  transitions and require strict reductions in mean, p95, and frozen
  high-jitter counts. The 27.5-second challenger at
  `outputs/joyai-temporal-state/20260813T182000Z-risk165-dual-observer-v30`
  passes both temporal contracts: source-motion mean falls 7.269%, p95 falls
  12.927%, and count-above-20 falls 7 to 4; candidate-self mean falls 4.611%,
  p95 falls 9.655%, and count-above-5.875135 falls 8 to 7. Exact endpoint,
  hand, flower, contact, and outside-window locks are recorded. This audit
  infrastructure is WORKING; the candidate itself remains PARTIAL below.
- Persistent workflow framework and flower long-video reference graph: the
  standard-library `phiagent.workflows` runtime provides named nodes,
  conditional edges, JSON-hashed thread checkpoints, streamed events,
  restart-safe review interrupts, failed-node retry, and shell-free existing-CLI
  adapters with strict physical-GPU selection records. The JoyAI-aware
  real-input replay at
  `outputs/workflows/20260813T032734Z-flower-long-video-joyai-v4` binds the existing
  660-frame/27.5-second video, manifest, adversarial report, high-resolution
  review, and promotion hashes through seven nodes and eight checkpoints. All
  twelve scoped visual gates pass, including non-frozen flower response and the
  20-second-and-later projected-contact visibility gate. The graph still exposes
  route-transition outliers at frames 479 and 559 plus the 23.33% legacy late
  hand-proxy diagnostic as three quality debts and schedules pinned
  JoyAI source-anchored `1+8n` causal challengers for `[463,495]` and
  `[543,575]` without weakening thresholds. The full local suite
  passes 835 tests with one optional MuJoCo skip, and targeted Ruff passes. This
  is WORKING workflow and visual-display audit infrastructure; it did not
  regenerate or retrain the model, and physical promotion remains false at 0/4
  physical gates with only one independent physical acquisition group.
- Perceptual 27.5-second VFM long-video demo: a persistent Wan full-timeline
  incumbent plus two official Wan2.2-Animate-14B late-window generations and
  three reviewed strict repairs are routed with deterministic overlap search;
  tracked source flowers and native background are restored outside the edit
  layer. The accepted 1280x720/660-frame/24-FPS artifact at
  `outputs/wan-perceptual-demo/20260812T150000Z-routed-wan14b-v4-seam-optimized`
  has post-decode flower exactness 1.0, background exactness 0.992429, flower
  dynamic fraction 0.980061, and 20-second-and-later projected-contact recall
  1.0. A corrected critic detects all color, topology, detachment, and ghost
  attacks across 585 sampled/contact frames, and high-resolution review passes
  the explicitly synthetic display scope. The legacy RGB-alpha image-space
  contract remains PARTIAL because its late hand-replacement proxy violates an
  anchor-derived limit on 23.33% of frames; two one-frame route discontinuities
  also remain visible under frame stepping. No metric geometry, force, exact
  telemetry, force closure, or real execution is claimed.

- Source flower/background preservation beyond 20 seconds: the object-factored
  v6 route projects the measured source scene outside a per-frame person edit
  support and restores source-visible flowers after explicit person-core/skin
  z-order resolution. The real 27.5-second/660-frame result at
  `outputs/wan-long-object-factored/20260812T151000Z-source-state-projection-v6`
  has 1.0 known-source exactness, 0.0 flower MAD, and 0.0 flower temporal
  residual both before encoding and after decoding its lossless FFV1 output;
  flower/person-core and flower/skin-negative overlaps are both zero pixels.
  Projection takes 16.4654 seconds at 40.0840 FPS on CPU, or 46.2571 seconds at
  14.2681 FPS including two complete encoded-output audits. Dense, gap-window,
  contact-window, and full-resolution review passes the scoped flower/background
  and source-human-leakage gates. This is WORKING only for preservation of known
  source state; late robot-layer artifacts, exact contact, per-stem identity,
  physics, and real execution remain PARTIAL.
- Wan-Animate-2 long-video generation infrastructure: matched real-input
  27.5-second/660-frame runs on `a800-4` now keep the 14B model resident across
  windows, validate disjoint physical GPU pairs and rendezvous ports
  transactionally, isolate compiled caches, preserve per-window hashes/timing,
  and recover completed GPU work without recomputation after a controller
  disconnect. The four-A800 throughput profile generated all ten windows in
  509.7154 seconds at 1.29484 useful FPS, versus 2094.7445 seconds and
  0.315074 FPS for the matched per-window-reload baseline: 4.1096x generation
  wall-time speedup. End-to-end wall time fell from 2153.1051 to 575.9873
  seconds (3.7381x), and measured A800 GPU-hours fell from 1.163747 to 0.566349
  (51.334%). Identical source/reference/prompt/model/checkpoint/config hashes are
  enforced by `scripts/summarize_long_video_infra_benchmark.py`. This is WORKING
  infrastructure throughput and cost evidence; it does not accept the final
  video's manipulation motion or contact physics.
- Model-derived RGB-D observation infrastructure: a second pinned DA3 run on
  physical A800 GPU 0 generates 55 previously unseen interleaved frames in
  3.6071 seconds at 15.2476 sampled FPS and 7.4967x source-video real time.
  Together with the first lattice it provides 110 synchronized proposals at
  4 Hz. Fifty-five alternating 4 cm virtual-camera RGB-D views have exact
  constructed extrinsics, 96.3955% mean visible-surface coverage, explicit
  per-run world frames, bound source/model/artifact hashes, and a
  warnings-as-errors numerical run. This is `WORKING` proposal/training
  infrastructure only: high-resolution review exposes disocclusion cracks,
  new occluded-surface coverage is zero, independent physical groups are zero,
  and absolute metric calibration remains false. The v9 compiler therefore
  remains 0/4 physical gates and the valid supervisor rejects 5/5 spoof attacks
  without promotion. See
  `outputs/foundation-contact/20260812T132000Z-model-derived-rgbd-virtual-views-v2`
  and `outputs/foundation-contact/20260812T138000Z-continual-supervisor-model-rgbd-v9`.
- Generated-observation authority and Qwen3-VL probe infrastructure: pinned local
  Qwen3-VL-4B and 8B models independently inspect the same 14 samples spanning
  all 27.5 seconds on physical A800 GPUs 0 and 1. Inference takes 63.9146 and
  62.4017 seconds, respectively, at 0.2190/0.2244 sampled FPS. Their four-field
  contact/flower/finger agreement is only 28.5714%: 4B labels flower motion from
  2.125 seconds onward while 8B labels contact with a static flower at all 14
  samples. The v11 compiler binds both reports and all hashes but grants zero
  physical authority; the v12 supervisor remains 0/4 physical gates, passes 5/5
  spoof attacks, and promotes nothing. A five-source lineage audit recognizes
  the original 660-frame real RGB video as one physical acquisition group for
  source flower/background motion, while DA3 virtual RGB-D, H3 output, and both
  Qwen reports add no independent physical group and no metric-camera, q/qdot,
  or force authority. This is WORKING fail-closed observation/triage software,
  not working physical reconstruction. See
  `outputs/foundation-contact/20260812T145000Z-qwen3vl-ensemble-audit-v1`,
  `outputs/foundation-contact/20260812T151000Z-compiled-qwen-observations-v11`,
  and `outputs/foundation-contact/20260812T154000Z-observation-authority-qwen-h3-v2`.
- Foundation-contact continual supervisor and SkillHone behavior loop: the
  standard-library supervisor audits the latest immutable pipeline report in 0.05405
  seconds, rejects all five configured evidence-spoof attacks, keeps all four
  physical gates false, and selects the independent metric-camera bridge as the
  next dependency-ready architecture experiment. Local Forgejo 16.0.2, Ollama,
  and mode-0600 SkillHone settings are configured. The authored skill improves
  a same-model/seed/split strict probe from 0/4 to 4/4, while final native
  SkillHone campaign passes 4/4 probe, 3/3 test, and 7/7 adversarial after
  zero-independent-anchor, prior-filled-cropped-joint, and asset-presence
  identity attacks were added. Its provenance-complete aggregate binds all
  14/14 strict decisions, the private evaluation revision, settings mode,
  runtime hashes, and commands while keeping physical promotion false. Routing
  Ollama directly through its JSON interface avoids the
  incompatible Write-tool path. A non-thinking instruct candidate is
  retained as rejected evidence because it regressed the adversarial split to
  1/4. This is `WORKING` supervisor/behavior infrastructure only; no physical
  video model was promoted.
- HRDexDB object-disjoint data foundation: revision
  `a46347556efd7ed87e70e7e87293b462d7253d6f` is public, non-gated, and
  CC-BY-NC-4.0. The verified 343,588,654-byte core pilot contains six train,
  two validation, and two sealed test objects with paired human/robot RGB,
  camera calibration, robot state/tactile timestamps, object poses, and meshes.
  Every selected robot scene has `grasp_success=true` and a frozen paired human
  episode. This is WORKING dataset infrastructure, not transfer quality.
- DROID held-out raw lineage: structured SequenceExample decoding recovers exact
  raw trajectory/recording prefixes for LeRobot episodes 21, 60, and 77. Exact
  tasks and frame counts match. Across all nine selected wrist/exterior stream
  comparisons, decoded-frame PSNR p05 is 33.201--36.297 dB and dHash Hamming
  p95 is 1--2 bits, passing frozen 25 dB/8-bit gates against three hashed
  LeRobot AV1 sources. Episode 60 uses identity exterior-camera assignment;
  episodes 21/77 explicitly swap exterior 1/2 under the frozen one-to-one
  assignment rule. This is WORKING lineage, not model quality. Raw DROID
  training/redistribution remains blocked on unresolved official rights.
- Persistent real-world-only SOTA monitoring:
  `scripts/monitor_sota_campaign.py` uses strict
  `<1 GiB and no process` GPU-free classification, 120-second capacity/job
  polling, durable heartbeat/event JSONL, and terminal artifact checks. Its
  configuration explicitly excludes RoboTwin, MuJoCo, SAPIEN rollouts,
  simulator success/contact labels, and Curobo. DROID SVO calibration is now a
  completed WORKING job. All three WorldArena/BWM seeds and the corrected
  20-episode/10,000-bootstrap aggregate evaluation and the strict DROID raw
  SVO/HDF5-to-SequenceExample alignment are complete.
  The v2 monitor ended cleanly after the original 12-hour BWM GPU wait timed
  out without inference; v3 was superseded after the corrected DROID run. The
  v4 completed seed42 generation/postprocessing; v5 completed both additional
  seeds; v6 observed the DROID waiter transition to success and exited WORKING
  at `outputs/sota-monitor/20260812T223000Z-real-world-only-v6`.
- Native agentic AC-WM engineering branch: `phiagent/acwm` defines a
  frame-explicit action contract that distinguishes `camera:*` skeleton and
  pointmap conditions from `robot_base:*` EEF/joint actions. Optional isolated
  adapters for pinned OSCAR, Boundless World Model, and Kinema4D revisions
  perform native-input routing, physical-GPU preflight, revision checks, batch
  execution, provenance capture, evaluation, bounded repair/rerouting, and a
  mandatory human acceptance gate. The OSCAR preparation path also pins its
  Cosmos-Reason1-7B and Wan2.1 VAE runtime assets for offline inference. This is
  WORKING software wiring with CPU regression coverage; it does not mean all
  three heavy models have run on the same real scene.
- Evidence-backed experience history: an append-only, standard-library JSONL
  ledger now preserves successful, partial, blocked, and not-started conclusions
  with evidence, limitations, next actions, and explicit supersession links.
  The ledger contains 558 historical records at the latest validation, including the v11-v18
  morphology, topology, object-preservation, depth-order, contact, and temporal
  regressions superseded by v19. The complete
  lightweight suite and targeted Ruff checks pass. The foundation-contact
  supervisor now adds one bounded, benchmarked SkillHone behavior-evolution
  loop; it does not establish autonomous physical video-model improvement.
- Guarded demo-video factory software loop: a standard-library record and
  cost-aware ridge router contract, immutable command-worker batch runner,
  physical-GPU selection/lease path, grouped holdout promotion, complete-recipe
  exploration mode, accepted-video index, and planner-distillation preference
  export, provenance-complete native AC-WM worker, and strict multi-scene
  campaign builder are implemented. The BWM production extension now adds a
  pinned WorldArena2.0 compiler, 104.9x compact transfer cache, physical-GPU
  sharding, strict reference/flow evaluation, storyboard generation, immutable
  model promotion, and measured cost reporting. A real task-disjoint campaign
  used `clean_table`, `fold_shirt`, and `pour_water` for training,
  `pour_over_coffee` for selection, and the untouched `wipe_table` task for
  final testing. The promoted 99,637,248-parameter action adapter at learning
  rate 5e-6 and epoch 2 raises test future SSIM from 0.825670 to 0.835889,
  lowers background MAD from 0.216235 to 0.187611, and passes every frozen
  non-regression gate on 4/4 samples. A three-A800 production run completed
  12/12 57-frame, 20-step samples in 112.594 seconds: 383.681 samples/hour,
  9.383 wall seconds/sample, and 7.779 GPU-hours/1,000 samples. Twelve focused
  remote tests and targeted Ruff checks pass. This is WORKING generated-video
  model+harness evidence, not proof of 3-D contact, task success, collision
  safety, or real-robot execution.
- CPU-only scene-aware replacement routing: a 180-frame, 30 FPS synthetic
  acceptance demo covers two independently identified subjects, explicit
  handedness, hand-and-forearm plus full-body replacement, two protected
  objects, object/robot occlusion, and two hard camera cuts. The accepted v2 run
  reports zero route diagnostics, zero changed values across 47,899,440 audited
  background RGB channel values, and zero changed values across 2,784,435
  protected-object channel values. The deliberately naive comparison changes
  542,633 protected-object channel values. This validates routing and
  compositing, not detector quality, photorealistic generation, or 3D contact.
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
- Local regression after shadow-skill evolution and phase-04A flower
  identity/contact expansion: 359 tests passed and
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
- Flower-preserved, proportion-calibrated full-length pose-rig visualization:
  `outputs/robot-person-replacement/pose-rig-runs/20260810T160000Z-smoothed-holding-bouquet-v19-accepted`
  combines v12's reviewed fixed 1.0x hand proportions with v14's exclusive
  cross-side pixel ownership and connected shoulder-to-hand chains. A complete
  bouquet-only RGBA layer replaces the incomplete source-flower masks; it is
  composited after the upper/lower arms and before both hands, bound inside the
  holding hand, and driven by a 2.5-frame zero-phase filter plus a symmetric
  12-pixel vector step bound. The accepted 1280x720/24 FPS video decodes all
  660 frames and has SHA-256
  `cf814cd716cb467cda5f72828bb7b485599aef6a6abf64cd5e5a54ec457bef05`.
  Both hand transforms remain fixed at 1.0x; both rendered limb chains remain
  connected in every frame; the bouquet is nonempty in 660/660 frames, has
  area p99/p01 ratio 1.002964, maximum target step 12.0 pixels, and minimum
  hand-over-bouquet contact of 537 pixels. Independent encoded validation finds
  zero full-frame and zero person-ROI transition outliers at threshold 4.0,
  with maximum ratios 2.6067 and 2.7375. Dense, early, full-size contact,
  diagnosed jump, filtered-step, and encoded peak reviews found no flower
  disappearance, hand split, scale breathing, visible jump, or hand-flower
  separation. This is WORKING for the declared 2D clip, not evidence of 3D
  depth, flower deformation, articulated finger grasping, force, collision
  safety, PhiZero inference, or real-robot execution.
- Full-length confidence-routed flower comparison:
  `outputs/flower-confidence-routed-comparison/20260811T035000Z-v44` applies
  the same successful policy as the short silver/graphite reference: validate
  one temporally coherent immutable candidate, then preserve that candidate for
  every frame. It globally selects the accepted v19 pose-rig video and performs
  only aspect-preserving presentation resize plus a two-row vertical stack; no
  per-frame repair, Poisson fusion, mask recomposition, or candidate mixing is
  enabled. The published comparison is
  `demo/showcase/real-flower-arranging-confidence-routed-robot-vertical.mp4`,
  672x768 at 24 FPS, decodes 660/660 frames over 27.5 seconds, and has SHA-256
  `d4f25075bd79c53062e36281e14caacb21e5a1f1734f747769551b4f7d5e1e09`.
  The source and robot timelines match exactly; immutable-candidate hash and
  acceptance gates pass; uniform storyboard and midpoint review show no gray
  human silhouette, source-hand return, missing bouquet, or background rewrite.
  This comparison is WORKING within the same 2D visualization scope as v19.
- Earlier full-length same-scene flower visualization:
  `outputs/robot-person-replacement/pose-rig-runs/20260810T020722Z-imagegen-base-v7`
  uses one reviewed arm-removed robot torso base and six independently rigid,
  pose-driven arm/hand pieces. The real 27.5-second source decodes 660/660
  output frames; background lock is 0.9999995, source blend inside the person
  clear is zero, pose misses and low-confidence frames are zero, there are no
  cross-dissolves or anchor cuts, and the maximum transition ratio is 2.7288
  below 4.0. Dense 28-frame review finds no human or prior gray inpaint block.
  This earlier baseline is WORKING only as an image-space visualization. Stem identity, depth
  order, grasp/contact physics, calibrated 3D motion, and real-robot execution
  are not verified.
- Full-length Wan-Animate-2 flower visualization:
  `outputs/wan-flower-animate2/20260810T-full660-distilled-overlap-v3/final-bridge32-r4-anchor-preserved-accepted`
  is now the preferred 27.5-second visual conversion. Ten 81-frame driving
  windows use the same official Wan-Animate-2 commit, hashed distilled weights,
  robot reference, prompt, 640x352 request, 10 steps, guidance 1.0, and seed 42
  as the strong three-second result. Nine additional half-window-offset jobs use
  the identical recipe to route around periodic local-frame resets. Stable-range
  seam search over all 19 windows and a four-frame-radius cosine blend reduce
  the encoded maximum transition ratio from the previous accepted 3.9027 to
  2.6564. Five diagnosed neighborhoods outside the quality anchor receive a
  bounded three-frame-radius cosine crossfade; their six transition energies fall
  from 2.1007-2.4525 to 1.0269-1.3613 after encoding. The 38-frame core from the
  proven three-second result remains unchanged before final encoding. The output
  decodes 660/660 frames at 624x352 and 24 FPS. Fifty-five uniform samples,
  consecutive before/after frames at all repaired neighborhoods, and thirteen
  residual-peak neighborhoods show one coherent silver-black robot with no
  visible source human, duplicate limbs/flowers, or new crossfade ghosts. The
  bridge-stage temporal consistency is 0.9688. This is WORKING only for
  image-space visualization. Motion preservation remains 0.2929, and exact stem
  identity, grasp/contact timing,
  calibrated motion, collision safety, and real-robot execution are not
  accepted.
- MiniMax-H3-guided early Wan continuity:
  `outputs/wan-h3-hybrid/20260810T-early-temporal-guide-v2-accepted` is the
  preferred candidate when judging the user-reported early jumps. The reviewed
  full-length H3 result is used only as a temporal guide because its robot
  identity gate failed; no H3 pixels enter the output. H3 and the real source
  jointly identify 19 unsupported Wan transition spikes in frames 0-236.
  Thirteen H3-timed local Wan bridges reduce their mean subject transition
  energy from 2.2283 to 1.3194 after encoding, and a wider second pass reduces
  residual frames 27/196 from 1.5996/1.9186 to 0.8963/1.2210. The two remaining
  threshold events at frames 22 and 46 were reviewed as continuous flower
  motion rather than pose cuts. Forty early samples, 55 full-video samples, and
  consecutive crops around every repaired neighborhood show no visible H3
  identity drift, duplicate limbs/flowers, or crossfade ghosts. The result
  decodes 660/660 frames at 624x352/24 FPS and protects the reviewed [259,297)
  Wan anchor before both encoding passes. The whole-video maximum transition
  ratio is not lower because it is dominated by unmodified start/later action
  transitions; this acceptance is explicitly limited to early continuity.

## PARTIAL

- JoyAI dual-observer stabilized flower demo: the published 660-frame candidate
  at `outputs/joyai-temporal-state/20260813T182000Z-risk165-dual-observer-v30`
  adds a candidate-self bidirectional residual consensus followed by a
  source-motion causal state pass. Both independent jitter audits improve as
  reported above, while late projected contact remains 11/11, the
  source-observed hold interval passes 147/147 visual-grasp frames and 147/147
  grasp-erasure attacks, and all four adversarial attacks are detected. A fresh
  isolated-runtime audit exposes a non-reproducible earlier claim: the unchanged
  5.981568 spatial-chroma-TV limit is failed by both the published incumbent
  (9.794221) and challenger (9.746221). The challenger therefore passes 13/14
  frozen image gates, not all gates; no threshold was relaxed. The incremental
  dual-observer stage runs in 116.1404 seconds (5.6828 FPS), while the complete
  historical quality stack is 555.5069 seconds (1.1881 effective FPS). This is
  single-scene 2-D visual evidence without metric depth, force closure,
  cross-scene replication, or real-robot execution.
- Geometry-grounded AC-WM redesign: a primary-source review of 2025--2026
  releases identifies FlowWAM robot flow, Kinema4D pointmaps, and OSCAR
  skeletons as the strongest executable spatial controls; vector-only BWM
  remains a baseline. PhiAgent now defines a camera-frame `robot_flow`
  representation and a pinned FlowWAM Stage-1 adapter that requires an encoded
  robot-only flow video plus URDF, camera calibration, and flow-producer
  provenance. The adapter pins source `f06fa460`, model revision `1e68f76c`,
  10,137,267,208 checkpoint bytes, and SHA-256 `e211e32b...e96c4`; heavyweight
  imports remain isolated. A matched target-versus-hold BWM guidance ablation
  proves `scale=1` is byte-identical to the release, then rejects scales
  1.5/2.0/3.0: every flow cosine remains negative, flow EPE and visual metrics
  regress, and the duplicate shadow remains. WorldArena counterfactual rebasing
  is also corrected from invalid Euler/gripper arithmetic to quaternion SO(3)
  composition. FlowWAM remains gated on the real Cobot-Magic scene because no
  held-out URDF/camera-to-flow producer has passed calibration; future RGB/flow
  is not used as a fake control. See `docs/GROUNDED_ACTION_WORLD_MODELS.md` and
  `outputs/bwm-action-guidance/20260813T030350Z-action-guidance-scale-sweep-summary.json`.
- Exact numeric BWM action control: a standard-library contract now accepts
  either 57 exact per-frame dual-arm EEF samples or explicit keyframes, preserves
  a named `robot_base:*` frame, supports explicit Euler/gripper and
  XYZ/quaternion profiles, separates action sample rate from video FPS, validates
  the exact channel/frame contract against matching statistics, and uses SLERP
  for quaternion keyframes. The legacy WorldArena compiler labels are corrected
  in derived metadata only after both four-value orientation blocks pass unit-
  quaternion checks; source numeric arrays and hashes are unchanged. A real
  held-out `wipe_table/episode_0` case uses source/action frames 296--352 at
  30 Hz, 0.484006 m of right-arm path, 0.126686 m terminal displacement, and
  zero values outside either the training min/max or p01/p99 bounds. Official
  BWM revision `738a8d3c` ran for 20 steps and seed 20260812 on directly
  reachable `a800-4` physical GPU 0, selected with 81,225 MiB free. The immutable
  run generated 57 H.264 frames at 896x672/24 FPS in 52.9085 seconds. Future
  SSIM is 0.85843, endpoint SSIM 0.84051, and motion-amplitude error 0.08721,
  but flow-direction cosine is only 0.14303 and dense storyboard review finds a
  large later shadow-like duplicate. The result is therefore generated and
  displayed but user-rejected/`PARTIAL`, not accepted action adherence, task
  success, or physical execution. See
  `outputs/numeric-bwm-real-inference/20260812T135323.974031Z-worldarena-wipe-episode0-official-seed20260812`.
- Expanded AC-WM statistical lane: a pinned 100-physical-episode WorldArena
  cache now yields 60 train, 20 validation, and 20 test rows. Eighty source
  videos contain one extra terminal video sentinel; the compiler excludes only
  that frame and keeps all selected windows inside the HDF5-aligned prefix.
  The transferred test bundle is 937,316 bytes and the new counterfactual suite
  has 20 physical `wipe_table` groups and 40 factual/swapped rows. The matched
  official-BWM/promoted-adapter seed-42 job is durably queued on a800-1 GPUs
  1/4 and records a 120-second capacity heartbeat; compatible GPUs remain
  occupied by other users. Physical A_swap references are still absent, so
  this remains a diagnostic, not executable causal GT.
- Raw DROID calibrated geometry smoke: one pinned 46,338,367-byte episode
  verifies all 77 HDF5 datasets across 338 finite aligned rows, commanded and
  measured robot-base Cartesian/joint/gripper labels, static exterior
  extrinsics, and a moving wrist transform. ZED 5.1 verifies all three camera
  serials, 1280x720 intrinsics/distortion, meter depth, and hashed depth
  samples. Calibrated depth splatting reaches 3.78%/3.05% target coverage and
  7.46/9.10 dB visible PSNR for exterior views 1/2. Structured raw-to-LeRobot
  lineage for held-out episodes 21/60/77 is now WORKING, including explicit
  episode-local exterior-camera assignment. Held-out SVO calibration/depth and
  raw-HDF5-to-LeRobot timing for episodes 21/77 are now WORKING. Strict-W
  remains `BLOCKED` because official raw-data rights, episode-60 raw
  availability, and learned held-out disocclusion/view quality are unresolved.
- Held-out raw DROID retrieval: episodes 21 and 77 each provide one trajectory,
  one metadata JSON, and three SVOs; all ten files (28,421,193 bytes total)
  match official GCS generation/MD5 records and have local SHA-256 provenance.
  The lineage-verified episode-60 prefix has zero public objects, and the public
  AUTOLab inventory omits 2023-07-31 entirely, so that episode remains
  `BLOCKED` rather than being substituted. A unified ZED 5.1 acceptance run
  processes all six episode-21/77 cameras on strict-free physical GPU 5 in
  8.54 seconds. Episode 21 has three 92-frame 1280x720 streams with sampled
  depth finite fractions 0.1420--0.6837; episode 77 has three 100-frame streams
  at 0.4789--0.7539. Serial, intrinsics, distortion, timestamp, SVO hash, and
  depth-artifact gates pass. A full-frame raw SVO/HDF5 audit then verifies all
  six cameras: PSNR p05 27.037--35.264 dB, dHash p95 1--3 bits, same-index
  advantage 1.999--11.224 dB over adjacent shifts, timestamp residual p95
  0.459--0.529 ms after a 41.416--41.538 ms constant offset, and exactly one
  terminal HDF5 row. The raw-to-LeRobot chain is WORKING. Four held-out
  depth-splat lower bounds remain `PARTIAL`: coverage is only 3.586%--19.129%
  and visible PSNR 7.500--16.439 dB, so raw splatting is conditioning evidence,
  not a novel-view solution. Overall coverage remains `PARTIAL` because episode
  60 and rights/learned-disocclusion gates are blocked.
- HRDexDB object-disjoint raw Wan validation: an apple train-object Inspire F1
  image conditions both validation objects; paired validation robot videos are
  evaluator references only. Banana scores 0.3978 motion, 0.9694 identity,
  0.0039 object, and 0.1166 temporal. Beige brush scores 0.3731 motion,
  0.9792 identity, and 0.0244 temporal; its low-chroma object gate is explicitly
  unresolved. Both validation cases reject the raw method, so sealed
  `box_pink`/`cactus` test objects remain unopened.
- Historical RoboTwin work is **EXCLUDED** from the active research scope by
  the real-world-only requirement. Its render/reset/21-seed inventory and
  failed MPlib attempts remain immutable engineering history, but they may not
  be used as training data, evaluation data, contact/task-success evidence,
  promotion evidence, or paper claims. Curobo transfer and all RoboTwin
  successors were stopped. Real A+/A_swap evidence now requires
  operator-approved physical execution with synchronized telemetry, video,
  contact/safety logs, and blind outcomes.
- Foundation-model-assisted metric contact reconstruction: pinned DA3 Nested
  now produces learned metric depth, intrinsics, and camera poses for 55
  full-span samples of the 27.5-second/660-frame candidate in 5.1189 seconds of
  inference, or 10.7445 geometry FPS and 5.2827x source-video real time. The
  one-time cold path including extraction and 21.8081-second model loading takes
  27.2883 seconds, approximately 1.0078x real time. A two-context audit bounds
  scale-ratio variation p95 to 0.31% and worst relative-depth residual p95 to
  2.80%; the learned proposal passes, but the physical camera stage remains
  false because this is not an independent absolute calibration. Exact G1 and
  bilateral Sharpa asset hashes pass. The 17-sample active-stem lift is rejected:
  its maximum segment-length CV is 1.857 versus the fixed 0.12 gate, exposing
  mask truncation/depth-edge failure. Full-q render reprojection and real
  sensor/solver forces are also missing. The compiler therefore reports all four
  physical stages false and preserves overall `PARTIAL`. An automatic
  architecture-level evolution plan proposes external metric calibration,
  full-asset analysis-by-synthesis, dynamic-point/rod optimization, and
  dependent sensor/inverse-dynamics force fusion without relaxing thresholds.
  A new uncertainty-weighted affine inverse-depth bridge requires at least 20
  registered observations from two independent groups, held-group error,
  bootstrap uncertainty, frozen exact-asset SHA, complete-`q`, and reprojection
  gates. It passes an end-to-end synthetic acceptance fixture but the real
  27.5-second input has zero independent metric anchors; the real bridge returns
  `PARTIAL` in 0.00712 seconds and emits no calibrated NPZ. The hash-bound
  recompile and continual supervisor confirm all 0/4 physical gates still fail
  and therefore leave the video/contact model `PARTIAL`.
  Thirty-seven focused tests pass. See
  `docs/FOUNDATION_CONTACT_PIPELINE.md` and
  `outputs/foundation-contact/20260812T122100Z-compiled-pipeline-camera-bridge-v5`.

- Calibrated simulated flower-contact bridge: the new direct-metric compiler
  path consumes hash-bound RGB-D without relabelling it as DA3 evidence and
  requires exact G1/Sharpa registry matches. The full run at
  `outputs/foundation-contact/20260812T201000Z-metric-flower-coupled-force-full660-v7`
  emits 660 frames/27.5 seconds at 24 FPS, complete 73-coordinate robot state,
  one persistent 12-node stem, force covariance, and fixed right-hand topology.
  Valid depth coverage is 1.0, maximum joint velocity is 2.31093 rad/s, maximum
  segment CV is 0.04265, and combined inverse-rod/contact-wrench residual p95 is
  0.000654 N. Pad forces are solved against the inverse-rod-required wrench;
  proximity and a copied balancing wrench cannot pass.
  All 300 contact frames pass exact distributed Sharpa-pad 6-D force closure
  and all 298 driven frames have causal stem response. Four raw failures at
  frames 401/405/455/466 use the first bounded `[-2 mm,0,0]` contact projection;
  all other frames are unchanged, and consecutive repair review shows no jump.
  Complete decode plus uniform/contact review pass. The independent compiler
  reports 4/4 physical stages and every cross-stage bundle-lineage gate
  `WORKING`. This supersedes the earlier v1 claim, whose six contacts were
  constructed and whose artifact lineage was incomplete.
  A target-independent exporter creates 12 physical-control training clips and
  four source-frame-disjoint validation clips with zero overlap. A rank-4 VACE
  smoke completes 12/12 steps and binds its 5,503,040-byte adapter SHA. The
  strict matched held-out evaluator reports contact ROI +0.000608 and contact
  motion +0.003968, but absolute contact/global similarities are only
  0.05898/0.00776. Human review rejects both arms for a green rounded mass,
  dark background, and color band instead of the pink flower and workspace.
  Remaining windows stop under the veto. This is
  `WORKING` simulated physical supervision only; the real Pexels replacement
  remains `PARTIAL` because its camera still has no independent metric
  observation.

- Latest open-model flower-state route: the 2026-08-13 audit verifies public
  revisions, weights, licenses and custom-input boundaries for DA3, MoGe-2,
  MegaSaM, CARI4D, V-DPM, SpatialTrackerV2, VideoManip, GMR,
  dex-retargeting, TrackDLO/MultiDLO, PhysTwin, DeformMaster, Wan and VACE;
  paper-only systems are not treated as dependencies. A standard-library
  router keeps monocular V-DPM/SpatialTracker output relative and selects the
  calibrated RGB-D/MultiDLO route only when independent scale exists. A new
  global multi-stem optimizer enforces persistent IDs, exact material lengths,
  fixed/free roots, temporal state, ID-swap attacks and occlusion covariance.
  On the real 17-frame `active-pink-stem-01` proposal, exact rod projection
  lowers segment CV from 1.85723 to `1.26e-14`, but observation-residual p95 is
  `1.5965` times the stem length versus the frozen 0.10 gate. It is therefore
  rejected and remains `PARTIAL`. Pinned V-DPM code and an A800 CUDA/import
  environment are prepared; real inference is `BLOCKED` because the 6.65 GB
  checkpoint is absent from all authorized caches and both official and mirror
  downloads time out. See `docs/FLOWER_LATEST_OPEN_MODELS_20260813.md` and
  `outputs/foundation-contact/20260813T030500Z-multistem-active-pink-v3`.

- Exact-asset robot trajectory bridge: a new optional-dependency-free solver
  and evidence compiler derive the exact 29-DOF G1 plus bilateral 22-DOF Sharpa
  schema from the three hash-bound MJCFs, fit full `q` and
  `camera_from_robot_base` through a supplied exact forward renderer, and reject
  data-Jacobian rank deficiency, large posterior uncertainty, wrong hashes,
  partial q, temporal gaps, weak held-out reprojection/silhouette evidence, and
  alternative-asset ambiguity. Ten focused solver/attack tests and the 38-test
  related foundation-contact suite pass with warnings treated as errors. The v2 real
  27.5-second run validates all three file hashes and all 73 joint definitions
  in 0.01075 seconds, but correctly returns `PARTIAL` and emits no trajectory:
  the view crops every leg joint, flowers occlude most finger joints, the exact
  generated embodiment is unverified, and no foundation-keypoint plus
  exact-render evidence bundle exists. v2 additionally rejects joint states
  without sensor/calibrated-view/physics authority and unnamed or unpinned
  proposal/render stacks. The bound pipeline remains 0/4 physical gates. Its
  continual supervisor remains valid, rejects all 5/5 semantic attacks, and
  does not promote. SkillHone now passes 4/4 probe, 3/3 held-out
  test, and 7/7 adversarial items at private revision `6e18538`; a failed old
  harness run is retained separately. See
  `outputs/foundation-contact/20260812T124800Z-exact-asset-full-q-real-v2`,
  `outputs/foundation-contact/20260812T125100Z-continual-supervisor-exact-asset-v7`,
  and `outputs/foundation-contact/20260812T123400Z-skillhone-exact-asset-adversarial-v9`.

- Long-video articulated contact and flower dynamics: the complete 27.5-second,
  660-frame hand-union candidate now has a fail-closed full-video audit and an
  architecture-level self-evolution loop. The audit processes all frames in
  13.6573 seconds at 48.326 FPS on the local CPU and finds 9/329 projected
  contact-driver frames without a flower response inside two frames; the longest
  frozen run is frames 88--91 (four frames, approximately 167 ms). Missing
  calibrated depth, surface gaps, external/support wrench, contact force,
  articulated metric hand state, and full per-stem identity are hard failures;
  2-D adjacency cannot satisfy them. The new
  `metric-articulated-rod-residual-v1` software core adds fixed joint topology,
  rooted deformable-stem physics, a friction-cone/6-D grasp-wrench force-closure
  certificate, causal response auditing, adversarial attacks, and complete
  architecture tournaments. Fifteen focused tests and Ruff pass; all response
  erasure, depth/force spoof, and broken-tree attacks are detected. Neither the
  current pixel architecture nor the unrendered new architecture is promoted,
  and the user's high-resolution finger/contact veto remains active. See
  `docs/LONG_VIDEO_CONTACT_DYNAMICS_EVOLUTION.md` and
  `outputs/wan-long-contact-dynamics/20260812T082000Z-first-principles-evolution-v1`.

- Wan-Animate-2 generation beyond 20 seconds: the continuity-first profile ran
  the complete 27.5-second source as one persistent two-A800 temporal chain with
  nine aligned source-camera-frame rolling references. It generated in
  822.7548 seconds at 0.802183 useful FPS, a 2.5460x speedup over the matched
  baseline, while A800 GPU-hours fell 60.723% to 0.457086. Removing the
  independent four-GPU chain reset reduced worst raw overlap best-seam MAD from
  14.2593 to 7.4452 (47.79%) and mean best-seam MAD from 5.2356 to 4.7994.
  However, same-time overlap disagreement accumulates late in the single chain;
  the radius-4 stitched candidate remains `review_required`, with maximum
  transition ratio 6.0800, temporal consistency 0.96397, and motion preservation
  0.26370. Dense and all-seam contact sheets retain one robot without an obvious
  source-human return, but this does not satisfy the prior accepted long-video
  transition quality or exact-action gates. Long-horizon replacement is
  therefore PARTIAL, not WORKING. Object-factored v6 removes flower/background
  error accumulation from the model's responsibility, but late artifacts inside
  the robot layer remain. A first-principles robot RGB-alpha-contact route at
  `outputs/wan-long-robot-contract/20260812T073750Z-first-principles-v1`
  rejects v6's late color drift, projects the more stable robot hypothesis onto
  exact source flowers/background, and fills only missing hand support. The final
  660-frame candidate has 0% late high-chroma/palette violations, 12/12 projected
  contacts, and 1.11% late hand-support violations; projection plus repair runs
  in 20.6982 seconds at 31.8869 FPS. All 11 scoped image-space gates and four
  adversarial attacks pass, but high-resolution review still finds intermittent
  finger morph/motion smear and ambiguous 3-D contact, so the full video remains
  PARTIAL. A rank-8 VACE adapter trained for 96 steps on physical A800 GPU 5 and
  was tested blind on frames 594--642: contact similarity improves from 0.3576
  to 0.4250, but edit-region similarity is only 0.1584, topology similarity only
  0.4581, and temporal similarity regresses from 0.7038 to 0.6792. Full model-only
  rollout was stopped; same-scene internalization and general capability remain
  PARTIAL. See `docs/LONG_VIDEO_ROBOT_LAYER_CONTRACT.md`.
- Four-track SOTA evidence campaign: the current round added physical-episode
  lineage, history-preserving EEF counterfactuals, inference-seed aggregation,
  cluster-aware bootstrap/Holm utilities, a task-disjoint video-to-action
  benchmark, a calibrated novel-view readiness gate, and a frozen Hand2Dex
  raw-baseline smoke. The round produced decision-relevant negative evidence,
  not a SOTA result. Every failed or blocked attempt is retained in the
  experience ledger.
- Lineage-safe AC-WM action audit: official BWM and the promoted
  `phiagent-bwm-worldarena2-action-adapter-v1` each generated four
  factual/action-swapped pairs at seeds 42, 314159, and 20260811. Donor future
  XYZ/Euler displacement was rebased onto the source history endpoint; all
  nine history states remained unchanged. Four clips collapse to only two
  physical `wipe_table` episodes. After averaging seeds within episode, the
  adapter raises factual future SSIM from 0.823330 to 0.829861
  (paired lower bound +0.003742), but flow endpoint error changes from 2.066107
  to 2.067347 pixels and the wrong-action SSIM/flow-margin lower bounds are
  -0.001428/-0.005080. The action-control audit therefore fails and remains
  `PARTIAL`; eef_abs is accurately described as realized absolute EEF
  trajectory conditioning, not low-level command control.
- WorldArena video-to-action labeling pilot: a task-disjoint offline
  RGB-to-14D-realized-EEF ridge pipeline, explicit robot-base frame, physical
  episode grouping, zero-delta/train-mean controls, and field-level
  abstentions are implemented and ran on real synchronized clips. The coarse
  model regressed translation delta RMSE from the zero baseline's 0.3977 cm to
  0.5101 cm and rotation error from 4.4345 to 12.9393 degrees. A spatial-grid,
  per-channel-alpha follow-up regressed further to 0.7177 cm and 37.2670
  degrees. Both are retained negative results. Uncalibrated flow-ridge tuning
  is closed; contact, phase, and video-only robot-base labels remain blocked.
- Calibrated DROID novel-view lane: the new `Strict-W` readiness audit confirms
  synchronized wrist/exterior RGB plus state/action timestamps. Structured raw
  lineage plus serial/calibration/intrinsics/distortion/depth provenance is now
  WORKING for held-out episodes 21 and 77. Episode 60 has no public raw tree,
  raw-data rights remain unresolved, and raw-HDF5-to-LeRobot video/action
  timestamp offsets are not yet verified. Geometry-aware
  GEN3C/TrajectoryCrafter/ReCamMaster comparisons therefore remain `BLOCKED`
  rather than being run on a partial camera contract. The older anchor-assisted
  VACE LoRA remains rejected and is not relabelled as strict ego-to-exo
  generation.
- Hand2Dex embodiment-transfer smoke: pinned Wan-Animate-2 distilled inference
  now supports concurrent jobs with pair-specific rendezvous ports, explicit
  machine rank/world size, and experiment-owned TorchInductor/Triton caches.
  On the three frozen seed-42 cases, raw outputs pass every motion and target
  identity proxy but fail every object and temporal gate: case scores are
  `(0.8080, 0.9928, 0.2112, 0.3148)`,
  `(0.7855, 0.9750, 0.6538, 0.2462)`, and
  `(0.7744, 0.9874, 0.0039, 0.3119)` for
  motion/identity/object/temporal. Complete pass rate is 0/3, so the frozen
  fail-fast rule stops seeds 43/44. Exploratory source-object factorization
  raises case-2 object consistency to 0.8767 but leaves temporal at 0.2776;
  case 3 does not improve. This repair is post-hoc and separately labelled.
- Batch low-cost demo-data production and self-evolution: the reusable harness,
  repository-local `evolve-demo-video-factory` skill, reference quality
  contract, held-group lightweight trainer, and future planner-distillation
  export now exist. Thirty existing real-input Ego repair candidates were
  migrated as six complete five-recipe episodes and replayed across three held
  action groups. Both learned and fallback routes measured 0% acceptance and
  five attempts/cost units, so the absolute acceptance gate correctly rejected
  promotion. These groups share one source interval and cannot substitute for
  two independent scenes. The objective remains PARTIAL because no accepted
  two-scene GPU tournament, promoted real-input router, or fresh held-scene
  production cost result exists. See `docs/DEMO_VIDEO_DATA_FLYWHEEL.md`.

- Agentic grounded AC-WM SOTA campaign: the latest-paper synthesis, pinned BWM
  and Wan2.2 revisions, explicit 16D-quaternion-to-14D-Euler action conversion,
  robot-base frame contract, embodiment/task-grouped split, overlapping-history
  clip compiler, seven hashed public-BWM compatibility patches, physical-GPU training
  launcher, paired-bootstrap all-baseline promotion gate, and genuine
  real-robot evidence validator are implemented. The active dataset lane is the
  100-episode real-robot WorldArena cache with a task-disjoint 60/20/20 split
  and 20-episode matched BWM suite. Historical simulator training smokes are
  excluded from current method, promotion, and paper evidence. All three
  matched seeds are complete. WorldArena action swaps use measured real
  trajectories but remain
  action-sensitivity diagnostics until both branches are physically executed.
  Balanced real-world training, cross-embodiment comparison, and physical-robot
  trials have not completed. Overall status is therefore PARTIAL; SOTA and
  real-robot operation are not claimed. See
  `docs/ACWM_SOTA_CAMPAIGN.md`.
- WorldArena wipe20 three-seed matched audit: official and promoted BWM each
  produce 120 videos across seeds 42/314159/20260811 for 20 independent
  physical episodes. Seeds are averaged within episode before the corrected
  10,000-bootstrap audit. Candidate factual future SSIM rises from 0.849294 to
  0.861907 (gain +0.012613, 95% lower bound +0.009622), while factual flow EPE
  regresses from 1.734004 to 1.739377 pixels (lower-is-better gain -0.005373,
  lower bound -0.016480). Wrong-action SSIM margin changes from -0.000159 to
  -0.004156 and wrong-action flow-EPE margin from -0.001791 to -0.010558; both
  gains and lower bounds are negative. `audit_passed=false`. The adapter is
  therefore not promoted for action control or SOTA.

- Historical metric-driven BWM adaptation (**EXCLUDED**): a physical-A800 run
  trained on simulator-derived released episodes, so neither its checkpoint nor
  its metrics are eligible for the active real-world-only campaign. It had trained the action
  encoder for ten steps on released episodes 40 and 41, then compared official
  BWM and two trained checkpoints on a fixed episode-42 window plus a
  wrong-action counterfactual. The five-epoch model improved causal action
  margin and optical-flow endpoint error but regressed future SSIM, background,
  temporal-gradient error, and terminal state. The two-step early-stop reduced
  the regressions but still lost future/terminal SSIM and flow direction. Both
  candidates are rejected and the status is PARTIAL. The 1344x380 comparison is
  retained only at
  `outputs/acwm-metric-driven-evaluation/paired-epoch0-20260811T1136/rejected/robottwin-reference-vs-official-vs-trained.mp4`
  has SHA-256
  `49c738bdda47e3fc8a98bb0ff7a2a14639ca7ec05f516537b358e9c8c93b44c5`;
  it is released-demo simulation evidence, not a real-robot video, and is
  excluded from training, promotion, paper claims, and the public showcase. The
  public AC-WM visualization instead
  uses the recorded EPIC-KITCHENS `P03_28` kitchen scene and its three accepted
  action-conditioned outputs.

- Real-scene robot-execution visualization:
  `demo/showcase/acwm-real-scene-vs-rendered-robot-execution-10s.mp4`
  synchronizes the unchanged ten-second real kitchen observation with rendered
  pour, shake, and handover robot effects in a labeled 2x2 comparison. The
  published artifact decodes as 240/240 frames at 1664x960 and 24 FPS; SHA-256
  is `b6135e06cbebc48f019eb93a68cdbdad829325b74510cff614b410b62fd709dc`.
  This is WORKING as a visual comparison only. The robot panels are generated
  counterfactuals, not physical-robot footage and not inference from the
  one-step BWM smoke checkpoint.

- Three-round GPU flower replacement using the cabbage-demo route:
  `outputs/gpu-flower-cabbage-route/20260811T045115Z-window110-198-multiround-v1`
  ran Wan2.2-Animate replacement with compiled SAM2 and the released
  relighting LoRA on physical A800 GPU 4 for real source frames 110--198.
  All three raw candidates are preserved. Seed 42 passes the dense sampled
  human-removal, two-hand, visible stem-contact, and flower-identity reviews;
  after deterministic correction from the native 88-frame/30 FPS emission to
  the source 89-frame/24 FPS timeline, it scores 0.8418 motion, 0.9819 target
  identity, and 0.7589 temporal consistency. Seeds 1051 and 2060 are rejected
  because they persistently regenerate a human face and skin, despite the
  controller ranking them above seed 42. Strict status remains PARTIAL because
  the generic single-object bouquet tracker reports object consistency 0 and
  full-speed shadow review is pending. The method is not expanded to the full
  27.5-second film, and the alpha-only shadow skills are not applied to this
  non-alpha 89-frame generative candidate.
- Native MiniMax-H3 identity-topology RSI: the new standard-library contract,
  rights-attributed Ref2VA dataset compiler, physical-GPU training launcher,
  frozen DINOv2 cross-check, whole-video and decoded-frame-digest-bound
  124-frame semantic reviewer with separate unique-left-origin,
  unique-right-origin, and head/neck-clearance gates, component-wise action
  non-regression, and finite
  RSI routing are implemented and tested. Action evidence must now match the
  assessed video SHA and share source, control, robot-reference, and mask hashes;
  one low action component can no longer hide regression in another. On the frozen
  `inspect-flower` demo, the NF4 H3 baseline fails left-shoulder attachment and
  stable proportions in 100 frames and its head-to-torso chain in 82 frames.
  Both rank-16 epoch-0 adapters remove the latter two failure categories, which
  is a measurable native structural adaptation, but left-shoulder failure
  remains in all 100 frames and fully passing frames stay 24/124. The safer r2
  candidate still changes worst-frame DINOv2 identity from 0.88338 to 0.86587
  and matched motion adherence from 1.0 to 0.96850; r1 regresses further.
  Reducing r2 LoRA scale to 0.5 restores identity to 0.88006 but also restores
  all baseline topology failures and leaves motion at 0.95468. Every candidate
  is therefore REJECTED for release. The completed r2 epoch-1 checkpoint keeps
  the same partial structural change and improves task motion, EPL,
  robot-identity proxy, and temporal scores, but object lock falls from 1.0 to
  0.00052, DINOv2 worst-frame identity falls to 0.78726, and matched
  pixel-motion adherence falls to 0.89493. The strict action ratio is 0.00116.
  A second epoch-1 replication retains the baseline 24/124 topology result and
  is also rejected: worst-frame appearance is 0.80049, matched pixel motion is
  0.93416, and component-wise action adherence is 0.96011. The rank-32 r3
  close-up round completed epoch 0, then three matched LoRA-scale candidates
  (1.0/0.5/0.25) all remained at 24/124 topology frames. Their matched motion
  ratios are 0.91237/0.90867/0.96881; scale 0.25 also fails the action gate at
  0.98682. Epoch 1 was stopped after 35/120 steps and produced no checkpoint.
  The r4b follow-up replaces same-renderer close-ups with three
  source/scene/identity-disjoint real-background texture domains and three
  disjoint held-out domains. Its corrected 12-clip train set passes the new
  domain contract; an earlier version is retained as rejected because visual
  review found a residual human forearm in one background ROI. A strict
  deterministic rank-16, `1e-5`, 36-step run completed on physical A800 GPU 0.
  Step 12 slightly raises DINOv2 mean/worst reference identity from
  0.90758/0.88338 to 0.90937/0.88996, but still passes only 24/124 topology
  frames, reduces matched motion to 0.97863 and action adherence to 0.82458,
  and has flower object lock 0.000419. Step 36 at scale 0.5 reproduces the same
  100 shoulder-root/head-clearance/proportion failures. All three r4b
  checkpoints are retained by hash but rejected; the learned route is stopped.
  Rejected learned candidates are now fail-closed at delivery, and fallbacks
  must pass the same current-task 12-gate assessment with exact source/control/
  reference/mask/baseline hashes. The earlier routed v19 artifact under
  `outputs/h3-identity-rsi/20260811T085300Z-r3-reject-structure-fallback-v2`
  is USER-REJECTED and superseded: its 124/124 topology result hid severe
  current-task regressions (0.445897 motion and 0.278505 action adherence).
  The hardened route under
  `outputs/h3-identity-rsi/20260811T090310Z-task-bound-delivery-block-v2`
  is WORKING as a safety mechanism: it records `blocked`, `output: null`, and
  emits no video because both candidate and fallback fail. The learned capability
  remains PARTIAL; no publishable general model, 3-D anatomy guarantee, or robot
  policy is claimed.

- Non-regressing learned flower-repair routing:
  `outputs/flower-repair-policy/20260810T205055Z-e65420b0` supersedes the v1
  aggregate-utility policy that hid a large action regression. Motion, EPL
  minimum, temporal consistency, and identity now have 0.01 hard regression
  limits; subject replacement has 0.02. Leave-one-action-out v2 selects the
  constrained oracle in 8/9 groups with 0.000006 mean regret, and both its first
  choice and guarded final result pass non-regression in 9/9 groups. Maximum
  selected motion/EPL regressions are 0.004713/0.005250; mean candidate count
  remains two instead of five. A real `inspect-flower` replay trained only on
  insert/handover preserves motion at 0.56575 versus 0.57002 raw, whereas the
  superseded aggregate winner fell to 0.48194. A deliberate old-policy stress
  replay rejects two regressing candidates before admitting the bounded result.
  The four-column demo is
  `demo-nonregression-v2c/heldout-inspect-nonregression-guard.mp4`. Status remains
  PARTIAL: flower lock is only 0.00052, absolute identity/motion/EPL gates still
  fail, and the workflow correctly records `REGENERATE_WORLD_MODEL_CANDIDATE`
  rather than trading action fidelity for flower restoration. See
  `docs/FLOWER_REPAIR_POLICY_EXPERIMENT.md`.

- Flower-task adaptation and staged-generation gate:
  the rank-4 smoke at
  `outputs/flower-task-adaptation/20260811T026000Z-real-window-ablation-v1`
  and the rank-8 rerun at
  `outputs/flower-task-adaptation/20260811T171000Z-real-window-rank8-ablation-v2`
  compare against zero-shot VACE on the same 17-frame real contact window,
  seed, prompt, robot/flower control, and 13.29--14.01% edit mask. The bounded
  GPU-4 rerun completed 96 rank-8 optimization steps over four epochs and wrote
  four 10,971,312-byte checkpoints. Its real-window outside-edit similarity is
  0.92148 versus 0.92105 zero-shot, while control-motion alignment regresses
  from 0.35681 to 0.34569. Uniform storyboard review still finds retained
  human head/torso, fragmented robot geometry, no two coherent hands, and no
  auditable hand--stem contact in either candidate. The available SAM2 flower
  signal is a union rather than persistent stem instances, so exact flower
  identity is also unverified. Both evaluations record 0/4 semantic gates and
  `REJECT_FULL_EXPANSION`.

  A later explicit-instance route is scoped **WORKING** on the same real input.
  `outputs/flower-contact-supervision/20260811T038000Z-real-contact-pairs-v6`
  uses a persistent single-pink-carnation SAM2 track, two independent robot-hand
  tracks, and contact-pair composition. Dense review passes complete human
  removal, two mechanical hands, clear stem contact, and flower identity, so its
  evaluator records `ALLOW_RELIGHTING_WINDOW`. The accepted continuous expansion
  is source `[272,378)` under
  `outputs/flower-full-expansion/20260811T051000Z-phase04a-contact-v3`: all 106
  frames pass the same four gates, with maximum 3.0-pixel active contact,
  12.98-pixel support contact under a 14-pixel support gate, and exact flower
  preservation before encode.

  Geometry-gated relighting is also scoped **WORKING** for those 106 frames at
  `outputs/flower-relighting/20260811T054000Z-phase04a-confidence-routed-v2`.
  Direct Wan LoRA videos are rejected; only their bounded low-frequency
  luminance is routed into the robot-safe interior. Flowers, prompted hands,
  protected contact, and outside-robot pixels remain exact before encode, and
  maximum temporal relighting residual is 1.4736 RGB MAE at the reviewed
  324--325 proposal join.

  The cabbage-demo replacement route now has two additional strict **WORKING**
  real-input anchors. Frames 110--198 under
  `outputs/gpu-flower-instance-supervision/20260811T062348Z-window110-198-real-instances-v1`
  pass 56/56 unchanged gates over three persistent named flowers, two robot
  hands, four phase-specific contact pairs, dense human-removal review, motion
  0.84177, identity 0.98186, and temporal consistency 0.75886. The accepted
  89-frame candidate has SHA-256
  `2afd1238a976b3b87b10ebbcbccb83ac3d5e3ca630af184738bb8bb2df00f255`.
  Frames 182--270 under
  `outputs/gpu-flower-instance-supervision/20260811T080000Z-window182-270-face-suppressed-v1`
  use deterministic source-face-control suppression, source-tracked flower
  restoration, the trained flower-repair policy, and a 48-pixel exact contact
  protection band. Independent post-repair SAM2 tracking on physical A800 GPUs
  4 and 5 passes 39/39 gates with motion 0.84103, identity 0.97999, temporal
  consistency 0.77454, and unchanged 8/15-pixel p90/maximum contact thresholds.
  Earlier temporal variants are retained as rejected because they raised the
  temporal proxy while regressing lower-hand-to-lime contact. These scoped
  anchors authorize further windows; they do not by themselves promote the
  full film.

  The first seed-42 overlap expansion and its source-face-suppressed retakes
  remain **PARTIAL**. Exact black source-face control removes the sampled human
  face, skin, sleeve, and torso from starts 0, 72, 326, 470, 542, and 571, and
  all six immutable raw Wan outputs decode 88 frames at 896x512/30 FPS. The
  proxy postprocessing truncates starts 0, 72, and 542 to 70 frames, so only the
  raw outputs may be retimed downstream. More importantly, the retakes redraw
  the flowers and have not passed named-instance, two-hand, contact, or temporal
  gates; none is stitchable yet.

  An initial frames-254--342 candidate remains retained as **PARTIAL**: its
  first left-hand prompt is empty in 79/89 frames and drifts to a pink
  background ribbon, while a lower re-prompt selects an ambiguous palm-root or
  wrist region. A dual-hand-constrained retake under
  `outputs/gpu-flower-window254-supervision/20260811T103500Z-dual-hand-v3-round0`
  supersedes that candidate for this window. After exact 89-frame/24-FPS
  retiming and source-tracked flower restoration, a new finger-only SAM2 prompt
  tracks the image-left articulated gripper for 89/89 frames with 9.69-pixel
  maximum step and 0.9118 mean adjacent IoU; the independent image-right
  gripper tracks 89/89 with 11.62-pixel maximum step and 0.9337 mean adjacent
  IoU. The unchanged evaluator passes 39/39 gates: motion is 0.84557, identity
  0.98207, temporal consistency 0.79876, and the two contact segments have
  p90/maximum distances of 0.0/14.87 and 3.0/12.04 pixels. Frames 254--342 are
  therefore a third scoped **WORKING** anchor, not authorization for unverified
  windows. Frames 398--486 now form a fourth scoped **WORKING** anchor under
  `outputs/gpu-flower-window398-joined/20260811T114500Z-v1`. The formerly
  unstable source interval is split into overlapping 398--442 and 442--486
  subwindows with one shared global-442 identity anchor. After source-instance
  restoration, protected temporal repair, and independent final-pixel GPU
  retracking, both subwindows pass every unchanged gate. Their aligned
  identity/motion/temporal scores are 0.98123/0.82088/0.76105 and
  0.97555/0.86435/0.80199. A rejected whole-frame midpoint seam is retained
  because it lowered pink-flower velocity agreement to 0.76190. The accepted
  seam instead keeps the left window exact and fades only three right-window
  non-object frames behind dilated flower/hand protection; its local peak is
  1.2261 times the full-window transition P90 and both post-stitch strict gates
  remain WORKING. The continuous 89-frame candidate has SHA-256
  `b53956ff0bf7d62fee7691d1489a06dc5ee599290b7e36c154c3048ce87ad6e6`.

  Seed-44 expansion now adds two overlapping **WORKING** subwindows under
  `outputs/gpu-flower-full-strict-expansion/20260811T122000Z-v1/window-0470`:
  global 485--524 and 514--558. Prompt-only seed-42 regeneration was proven
  byte-identical because the pinned backend ignores the accepted prompt, while
  seeds 43/44 were real latent variants and seed 44 retained the best object
  and temporal proxies. The final lossless left window passes 26/26 unchanged
  gates with identity/motion/temporal 0.97658/0.85621/0.78799; its bouquet and
  support hand adjacent IoUs are 0.93864/0.94314 and contact is visible in
  40/40 frames. The final right window uses lossless-gated round 09, a 16-pixel
  flower-context lock, and source-mask-derived lime-bud correction prompts. It
  passes 26/26 gates with 0.97603/0.82368/0.92262, flower velocity agreement
  0.83721, hand IoUs 0.88399/0.94264, and contact visible in 45/45 frames.
  Compatibility encoding is no longer allowed to select a repair round after
  it falsely raised round-07 motion above 0.82 while the lossless artifact was
  only 0.81660.

  The continuous 398--558 join is still **PARTIAL** under
  `outputs/gpu-flower-398-558-strict-expansion/20260811T160500Z-v2-prefix484`.
  Its 484--485 fade and 517--518 cut reduce seam MAE to 1.85388 and 0.66437,
  but the post-stitch 485--524 window scores only 0.81521 motion and 0.73137
  temporal consistency, and bouquet-hand adjacent IoU is 0.86712 below the
  unchanged 0.88 threshold. Dense and single-endpoint relabeling attempts fall
  further to 0.84518 and 0.85579. The preview is retained for comparison but
  is not promoted; the two WORKING subwindows remain frozen independently.

  Strict replacement full-film status remains **PARTIAL**. The identity track is deliberately cut
  before the 377--378 discontinuity; `[378,398)` is an unobservable
  release/occlusion transition. Fresh `[398,448)` prompts fail semantic review:
  one supposed hand track selects flower material, the other a wrist/forearm
  patch, and the scissors probe selects hand/forearm pixels before becoming
  empty for 30/50 frames. Its decision is
  `BLOCK_RELIGHTING_REQUIRE_CORRECT_TOOL_AND_TWO_GRIPPER_SUPERVISION`; phases
  04B/04C and the remaining 27.5-second film were not relit. See
  `docs/FLOWER_TASK_ADAPTATION.md` for the evidence and next gate.

  A separate full-film **WORKING** 2-D delivery is now available at
  `outputs/flower-full-delivery/20260812T001000Z-shadow-epl-temporal-v1`.
  It inherits the reviewed 660-frame v71 shadow/background parent, applies the
  accepted LoRA luminance residual only to `[272,378)`, and uses the trained
  flower-repair policy to select an EPL-local, hard-background-locked,
  flower-restoring crossfade recipe over global noise smoothing. Source/EPL
  diagnostics identify only frames 97, 193, 294, 405, 507, and 562 as
  unsupported window jumps. All six transition energies fall, maximum ROI
  transition energy drops from 11.2966 to 7.5638, and the mean modified area is
  1.0317%. Outside-safety pixels, all 660 flower masks, and the accepted
  phase-04A contact masks are exact before encode. Dense and consecutive-frame
  review finds no new ghost, flower loss, returning human forearm, or visible
  repaired-window jump. The 832x480/24 FPS candidate decodes 660/660 frames and
  has SHA-256
  `29d07167d45df13ffa20cd6a22acd52610bb3e747fbbc1f67d1052fbbc704969`.
  The newly trained topology LoRA contributes no pixels because its held-out
  identity, topology, and motion gates failed. This full-length WORKING label is
  for 2-D shadow/background and temporal quality only; explicit full-film
  single-flower/tool contact remains PARTIAL.

- Real-scene OSCAR-2B action-conditioned bowl run:
  `outputs/acwm-open-models/20260810T143008Z-ae32011f` contains three matched
  81-frame, 640x480, 15 FPS outputs generated on physical A800 GPU 2 from one
  Hand2Dex-2 real first frame. All use OSCAR source
  `4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb`, OSCAR-2B revision
  `c9781ffa7dd8556d862d7d9f338a2ea008a58ca6`, 35 steps, guidance 6.0, and seed
  20260810; only the camera-pixel skeleton condition changes. `lift-up` passes
  with action/embodiment/object/temporal/background scores
  0.9937/0.9680/0.9753/0.8659/0.9233. The original `slide-right` automatic
  scores are 0.8401/0.9515/0.9506/1.0000/0.9203, but posthoc user review found
  severe late-frame hand/finger distortion, superseding its earlier visual
  acceptance. The raw OSCAR batch is therefore only 1/3 accepted.
  `slide-left` is rejected at 0.2848 action adherence and 0.6409 temporal
  consistency because the robot retracts without moving the bowl. A stronger
  leftward terminal-wrist repair in
  `outputs/acwm-open-models/20260810T144421Z-efdb8fea` also fails. The overall
  A specialized repair at
  `outputs/acwm-hand-structure/20260810T162000Z-oscar-slide-right-lock-v1`
  uses pinned SAM2 to extract one connected canonical hand from the shared
  first frame, then applies a fixed 0.75-scale rigid projection driven by the
  existing camera-pixel action. Its numeric gates pass, but later user review
  supersedes that acceptance because the whole hand visibly translates like a
  pasted rigid layer. The repair is now retained only as rejected evidence.
  A new native OSCAR run at
  `outputs/acwm-open-models/20260810T155518Z-06311bc4` uses the successful lift's
  vertical trajectory as a template, adds a rightward arc, and regenerates the
  complete articulated arm and hand. Automatic scores are
  0.9492/0.8557/0.9259/1.0000/0.9131; a dense 16-frame and full-video review
  confirms one attached arm/hand, continuous joint motion, stable bowl contact,
  clear upward-right object displacement, and a fixed workcell background. The
  agentic demo remains PARTIAL (2/3 selected action types), and establishes
  neither arbitrary action reliability nor robot-base control, contact physics,
  or real-robot execution. BWM and Kinema4D remain gated because their native
  14D robot actions or calibrated pointmap/URDF inputs are absent.
- The formerly accepted v11 pose-rig result at
  `outputs/robot-person-replacement/pose-rig-runs/20260810T133000Z-alpha-texture-robust-pose-v11-accepted`
  is superseded and PARTIAL after user review exposed hand-size breathing that
  its transition-only acceptance missed. Exact wrist-index endpoint mapping
  bypassed the previous scale clamp: the right hand used a 7.32-pixel anchor
  direction and ranged from 0.027x to 4.922x scale, while its maximum frame
  scale step reached 0.827x. The robust pose filtering and robot-only texture
  remain valid components, but v11 is not accepted for stable robot morphology.

- The formerly accepted v12 pose-rig result at
  `outputs/robot-person-replacement/pose-rig-runs/20260810T135112Z-morphology-locked-pose-v12-accepted`
  is superseded and PARTIAL after user review exposed split hands and floating
  arm fragments. v12 fixed hand-size breathing, but its crossed-arm texture was
  sampled with six overlapping capsule masks, so some finger, opposite-arm,
  and torso alpha pixels were transformed in more than one rigid layer.
  Transition and hand-scale gates did not measure this topology error. Its
  robust pose repair and morphology lock remain reusable; v14 replaces the
  texture and adds exclusive ownership plus per-frame connected-chain gates.

- The formerly accepted v14 pose-rig result at
  `outputs/robot-person-replacement/pose-rig-runs/20260810T142228Z-disjoint-chain-v14-accepted`
  is superseded and PARTIAL after user review found that the manipulated flowers
  were completely absent and the fixed 0.62x hands were perceptually too small.
  Its disjoint-mask and connected-chain topology remains reusable in v19, but
  v14 is not an acceptable flower-arranging visualization. The intervening
  v15-v18 experiments are retained as PARTIAL evidence: v15's tracked/GrabCut/
  SLIC masks missed most of the bouquet; v16 used wrong forearm/flower depth
  order; v17 lost hand contact when using the dual-wrist midpoint; and v18 kept
  contact but failed the 20-pixel continuity gate with a 34.17-pixel bouquet
  jump. v19 supersedes all four without erasing those failures.

- User-rejected real-scene AC-WM bowl demo:
  `outputs/acwm-bowl-h3/20260810T123300Z-hand2dex2-v1` contains three matched
  MiniMax-H3 NF4 outputs for slide-left, slide-right, and lift-up in one captured
  Hand2Dex-2 scene. Its automated image-plane proxies still measure separated
  terminal centers, 5.3-7.0 pixel terminal errors, 124/124 bowl detections, and
  a 247.0 pixel endpoint-separation floor. These numbers do not establish robot
  appearance, articulated motion, contact, or physical quality. User visual
  review found all three generated robots poor and their action differences
  insufficiently convincing. The run is therefore PARTIAL and retained only as
  a negative MiniMax-H3 baseline for comparison with a native action-conditioned
  world model.

- MiniMax-H3 explicit-trajectory real-scene action-control v3:
  `outputs/minimax-h3-action-control/20260810T055500Z-control-video-v3`
  compiles the same three language instructions into named, per-frame left/right
  shoulder-elbow-wrist trajectories in `camera:source_anchor_pixels`; their
  minimum pairwise two-wrist RMS separation is 81.1222 pixels. Each A800 GPU 7
  H3 run receives the reviewed high-quality robot identity image, one real-scene
  image, and its action-specific 124-frame control video with the same NF4 H3
  weights, 20 steps, seed 20260810, and pinned DiffSynth `b1c02ce` source. All
  three outputs decode at 832x480/24 FPS. Dense visual review shows a coherent
  robot and real flower-workshop scene, with visibly different insertion,
  chest-level handover, and raised-wrist inspection poses. After five repair
  rounds, pairwise full-frame MAD has a 3.6695 floor and active-pixel MADs are
  42.98-49.68; background and object locks are approximately 1.0, temporal
  consistency is 0.9193-0.9466, and identity improves to 0.6880-0.7125. Strict
  acceptance still rejects all three: identity remains below 0.72 and explicit
  control-motion adherence is 0.4819-0.5804 with EPL minima 0.4495-0.5757.
  The demo therefore presents v3 as PARTIAL: macro actions are visually
  separated, but exact wrist paths, grasp transfer, 45-degree rotation, contact
  physics, and robot execution are not accepted.

- User-rejected ten-second EPIC-KITCHENS Ego bottle action-control comparison:
  `outputs/minimax-h3-long-ego-action-demo/20260811T070900Z-epic-blue-bottle-v1`
  uses the licensed public `P03_28` interval at 25.83--35.83 seconds rather than
  the legacy flower scene. The verified official source MD5 is
  `801bf911ae9eb4293abec88480f17c4c`; the prepared source is exactly 240 frames
  at 24 FPS and records the EPIC-KITCHENS-100 annotation/downloader revisions,
  CC BY-NC 4.0 license, commands, hashes, and endpoint frames. Pour, shake, and
  handover compile to explicit `camera:epic_kitchens_p03_28_pixels` robot-hand,
  bottle-pose, and holder-state traces. Six physical-GPU-2 H3 NF4 generations
  completed with seed 20260811, 20 steps, 832x480, two 124-frame windows, an
  eight-frame overlap, and per-action frame-116 continuation references. The
  packaged comparison and three variants decode to exactly 240 frames / 10.0
  seconds, and their minimum pairwise full-frame MAD is 14.6579. All six strict
  window evaluations remain rejected: conservative source-skin/EPL minima are
  0.0675--0.3353, and following-window pour bottle lock is 0.0675; background,
  robot-material, motion, and temporal gates pass. Best-seam subject MAD is
  21.2237 for pour, 22.3311 for shake, and 17.5369 for handover, with a visible
  pose reset near five seconds. A domain-matched 30-row held-action ridge repair
  router and alpha sweep were trained but rejected: exact held-action recipe
  selection stayed 1/6 and no candidate in any window satisfied both the 0.95
  background gate and the capability non-regression contract. The rejected
  checkpoint is not routed into the demo; a deterministic tight-support repair
  reduces mean support from about 57% to about 25%. Subsequent user review found
  visible human-hand/sleeve ghosts and unacceptable blur in all three videos.
  This route is PARTIAL and rejected; it is not the current visual demo.

- Current robot-factored EPIC Ego visual recovery:
  `outputs/acwm-ego-quality-recovery/20260811T083013Z-cabbage-route-v1` uses the
  same verified 240-frame `P03_28` interval and three H3 action drivers, but does
  one joint Wan2.2-Animate replacement with five generated history frames. A
  driver subject mask plus fail-closed lower-frame guard prevents source-person
  pixels from returning; source face control, source alpha repair, object
  overwrite, temporal filtering, and post blur are disabled. The three physical
  A800 runs on GPUs 0/2/3 record seed 42, pinned revisions, commands, masks,
  logs, and outputs. All variants decode at 880x512, 240 frames, 24 FPS. Dense
  review covers 60 frames per action plus full-resolution boundary/key frames
  and finds no visible human hand, skin, sleeve, or translucent residual. New to
  old foreground p10 sharpness ratios are 1.6177 pour, 1.2345 shake, and 1.4986
  handover; safe-background/source ratios are 0.9287, 0.8781, and 0.9007. Mean
  pairwise full-frame MAD is 22.4176--26.6564, with every frame above 2 MAD.
  Status is WORKING for visual human-to-robot recovery and still PARTIAL for
  embodied action control: the outputs are image-space generated videos without
  calibrated robot state, contact, force, collision, or real execution. The
  public demo is `demo/showcase/acwm-ego-robot-factored-actions-10s.mp4`; its
  adjacent old/new comparison and manifest retain the rejected baseline.

- Same-scene instruction-conditioned EPIC Ego task comparison:
  `outputs/acwm-ego-multitask/20260811T101400Z-epic-p03_28-v4` packages the
  accepted handover result with new unscrew-cap and faucet/rinsing actions. The
  two new candidates use task-state-valid H3 NF4 recursive continuation tails;
  reset prefixes are excluded and nearest decoded frames provide bounded
  retiming. No cross-dissolve, generated frame interpolation, blur, temporal
  filtering, alpha repair, or source-person restoration is used.
  All three outputs decode to 240 frames at 24 FPS. Dense human, blur, and task
  review passes. Foreground p10 ratios against accepted visual baselines are
  1.0000 handover, 2.1165 unscrew, and 1.3818 rinse; background/source p10
  ratios are 0.8971, 1.1231, and 1.0457. Pairwise mean full-frame MAD is
  36.6373--50.9219 and every frame exceeds 2 MAD. The fixed evaluation records
  `WORKING` for the bounded camera-frame visual comparison. Physical contact,
  force, calibrated robot state, and real execution remain unverified. The
  primary public artifact is
  `demo/showcase/acwm-ego-instruction-task-comparison-generated-only-10s.mp4`;
  the adjacent 2x2 artifact retains the real source as explicit provenance.

- Real first-person to third-person paired-data demo:
  `outputs/droid-ego-third-person/20260811T135000Z` packages synchronized DROID
  episodes 21, 60, and 77 as one wrist-camera first-person input plus two
  exterior-camera third-person targets. The three full AV1 source streams each
  decode to 320x180, 32,212 frames, 15 FPS, and 2,147.466667 seconds. Selected
  episode lengths are 92, 144, and 100 frames. Within each episode, all three
  cameras have the same dataset-frame interval and exact start/end timestamp
  range, so maximum cross-camera timestamp-range delta is 0 and all 3/3
  trajectory-lineage gates pass. Posters and complete videos were visually
  reviewed for the same Franka, object, scene, and action across cameras. Status
  is WORKING for synchronized paired targets and public demo packaging. The
  exterior streams are measured ground truth, not model output. Learned
  generation is evaluated separately below.

- First-person/third-person provenance labels:
  `outputs/droid-ego-third-person/20260811T140500Z-labels-v2` adds permanent
  in-frame labels to all three task videos and the ten-second reel. The wrist
  panel is `REAL CONDITION / FIRST-PERSON`; both exterior panels are
  `REAL TARGET A/B / THIRD-PERSON`; the footer states
  `OUR GENERATED VIDEO: NOT AVAILABLE / NOT STARTED`. The comparison layout
  and labels are PhiAgent-authored, while every task-view pixel remains a
  measured DROID camera frame. Status is WORKING for provenance-safe demo
  labeling. This historical artifact predates the separate learned candidate.

- DROID first-person to third-person model adaptation:
  `outputs/droid-view-lora/20260811T152900Z-dataset-v10-full91-target-a-crf24`
  freezes 91 episode-disjoint target-A training pairs and heldout episodes
  21/60/77. A rank-16 PhiAgent DROID View LoRA on the pinned
  Wan2.1-VACE-1.3B base completed 182 steps on physical A800 GPU 6. Each
  heldout generation uses a real wrist video, one real target-view anchor, and
  task text; the synchronized real target is withheld until evaluation. The
  selected checkpoint aggregates to 0.520788 full-frame SSIM, 0.376567
  subject-ROI SSIM, 0.619882 subject edge F1, -0.262506 motion correlation,
  and -0.342496 static-anchor gain. Status is PARTIAL / NOT ACCEPTED because
  every episode fails at least one identity/motion gate. A 273-example
  start/middle/end route improves motion correlation to 0.374912 but regresses
  full-frame/subject SSIM to 0.390229/0.270019, so it is rejected. The public
  four-column reel permanently labels both real conditions, the model output,
  and the post-generation real heldout target.

- DROID SOTA migration after the rejected Predict2 adaptation: a fresh
  low-learning-rate Predict2 LoRA and inference residual scaling improved
  validation motion while keeping one episode near the base appearance, but
  the complete two-episode validation still failed absolute subject-identity,
  static-anchor, and motion gates. No final holdout or demo was promoted. The
  replacement route now targets the official Cosmos3-Nano image-to-video
  generator with the native DROID 2x2 layout. Its new controller selects and
  records physical GPUs, pins the exact framework/checkpoint revisions, passes
  only the disclosed real composite first frame and task text, writes a unique
  experiment directory, and labels every continuation as `OUR GENERATED
  VIDEO`. A second controller prepares official Cosmos3 JSONL and forces the
  post-training distribution to 100% first-frame I2V at 480p/93 frames with
  zero condition dropout. An acceptance-gated evaluator maps every named view
  through an explicit 2x2-tile pixel-frame transform, and the demo builder
  refuses any validation result below `WORKING`. A checkpoint verifier binds
  the pinned revision, indexed byte counts, and safetensors headers. Forty-nine
  current Cosmos3-focused CPU tests pass. On the A800 host, the pinned official
  framework environment imports
  its training, inference, export, FlashAttention, NATTEN, and Transformer
  Engine modules, and a standalone CUDA 12.8 BF16 matrix preflight passes on
  physical GPU 5. The broader framework import process required forced teardown
  after writing its successful import/CUDA evidence, so that check remains
  PARTIAL. The pinned Cosmos3-Nano weights, text tokenizer, Wan VAE, and sound
  tokenizer have all passed exact revision, indexed tensor-data byte-count, and
  full-SHA-256 verification. The official local-first HF-to-DCP conversion at
  `outputs/cosmos3-dcp-conversion/20260812T134515Z-fb17ea39` succeeded and wrote
  30,348,835,273 bytes. A two-A800, two-step, 33-frame full-SFT smoke at
  `outputs/cosmos3-droid-sft-training/20260812T135857Z-69912495` loaded all 804
  base tensors, optimized 6,965,486,784 parameters, produced finite gradients
  and losses, and saved two checkpoints. This proves real model loading and
  backpropagation but does not validate the 93-frame objective. Formal full SFT
  is NOT STARTED because the recipe
  requires exactly eight sufficiently free GPUs and only two met the 60,000 MiB
  free-memory gate. The official-configuration-derived `formal_lora` route keeps
  the 500-step/93-frame objective, uses rank-16/alpha-32 generation-MoE adapters,
  full activation checkpointing, context parallelism 2, and adapter-only
  optimization. Its matched two-step 93-frame probe at
  `outputs/cosmos3-droid-sft-training/20260812T140659Z-da2b7841` optimized only
  15,335,424 LoRA parameters, produced finite gradients and losses, saved a
  checkpoint, and peaked at about 52.4/45.5 GB on physical A800 GPUs 3/5. The
  formal 500-step run at
  `outputs/cosmos3-droid-sft-training/20260812T141044Z-d99ad85f` completed all
  500 synchronized optimization steps on those GPUs without OOM and saved five
  29 GB DCP checkpoints. Training ran from 22:11:37 to 03:23:20 Asia/Shanghai;
  the first 50 logged losses average 0.23259 and the final recorded gradient norm
  is finite at 0.02075. The integrated post-training export first failed because
  training CP=2 was invalid for the single-process exporter. An immutable
  recovery at
  `outputs/cosmos3-droid-sft-export-recovery/20260813T021956Z-921364a6`
  uses an export-only CP=1 config copy and bundles seven hash-verified local
  tokenizer/processor files; it succeeds with 30,354,906,417 bytes across seven
  safetensors shards. Status is WORKING for formal LoRA training and export,
  but PARTIAL / NOT ACCEPTED for the requested viewpoint-generalization quality
  described below. The final holdout remains untouched.

- True wrist-only first-person to third-person Cosmos3 lane: this is separate
  from the anchor-conditioned 2x2 upper-bound lane. Each derived SFT sequence
  contains one resized real wrist-camera frame at frame 1 and only synchronized
  real `exterior_1` or `exterior_2` frames thereafter; the condition contains
  zero third-person pixels. The completed split contains 344 training sequences
  and four episode-disjoint validation sequences, with final holdout still
  excluded. The all-record pixel-lineage audit at
  `outputs/cosmos3-droid-wrist-audit/20260812T133300Z-all348-pixel-lineage`
  accepts 348/348 records: minimum condition-to-wrist SSIM is 0.984447, minimum
  future-to-named-exterior SSIM is 0.985167, and minimum wrist-over-exterior
  margin is 0.342924. Training is hard-bound to `condition-mode=wrist_only`,
  rechecks physical GPU state at launch, and binds the accepted audit plus exact
  dataset-contract hash. A separate
  strict evaluator requires condition-frame identity, an actual wrist-to-target
  view-switch margin, full-frame/subject/edge similarity, motion correlation,
  motion magnitude, and static-anchor gain for every validation sample and both
  target views. Its demo builder refuses all non-`WORKING` evaluations and labels
  `REAL CONDITION / FIRST-PERSON`, `OUR GENERATED VIDEO / THIRD-PERSON`, and
  `WITHHELD REAL TARGET / EVALUATION ONLY` permanently in-frame. Four of four
  episode-disjoint validation generations from the exported iteration-500 model
  complete at 480p/93 frames/16 FPS/35 UniPC steps, with zero real third-person
  future frames passed to the model. Strict evaluation at
  `outputs/cosmos3-droid-wrist-validation-evaluation/20260813T022900Z-iter500-seed20260812`
  returns PARTIAL / NOT ACCEPTED: aggregate mean full-frame SSIM is 0.336849,
  subject-ROI SSIM 0.271958, subject edge F1 0.134844, motion correlation
  0.169356, motion magnitude ratio 1.517745, and static-anchor SSIM gain
  -0.512556. Only the motion-magnitude-range gate passes; the aggregate
  condition-frame SSIM is 0.850512, first-future target SSIM 0.331660, and
  wrist-to-target view-switch margin -0.050908, so subject identity and the
  requested camera switch are not accepted. The public demo builder correctly
  refuses this result. A separate red-bannered diagnostic comparison at
  `outputs/cosmos3-droid-wrist-diagnostic-comparison/20260813T023300Z-iter500-not-accepted`
  shows the real wrist condition, OUR GENERATED VIDEO, and withheld real target
  without promotion. Fifty-six Cosmos3-focused CPU tests and Ruff pass.

- Superseded identity-locked viewpoint stress demo:
  `outputs/acwm-viewpoint-generalization/20260811T133735.526957Z` derives four
  synchronized camera-frame views from each reviewed handover, unscrew, and
  rinse video: the source view, fixed -12/+12 degree projective perturbations,
  and a continuous +/-8 degree sweep. All four published videos decode to
  1280x870, 240 frames, 24 FPS, and 10.0 seconds. Every output view uses the
  same decoded source frame, so the robot shell, five-finger topology, joints,
  bottle, action state, and background have deterministic pixel lineage rather
  than being regenerated independently. Across 24 sampled frames per task and
  each derived view, the worst central-frame round-trip PSNR is 33.5567 dB,
  maximum MAE is 2.1828, and minimum bounded similarity is 0.99144; all three
  task-level identity-lineage gates pass. The demo page exposes a three-task
  reel, three complete per-task clips, the manifest, and an ImageGen identity
  anchor. Status remains PARTIAL and this run is superseded because it is an explicit 2-D projective
  camera stress test, not learned novel-view synthesis, calibrated 3-D
  reconstruction, physical multi-camera capture, contact physics, or physical
  robot execution.

- Historical superseded flower-scene ten-second action-control attempt (not the
  current main AC-WM demo):
  `outputs/minimax-h3-long-action-demo/20260811T043100Z-10s-v1` compiles
  insert-flower, handover-flower, and inspect-flower into contiguous typed
  phases with explicit camera-frame trajectories and object-holder transitions.
  Each action uses two legal 124-frame H3 NF4 windows at 24 FPS with an
  eight-frame overlap; window 2 receives that action's generated frame 116 as a
  third reference, never a different action's state. All six A800 GPU-3 outputs
  complete, and the packaged comparison plus three individual clips each decode
  to exactly 240 frames / 10.0 seconds. The action-distinctness proxy passes with
  a 4.901 minimum pairwise full-frame MAD (individual pairs 4.901--9.601), but
  best-seam subject MAD remains 5.782 for handover, 9.282 for inspect, and 15.007
  for insert. All six strict window evaluations are rejected: conservative
  action/EPL minima range from 0.1667 to 0.5524, robot identity remains below the
  0.72 gate, and flower lock is approximately 0.00048--0.00060. The replicated
  non-regression repair router preserves capability and reduces safe candidate
  evaluation when possible, but correctly requests world-model regeneration
  rather than hiding these failures. Status is PARTIAL; macro actions are
  visibly different, while the window seam, exact path, grasp transfer, contact
  physics, and real-robot execution are not accepted. Human review remains
  pending. The portable evidence is
  `demo/showcase/minimax-h3-long-action-comparison-10s-partial.mp4` with its
  adjacent manifest.

- MiniMax-H3 real-scene language action-control comparison:
  `outputs/minimax-h3-action-control/20260810T051500Z-anchor-action-v2`
  generated three matched 124-frame insert, handover, and inspect variants on
  physical A800 GPU 7 with the pinned DiffSynth `b1c02ce` source, third-party
  NF4 H3 weights, 20 steps, seed 20260810, one robot identity image, and one
  scene anchor extracted from the existing real source video. Removing the
  full source video from H3's temporal reference raised the raw pairwise
  full-frame MADs from 2.73-2.82 in the rejected video-reference attempt to
  8.43, 21.40, and 22.45. After five bounded agent repair rounds per action,
  the comparison remains frame-aligned at 832x480/24 FPS and has a pairwise
  distinctness floor of 3.0356; all best candidates have background/object
  locks approximately 1.0 and subject replacement approximately 0.8997.
  Robot-identity proxies remain 0.5975-0.6123 below the 0.72 gate, every action
  candidate is rejected by strict acceptance, and complete semantic review of
  the requested insert/hand-transfer/wrist-rotation events is still pending.
  Source-motion metrics are diagnostic only because these prompts intentionally
  override the source person's action. User review rejected this v2 result as a
  featured demo because robot quality was poor and action differences were not
  visually clear enough; v3 above supersedes it for display. It remains negative
  evidence, not accepted precision action control, contact physics, or real-robot
  execution.

- Full-length MiniMax-H3 + EPL flower replacement:
  `outputs/minimax-h3-long-flower/20260810T044200Z-h3-epl-continuation-v3-recovery`
  covers all 660 source frames at 832x480/24 FPS with seven overlapping H3
  NF4 Ref2VA windows.  Each new window receives the preceding window's exact
  overlap pose as a second image reference.  Against independently generated
  windows, all six reviewed boundaries improve: same-time subject MAD falls by
  8.5%-58.7%, and best-seam MAD falls by 12.8%-56.9%.  Agentic stitching rejects
  the visually discontinuous short-result pixel anchor, keeps its EPL/physics
  phase evidence only, hard-locks source background and flower pixels after
  every lossy intermediate decode, and selects the fourth low-motion optical-flow
  round.  The final maximum seam-transition ratio is 3.963 below the 4.0 gate;
  background/object locks are 1.0, source-face and subject replacement pass,
  temporal consistency is 0.8550, all 660 frames decode, and dense plus
  consecutive seam review passes with no visible source-human return.  The
  final manifest is nevertheless correctly REJECTED/PARTIAL: robot-reference
  structure 0.6176, motion preservation 0.6020, and the EPL minimum 0.5977 miss
  their 0.72/0.72/0.62 gates.  The separate successful insertion evidence is a
  five-second pre-grasped authored MuJoCo trajectory; it is not proof that this
  generated video is physically executable. A later visual-cleanup attempt at
  `outputs/minimax-h3-shadow-removal/20260810T143000Z-safety-union-no-hand-v25-accepted`
  passed its pixel-level halo proxies but failed subsequent user review because
  the fixed safety union erased robot hands and manipulated flowers. That
  attempt is therefore PARTIAL and is not an accepted demo.
  The replacement workflow at
  `outputs/minimax-h3-shadow-removal/20260811T020000Z-layered-v42-pending`
  is the latest PARTIAL candidate. It separates H3 body, cross-confirmed left
  and right mechanical limbs, source and generated flower instances, and
  pose-derived human-skin negatives; removes small flower-mask components;
  temporally corrects isolated SAM2 seed-frame expansions; and uses local
  Poisson background fusion instead of a person-shaped alpha seam. The encoded
  result decodes 660/660 frames at 832x480/24 FPS. Automatic body, limb, source
  flower, generated flower, strict-flower, human-skin, halo, and decode gates
  pass; human-skin retention is 0.0015443 and minimum adjacent robot-mask IoU
  is 0.7480. Subsequent full-video user review rejected v42 because its flicker
  is severe. The matched temporal audit at
  `outputs/minimax-h3-shadow-removal/20260811T033000Z-flicker-audit-v43`
  shows mean temporal jerk rising from 0.6975 in the H3 input to 1.0891 after
  v41 alpha composition and 1.6797 after v42 per-frame seamless composition.
  Its exact-copy gates and mask IoU therefore missed the visible RGB boundary
  instability. v42 remains PARTIAL negative evidence and is not accepted.
- Conservative return to the user-preferred H3+EPL track:
  `outputs/minimax-h3-shadow-removal/20260811T044500Z-conservative-arm-shadow-v46-accepted`
  starts from the original stable H3 video rather than any v41/v42 composite.
  It preserves the raw frame everywhere except a low-saturation neutral band
  inside the fixed person safety region, near independently tracked robot arms,
  and outside a temporally unioned robot/hand/flower protection matte. That band
  is mixed only partially with one fixed clean plate; no source frames, Poisson
  fusion, generative repair, or semantic-layer replacement are used. The edit
  changes 1.57396% of pixels on average and reduces the edited band's mean
  clean-plate distance by 27.785%. All protected and outside-safety pixels are
  exact before encoding on 660/660 frames. Full-frame mean jerk changes only
  from 0.69753 to 0.71154, and safety-ROI jerk from 3.18538 to 3.20164; both
  pass the bounded-regression gates and remain far below v41/v42. Dense and
  enlarged risk-frame review passes with both hands, held flowers, robot edges,
  and background continuity intact. The published 27.5-second comparison is
  `demo/showcase/real-flower-arranging-h3-conservative-shadow-vertical.mp4`
  with SHA-256
  `dc38b38d2fa3d6a890b83ba07582b21b93eda1adee0fc4f235cda5716995b8d6`.
  This is WORKING only as conservative cleanup of this generated 2D video.
- Wider/lighter H3 shadow and compact residual-hand cleanup:
  `outputs/minimax-h3-shadow-removal/20260811T151500Z-wide-light-compact-hand-v59-accepted`
  was initially promoted over v46 for this specific request.
  The neutral shadow domain grows from the earlier 44-pixel conservative
  neighborhood to 68 pixels while its maximum blend strength falls from 0.55
  to 0.34. Human-skin negatives use an independent long-reach domain, but are
  retained only when a 300--15,000-pixel connected component is centred in the
  declared camera-frame arm-contact ROI and has at least 0.5% overlap with the
  tracked arm neighborhood. Per-component hulls and a five-frame high-alpha
  temporal maximum remove the intermittent palm/finger outline; low-chroma
  robot metal and green/yellow flower cores retain precedence. All nine
  acceptance gates pass on 660/660 frames. Mean modified area is 4.0296% with
  a 5.9380% maximum, and edited-band clean-plate distance falls by 25.469%.
  Full-frame mean jerk is 0.71943 versus 0.69753 in raw H3, and union-ROI mean
  jerk is 1.73114 versus 1.69504; both pass their bounded-regression gates.
  The initial sparse review sampled early 24/48/60/72/96/120 closeups and only
  12 later frames through frame 659. Subsequent denser full-timeline user review
  rejects that conclusion: recognizable human hands/forearms remain around
  frames 135--240 and 405--474, with an especially clear horizontal forearm at
  frame 462. The global skin connected-component pass had merged that forearm
  with flower/table-coloured pixels and discarded it under the maximum-area
  gate. Pixels outside the union safety domain and the final protection matte
  remain exact before encoding, but those automatic gates did not measure this
  semantic failure. The deterministic v58/v59 robot
  outputs share SHA-256
  `c0627c86cfe5a8d923bcd0d178d53eaa8292c12129a510b2800368fdb7669483`.
  The published 27.5-second comparison is
  `demo/showcase/real-flower-arranging-h3-wide-light-shadow-hand-clean-vertical.mp4`
  with SHA-256
  `73aaa7b2544215025fc2582de3739ba4e5e2b71b7fb562ee3823a4e914c65179`.
  The published file is retained only as reproducible negative evidence and is
  now PARTIAL; it must not be treated as the current accepted display candidate
  or as evidence of physical robot execution.
- Broad lower-person coverage with strict robot/flower/vegetation protection:
  `outputs/minimax-h3-shadow-removal/20260811T044300Z-wide-full-person-covered-v78-accepted`
  is the current display candidate for this request. It retains v74's reviewed
  full-strength residual hand/forearm track and weak 102-pixel neutral cleanup,
  then adds an independent source-person semantic layer only inside camera-frame
  ROI `(350,165,780,475)`. That layer uses 48-pixel spatial dilation, a +/-2-frame
  temporal union, 18-pixel feather, and 0.88 light-graphite strength, so the
  waist, garment hem, and complete arm workspace remain inside a deliberately
  large soft shadow instead of intermittently exposing source-human pixels.
  Robot body/limb/wrist and flower masks have 10-pixel protection, and generated
  green vegetation has an additional 3-pixel hard-protection matte. The broad
  material keeps 84% of local luminance and removes color rather than creating
  a deep silhouette. Mean wide-person coverage is 8.4043% of the frame; total
  mean modified area grows from v74's 5.4105% to 11.5506%, with a 14.7759%
  maximum. Wide-region mean chroma falls from 13.2482 to 7.8594 (40.7%), while
  the independent clean-plate band still reduces its distance by 16.9760%.
  All 11 gates pass on 660/660 frames, including exact robot/flower/outside-
  safety preservation, exact green-protection inheritance, bounded full-frame
  and ROI jerk, calibrated chroma suppression, and human review. Review covers
  the full timeline at 1 FPS, every 12th lower-workarea frame, every frame in
  110--136, 235--255, 384--405, and 475--492, plus 24 high-motion frames.
  The v77 prepromotion and v78 accepted robot videos are byte-identical with
  SHA-256 `b9201ae32206f48f5aa0a83cd3b8d3b13a8f7e06cb7a4ea190488a494da376ff`.
  The published 27.5-second comparison is
  `demo/showcase/real-flower-arranging-h3-wide-full-person-covered-vertical.mp4`
  with SHA-256
  `fb700bfa15cb9662d7bd436392236b510e1bc6cffffcc0e0df77689e2332de5d`.
  It is H.264 at 672x768, 24 FPS, and 660 decoded frames. This remains reviewed
  2-D generated/composited video evidence, not physical robot execution.
- Softer, wider H3 shadow with strict full-timeline hand containment
  (superseded by v78, retained as reproducible history):
  `outputs/minimax-h3-shadow-removal/20260811T040500Z-softer-wider-strict-containment-v74-accepted`
  was the prior display candidate for this request. It keeps the accepted v71
  residual-hand material track, 1.0 replacement strength, and 4-pixel feather
  exactly unchanged on all 660 frames, while widening only the independent
  weak neutral cleanup neighborhood from 90 to 102 pixels and reducing its
  maximum clean-plate blend strength from 0.21 to 0.16. A direct per-frame
  comparison reports zero residual-arm coverage regressions and zero coverage
  changes versus v71. Every-frame review again covers 110--136, 235--255,
  384--405, and 475--492; dense three-frame review covers both complete
  interaction phases, and the full timeline is reviewed at 1 FPS. No exposed
  human hand/forearm, erased flower, or missing robot finger is visible in
  those checks. All nine acceptance gates pass on 660/660 frames. Mean modified
  area is 5.4105%, edited-band clean-plate distance falls by 18.139%, full-frame
  mean jerk is 0.72162, and union-ROI mean jerk is 1.62702. The independently
  reproduced v73/v74 robot outputs share SHA-256
  `f77991cca1a586aa08b1dbf760a77d091886a804f9b737e89e779ca1e5697295`.
  The published 27.5-second comparison is
  `demo/showcase/real-flower-arranging-h3-soft-wide-hand-contained-vertical.mp4`
  with SHA-256
  `7a187d9ed2653ca2f8809e294ea4c5227d5945597f0a547bb768f1f311df62ea`.
  This is a reviewed 2-D video cleanup, not evidence of physical robot
  execution. Continuous playback control through the computer-use skill timed
  out after opening QuickTime's file chooser, so continuity acceptance is based
  on full decoding, dense visual sheets, and bounded temporal metrics.
- Reusable H3 shadow evolution skills: two auto-discovered personal skills now
  separate fast-motion shadow attachment from residual-shadow intensity. The
  `$repair-shadow-motion-lag` skill at
  `/Users/jiangyuhua/.codex/skills/repair-shadow-motion-lag` median-filters
  camera-frame arm centroids, uses a continuous 6--18 pixel confidence ramp,
  and warps only alpha below 0.70. Its first hard-switch full-film run is
  retained as PARTIAL because ROI jerk regressed by 10.87%. The evolved run at
  `outputs/minimax-h3-shadow-removal/20260811T130000Z-skill-motion-lag-v75-continuous-pending`
  aligns 149/660 high-motion frames; all independent automatic and human gates
  pass, protected/outside-safety pixels remain exact preencode, and ROI jerk is
  1.68587 versus v71's 1.68049 (ratio 1.00320). The video SHA-256 is
  `a07d74b91045a012606700d1a892002d3a83db9a5a7f15a8f09a2daf796edcf6`.

  The `$lighten-robot-shadows` skill at
  `/Users/jiangyuhua/.codex/skills/lighten-robot-shadows` then changes only the
  inherited low-alpha plate blend (`gain=1.20`, `cap=0.32`) and keeps motion,
  spatial masks, residual-arm material, and alpha at or above 0.70 unchanged.
  Its accepted combined candidate at
  `outputs/minimax-h3-shadow-removal/20260811T133000Z-skill-shadow-lighten-v1-gain120-pending`
  lowers absolute post-cleanup plate MAE by 4.39%, increases bounded plate alpha
  by 6.63%, increases mean edit area by 5.65%, and changes ROI jerk by only
  0.14%; all eight independent gates pass after high-motion and full 1-FPS
  review. The video SHA-256 is
  `13eefbe413becf09e1dea8335e090b37c8308e3c08bb787cd944d1435b7b1e5b`.
  These skills and runs are WORKING for the pinned v71-derived 2-D chain. They
  do not automatically supersede the independently accepted v74 display
  candidate without a direct user-selected comparison.
- Wider, lighter H3 shadow with full residual-human contour replacement
  (superseded by v74, retained as reproducible history):
  `outputs/minimax-h3-shadow-removal/20260811T234000Z-wider-lighter-contour-arm-v71-accepted`
  supersedes v59 as the current display candidate for this request. The neutral
  cleanup neighborhood is widened to 90 pixels while maximum clean-plate blend
  strength is reduced to 0.21. Reviewed middle and late residual-human forearms
  are no longer replaced by a fixed plate: camera-frame polygon tracks convert
  their source luminance into a cool silver robot material, while coherent
  green and pink flower components remain in front. The late arm uses an
  eight-vertex contour rather than v65's rejected tall rectangle. Every-frame
  review covers frames 110--136, 235--255, 384--405, and 475--492; the two
  material ramps begin before the observed human-arm entrances, and no exposed
  human skin, large gray hole, or broad metal plate remains in those windows.
  Both outputs decode 660/660 frames. Mean modified area is 5.2397%, edited-band
  clean-plate distance falls by 20.534%, full-frame mean jerk is 0.72219 versus
  0.69753 in raw H3, and union-ROI mean jerk is 1.68049 versus 1.63732; all nine
  acceptance gates pass. The deterministic v70/v71 robot outputs share SHA-256
  `b64d324d84e3bc1de5d542c3c46ccefd65b0fb4a55660dbf3673007d992ffa7e`.
  The published 27.5-second comparison is
  `demo/showcase/real-flower-arranging-h3-wider-lighter-contour-hand-clean-vertical.mp4`
  with SHA-256
  `b4aa9542f54f4e165d4276a8fbba30d34322869ca7bd1a2b1028d2baed44219a`.
  This result is WORKING only as a reviewed 2-D video cleanup; it is not
  evidence of physical robot execution, and stylized faceting can remain under
  dense flower occlusion.
- MiniMax-H3 flower replacement validation:
  `outputs/minimax-h3-flower-validation/20260810T021100Z-nf4-ref2va-r3`
  ran the pinned DiffSynth H3 source at commit `b1c02ce` with third-party NF4
  Ref2VA weights on physical A800 GPU 7. The real 124-frame source window and
  output decode at 832x480/24 FPS; the raw H3 output SHA-256 is
  `a0ef201abb43d15fc400c437bbada5969f4af01f219bc731d562db51987938b9`.
  Five EPL/agentic repair rounds found and rejected a source-face reintroduction,
  then produced a face-safe background/object-locked result. Background and
  object locks are 1.0, subject replacement is 0.8997, source-face replacement
  is 0.99996, and temporal consistency is 0.9311. Robot identity 0.6005,
  motion 0.6185, and manipulate-phase EPL minimum 0.6093 remain below the
  0.72/0.72/0.62 gates, so the result is not the default renderer. Against the
  matched pose-rig v7 window it improves motion from 0.3346 to 0.6185, EPL
  minimum from 0.3337 to 0.6093, and mean score from 0.7182 to 0.8084. This is
  one 5.17-second NF4 validation window, not the complete 27.5-second demo,
  official BF16 H3, PhiZero inference, contact physics, or robot execution.
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
- Real-world flower-arranging observation demo:
  `demo/showcase/real-flower-arranging.mp4` publishes the uncut 27.5-second,
  660-frame Pexels 5893642 real-camera demonstration at 1280x720 and 24 FPS.
  It contains real flowers, tools, contact, and workspace interaction and is now
  the featured flower case on the demo page. This is WORKING as a real-world
  human observation input only. Robot perception, policy inference, and
  real-robot flower arrangement remain NOT STARTED; the rejected screen-space
  and MuJoCo visualizations are not presented as substitutes.
- Contact-window Wan-Animate-2 flower replacement:
  `outputs/wan-flower-animate2/20260810T-hard-contact-distilled` uses official
  Wan-Animate-2 source commit `3ad2fef7d61d6200c9c653e0fe47be7616b323f3`,
  the hashed distilled checkpoint, physical A800 GPUs 0-1, and the real source
  frames 236-316. The raw 80-frame result became the quality anchor for the
  accepted full-length visual route, but strict task evaluation rejected it:
  motion preservation is 0.5413 below 0.72, robot identity is 0.5965 below
  0.72, subject replacement is 0.8548 below 0.88, and manipulate-phase EPL
  minimum is 0.5155 below 0.62. Temporal consistency is 0.9541. This is
  PARTIAL visual evidence only, not contact reconstruction, policy inference,
  or real-robot flower arranging.
- Full-length occlusion-aware compositor v4b:
  `outputs/robot-person-replacement/motion-keyframe-runs/20260810T-occlusion-aware-v4b-mediapipe`
  decodes all 660 real frames and uses current-frame person segmentation plus
  thresholded visible-change and source-like-human residual gates. Background
  and protected-object pixel locks are 1.0, but visible subject replacement is
  0.7901 and human leakage risk is 0.1820; 16-frame review finds residual face,
  shirt, and arms early in the clip. The corrected gate rejects the artifact.
  This is PARTIAL engineering progress, not a deliverable conversion.
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
