# Agentic grounded AC-WM campaign

Evidence date: 2026-08-12. Current status: **PARTIAL**.

The requested end state is not yet established. This campaign turns it into a
falsifiable procedure: a candidate is called stronger only after paired frozen
tests show it beats every named reproducible baseline, and a “real-robot demo”
requires physical execution recorded after the model prediction. A simulated or
photorealistic generated clip is not relabelled as a real-robot result.

The active campaign is **real-world-only**. RoboTwin, MuJoCo, SAPIEN rollouts,
simulator states/contacts/task-success labels, and simulator-derived
counterfactual references are excluded from training, promotion, and paper
evidence. Their old artifacts remain immutable historical records only.

## Research synthesis

- [XEWorld (arXiv:2608.05799)](https://arxiv.org/abs/2608.05799) motivates an
  explicit unseen-embodiment split, strict spatiotemporal action alignment, and
  separation of appearance adaptation from dynamics learning.
- [Boundless World Model (arXiv:2607.29302)](https://arxiv.org/abs/2607.29302)
  supplies the strongest practical starting checkpoint and its 14D action,
  57-frame, 9-history-frame interface. The pinned public README still labels the
  training release “Coming soon,” so the local training entry point is treated
  as incomplete engineering evidence rather than an official recipe.
- [Mem-World (arXiv:2606.18960)](https://arxiv.org/abs/2606.18960) makes 4D
  memory a required long-horizon ablation. It is not implemented in this round.
- [MiraBench (arXiv:2605.29360)](https://arxiv.org/abs/2605.29360) motivates
  separate physics and action-following gates; visual fidelity cannot mask a
  causal failure.
- [WorldArena 2.0 (arXiv:2605.17912)](https://arxiv.org/abs/2605.17912) and the
  [original WorldArena (arXiv:2602.08971)](https://arxiv.org/abs/2602.08971)
  motivate the evaluation protocol. The active lane uses only recorded
  real-robot WorldArena videos and measured trajectories; simulator subsets and
  synthetic task-success labels are excluded.
- [SC3-Eval (arXiv:2606.18610)](https://arxiv.org/abs/2606.18610) motivates the
  correct-action versus counterfactual-action consistency check used below.
  [VBench](https://github.com/Vchitect/VBench) supplies the complementary
  subject, background, flicker, and motion-quality dimensions; task and causal
  metrics remain separate from appearance quality.

## Implemented in this round

1. `build_worldarena_bwm_test_bundle.py` and
   `build_bwm_counterfactual_suite.py` compile recorded real-robot WorldArena
   RGB plus measured absolute EEF trajectories into BWM's reviewed 14D
   Euler-XYZ contract. Every action names its robot base. Task and physical
   episode groups cannot leak across train, validation, and test.
2. Packed MP4 and Parquet offsets remain separate. Seven reviewed, hashed
   compatibility patches fix the public BWM release's offset overwrite,
   disabled-text tokenizer construction, unsupported training-builder argument,
   failure to expand the Wan sharded-weight index, a missing DiffSynth
   dataset-runner flag, and incorrect list-style dimension access on the
   documented `(V,C,T,H,W)` video tensor. The final patch generalizes the
   one-latent DiffSynth SFT loss to exclude all encoded BWM history latents from
   the denoising target. The pinned public source is therefore an auditable
   starting point, not an unmodified official training recipe.
3. `train_agentic_bwm.py` inspects and records physical GPU state before setting
   `CUDA_VISIBLE_DEVICES`. It also writes and passes an experiment-owned
   single-GPU Accelerate configuration, preventing user-level `gpu_ids` from
   disagreeing with the recorded physical GPU. It creates a new immutable attempt directory with
   configuration, command, source/model revisions, package versions, seed, logs,
   and outputs. The first stage trains only the action encoder; joint DiT/action
   fine-tuning is an explicit later stage.
4. `evaluate_acwm_promotion.py` requires paired per-trial gains, at least 20
   trials, and a positive one-sided bootstrap lower confidence bound for every
   mandatory metric against every declared baseline.
5. `validate_real_robot_acwm_demo.py` requires calibrated pre-execution
   prediction, synchronized 14D telemetry, physical execution video, safety
   logs, blind outcome review, and hashes. The current protocol is `NOT STARTED`
   because this workspace has no controllable robot or operator approval.

## Evidence obtained

- Focused CPU tests and targeted Ruff checks pass for real-data conversion,
  training launch, promotion, verified model adoption, and real-robot evidence.
- The real WorldArena cache contains 100 physical episodes with a task-disjoint
  60/20/20 split. A compact 20-episode suite has 20 independent
  factual/history-preserving cross-episode action-swap pairs.
- An earlier small three-seed/two-physical-episode audit preserves the adapter's factual
  future-SSIM gain from 0.823330 to 0.829861, but flow EPE slightly regresses
  and both action-sensitivity confidence bounds cross zero. The adapter is not
  promoted for action control.
- The larger seed42 wipe20 run completes official/candidate factual and swapped
  videos for 20 independent physical episodes. A corrected 10,000-bootstrap
  audit improves factual future SSIM from 0.852103 to 0.864253 (gain +0.012150,
  95% lower bound +0.009287), but factual flow EPE and both wrong-action
  primary margins regress. The all-primary gate is false and action promotion
  is rejected. Seeds 314159 and 20260811 also complete. After averaging all
  three seeds within each physical episode, factual SSIM gain remains
  significant (+0.012613; lower bound +0.009622), while factual flow EPE and
  both wrong-action margins still regress. The final three-seed action audit
  remains rejected.
- The three Wan2.2 DiT shards, T5 encoder, VAE, index, and configuration in an
  existing remote cache were byte-checked against the pinned official revision.
  The verifier created an experiment-owned symlink view and recorded every
  file's byte count and SHA-256 under
  `checkpoints/Wan2.2-TI2V-5B-verified/.phiagent-verification.json` on `a800-1`.
- The isolated BWM runtime imports with PyTorch 2.8.0+cu128, DiffSynth 2.0.11,
  Transformers 4.51.3, and safetensors 0.7.0. The Transformers stack is pinned
  because the otherwise unbounded resolver selected an incompatible 5.x
  release during the first attempt; that failed attempt remains in the ledger.
- Official BWM `step-12000.safetensors` was verified at 10,051,484,872 bytes
  with SHA-256
  `75f863b9474d6e74934db45bb85728fef0adece3d123c667b78349bdade9c7f3`.
- Historical one-step simulator training attempts exposed gaps in the public
  release and validated some software plumbing, but their checkpoints and
  metrics are excluded from the active real-world-only campaign. A balanced
  real-world training acceptance run has not completed.

## Historical simulator diagnostic (excluded from the active campaign)

The following record is retained only for append-only failure history. It may
not support the method, promotion, SOTA comparison, or paper. A second
adaptation used released demo episodes 40 and 41 only and held out the
fixed episode-42 frame window 64--80. The run completed ten action-encoder
optimization steps (five epochs) on physical A800 GPU 0 and saved five immutable
checkpoints. The final adapter contains 99,637,248 finite parameters and has
SHA-256
`4f2a41e8f14642a794a5a06bf9adf83f83ea673d6c1b062650b150e3028152d9`.
Official and adapted BWM rollouts used the same Wan base, DiT weights, first
frame, 14D actions, seed, 17-frame horizon, and 20 inference steps. A second
rollout held the first frame fixed but substituted episode-40 actions, allowing
an action-causal margin to be measured rather than assuming visual similarity
proves action following.

The five-epoch candidate improved action-causal margin from 0.00564 to 0.00788
and reduced 224x168 optical-flow endpoint error from 0.4140 to 0.3718 pixels.
It was rejected because future SSIM fell from 0.95836 to 0.94930, background
MAD increased from 0.00708 to 0.00753, temporal-gradient error increased from
0.001591 to 0.001764, and endpoint SSIM fell from 0.94690 to 0.93594. Early
stopping at epoch 0 reduced the regressions and improved action-causal margin
to 0.00676, but still lost future SSIM, endpoint SSIM, and flow-direction
cosine. It also failed promotion. These are one-window diagnostics on released
demo data, not an unbiased test-set result.

The rejected comparison is retained only as internal evidence at
`outputs/acwm-metric-driven-evaluation/paired-epoch0-20260811T1136/rejected/robottwin-reference-vs-official-vs-trained.mp4`
(1344x380, 24 FPS, 67 frames; SHA-256
`49c738bdda47e3fc8a98bb0ff7a2a14639ca7ec05f516537b358e9c8c93b44c5`).
It is deliberately labelled as rejected held-out simulation evidence. It is not
physical-robot footage and is not eligible for active training, promotion,
paper claims, AC-WM showcase, or delivery.

## Public real-world-scene AC-WM visualization

The public AC-WM visualization uses the real EPIC-KITCHENS-100 `P03_28`
first-person kitchen interval from 25.83 to 35.83 seconds. The source-plus-result
grid is `demo/showcase/acwm-ego-instruction-task-comparison-10s.mp4`; the
generated-only view is
`demo/showcase/acwm-ego-instruction-task-comparison-generated-only-10s.mp4`.
The three camera-frame action conditions are bottle handover, cap unscrewing,
and faucet/rinsing pose. The frozen evidence reports 240 frames at 24 FPS,
pairwise full-frame MAD 34.62--53.89, quality non-regression, reproducible
lineage, and passed dense human-residual, blur, and terminal-task reviews. This
is a generated action visualization in a recorded real scene, not calibrated
physical-robot execution or a new world-model fine-tune claim.

## Remaining acceptance sequence

1. Reserve uncontended GPU capacity and run the balanced multi-embodiment
   training campaign with matched seeds.
2. Evaluate public BWM, Ctrl-World, OSCAR, Kinema4D, and candidate ablations on
   exactly the same frozen suites. Do not select on test data.
3. Add the Mem-World-style 4D memory ablation only after the baseline is
   reproduced.
4. Obtain hardware/operator access and execute at least three genuine demo
   trials; obtain 20 paired trials before allowing the promotion gate to run.

Until all four are complete, the honest result is **PARTIAL**, not SOTA.

## Rendered real-scene comparison

For visual inspection only,
`demo/showcase/acwm-real-scene-vs-rendered-robot-execution-10s.mp4` places the
unchanged real kitchen observation beside synchronized rendered pour, shake,
and handover robot effects. It is 240 frames at 1664x960 and 24 FPS with
SHA-256 `b6135e06cbebc48f019eb93a68cdbdad829325b74510cff614b410b62fd709dc`.
The labels and footer explicitly state that the robot panels are simulated.
This artifact is not physical robot evidence and is not attributed to the
one-step BWM smoke checkpoint.
