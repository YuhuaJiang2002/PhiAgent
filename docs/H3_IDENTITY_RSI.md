# Native identity-topology RSI for MiniMax-H3

This track treats **native identity consistency** as a learned property of the
H3 Ref2VA denoiser: one referenced robot should remain one coherent robot over
the complete generated video.  It is not a post-process that hides a detached
arm in selected keyframes.

The first target is intentionally narrow and falsifiable.  Given a robot
reference, a real flower-workcell scene, and an explicit action-control video,
the adapted model must preserve:

- one robot subject and one head-to-torso chain;
- exactly two continuous shoulder-to-hand chains with unique shoulder origins;
- stable robot proportions with no human residue or extra limbs;
- the baseline action, scene, and temporal capabilities.

## Promotion contract

Every candidate is compared with the unchanged H3 backbone under the same
reference, scene, action control, seed, frame count, resolution, and inference
steps.  Promotion requires all gates, not a weighted average:

| Gate | Requirement |
| --- | --- |
| Full-frame topology | 100% of decoded frames pass every semantic topology gate with review confidence at least 0.95 |
| Identity gain | conservative identity floor improves by at least 0.02 |
| Identity floor | candidate floor is at least 0.62 |
| Motion non-regression | no more than 0.01 below the matched backbone |
| Action non-regression | conservative minimum of motion, EPL, and object lock is no more than 0.01 below the matched backbone |
| Scene non-regression | no more than 0.005 below the matched backbone |
| Temporal non-regression | no more than 0.01 below the matched backbone |

The topology evidence is bound to the exact candidate SHA-256.  DINOv2 masked
similarity is recorded as an independent appearance/worst-frame cross-check;
it cannot replace the semantic head, torso, shoulder, and limb-count review.

## Bounded RSI policy

RSI here means an auditable engineering loop, not an unbounded model rewriting
its own objective:

1. compile rights-attributed, topology-positive Ref2VA clips while keeping
   validation and test subjects out of optimization;
2. train one reviewed LoRA point on `qkv_proj,out_proj`;
3. generate the frozen held-out action and score every hard gate;
4. accept and stop, or append the rejection to `experiences/ledger.jsonl`;
5. choose the next reviewed point from the measured failure: capability
   regression routes to a lower learning rate, while topology/identity underfit
   routes to more capacity;
6. terminate when a candidate passes or the finite search space is exhausted.

The current reviewed search space is defined in
`phiagent/training/h3_identity_rsi.py`.  The optimizer cannot invent new
hyperparameters or see hidden evaluation labels during a round.

## Reproducible entry points

Prepare only the training split:

```bash
python scripts/prepare_h3_identity_dataset.py \
  --config configs/h3_identity_topology_rsi_v2.json \
  --output-root outputs/h3-identity-datasets
```

Preflight or execute one immutable native Ref2VA LoRA round:

```bash
python scripts/train_h3_identity_rsi.py \
  --dataset-dir outputs/h3-identity-datasets/<dataset-run> \
  --diffsynth-repo external/DiffSynth-Studio \
  --model-base-path checkpoints/h3-models \
  --python .venv-h3/bin/python \
  --gpu <physical-gpu-index> \
  --round-name r2-conservative-r16 \
  --lora-rank 16 \
  --learning-rate 2e-5 \
  --dataset-repeat 12 \
  --num-epochs 2 \
  --seed 20260811 \
  --execute
```

Compile digest-bound full-frame topology evidence, then apply the promotion
contract:

```bash
python scripts/compile_robot_topology_review.py \
  --video <candidate.mp4> \
  --plan <full-frame-review.json> \
  --output-dir <new-review-dir> \
  --build-demo --allow-rejected

python scripts/assess_h3_identity_rsi.py \
  --candidate-video <candidate.mp4> \
  --baseline-metrics <matched-metrics.json> \
  --candidate-metrics <matched-metrics.json> \
  --baseline-topology-evidence <baseline-topology-evidence.json> \
  --topology-evidence <candidate-topology-evidence.json> \
  --output-dir <new-assessment-dir> \
  --allow-rejected
```

The GPU evaluator additionally requires matched baseline/candidate action
evaluations and both videos' digest-bound topology evidence. Its immutable
output records the selected physical GPU, input hashes, pinned DINOv2 revision,
and local model-file hashes.

## Current honest evidence

The unchanged NF4 H3 held-out `inspect-flower` baseline passes topology in only
24/124 frames.  It fails unique left-shoulder attachment and stable proportions
in 100 frames, and the single head-to-torso chain in 82 frames.  The six-step
rank-8 smoke adapter leaves the same failure histogram and is rejected.

Rank-16 epoch-0 checkpoints provide the first narrow, measured adaptation
effect: both r1 and lower-learning-rate r2 retain only the 100 left-shoulder
failures.  The 82 head-to-torso failures and 100 stable-proportion failures are
absent in full-frame digest-bound review.  This is a partial native structural
improvement, but fully passing frames remain 24/124 because the shoulder gate is
a hard conjunction.

The improvement is not releasable.  r1 worst-frame DINOv2 identity falls from
0.88338 to 0.78904 and matched motion adherence is 0.90831.  r2 limits those
regressions to 0.86587 and 0.96850, but still misses the contract.  Reducing r2
LoRA strength to 0.5 recovers worst-frame identity to 0.88006, while motion is
still 0.95468 and all three baseline topology failure categories return.  Every
candidate is therefore rejected and preserved as evidence; no optimizer run is
labelled as an improved release model merely because training completed.

The completed r2 epoch-1 checkpoint retains the same narrow structural effect:
only the 100 left-shoulder failures remain, so the full conjunction is still
24/124. Its task evaluator improves motion from 0.48194 to 0.56464, EPL from
0.44953 to 0.50208, robot-identity proxy from 0.71250 to 0.72106, and temporal
consistency from 0.94592 to 0.95293. Those gains do not constitute a promotion:
object lock collapses from 1.0 to 0.00052, frozen DINOv2 worst-frame identity
falls from 0.88338 to 0.78726, and matched pixel-motion adherence is 0.89493.
The conservative action-adherence ratio is therefore 0.00116, and the final
candidate fails identity gain/floor, topology, motion, and action gates.

The next data revision should add camera-matched raised-arm positives, explicit
shoulder-origin crops, and hard negatives for arms emerging from the neck or
head.  Additional identities, scenes, and actions remain mandatory before any
publication claim.

## Release boundary

The present runtime uses third-party pre-quantized NF4 H3 weights.  Any adapter
release must identify the exact base revision and quantization, ship the
upstream license and required notices, mark modified files, retain the AUP and
territorial restrictions, publish training-data provenance, and include the
failed/non-regression evidence.  Review the current
[MiniMax-H3 license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
before distribution.  No result in this track should be described as a general
H3 replacement, physical robot policy, or universal identity solution without
additional identities, scenes, actions, official-BF16 testing, and real-robot
acceptance evidence.
