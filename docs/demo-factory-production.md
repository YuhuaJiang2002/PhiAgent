# Agentic demo-video factory production model

Evidence date: 2026-08-11 UTC.

## Final model

The promoted model is `phiagent-bwm-worldarena2-action-adapter-v1`:

- base: pinned Wan2.2-TI2V-5B plus official Boundless World Model
  `step-12000`;
- learned component: a 99,637,248-parameter action encoder, 199,275,904 bytes;
- recipe: 3 epochs, learning rate 5e-6, seed 20260811, action-only training;
- input contract: 14-D absolute EEF pose in
  `robot_base:worldarena2-cobot-magic-max-end-pose`, 9 history frames, 57 total
  rollout frames;
- immutable remote package:
  `/data0/jiangyuhua/PhiAgent-0/outputs/demo-factory-models/bwm-worldarena2-action-adapter-v1`;
- local evidence mirror:
  `outputs/demo-factory-models/bwm-worldarena2-action-adapter-v1/model.json`.

The adapter SHA-256 is
`da06a649aa0ccee950964dc32525b2b3ed66b61276eb4ba380b938e66237a009`.
The deployment-ready merged checkpoint SHA-256 is
`cbc6ce18e479d8604a12142766938b018f8f5f33ab2c88bb649c12c1510eed5c`.

## Acceptance evidence

The split is task-disjoint, rather than random clips from the same task:

| Split | Tasks | Samples | Used for |
|---|---|---:|---|
| Train | `clean_table`, `fold_shirt`, `pour_water` | 12 | action-adapter fitting |
| Validation | `pour_over_coffee` | 4 | recipe/checkpoint selection |
| Test | `wipe_table` | 4 | one-time final acceptance |

The chosen model passed every frozen validation and test gate. On the untouched
test task, future SSIM improved from 0.825670 to 0.835889, endpoint SSIM from
0.806228 to 0.814920, background MAD fell from 0.216235 to 0.187611, and flow
endpoint error fell from 2.093513 to 2.070462. All four samples improved in
future SSIM. The earlier 2e-6 adapter improved all four validation samples but
missed the predeclared +0.002 mean-SSIM margin, so it was not promoted.

## Measured production time

The production benchmark used three physical NVIDIA A800-SXM4-80GB GPUs. Each
worker loaded the final model and generated four samples; the harness combined
all 12 outputs without copying their payload bytes.

| Quantity | Measured result |
|---|---:|
| Samples completed | 12/12 |
| Frames per sample | 57 |
| Generated duration per sample | 2.375 s at 24 FPS |
| End-to-end wall time | 112.594 s |
| Aggregate throughput | 383.681 samples/hour |
| Wall time per sample, 3 GPUs | 9.383 s |
| GPU time per sample | 28.005 GPU-s |
| GPU time per 1,000 samples | 7.779 GPU-hours |
| Wall time per 1,000 samples, 3 GPUs | 2.606 hours |

For a GPU price `P` per hour, the measured inference cost estimate is
`sample_count * 28.005 / 3600 * P`. The benchmark excludes upstream scenario
authoring and queue delays. Larger batches should amortize model loading at
least as well, but that projection must be re-measured if frame count,
inference steps, resolution, hardware, or concurrency changes.

## Production command

Run from `/data0/jiangyuhua/PhiAgent-0` on the A800 host:

```bash
python3 scripts/run_bwm_factory_batch.py \
  --repository external/boundless-world-model \
  --base-model checkpoints/Wan2.2-TI2V-5B-verified \
  --checkpoint outputs/demo-factory-models/bwm-worldarena2-action-adapter-v1/weights/merged-checkpoint.safetensors \
  --metadata /absolute/path/to/batch.jsonl \
  --dataset-base-path /absolute/path/to/dataset-root \
  --action-stats /absolute/path/to/action-stat.json \
  --experiment-root outputs/demo-factory-production \
  --label production-batch \
  --gpu 0 --gpu 2 --gpu 3 \
  --minimum-free-gpu-mib 61000 \
  --num-frames 57 \
  --num-inference-steps 20
```

Each invocation creates a new immutable campaign directory and records the
physical GPU inventory, leases, commands, environment, per-shard result,
throughput, and SHA-256 of every generated video. Partial shard failures are
reported as `BLOCKED`; they are never silently counted as completed samples.

## Claim boundary

This model+harness is accepted for audited generated-video data production.
The evaluation compares generated video with lossy 384x288 real-robot reference
clips using pixel and optical-flow metrics. It does not establish calibrated
3-D geometry, contact forces, collision safety, task completion, or deployment
on a physical robot. WorldArena2 `end_pose` is treated as dataset-declared
robot-base EEF data; no independent calibration was available.
