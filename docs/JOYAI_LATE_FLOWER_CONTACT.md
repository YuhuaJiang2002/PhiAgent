# JoyAI late flower-contact prompt and noise-seed sweep

Status: `PARTIAL`.

This case uses the pinned 2026-08-11 JoyAI-Video-Edit release as a direct
real-video-to-robot-video proposal generator. It strengthens the editing prompt
around persistent hand–stem contact, generates four complete 27.5-second
candidates, applies contact and structure gates before temporal tie-breaking,
and packages a synchronized source/result comparison.

The published video is available on the [PhiAgent demo page](https://yuhuajiang2002.github.io/PhiAgent/#joyai-late-contact).
Its left panel is the real human source and its right panel is the selected
JoyAI robot result. SAM masks are used only by offline measurement; no SAM mask
is used to modify the generated pixels.

## Frozen model and protocol

- Model: `jdopensource/JoyAI-Video-Edit`
- Repository revision: `3478e4b8c9a79fe935157d1d477cd3e57bb41f1f`
- Weights revision: `e14d9ac50d4ad8e9f91b655bfab270c02a43923b`
- DiT SHA-256: `b3904b6fda53d13b230918bb616f322d12cfb2337b0e8d9dc203cdabc36605ba`
- Prompt: [`configs/joyai/late_flower_contact_prompt_v1.txt`](../configs/joyai/late_flower_contact_prompt_v1.txt)
- Selection contract: [`configs/joyai/late_flower_contact_seed_sweep_v1.json`](../configs/joyai/late_flower_contact_seed_sweep_v1.json)
- Seeds: `17, 42, 73, 101`
- Inference steps: `2`
- Generation: one uninterrupted 665-frame causal stream per seed, trimmed back
  to the source's 660 frames after removing the five cloned tail frames
- Output: 1280×720, 24 FPS, 27.5 seconds

Each GPU entry point records the physical GPU selection and sets
`CUDA_VISIBLE_DEVICES`. The server launcher refuses missing revisions,
incomplete checkpoints, insufficient GPU memory, an invalid CUDA runtime, or an
existing output directory.

## Generation

Prepare the 660-frame source as the shortest complete JoyAI causal stream:

```bash
python scripts/prepare_joyai_full_stream.py \
  --source-video SOURCE.mp4 \
  --output-dir RUN/preparation \
  --expected-source-frames 660 \
  --fps 24 \
  --source-width 1280 --source-height 720 \
  --model-width 1248 --model-height 720 \
  --crop-left 16 --crop-top 0 \
  --chunk-frames 8
```

Launch the pinned two-GPU server. The repository and checkpoints are external
and are not vendored by PhiAgent:

```bash
python scripts/launch_joyai_video_edit_server.py \
  --repository EXTERNAL_JOYAI_3478e4b8 \
  --checkpoint-root CHECKPOINTS_JOYAI_0811 \
  --output-dir RUN/server \
  --python JOYAI_PYTHON \
  --physical-gpu 4 --physical-gpu 5 \
  --host 127.0.0.1 --port 18080 \
  --overlay patches/joyai-video-edit/0811-torch291-dynamic-vae-unwrap.patch
```

Generate each frozen seed as a complete stream. Use a new output directory for
every seed:

```bash
python scripts/run_joyai_video_edit_client.py \
  --server-url ws://127.0.0.1:18080/ws \
  --input-video RUN/preparation/joyai-full-stream-input-ffv1.mkv \
  --output-dir RUN/inference-seed-17 \
  --reference-image ROBOT_REFERENCE.png \
  --prompt-file configs/joyai/late_flower_contact_prompt_v1.txt \
  --width 1248 --height 720 --fps 24 \
  --expected-frames 665 \
  --seed 17 --num-inference-steps 2 \
  --output-quality 95 --throughput-mode \
  --output-artifacts both
```

Restore the source-width side borders and remove causal tail padding:

```bash
python scripts/finalize_joyai_full_stream.py \
  --source-video SOURCE.mp4 \
  --joyai-video RUN/inference-seed-17/joyai-proposal-lossless.mkv \
  --output-dir RUN/final-seed-17 \
  --expected-frames 660 --fps 24 \
  --source-width 1280 --source-height 720 \
  --model-width 1248 --model-height 720 \
  --crop-left 16 --crop-top 0
```

## Selection and audit

The selector does not mix different seeds frame by frame. A seed must first
pass late projected contact, dense persistent grasp, source hold observability,
grasp-erasure rejection, robot structure/identity, and adversarial attacks.
Only hard-gate passers may be ranked by temporal jitter. Threshold relaxation
and mean-score overrides are forbidden.

In this run no seed passed every hard gate. Seeds 17 and 42 tied for the best
H.264 screening contact result, and seed 17 won the declared temporal
tie-break:

| Seed | Late contact | Persistent grasp (screening) | Self-flow jitter P95 | Self-flow jitter mean |
|---:|---:|---:|---:|---:|
| 17 | 9/11 | 146/147 | 6.0703 | 2.3448 |
| 42 | 9/11 | 146/147 | 6.2785 | 2.3938 |
| 73 | 9/11 | 142/147 | not ranked | not ranked |
| 101 | 9/11 | 145/147 | not ranked | not ranked |

The authoritative FFV1 re-audit of seed 17 measured 9/11 late projected
contact and 145/147 persistent visual grasp. H.264 had filled one single-pixel
gap, which is why its screening count is one frame higher. Lossless failures
remain at frames 574 and 637 for persistent grasp and frames 627–628 for the
late projected-contact samples. All four injected audit attacks were detected.

The previous direct JoyAI result measured 5/11 late projected contact, so the
prompt-enhanced candidates improve the image-space diagnostic to 9/11. They do
not satisfy the frozen 100% gate and are not promoted.

Run the strict audit with [`scripts/audit_robot_layer_long_video.py`](../scripts/audit_robot_layer_long_video.py),
measure independent self-flow jitter with
[`scripts/audit_joyai_rv2v_temporal_stability.py`](../scripts/audit_joyai_rv2v_temporal_stability.py),
and build the labeled comparison with
[`scripts/build_persistent_grasp_comparison.py`](../scripts/build_persistent_grasp_comparison.py).

## Honest scope

The result is a one-scene 2-D camera-frame visual diagnostic. It provides no
metric depth, robot-base trajectory, collision, force-closure, sensor contact,
or real-robot execution evidence. The published case is intentionally marked
`PARTIAL` and includes the remaining failed frames rather than trimming them
out.
