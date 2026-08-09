# Agentic PhiZero proxy

This is an explicitly approximate route for making progress while the official
PhiZero tokenizer, decoder, adaptation checkpoint, and code are unavailable. It
must never be reported as execution or reproduction of PhiZero.

## Strategy

1. Use one official `hand2dex` human-hand source video.
2. Extract or independently create several Sharpa first-frame candidates.
3. Generate a candidate ensemble with released motion-transfer backends. The
   first implemented backend is the pinned Wan2.2-Animate adapter. Its default
   is replacement mode, which preserves source pixels outside the estimated
   character mask instead of regenerating the complete scene.
4. Run a local evaluator agent on each candidate and the official transferred
   reference. It must report four independent scores in `[0, 1]`:
   motion preservation, target identity, object consistency, and temporal
   consistency.
5. Accept only candidates that pass every configured threshold. Otherwise the
   repair agent selects the strongest candidate and changes the seed within a
   bounded round budget. The pinned native WanAnimate generator ignores its
   accepted prompt argument, so textual feedback is recorded but does not
   condition this backend.
6. Persist every proposal, backend run, score, diagnosis, hash, package version,
   Git state, and final decision.

This approximates the paper's result at the video level. It does not recreate
the learned FSQ physical-language representation or unchanged-token decoding.

Wan-Animate-2 is a second, separately labelled proxy backend. Unlike the original
Wan-Animate preprocessing path, the official Apache-2.0 model directly consumes a
reference image and driving video. This removes the human pose/face-control blocker
for a real Sharpa reference image, but it remains a general character-animation model:

```text
human driving video + official Sharpa image
  -> pinned Wan-Animate-2 on two selected A800 GPUs
  -> local motion/identity/object/temporal evaluation
```

Run strict preflight before inference:

```bash
python scripts/prepare_wan_animate2.py
./scripts/bootstrap_wan_animate2_environment.sh

python scripts/run_wan_animate2.py \
  --source-video external/PhiZero-reference/hand2dex_1_source.mp4 \
  --reference-image inputs/sharpa-reference.png \
  --prompt "A Sharpa Wave dexterous hand manipulates the demonstrated object." \
  --repo external/Wan-Animate-2 \
  --checkpoint-dir checkpoints/Wan2.2-Animate-2-14B \
  --python .venv-wan-animate2/bin/python \
  --preflight-only
```

The runner verifies source and model revisions, hashes inputs and checkpoint files,
selects exactly two physical GPUs, records `CUDA_VISIBLE_DEVICES`, and requires an
actual non-empty `results.mp4`. Every run is labelled
`wan_animate2_proxy_not_official_phizero`.

## Local evaluator contract

The built-in evaluator is `scripts/local_video_evaluator.py`. It uses local
ffmpeg decoding at 64x64 and 8 FPS by default, adds no Python dependency, and
never uploads video. It computes:

- motion preservation from blockwise source/candidate transition vectors;
- target identity from the Sharpa first-frame anchor and official-reference
  structural similarity;
- object consistency as official-reference edge similarity;
- temporal consistency from excess second-order frame jerk over the reference.

The object metric is explicitly a visual proxy, not object detection or
instance-level contact verification.

The evaluator is invoked without a shell:

```text
EVALUATOR \
  --source SOURCE.mp4 \
  --reference OFFICIAL_TRANSFERRED.mp4 \
  --target-image SHARPA.png \
  --candidate CANDIDATE.mp4 \
  --metadata BACKEND_METADATA.json
```

It must write exactly one JSON object to stdout:

```json
{
  "evaluator": "local-evaluator-name-and-revision",
  "motion_preservation": 0.82,
  "target_identity": 0.88,
  "object_consistency": 0.79,
  "temporal_consistency": 0.76,
  "diagnoses": ["minor fingertip deformation"],
  "evidence": "/absolute/path/to/persisted/evaluation.json"
}
```

The optional evidence path must exist. Evaluator failures and malformed,
missing, non-finite, or out-of-range scores fail the run rather than becoming
success-shaped defaults. A local evaluator is required so project videos are not
silently uploaded to third parties.

## Run

After preparing the pinned reference assets and a Sharpa target image:

```bash
ffmpeg -i external/PhiZero-reference/hand2dex_1_transferred.mp4 \
  -frames:v 1 inputs/hand2dex_1_sharpa_first_frame.png

python scripts/run_agentic_phizero_proxy.py \
  --source-video external/PhiZero-reference/hand2dex_1_source.mp4 \
  --reference-video external/PhiZero-reference/hand2dex_1_transferred.mp4 \
  --target-image inputs/hand2dex_1_sharpa_first_frame.png \
  --target-image inputs/hand2dex_1_sharpa_variant.png \
  --seed 42 \
  --seed 1051 \
  --evaluator scripts/local_video_evaluator.py \
  --wan-repo external/Wan2.2 \
  --checkpoint-dir checkpoints/Wan2.2-Animate-14B \
  --mode replacement \
  --experiment-root outputs/phizero-agentic-proxy
```

Run with `--preflight-only` before GPU inference. The default threshold vector is
`[0.75, 0.80, 0.75, 0.75]`; it is a proxy engineering gate, not a paper metric.
`ffmpeg` must be on `PATH`; the evaluator fails with an actionable preflight
message when it is absent.

Replacement mode additionally requires the checkpoint's SAM2 assets and uses
the released relighting LoRA by default. Inspect `src_mask.mp4` before trusting
a candidate: segmentation errors can retain human pixels or include the held
object in the replaced region. Use `--mode animation` only for a full-frame
regeneration baseline.

## Next adapters

- An optional local Qwen-VL evaluator for semantic identity and object-contact
  review beyond the deterministic built-in metrics.
- DINO/segmentation tracking for Sharpa and object identity drift.
- Optical-flow correspondence for source-to-candidate motion preservation.
- A Cosmos candidate backend when a deterministic control video can be derived
  without pretending that it is PhiZero physical language.

Each adapter remains optional and isolated so importing `phiagent` stays CPU-only.
