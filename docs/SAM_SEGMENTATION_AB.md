# SAM2 / SAM3.1 segmentation A/B harness

The T-shirt video evaluator runs two pinned trackers on byte-identical JPEG
tracking frames, lossless shared scoring frames, and identical initial binary
masks:

- SAM2 is the authoritative evaluator for the existing epoch.
- SAM3.1 Object Multiplex is a shadow evaluator.
- Both workers run concurrently on distinct physical GPUs and may use separate
  Python environments.
- SAM2 thresholds are emitted for SAM3.1 as diagnostics only. They never become
  a SAM3.1 pass/fail decision.

This separation preserves historical comparability while collecting evidence
for a possible future evaluator epoch.

## Optional environments and checkpoints

SAM2 remains pinned by `requirements/sam2.txt`. Install
`requirements/segmentation-ab.txt` in the SAM2 environment so CPU preparation
and scoring use the pinned NumPy and OpenCV versions. Install SAM3.1 in a
separate Python 3.12 environment with a CUDA-compatible PyTorch 2.7 or newer
build, then install `requirements/sam31.txt`; that file includes the same shared
dependencies. The core `phiagent` package does not import either model.

Request access to `facebook/sam3.1` on Hugging Face and download
`sam3.1_multiplex.pt` into the ignored
`checkpoints/sam3.1/` directory. The committed shadow config pins:

- source commit `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`;
- Hugging Face revision `daa63191845a41281374e725f4c9e51c7a824460`;
- checkpoint SHA-256
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.

SAM3.1 uses Meta's SAM License rather than SAM2's Apache-2.0 license. Do not
vendor either model repository or checkpoint.

## Run

Use a new output directory for every attempt:

```bash
python scripts/evaluate_joyai_tshirt_segmentation_ab.py \
  --video /absolute/path/to/candidate.mp4 \
  --task-config configs/physical_video/tshirt_left_stage_segmentation_ab_v1.json \
  --sam31-config configs/physical_video/sam31_multiplex_shadow_v1.json \
  --sam2-python /absolute/path/to/sam2-env/bin/python \
  --sam31-python /absolute/path/to/sam31-env/bin/python \
  --sam2-gpu 0 \
  --sam31-gpu 1 \
  --output-dir outputs/tshirt-segmentation-ab/NEW-RUN-ID
```

If GPU indices are omitted, the harness selects two distinct eligible physical
GPUs. Before launch it acquires repository GPU leases in sorted physical-index
order, rechecks free memory while holding both leases, and keeps the leases until
both workers exit. It records the full inventory, active compute processes,
selected physical indices, lease paths, `CUDA_VISIBLE_DEVICES`, commands,
source/checkpoint hashes, Git state, package versions, seed, logs, masks,
storyboards, timings, and peak CUDA memory.

The shared-input stage decodes the video once and stores JPEG tracking-frame,
lossless scoring-frame, and initial-mask hashes. Both workers verify every hash
before loading a model. An authoritative SAM2 failure produces a `BLOCKED` run.
A SAM3.1-only failure produces a nonzero `PARTIAL` outcome while retaining the
completed SAM2 result and decision.

## Interpret results

`comparison.json` reports same-object frame IoU, mask disagreement, area ratio,
centroid distance, potential label-takeover frames, incumbent-threshold
disagreements, elapsed time, and peak CUDA memory. These are agreement metrics,
not accuracy metrics: either tracker can be wrong.

SAM3.1 promotion remains disabled until a frozen set of native-resolution,
human-labeled keyframes measures boundary quality, missed masks, identity
switches, and wrong-object takeover. A promoted model requires a new evaluator
epoch and independently registered thresholds.
