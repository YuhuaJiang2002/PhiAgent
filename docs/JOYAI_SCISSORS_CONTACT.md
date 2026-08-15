# JoyAI scissors and late-contact pipeline

Status: **PARTIAL** (2026-08-13).  The final label stays PARTIAL because the
image-space contact test is not 3-D contact or force-closure evidence, and the
small scissors handle rings are not independently observable in every native
frame.

## What changed

The full-stream prompt treats the florist scissors as a distinct rigid object,
not a texture on the hand.  `HeldToolContract` pins its source interval
(frames 398–447), holder (`robot_right_hand`), topology, and ten native-resolution
review frames.  Tool review is a human veto and never an automatic promotion.

The JoyAI proposal is one uninterrupted 665-frame causal session.  Five cloned
tail frames complete the model's `1 + 8n` chunk contract and are removed without
interpolation to recover the exact 660-frame, 24 fps, 27.5 s timeline.  A
tool-bearing robot reference is a semantic conditioning anchor only; it is not
accepted as evidence about generated frames.

Late hand–flower repair is audit-driven.  It selects only source-required
contact frames that the candidate misses, expands them into small temporal
intervals, and copies donor pixels only where they add a robot replacement
inside tracked hand support.  Flower pixels are immutable.  If a residual
one-pixel image-space gap remains, `project_missing_contact` grows existing
generated-hand pixels through tracked hand support and stops immediately when
the fixed projected-contact invariant passes.  It cannot cross the protected
flower mask and it carries `physical_evidence = false` semantics.

## Reproducible measurements

- Official JoyAI full-stream proposal: 665 generated frames in 105.692 s,
  **6.292 generated fps** (3.81× real time for a 24 fps video).
- The scissors-conditioned raw candidate reaches 6/11 late projected contacts,
  up from the reported 5/11 baseline but still below the fixed 95% gate.
- Final deterministic union: 660 frames in 12.761 s, **51.720 repair fps**.  It
  changes 25,548 donor pixels across 14 audit-selected frames, never writes a
  protected flower pixel, and needs no synthetic contact-pixel growth.
- The exact final audit reaches **11/11 = 100%** projected-contact recall at or
  after 20 s.  Ten of eleven image gates pass; late hand edge energy remains
  below its anchor-fitted lower bound in 22/180 late frames (12.22%, over the
  allowed 10%).  Color, contact, and topology attacks are detected, while the
  structure-ghost attack is not, so adversarial promotion remains rejected.
- Native scissors review finds a hand-bound black/silver tool trajectory across
  the required interval, but the two handle rings and finger-through-ring state
  are not separable in every sampled frame; this gate remains PARTIAL.

## 100-hour capacity estimate

The planning baseline uses the measured A800 rate above without extrapolating a
new model speed: 665 generated frames in 105.692 seconds, or 6.2919 generated
fps. At 24 fps, 100 source hours contain 8,640,000 frames. If the source is
partitioned into the already tested 660-frame (27.5-second) independent clips,
the `1 + 8n` causal contract produces 8,705,451 model frames across 13,091
sessions. Tail padding adds 65,451 frames, or 0.758%.

This is 384.33 A800-hours of generation at 100% utilization. The measured final
union adds 46.40 worker-hours at 51.7201 fps and can run as a pipeline behind
generation. The following calendar estimates assume 85% utilization, zero
unmeasured session-start overhead, independent clips, resident model services,
and enough postprocess workers to keep that stage off the critical path:

| A800s | Postprocess workers | Critical-path hours | Days |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 452.16 | 18.84 |
| 4 | 1 | 113.04 | 4.71 |
| 8 | 1 | 56.52 | 2.35 |
| 16 | 2 | 28.26 | 1.18 |
| 32 | 4 | 14.13 | 0.59 |

Eight A800s plus one postprocess worker are the recommended first production
shape: their measured-rate stages are nearly balanced at 56.52 and 54.59 hours.
Reserve at least three calendar days before reruns and quality review. First run
a one-hour pilot, because five seconds of unmeasured per-session overhead would
add 18.18 A800-hours to the full campaign and multi-GPU scaling has not yet been
measured.

The client now provides `--throughput-mode`. It preserves the prompt, seed,
sampling steps, output JPEG quality, and causal ordering while changing only
transport and artifact handling:

- decode input lazily instead of retaining an entire clip in memory;
- submit one eight-frame causal chunk before waiting for an ACK;
- sample frame-level protocol logs every 240 frames while retaining boundary
  and control events;
- append returned JPEG packets to one MJPEG spool instead of creating one file
  per frame;
- encode only the review artifact and remove the spool after a successful mux.

Explicit flags override each throughput default. Use
`--output-artifacts lossless` or `both` when an H.264 review artifact is not
suitable for the downstream dataset.

At an assumed 50 Mbit/s review bitrate, 100 hours need about 2.25 decimal TB
(0.90 TB at 20 Mbit/s; 4.50 TB at 100 Mbit/s). The CRF-8 encoder is
content-dependent, so the one-hour pilot must replace this assumption with its
measured bitrate. At an assumed 200 KiB per protocol JPEG, the campaign writes
about 1.78 TB of transient JPEG payloads. The MJPEG path bounds live staging to
about 136 MB per worker, or 1.09 GB across eight workers, rather than retaining
8.7 million small files.

Reproduce the estimate:

```bash
python scripts/estimate_joyai_replacement_capacity.py \
  --output-dir outputs/joyai-capacity/<new-run> \
  --video-hours 100 \
  --fps 24 \
  --average-clip-seconds 27.5 \
  --gpu-count 8 \
  --gpu-utilization 0.85 \
  --postprocess-workers 1
```

The estimate and CPU implementation tests are complete. The optimized protocol
path and its scaling remain `PARTIAL` until the one-hour A800 pilot measures
session overhead, output bitrate, GPU utilization, failures, visual-quality
non-regression, and end-to-end throughput.

Primary local evidence:

- `outputs/joyai-flower-edit/20260813T105500Z-full665-scissors-contact-seed42-v2`
- `outputs/joyai-flower-edit/20260813T110000Z-full660-scissors-contact-final-v1`
- `outputs/joyai-flower-edit/20260813T105500Z-contact-projected-union-v2`
- `outputs/joyai-flower-edit/20260813T104500Z-contact-projected-audit-v1`
- `outputs/joyai-flower-edit/20260813T113000Z-scissors-contact-projected-union-v2`
- `outputs/joyai-flower-edit/20260813T113500Z-scissors-contact-projected-audit-v2`
- `outputs/joyai-capacity/20260815T160840Z-100h-a800-v1`

## Acceptance boundary

The current 2-D audit establishes only projected adjacency, replacement
coverage, topology proxies, palette/skin leakage limits, and adversarial
sensitivity.  It does not establish metric depth, contact force, collision-free
kinematics, or force closure.  Those require an independent depth/telemetry or
verified physics source and remain outside this visual-data milestone.
