# JoyAI egocentric silver five-finger robot-arm replacement

Status: `PARTIAL`.

This case applies the pinned JoyAI-Video-Edit 0811 RV2V path to two complete
first-person, bimanual manipulation videos. It replaces visible human arms with
polished silver five-finger robot arms and publishes synchronized source/result
comparisons. The two inputs last 35.917 and 40.167 seconds after normalization,
which makes them longer than the previously published 27.5-second flower case.

The published media are available on the
[PhiAgent demo page](https://yuhuajiang2002.github.io/PhiAgent/#joyai-ego-tabletop).
The left panel is the original human source and the right panel is the selected
robot result labeled `PhiAgent`. Publication of both source scenes was authorized
by their owner.

## Published cases

| Case | Normalized timeline | Manipulation | Selected seed | Published right panel |
|---|---:|---|---:|---|
| Tabletop object arrangement | 862 frames, 35.917 s | Alarm clock, containers, and paper packages | 101 | Direct RV2V stream with localized source-background projection only around the requested 19-25 s repair interval |
| Device and cable handling | 964 frames, 40.167 s | Head-mounted device, straps, phone, cables, and connectors | 73 | Selected uninterrupted direct RV2V stream |

Both comparisons are 1280x360, 24 FPS, H.264 High, `yuv420p`, video-only, and
front-loaded for progressive browser playback. Exact hashes and presentation
metadata are recorded in
[`evaluation.json`](../demo/showcase/joyai-ego-silver-five-finger-v2/evaluation.json).
Revision 2 corrects the baked result label to `PhiAgent` in both videos and
replaces the initially promoted full-timeline tabletop restoration with the
requested localized 19-25-second refinement. Revision 1 remains archived for
provenance but is no longer linked from the demo page.

## Frozen model and recorded protocol

- Model: `jdopensource/JoyAI-Video-Edit`
- Repository revision: `3478e4b8c9a79fe935157d1d477cd3e57bb41f1f`
- Weights revision: `e14d9ac50d4ad8e9f91b655bfab270c02a43923b`
- DiT SHA-256: `b3904b6fda53d13b230918bb616f322d12cfb2337b0e8d9dc203cdabc36605ba`
- Inference: 1248x720, 24 FPS, two steps, eight-frame causal chunks
- Seeds: `17, 42, 73, 101`, with no framewise seed mixing
- Corrected publication protocol: [`configs/joyai/ego_silver_five_finger_rv2v_v2.json`](../configs/joyai/ego_silver_five_finger_rv2v_v2.json)
- Prompts: [tabletop](../configs/joyai/ego_silver_five_finger_clip_a_prompt_v1.txt) and [device/cable](../configs/joyai/ego_silver_five_finger_clip_b_prompt_v1.txt)
- Exact source-aligned reference images: [tabletop](../demo/showcase/joyai-ego-silver-five-finger-v2/tabletop-object-arrangement-reference.png) and [device/cable](../demo/showcase/joyai-ego-silver-five-finger-v2/device-cable-handling-reference.png)

The configuration is a record of the completed run, not a claim that this exact
file was frozen before the original inference. The prompts, source hashes,
reference hashes, model revisions, seeds, and measured timings were preserved by
the experiment artifacts and are reported without absolute internal paths.

## Direct RV2V generation

Use the same full-stream tools as the published flower case. Prepare the source
with [`scripts/prepare_joyai_full_stream.py`](../scripts/prepare_joyai_full_stream.py),
launch the pinned two-GPU service with
[`scripts/launch_joyai_video_edit_server.py`](../scripts/launch_joyai_video_edit_server.py),
and run each seed with
[`scripts/run_joyai_video_edit_client.py`](../scripts/run_joyai_video_edit_client.py).
The tabletop source has 862 normalized frames and is padded to 865 model frames;
the device source has 964 normalized frames and is padded to 969. Remove the
three or five cloned tail frames and restore the 1280x720 source width with
[`scripts/finalize_joyai_full_stream.py`](../scripts/finalize_joyai_full_stream.py).

Each candidate is one uninterrupted causal stream. The selector never mixes
different seeds frame by frame. Four full candidates were generated for each
clip and reviewed at matched timestamps. The recorded selection used exact
timeline validation, two-second contact sheets, four-seed comparisons,
upper-frame source-preservation PSNR, and upper-frame temporal differences.
Tabletop seed 101 and device seed 73 were selected.

## Disclosed localized tabletop refinement

The direct tabletop RV2V result preserved the manipulation but softened and
rewrote some non-robot scene detail in the user-selected 19-25-second interval.
The corrected published tabletop right panel uses the direct seed-101 stream
outside that interval and adds a deterministic localized source-state projection:

1. SAM2 tracks the human arms in the normalized source and the robot arms in the
   selected seed-101 result over frames 444-612, or 18.5-25.5 seconds.
2. The per-frame union of both tracks is dilated by five pixels and feathered
   with sigma 1.5.
3. Frames 456-599, corresponding to the requested 19-25-second core, use full
   localized refinement. The 12 frames on each side provide a gradual transition.
4. Robot-support pixels remain exactly those of the selected RV2V candidate;
   safe pixels come from the corresponding source frame at the same timestamp.
5. Frames outside 444-612 and the two transition endpoints remain pixel-exact to
   the direct seed-101 candidate. No static clean plate or frozen frame is used.

The 862-frame localized FFV1 audit measured zero maximum decoded difference
outside the transition window, at both transition endpoints, in the protected
robot union, and in the core safe-source region. Safe-region mean absolute error
from the source decreased from 8.6296 to 0.0 in the core, and all four boundary
transitions remained below the 2x spike gate. These are camera-frame checks, not
physical evidence, and native-resolution visual review remains authoritative at
arm/object boundaries.

The device/cable result does not use this source-pixel projection and remains the
selected direct seed-73 RV2V stream.

## Measured timing

The recorded service used two NVIDIA A800 80GB GPUs, with the main model and VAE
on separate devices. The service cold start took 193.691 seconds.

| Case | Pure model edit | Effective model rate | Selected client wall time | Warm direct file pipeline |
|---|---:|---:|---:|---:|
| Tabletop | 113.316 s | 7.634 FPS | 139.37 s | 275.67 s |
| Device/cable | 128.609 s | 7.534 FPS | 155.59 s | 292.34 s |

The localized tabletop refinement's recorded primary compute stages took 298.76
seconds: 14.52 seconds for lossless source normalization, 63.05 and 64.98 seconds
for source and candidate SAM2 propagation, and 156.21 seconds for dual-format
composition and validation. Generating all eight full RV2V candidates serially
took 1180.06 seconds. Mask review, reporting, and file transfer are not included
in the direct pipeline timings.

At roughly 7.5-7.6 model FPS, this configuration is about 3.1-3.2 times slower
than a 24 FPS source before file preparation and final encoding. A resident
service removes cold-start cost but does not make the model realtime.

## Honest scope

The result remains `PARTIAL`. It demonstrates long, continuous, first-person
visual replacement in two scenes, not robot execution. The review sampled the
complete timeline at fixed intervals but does not assert that every generated
finger and contact pixel passed manual inspection. Thin cables and finger
occlusions in the second clip remain the highest-risk regions. The videos provide
no metric depth, robot-base trajectory, joint feasibility, collision safety,
force closure, sensor contact, or real-robot evidence.
