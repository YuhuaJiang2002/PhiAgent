# Long-video perceptual VFM harness

Evidence date: 2026-08-12.

## Scope decision

The current product target is **perceptually plausible synthetic video data**.
For that target, calibrated RGB-D, exact robot `q/qdot`, a URDF-bound trajectory,
and measured or solver contact force are not mandatory inputs. They remain
mandatory only for metric depth, force closure, collision safety, or real-robot
execution claims. A model-generated depth/force/telemetry channel may condition
generation or diagnose failures, but it is never relabelled as a measurement.

The display contract therefore uses visible hard gates:

- at least 20 seconds and a complete decodable timeline;
- source background and flower pixels locked outside the edit layer;
- non-frozen source flower response motion;
- no visible human residue;
- stable robot identity and wrist/hand topology;
- no intermittent smear or detached hand;
- explicit adversarial attack detection; and
- high-resolution human review as a veto.

An average score cannot override any failed gate.

## Foundation-model allocation

The harness assigns models by failure mode instead of asking one model to own
the full 27.5-second state:

| Responsibility | Model or deterministic layer |
| --- | --- |
| fast full-timeline robot proposal | persistent multi-GPU Wan Animate 2 path |
| high-risk hand/contact windows | official Wan2.2-Animate-14B replacement |
| bounded masked-edit challenger | VACE; currently rejected on this case |
| semantic failure-window proposals | Qwen3-VL 4B/8B; no promotion authority |
| future long-continuation challengers | LongCat-Video, SkyReels-V2, MAGI-1 |
| flowers/background/response motion | immutable source-state layers |

The agentic loop is:

```text
fast incumbent
  -> mine late/high-resolution failures
  -> generate overlapping official-14B windows
  -> reject malformed windows
  -> route reviewed windows only inside person support
  -> restore source flowers/background
  -> audit full timeline, seams, hands, attacks, and cost
  -> DISPLAY_READY or PARTIAL
```

This avoids recurrently feeding a complete generated frame back into the next
window. Long-horizon state that is already observed (flowers, background,
camera timing) is not regenerated. Only the uncertain robot edit layer is
sampled again.

## Implemented infrastructure

- `phiagent/agent/perceptual_video_harness.py`: fail-closed display promotion.
- `phiagent/rendering/wan_animate.py`: pinned official Wan adapter, GPU lease,
  provenance, face-control suppression, and uncompressed PPM reference
  transport that avoids upstream libpng/zlib failures.
- `scripts/build_routed_vfm_long_demo.py`: overlapping reviewed-window router,
  source flower/background locks, lossless artifact, independent post-encode
  lock audit, seam metrics, and high-resolution review sheets.
- `scripts/audit_robot_layer_long_video.py`: frozen anchor-derived image-space
  gates and color/structure/contact/topology attacks.

The generated demo is still synthetic visual evidence. It does not establish
metric contact, force closure, or executable robot behavior.

## Accepted scoped demo

The v4 route is `DISPLAY_READY` for the synthetic-display contract. It combines
the persistent full-timeline incumbent with reviewed official Wan2.2-Animate
windows, chooses hard switches by a deterministic overlap cost, and restores
the tracked source flowers after every edit. The 1280x720 artifact contains
660 frames at 24 FPS (27.5 seconds).

Measured evidence:

- compositor: 54.6897 seconds, 12.0681 FPS;
- post-decode flower exactness: 1.0;
- post-decode native-background exactness: 0.992429;
- source-flower dynamic transition fraction: 0.980061;
- 20-second-and-later projected contact recall: 1.0; and
- all four adversarial attack families detected across 585 sampled/contact
  frames.

The older all-in-one RGB-alpha gate remains `PARTIAL`: its anchor-derived
source-hand replacement coverage limit is violated on 23.33% of late frames.
That proxy is retained as evidence and is not silently relaxed. The scoped
display decision instead requires explicit high-resolution hand/topology review
and a separate no-human-residue gate. Two one-frame identity route changes are
visible under frame stepping but do not synthesize double or melted anatomy;
the tested optical-flow bridge was rejected because it did.

The measured official 14B tail and bridge runs took 511.5277 and 476.9168
seconds for 88 raw frames (0.1720 and 0.1845 FPS). The persistent full proposal
took 509.7154 seconds for 660 frames (1.29484 FPS). The final dual-encode
compositor took 54.6897 seconds. Running those three independent model paths
serially corresponds to 1552.8495 seconds (0.4250 delivered FPS). Scheduling
them concurrently on disjoint leased GPUs has a measured-component critical
path estimate of 566.2173 seconds (1.1656 delivered FPS); this is a scheduling
projection, not a separately timed end-to-end run. Total measured model compute
is 0.84092 A800 GPU-hours, 27.74% below the old per-window-reload baseline but
48.48% above the fast incumbent alone.
