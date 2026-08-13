# Long-video robot-layer and contact contract

Evidence date: 2026-08-12. This document reports a real 27.5-second input,
not a synthetic unit-test result. The complete video remains **PARTIAL**.

## First-principles diagnosis

A source-conditioned human-to-robot video contains three different state classes:

```text
Y_t = source scene/object state S_t outside edit support
    + robot appearance and articulated geometry R_t inside edit support
    + interaction/contact state C_t at the robot-object boundary.
```

Regenerating all three classes makes known source pixels stochastic and makes every
rolling window condition on errors from the preceding window. A last-frame anchor
reduces resets, but it is also an error channel: color bias, malformed fingers, or
wrong contact become part of the next model input. Long duration is therefore not
primarily a stitching problem. It is recurrent distribution shift plus an
under-specified state representation.

The implemented contract makes the responsibilities explicit:

- source flowers and background are measured state and are copied by construction;
- the learned target is a robot RGB edit layer plus alpha, not a complete frame;
- a canonical reference is an identity anchor, not a rolling replacement for memory;
- the control video has fixed semantics: red is robot support/alpha, green is the
  protected-object boundary, and blue is required projected contact; and
- appearance, articulated support, and contact are gated separately after 20 seconds.

Every mask and control names its camera-pixel frame. The image-space contact gate
only establishes 2-D adjacency. It does not establish depth order, force closure,
collision safety, or executable robot contact.

## Primary literature used

- [Vera](https://arxiv.org/abs/2606.23610) identifies full-pixel regeneration as
  the source of unintended edits and predicts an edit layer with alpha. This is the
  basis for removing known flowers/background from the generator's state budget.
- [Rolling Forcing](https://arxiv.org/abs/2509.25161) treats long-video failure as
  accumulated self-history error and uses joint multi-frame denoising, an identity
  attention sink, and training on generated histories. A rolling last frame alone
  does not implement these mechanisms.
- [Rethinking Temporal Consistency in Video Object-Centric Learning](https://arxiv.org/abs/2605.03650)
  motivates deterministic grounded correspondence rather than asking a temporal
  predictor to rediscover object identity. The present route consumes tracked
  person, limb, hand, and flower support directly.
- [Beyond Consistency](https://arxiv.org/abs/2606.08780) decomposes long videos into
  semantic clips linked by anchors. The curriculum uses short clips with immutable
  state semantics and reserves late clips for validation.
- [WorldTrace](https://arxiv.org/abs/2608.07408) reports that extrapolating temporal
  positions beyond the trained horizon is out of distribution and introduces
  addressable memory. The current Wan runtime does not expose a compatible KV-cache
  API, so this work implements external addressed tracks/anchors and does **not**
  claim native WorldTrace memory integration.
- [GenHOI](https://arxiv.org/abs/2606.12995) turns contact events and regions into
  object-centric constraints; [Dream2Act](https://arxiv.org/abs/2603.19709) argues
  for morphology-consistent robot contact instead of direct human retargeting; and
  [GenVid2Robot](https://arxiv.org/abs/2607.09191) supports closing the loop from
  generated video to robot-grounded supervision.
- [TOC-Bench](https://arxiv.org/abs/2605.09904) evaluates identity, state, occlusion,
  and interaction on object tracks. This motivates the object-track-grounded audit
  rather than a global temporal score.

## Real 27.5-second repair

The failing object-factored v6 candidate is rejected after thresholds are fit only
on frames 259--296; no frame at or after frame 480 is allowed to fit its own gate.
After 20 seconds, v6 violates the high-chroma gate on 81.67% of frames, violates
the canonical-palette gate on 100%, and reaches only 11/12 required projected
contacts.

The replacement route projects the more stable legacy robot layer onto the exact
source flower/background state, then fills only missing hand pixels from the second
generated hypothesis. Flower pixels are protected. The bounded union changes
23,243 pixels in 51/660 frames, at most 727 pixels in one frame.

| Late, frame >= 480 | v6 | repaired candidate |
| --- | ---: | ---: |
| high-chroma gate violation | 81.67% | 0% |
| canonical-palette violation | 100% | 0% |
| projected 2-D contact | 11/12 | 12/12 |
| hand-support violation | not selected | 1.11% |
| flower/background known-state error | 0 lossless | 0 lossless |

The final automatic audit passes 11 scoped image-space gates. Its four attacks are
also detected: magenta material shift, arm erasure, contact detachment, and a
shifted/ghost hand. This is stronger evidence than an unperturbed score, but it is
not sufficient evidence for a coherent articulated hand.

High-resolution human review still finds intermittent finger morph or motion smear,
and exact finger--stem depth order is ambiguous. Consequently, the complete video
is **PARTIAL**. Only late material stability, gross robot topology, exact source
flower/background preservation, and declared projected 2-D contact are accepted.

## Throughput

On the measured CPU route, source-state projection takes 15.1542 seconds
(43.5522 FPS) and the bounded hand union takes 5.5440 seconds (119.0485 FPS).
Together they process all 660 frames in 20.6982 seconds, or **31.8869 FPS** and
0.7527x real time. Including the complete four-attack audit takes 51.2139 seconds,
or **12.8871 FPS** and 1.8623x real time.

This is post-generation repair/evaluation throughput. It must not be confused with
raw Wan sampling: the previously accepted four-A800 generation infrastructure
produces 660 useful frames in 509.7154 seconds, or **1.29484 useful FPS**.

## Attempted model internalization

The curriculum contains twelve pre-20-second training clips and four temporally
held-out clips at or after frame 480. Each sample is 17 frames at 448x256 and 8 FPS
with the same RGB-alpha-contact semantics. An A800 physical GPU 5 completed a
rank-8 VACE LoRA run: learning rate 2e-5, four epochs, dataset repeat two, seed
20260812, and 96 optimizer steps in 524.18 seconds. The epoch-3 checkpoint SHA-256
is `2927fb3e...fcc01`.

The blind comparison uses frames 594--642 (24.75--26.75 seconds), with identical
input, reference, prompt, seed, 20 steps, and denoising strength. Only LoRA loading
changes.

| Held-out proxy | zero-shot | rank-8 LoRA |
| --- | ---: | ---: |
| contact similarity to teacher | 0.3576 | 0.4250 |
| topology-edge similarity | 0.4362 | 0.4581 |
| edit-region similarity | 0.1275 | 0.1584 |
| temporal similarity | 0.7038 | 0.6792 |
| source similarity outside edit region | 0.9245 | 0.9249 |

The adapter changes the output and improves three local proxies, but it fails the
minimum edit-region similarity, minimum topology similarity, and temporal
non-regression gates. Human review also shows wrong robot scale/identity and
invented oversized flowers. Full 27.5-second model-only expansion is therefore
stopped. This is **PARTIAL** same-scene adaptation, not a general learned capability.

Generalization requires multiple scenes, robot morphologies, manipulation actions,
occlusion patterns, and contact objects; contact/depth or force supervision; training
on self-generated histories out to at least 30 seconds; and a native addressable
memory/identity-sink path. None of those claims is promoted by this experiment.

## Reproducibility artifacts

- Contract and metrics: `phiagent/rendering/robot_layer_contract.py`
- Full-video audit: `scripts/audit_robot_layer_long_video.py`
- Bounded repair: `scripts/repair_robot_hand_layer.py`
- Curriculum: `scripts/build_robot_layer_contact_curriculum.py`
- Held-out comparison: `scripts/evaluate_robot_layer_contact_lora.py`
- Experiment root:
  `outputs/wan-long-robot-contract/20260812T073750Z-first-principles-v1`
- Review video:
  `hand-union-repair-v1/robot-hand-union-27p5s.mp4`
- Adapter audit:
  `heldout-adapter-evaluation-v1/evaluation.json`

