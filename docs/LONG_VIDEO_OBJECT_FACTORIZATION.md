# Object-factored generation for videos longer than 20 seconds

Evidence date: 2026-08-12. Overall video status: **PARTIAL**. Source flower and
background preservation status: **WORKING** for the declared 27.5-second input.

## Failure model from first principles

The requested task is replacement, not unconstrained video generation. At time
`t`, the source already measures background `B_t`, flower state `F_t`, source
person support `H_t`, and their camera motion. The generator only needs to
estimate the replacement robot layer `R_t` and its visibility. Regenerating
`B_t` and `F_t` makes a stochastic model estimate variables that are already
known, then rolling-reference inference feeds those errors back into later
windows.

The output is therefore defined as a layered state equation:

```text
Y_t = R_t                         on edit support E_t
Y_t = source_t                    outside E_t
Y_t = source_t                    on visible flower layer V_t
V_t = flower_track_t - person_core_t - source_skin_occluder_t
```

This separates two error classes that one temporal score cannot distinguish:

1. robot-generation error, which remains model-dependent; and
2. known-object/background error, which can be eliminated by construction.

It also makes depth order explicit. When flower and source-person masks conflict,
the generated robot wins in the eroded person core, while flower pixels remain
eligible at object/person boundaries. A tracked source-skin negative prevents a
contaminated flower-union mask from copying a source face or hand onto the robot.

## Literature basis

- [Vera](https://arxiv.org/abs/2606.23610) identifies full-pixel regeneration as
  the content-preservation failure and generates an edit layer plus alpha matte.
  The implemented route applies the same preservation-by-construction principle
  to the existing Wan result.
- [Rethinking Temporal Consistency in Video Object-Centric Learning](https://arxiv.org/abs/2605.03650)
  treats identity as deterministic correspondence instead of an expensive learned
  predictor. The implementation consumes measured person, hand, and flower tracks
  in named camera-pixel frames rather than asking the diffusion history to remember
  the flowers.
- [Rolling Forcing](https://arxiv.org/abs/2509.25161) attributes long-rollout
  degradation to error accumulation and proposes joint multi-frame denoising,
  initial-frame attention sinks, and training on self-generated histories. These
  are the next model-training requirements for the remaining robot layer; a raw
  last-frame rolling reference is not equivalent.
- [History-Guided Video Diffusion](https://arxiv.org/abs/2502.06764) supports
  variable-length histories and shows that history guidance can stabilize very
  long rollouts. This motivates a 5--30 second robot-layer curriculum rather than
  only fixed 81-frame windows.
- [SAM2Long](https://arxiv.org/abs/2410.16268) maintains multiple segmentation
  paths through a memory tree for long-video object segmentation. The current
  precomputed tracks are sufficient for this input, while a production service
  should replace single-path union tracking with uncertainty-aware alternatives.
- The official [VACE](https://github.com/ali-vilab/VACE) interface supports masked
  video-to-video editing. It is a suitable regional generator adapter, but its
  short-clip operating route still requires the same long-history and layer-state
  controls; masking alone does not solve robot drift.

## Implemented path

`scripts/build_object_factored_long_video.py` performs a deterministic CPU pass:

1. decode the 1280x720 source into `camera:wan_output_624x352` using the recorded
   Wan resize/crop;
2. remap 660 source-person, source-hand, and source-flower masks from
   `camera:source_aligned_832x480` through an explicit shared 1280x720 camera
   frame;
3. copy generated pixels only on the dilated per-frame source-person support;
4. project accepted source-visible flower pixels last after person-core and skin
   occlusion conflict resolution;
5. write a review H.264 and a lossless FFV1 artifact; and
6. decode both completed artifacts and repeat the preservation and temporal gates.

The coordinate transform and layer-order rules are unit tested in
`tests/test_object_factored_long_video.py`. Importing `phiagent` still does not
import NumPy, OpenCV, CUDA, or a model runtime.

## Full 27.5-second result

Authoritative experiment:
`outputs/wan-long-object-factored/20260812T151000Z-source-state-projection-v6`.

| Measurement | Original generated candidate | Object-factored v6 |
| --- | ---: | ---: |
| Frames / duration | 660 / 27.5 s | 660 / 27.5 s |
| Flower MAD on accepted visible layer | 26.0711 | 0.0 pre-encode and lossless decode |
| Flower temporal residual MAD | 3.65067 | 0.0 pre-encode and lossless decode |
| Known-source exact fraction | not invariant | 1.0 pre-encode and lossless decode |
| Flower/person-core overlap | not controlled | 0 pixels |
| Flower/skin-negative overlap | not controlled | 0 pixels |
| Projection speed | n/a | 40.0840 FPS, 16.4654 s |
| Projection plus two complete decode audits | n/a | 14.2681 FPS, 46.2571 s |

The review MP4 is deterministically identical to the visually reviewed v5 MP4
(SHA-256 `caa1eae8...`). A 30-frame dense sheet, 12 full-resolution samples,
the seven-frame source-flower-track gap neighborhood, and the main contact window
show stable source flowers/background and no repeated v3/v4 source-face leakage.

## Honest boundary and next model work

The flower/background problem is solved for this source-conditioned replacement
because those states are no longer generated. The complete video is still
**PARTIAL**: late color/structure artifacts already present inside the Wan robot
layer remain, and exact hand--stem contact, per-stem identity through all
occlusions, 3-D physics, and real-robot execution are not accepted.

The next learned model must generate `robot RGB + alpha`, not a full frame; retain
a canonical identity attention sink; train on its own robot-layer histories;
accept variable history lengths from 5 to at least 30 seconds; and expose explicit
object visibility/contact state. A new 20-second-or-longer real-input run must
pass robot topology, identity, action/contact, and object-track gates before that
remaining capability can be marked WORKING.

