# Flower task adaptation experiment

Evidence date: 2026-08-11. Overall status: **PARTIAL**.

## Objective

Test whether task adaptation, explicit bimanual contact/flower-instance contracts,
phase-wise generation, and immutable-candidate preservation are enough to raise
the original 27.5-second flower-arranging replacement toward the short cabbage
demo's visual quality. A short real critical window is required to pass before
any full-length expansion.

## Implemented route

1. `compile_flower_task_contract.py` compiles all 660 source frames into named
   left/right phases, contact constraints, immutable generation windows, and
   explicit `camera:source_pixels`, `robot:base`, and `object:flower` frames.
2. `generate_flower_task_vace_dataset.py` creates a small paired VACE task
   dataset in which the left hand holds a bouquet while the right hand performs
   approach, grasp, manipulate, release, and retract phases. The active flower
   is static before grasp, rigidly attached during contact, and static after
   release.
3. `train_sharpa_vace_lora.py` verifies the physical GPU, pinned DiffSynth
   revision, four VACE checkpoint hashes, and dataset manifest before training.
4. `prepare_real_flower_vace_window.py` selects source frames 272--320, projects
   the contact-calibrated robot control into the camera frame, protects the
   existing flower union, and permits editing only in a 13.29--14.01% regional
   mask.
5. `run_sharpa_vace_inference.py` generates matched zero-shot and task-LoRA
   candidates with the same prompt and seed, then restores every pixel outside
   the edit mask from the original candidate before encoding.
6. `evaluate_real_flower_task_window.py` combines preservation/motion proxies
   with mandatory semantic storyboard checks. Frame-level candidate mixing is
   forbidden.

## Evidence

The real 660-frame contract found seven right-hand/flower-union proximity
intervals: `[0,10]`, `[47,81]`, `[142,170]`, `[236,257]`, `[272,448]`,
`[499,631]`, and `[650,656]`. It is not claim-ready: the available SAM2 signal
is a flower union with seven empty frames, not persistent per-stem instances;
contact and occlusion depth remain proxy evidence.

The remote task-LoRA smoke train completed 12/12 steps on physical A800 GPU 1
and produced a 5,503,040-byte rank-4 checkpoint with SHA-256
`6f1e7fec93f761278298630745383f6a0508daf1dd703c4aef7eb0bc8995ae00`.
This establishes trainability only. On a matched synthetic held-out clip, the
LoRA changed contact-ROI similarity from 0.13191 to 0.13444 and contact-motion
similarity from 0.55595 to 0.55911. The gains are positive but very small, and
absolute image quality is low.

The mandatory real critical-window test used 17 frames at 448x256 and 8 FPS,
20 steps, denoising strength 0.75, and seed 20260811. The comparison and
evaluation are under
`outputs/flower-task-adaptation/20260811T026000Z-real-window-ablation-v1`.

| Real-window metric | Zero-shot | Task LoRA |
| --- | ---: | ---: |
| Outside-edit similarity to input | 0.92222 | 0.92227 |
| Control-motion alignment in edit region | 0.35999 | 0.35885 |
| Mean change inside edit region | 0.17368 | 0.17163 |

The LoRA regresses control-motion alignment by 0.00114 and differs from
zero-shot by only 1.8666 RGB levels on average. More importantly, the uniform
storyboard shows the human head and torso in both candidates, fragmented robot
geometry, no two coherent robot hands, and no auditable hand--stem contact.
Persistent flower identity also cannot be established from the union mask. All
four semantic gates fail, so `evaluation.json` records
`REJECT_FULL_EXPANSION`.

### Rank-8 bounded optimization rerun

A second bounded train ran on physical A800 GPU 4 with rank 8, learning rate
`5e-5`, four epochs, dataset repeat 2, 17 frames, and 448x256 resolution. It
completed 96 optimization steps and wrote four 10,971,312-byte checkpoints. The
final `epoch-3.safetensors` SHA-256 is
`7d5bae3977b0482a4aeb61ed335b706b0bda714efbdf24a55388566ee705deaa`.
The completed training run is under
`outputs/flower-task-adaptation/20260811T165000Z-vace-lora-rank8-e4-v4`.

The same real window, zero-shot candidate, prompt, seed, regional mask, control,
20 inference steps, and denoising strength were then reused for a final-checkpoint
comparison under
`outputs/flower-task-adaptation/20260811T171000Z-real-window-rank8-ablation-v2`.

| Rank-8 real-window metric | Zero-shot | Rank-8 task LoRA |
| --- | ---: | ---: |
| Outside-edit similarity to input | 0.92105 | 0.92148 |
| Control-motion alignment in edit region | 0.35681 | 0.34569 |
| Mean change inside edit region | 0.17360 | 0.14046 |

The stronger adapter improves the background-preservation proxy by `0.00043`
but regresses control-motion alignment by `0.01112`. Uniform review of frames
0, 4, 8, 12, and 16 still finds the human head and torso, fragmented robot
geometry, no two coherent robot hands, no causal stem contact, and no persistent
flower-identity evidence. It therefore remains `PARTIAL` with 0/4 semantic gates
and again records `REJECT_FULL_EXPANSION`.

## Explicit instance/contact supervision result

The failed generative route was superseded for the critical geometry gate by
explicit SAM2 instances and exact candidate composition. The accepted critical
window contains source frames 272, 275, ..., 320. `active-pink-stem-01` has no
empty masks, adjacent IoU 0.6652--0.9037, and a maximum centroid step of 5.66
pixels. Two independently prompted H3 hand instances also have no empty masks;
their minimum adjacent IoUs are 0.8599 and 0.8116.

After five retained partial compositions, v6 under
`outputs/flower-contact-supervision/20260811T038000Z-real-contact-pairs-v6`
passes all automatic and dense human gates:

| Critical-window gate | Evidence | Result |
| --- | --- | --- |
| Complete human removal | largest retained audited component at most 54 pixels; storyboard review | WORKING |
| Two mechanical hands | minimum prompted hand areas 1,768 and 847 pixels | WORKING |
| Clear stem contact | maximum active hand--stem distance 1.40 pixels | WORKING |
| Flower identity retained | exact active-flower preservation is 1.0 before encode | WORKING |

`geometry-gate-evaluation.json` therefore records `ALLOW_RELIGHTING_WINDOW`.
The result is scoped to this real input window; it is not a claim about the
full film.

## Continuous single-flower expansion

The first full-range stem probe over source frames 272--447 produced 176/176
masks, but dense review and boundary metrics reject a single identity across the
whole interval. The same pink carnation is observable only through source frame
377. At 377--378 the centroid jumps 19.10 pixels and adjacent IoU falls to
0.4105; later discontinuities correspond to release, occlusion, and a scissors
subaction. The old union-proxy phases 023/024 are therefore split into:

- phase 04A, `[272,378)`: continuous `active-pink-stem-04`, accepted;
- phase 04B, `[378,398)`: release/occlusion transition, flower identity not
  observable and never propagated;
- phase 04C, `[398,448)`: scissors trimming, requiring fresh tool and gripper
  supervision.

Phase 04A under
`outputs/flower-full-expansion/20260811T051000Z-phase04a-contact-v3` covers all
106 consecutive frames. Its maximum active contact distance is 3.0 pixels,
maximum support distance is 12.98 pixels under a separate 14-pixel support
gate, both hand tracks remain nonempty, and exact active-flower preservation is
1.0. Dense review of every frame passes complete human removal, two robot hands,
clear stem contact, and flower identity. The geometry evaluator records
`ALLOW_PHASE_04A_RELIGHTING`.

## Geometry-gated relighting

Direct official Wan LoRA generations were retained as rejected proposals because
they regenerate flower and scene geometry. On phase 04A, two 53-frame proposals
score only 0.3680 and 0.3333 object consistency. The accepted route uses only
their bounded low-frequency luminance proposal inside a robot-safe interior and
copies flower, hands, contact, table, and all outside pixels exactly before
lossless encode.

The final 106-frame artifact is under
`outputs/flower-relighting/20260811T054000Z-phase04a-confidence-routed-v2`.
All automatic and dense human gates pass. Flower, prompted-hand,
protected-contact, and outside-robot exact fractions are all 1.0 before encode;
the LoRA luminance error falls from 25.5780 to 25.1384. The maximum temporal
relighting residual is 1.4736 RGB MAE at the join from source frame 324 to 325,
below the 1.5 gate and explicitly reviewed with amplified deltas. This is a
subtle, geometry-preserving relight, not a 3-D illumination reconstruction.

## Full-film boundary

Fresh phase-04C prompts do not pass. The two robot-hand masks are numerically
smooth over 50/50 frames, but all-frame review shows that one tracks flower
material and the other a wrist/forearm patch rather than two distinct grippers.
The scissors probe is empty for 30/50 frames, and its nonempty masks select the
source hand/forearm instead of the tool. Phase 04C therefore remains `PARTIAL`
with `BLOCK_RELIGHTING_REQUIRE_CORRECT_TOOL_AND_TWO_GRIPPER_SUPERVISION`.
No relighting is applied across phases 04B or 04C, and no full-film persistent
flower/contact claim is made.

## Full-length shadow/EPL/temporal delivery

The separate full-length visual-delivery route under
`outputs/flower-full-delivery/20260812T001000Z-shadow-epl-temporal-v1` uses the
reviewed v71 H3+EPL candidate as its immutable parent. That parent already
passes nine shadow/background gates over all 660 frames: the neutral cleanup is
bounded to the person safety region, flower/robot content is protected, and
reviewed residual human forearms are converted to narrow source-luminance-driven
silver contours rather than replaced by a broad clean-plate patch.

The newly trained H3 topology LoRA is not used for pixels. Its held-out
assessment fails identity gain/floor, topology, and motion non-regression, so
applying it to the full film would exchange temporal appearance for visibly
wrong robot structure. Instead, the trained flower repair policy acts as a
non-regression route selector. It ranks the four full-film recipes as follows:

| Policy route | Predicted constrained utility |
| --- | ---: |
| EPL-local crossfade with background/flower lock | 240.7984 |
| EPL-local flow with background/flower lock | 216.5801 |
| No temporal repair | 72.7996 |
| Global noise smoothing | 0.1449 |

Source video and EPL-local statistics jointly isolate six unsupported generated
window jumps at frames 97, 193, 294, 405, 507, and 562. Only two-frame-radius
robot neighborhoods are cosine-bridged. Static background outside the safety
region, all full-film flower masks, and the phase-04A stem/two-hand contact masks
are restored exactly before encoding. The accepted phase-04A LoRA luminance
residual is also carried into source frames `[272,378)` with an eight-frame
taper and an 8-level per-channel bound; it is not extrapolated into unverified
phases.

All six repaired transition energies improve:

| Frame | Before | After |
| ---: | ---: | ---: |
| 97 | 9.0889 | 4.6692 |
| 193 | 6.1782 | 3.9459 |
| 294 | 11.2966 | 6.4931 |
| 405 | 8.5869 | 5.0373 |
| 507 | 9.0828 | 5.0905 |
| 562 | 9.4107 | 4.8179 |

Maximum ROI transition energy falls from 11.2966 to 7.5638. The mean modified
fraction is 1.0317%. Both the 1-FPS full timeline and consecutive frames around
every repair pass review without new crossfade ghosts, flower loss, a returning
human forearm, or a visible repaired-window jump. The compatibility candidate
SHA-256 is
`29d07167d45df13ffa20cd6a22acd52610bb3e747fbbc1f67d1052fbbc704969`.
This is `WORKING` for the complete 27.5-second 2-D shadow/background and temporal
delivery. It does not upgrade the unverified phase-04B/04C contact geometry.

## Conclusion and next gate

Explicit flower identity, hand instances, contact pairing, and phase splitting
do raise a real 106-frame section far beyond the failed VACE-only attempts: the
same segment now passes all four geometry gates and a geometry-preserving
relighting gate. This is meaningful progress toward the cabbage-demo standard,
and the 27.5-second visual delivery now passes its separate shadow/background
and temporal gate. Full-film interaction geometry remains `PARTIAL`, because
flower identity is unobservable through phase 04B and the phase-04C
scissors/two-gripper probes fail semantic review.

The next expansion must re-anchor the scissors and both grippers with
tool-specific supervision, then establish tool--bouquet contact without carrying
the phase-04A flower ID through an occlusion. Only after that geometry gate passes
may phase-04C relighting run. Other full-film action intervals require the same
independent instance/contact acceptance sequence.
