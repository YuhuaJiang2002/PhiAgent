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

## Conclusion and next gate

The method is structurally appropriate, but neither the smoke adapter nor the
96-step rank-8 adapter and union mask bring the original flower video close to
the cabbage demo. The 27.5-second generation was deliberately not launched
after both critical-window failures.

The next attempt should first obtain persistent per-stem instance tracks and
real paired robot/flower supervision spanning approach, grasp, manipulation,
release, and occlusion transitions. A replacement model must then pass the same
real critical window with complete human removal, two coherent hands, visible
causal stem contact, and retained flower identities. Relighting should be
applied only after geometry/contact passes; full-length phase stitching should
start only after that window gate succeeds.
