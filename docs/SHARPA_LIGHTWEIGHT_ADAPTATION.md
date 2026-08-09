# Lightweight Sharpa adaptation exploration

This track asks whether small, explicitly bounded adapters can improve the visible
human-hand-to-Sharpa result while the official PhiZero implementation and checkpoints
remain unavailable. Every result is labelled
`sharpa_lightweight_adaptation_not_official_phizero`.

## Why training is needed

The zero-shot Wan replacement baseline can anchor the first frame to a Sharpa image,
but the measured case-1 runs do not reliably preserve object contact or long-horizon
appearance. A first-frame edit supplies appearance evidence only; it does not teach the
generator stable Sharpa geometry, articulation, or object-relative motion.

Lightweight training is therefore justified, but one adapter cannot establish every
claim. The exploration separates these arms:

| Arm | Trainable data | Question |
| --- | --- | --- |
| `zero_shot` | none | How far does released replacement mode already go? |
| `appearance_lora` | Sharpa identity images only | Does identity adaptation improve geometry without motion supervision? No validated trainer is integrated yet. |
| `animate_lora` | target, pose-control, and face-control video triplets | Does the documented DiffSynth Animate LoRA path improve animation-mode transfer? |

Use the same base checkpoint, source clips, first frames, seeds, inference steps,
candidate budget, evaluator, and confirmation set in all arms. Adapter rank is a
secondary ablation; start with ranks 8 and 32 rather than reporting the best rank from
an unrestricted search.

`animate_lora` is not a contact-specific objective and does not support Wan replacement
mode. It must not be described as a motion/contact adapter without a dedicated contact
signal and ablation.

## Data boundary

- Project-owned or explicitly licensed Sharpa identity images may train the appearance
  adapter.
- Project-owned or explicitly licensed Sharpa manipulation clips may train the
  motion/contact adapter.
- The three official PhiZero transferred videos are evaluation-only. Training on them
  would invalidate the intended comparison and is rejected by the manifest schema.
- Freeze train, validation, and confirmation splits by content hash before GPU work.
- Do not use confirmation results for prompt, seed, mask, checkpoint, or rank selection.

Create a JSON spec:

```json
{
  "experiment_id": "sharpa-appearance-r8-v1",
  "arm": "appearance_lora",
  "assets": [
    {
      "asset_id": "sharpa-front-001",
      "path": "data/sharpa/front-001.png",
      "split": "train",
      "kind": "identity_image",
      "source_uri": "local://capture-session-001",
      "rights_basis": "project-owned capture"
    },
    {
      "asset_id": "hand2dex-1-reference",
      "path": "external/PhiZero-reference/hand2dex_1_transferred.mp4",
      "split": "confirmation",
      "kind": "reference_video",
      "source_uri": "https://phi-zero.github.io/",
      "rights_basis": "official public evaluation reference"
    }
  ]
}
```

Freeze the resolved paths, sizes, and hashes before training:

```bash
python scripts/prepare_sharpa_adaptation_manifest.py \
  --spec configs/sharpa-appearance-r8-v1.json \
  --output outputs/sharpa-adaptation/sharpa-appearance-r8-v1/manifest.json
```

The output directory must be new. The manifest tool refuses overwrite, duplicate
content, incompatible supervision, and training use of reference videos.

An `animate_lora` spec additionally groups each training row explicitly:

```json
{
  "experiment_id": "sharpa-animate-r32-v1",
  "arm": "animate_lora",
  "assets": [
    {"asset_id": "target-001", "path": "target.mp4", "split": "train", "kind": "target_video", "source_uri": "local://capture-001", "rights_basis": "project-owned capture"},
    {"asset_id": "pose-001", "path": "pose.mp4", "split": "train", "kind": "pose_control_video", "source_uri": "local://capture-001-pose", "rights_basis": "project-owned derivative"},
    {"asset_id": "face-001", "path": "face.mp4", "split": "train", "kind": "face_control_video", "source_uri": "local://capture-001-face", "rights_basis": "project-owned derivative"}
  ],
  "animate_examples": [
    {
      "example_id": "capture-001",
      "target_video_asset_id": "target-001",
      "pose_video_asset_id": "pose-001",
      "face_video_asset_id": "face-001",
      "prompt": "A Sharpa dexterous hand manipulates an object."
    }
  ]
}
```

## Training order

1. Run the existing zero-shot replacement baseline on all three cases.
2. Train an appearance LoRA on diverse Sharpa views, lighting, and articulation. Keep
   the video model and motion controls otherwise frozen.
3. Only if identity improves without adequate contact fidelity, train a separate
   temporal adapter on Sharpa manipulation clips. Do not infer contact learning from
   identity images.
4. Select masks, rank, and checkpoint on validation data, then run confirmation once.

The reviewed training candidate is DiffSynth-Studio commit
`b1c02ce76aabc989f6bf534756b2da84532249e5`, whose Apache-2.0 example trains a rank-32
DiT LoRA for Wan2.2-Animate-14B in animation mode. Its tested configuration uses eight
80 GiB GPUs and requires one target video, pose-control video, face-control video, and
prompt per example. The first target-video frame becomes the input identity image.

The documented entry point does not support identity-image-only or replacement-mode
training. The official Wan documentation also warns that ordinary Wan2.2 LoRAs may
behave unexpectedly with Wan-Animate. These paths remain separate until measured.

Prepare and verify the pinned trainer source:

```bash
python scripts/prepare_diffsynth_wan_animate.py
```

Run strict preflight after freezing an `animate_lora` manifest and installing
DiffSynth in an isolated GPU environment:

```bash
python scripts/train_sharpa_animate_lora.py \
  --manifest outputs/sharpa-adaptation/animate-v1/manifest.json \
  --diffsynth-repo external/DiffSynth-Studio \
  --checkpoint-dir checkpoints/Wan2.2-Animate-14B \
  --gpu 0 --gpu 1 --gpu 2 --gpu 3 --gpu 4 --gpu 5 --gpu 6 --gpu 7
```

Preflight is the default. Add `--execute` only after reviewing its immutable
experiment record and generated command. The runner verifies the DiffSynth commit and
license, the pinned Wan checkpoint revision and files, eight distinct physical GPUs,
free memory, exact Animate metadata columns, and asset hashes. It sets and records
`CUDA_VISIBLE_DEVICES` before launch.

The first local smoke dataset contains eight 640x480 frames extracted from the accepted
synthetic Sharpa MuJoCo rollout. It validates the appearance-manifest path only; one
fixed-camera synthetic clip is not adequate appearance training data and cannot be fed
to the reviewed Animate trainer without real pose/face controls.

## Acceptance

Report all four proxy dimensions separately: motion preservation, target identity,
object consistency, and temporal consistency. In addition:

- manually review object-in-hand state at fixed timestamps;
- report mask coverage and visible retained-human-hand pixels;
- report adapter parameters, training samples, steps, GPU hours, and inference cost;
- include all three cases and fixed seeds;
- retain rejected checkpoints and negative results.

An appearance-only improvement supports only an identity claim. A proxy video does not
establish physical robot execution or exact PhiZero reproduction.
