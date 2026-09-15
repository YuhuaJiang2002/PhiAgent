# PlenopticDreamer V1: camera-conditioned DiT for ego-to-exo video

This directory is an independent V1 implementation. It does not replace or
modify the existing geometry/H3 pipeline in the parent directory.

**Status: PARTIAL.** The source package contains the training, checkpoint,
inference, data, captioning, context-parallel, and verification paths used by
the PlenopticDreamer stage-one recipe. The final 16,000-step reproduction and
its four-case fixed validation completed on H20 GPUs, and the exact final
source overlay, checkpoint identity, and lightweight evidence are hash-bound
under `releases/step-016000`. The weights themselves are not included, and the
mixed calibrated-view results do not accept arbitrary ego-to-exo generation.

## What V1 implements

V1 fine-tunes the public 2B Cosmos camera DiT to predict a calibrated target
view from one to four synchronized source views. During training, source and
target videos are encoded by the frozen VAE. The target latent occupies the
middle view slot, is hidden by the conditioning mask, and supplies the
rectified-flow target. At inference, target RGB is zero and only source videos,
camera calibration, and text are supplied.

The main model implementation is
[`tools/plenoptic_model.py`](tools/plenoptic_model.py):

- `SpatialCameraDIT` keeps every temporal latent and shards the patch-grid width
  across a node-local context-parallel group.
- Each rank slices the exact global 3-D RoPE and camera-ray grid for its spatial
  shard. Sparse attention and per-block absolute embeddings are rejected.
- Only DiT self-attention and camera-encoder parameters are trainable; the base
  model, cross-attention, VAE, and text encoder remain frozen.
- The progressive schedule trains `k=1` through `k=4` without repeating source
  views to imitate a larger context.

This is a calibrated novel-view generator. A real ego video still needs a
synchronized source-camera model and a chosen target-camera trajectory. V1 does
not infer metric calibration or an exo trajectory from arbitrary RGB alone.

## Contents

- `tools/train_plenoptic.py`: stage-one rectified-flow training, progressive
  context, online captions, gradient synchronization, and exact resume.
- `tools/infer_stage1.py`: ego/source-to-exo target generation with target RGB
  withheld.
- `tools/plenoptic_data.py`: SynCam/MultiCam video and camera geometry.
- `tools/plenoptic_distributed.py`: node-local spatial CP and cross-node DP.
- `tools/prepare_plenoptic.py`, `tools/download_weight_dataset.py`, and
  `tools/prepare_datasets.py`: pinned asset download, verification, extraction,
  and scene manifests.
- `tools/launch.py`: physical-GPU selection and per-run provenance capture.
- `configs/basic_stage1_24gpu.json`: the V1 24-H20 recipe.

Model repositories, weights, datasets, captions, caches, and outputs stay in
ignored directories and are never committed.

## Final reproduction snapshot

[`releases/step-016000`](releases/step-016000) is the immutable historical
package for the completed run. It contains the 36-file custom source overlay,
four exact PhiAgent integration files, 15 training/evaluation evidence files,
six real-input integration evidence files, a 61-file SHA-256 inventory, the
exact upstream Cosmos commit and source-archive hash, the resolved 24-GPU
recipe, and the final checkpoint and inference-snapshot identities. It also
preserves the fixed-suite runner and reporting tools used for the final
evaluation.

The final resume checkpoint is 6,703,192,745 bytes with SHA-256
`f7ad29d7e91e1e9674335b2368b079f63c8051e9d97a122b02bce97cac93d6b3`.
It is external to this repository. Verify the package, and optionally a
separately transferred checkpoint, using only the standard library:

```bash
python tools/verify_final_release.py
python tools/verify_final_release.py --checkpoint /path/to/step-016000.pt
```

The release binding is retrospective: the training checkpoint sidecar did not
contain a source-tree digest. The package therefore preserves the exact retained
final workspace and binds it at release time; it does not claim an impossible
in-checkpoint build attestation after the fact.

The captured custom overlay must be applied on top of the pinned public Cosmos
Transfer 2.5 commit. Pulling the official repository recreates the base, but
does not recreate the reproduction-specific model, trainer, progressive
context, spatial-CP, or validation changes by itself.

## Pinned dependencies and assets

All immutable revisions are also recorded in [`third_party.json`](third_party.json).
Prepare the upstream source inside this directory:

```bash
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
git -C cosmos-transfer2.5 checkout 2ff49d0717af02057ae79bc75c00fbff9da1b4e7
cd cosmos-transfer2.5
uv sync --extra=cu128
source .venv/bin/activate
cd ..
```

The pinned model and dataset revisions are:

| Asset | Revision |
| --- | --- |
| `nvidia/Cosmos-Predict2.5-2B` base | `15a82a2ec231bc318692aa0456a36537c806e7d4` |
| `nvidia/Cosmos-Predict2.5-2B` VAE | `f176dc95b4a70f53ce01c4b302851595e7322b00` |
| `nvidia/Cosmos-Reason1-7B` | `3210bec0495fdc7a8d3dbb8d58da5711eab4b423` |
| `KlingTeam/SynCamVideo-Dataset` | `74e4fcaf4f20981ed67c2c129a50f01fb600e5c4` |
| `KlingTeam/MultiCamVideo-Dataset` | `4e7d995480cfa915f7c3721e616ed14789d87ff7` |

Create the lightweight download environment, store a Hugging Face read token
interactively if the gated base requires it, and start the resumable verifier:

```bash
python3 tools/prepare_plenoptic.py setup
python3 tools/prepare_plenoptic.py login
python3 tools/download_weight_dataset.py start all
python3 tools/download_weight_dataset.py status
```

The downloader fixes every repository revision, verifies the returned file
metadata and complete content hash, extracts the two datasets, and writes scene
manifests. Keep the upstream training environment active for model work.

## Training

`tools/launch.py` is the supported GPU entry point. It queries `nvidia-smi`,
resolves physical indices to full GPU UUIDs, rejects busy or undersized devices,
sets `CUDA_VISIBLE_DEVICES`, verifies the pinned Cosmos checkout, requires a new
run directory, and records the command, configuration, Git state, host, package
versions, selected GPUs, logs, seed, and outputs.

The 24-GPU configuration uses three nodes, eight GPUs per node, spatial CP=8,
DP=3, 81 frames at 432×768, and 16,000 optimizer updates. Its context schedule
is `k=1` through step 10,000, `k=2` through 14,000, `k=3` through 15,000, and
`k=4` through 16,000. Use the same shared checkout and run directory on all
nodes. Start rank zero first, then ranks one and two within 60 seconds:

```bash
# node 0
python tools/launch.py train \
  --gpus 0,1,2,3,4,5,6,7 \
  --run-dir runs/stage1-v1-001 \
  --config configs/basic_stage1_24gpu.json \
  --nnodes 3 --node-rank 0 --master-addr 192.0.2.10

# nodes 1 and 2: use the same arguments and set --node-rank to 1 or 2.
```

The default preflight requires 60,000 MiB free on every selected GPU. Set a
different explicit threshold only after measuring the target model and input.
To resume the same experiment, reuse its run directory and exact checkpoint:

```bash
python tools/launch.py train \
  --gpus 0,1,2,3,4,5,6,7 \
  --run-dir runs/stage1-v1-001 \
  --config configs/basic_stage1_24gpu.json \
  --nnodes 3 --node-rank 0 --master-addr 192.0.2.10 \
  --resume runs/stage1-v1-001/latest.pt
```

Resume restores the trainable weights, AdamW state, CPU/CUDA RNG state for every
rank, data identity, and exact caption ledger. `--weights-from` loads adaptation
weights into a new experiment without claiming exact continuation.

## Inference and verification

Generate one held-out calibrated target view with a new output directory:

```bash
python tools/launch.py infer \
  --gpus 0,1,2,3,4,5,6,7 \
  --run-dir runs/infer-v1-001 \
  --checkpoint runs/stage1-v1-001/latest.pt \
  -- --dataset syncam --split val --scene-index 0 --target-camera cam10
```

The CPU release contract does not import PyTorch or require a GPU:

```bash
python -m unittest discover -s tests -v
```

The final fixed validation used four target-withheld calibrated cases, 81
frames each. Averaged generated-versus-target metrics were RGB MSE `0.072432`,
MAE `0.194359`, PSNR `12.251051 dB`, and luminance SSIM `0.274137`. The
copy-first-source baseline was RGB MSE `0.069551`, MAE `0.193966`, PSNR
`12.105252 dB`, and SSIM `0.261656`. Results were mixed by dataset and case:
the model beat the copy baseline on both SynCam cases but lost on both MultiCam
cases. These are monitoring metrics, not paper benchmark scores or an
ego-to-SIM-to-exo acceptance result.

The final weight also ran target-blind on all 81 frames of calibrated TACO
sequence `20230927_032`, ego camera to fixed camera `21218078`. Relative to the
step-010000 attempt, PSNR improved by `1.052051 dB` and SSIM by `0.044916`.
Relative to copying the ego source, however, it gained only `0.054478 dB` and
`0.007506` SSIM. The run supplied one unique ego RGB view, repeated across the
final checkpoint's four source slots. Full chronological review rejects the
output: it remains a close overhead, source-like view rather than the requested
wide fixed exo view containing the complete person, table, chair, and room.
This checkpoint may be kept as a low-trust visual proposal, but it is not an
accepted ego-to-exo or ego-to-SIM-to-exo stage.

The explicit GPU checks use small randomly initialized Camera DiTs; they test
spatial-CP forward/backward equivalence and CP/DP gradient, freeze, checkpoint,
and RNG behavior. They do not establish pretrained video quality:

```bash
python tools/launch.py verify-spatial \
  --gpus 0,1 --run-dir runs/verify-spatial-v1-001 --min-free-mib 10000

python tools/launch.py verify-training \
  --gpus 0,1 --run-dir runs/verify-training-v1-001 \
  --context-parallel-size 2 --min-free-mib 10000
```
