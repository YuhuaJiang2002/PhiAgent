# Environment audit

Audit date: 2026-08-08 (Asia/Shanghai)

## Remote A800 candidates

The originally requested phi-a800 alias remains absent. The user subsequently
authorized the configured Kingsoft Cloud aliases a800-1, a800-2, and a800-3.

- a800-1: reachable Ubuntu 22.04 host VM-16-9-ubuntu, 8 NVIDIA
  A800-SXM4-80GB GPUs, driver 535.161.07. GPUs 4-7 were completely idle at the
  selection check. /data0 has about 3.4 TiB free. Selected execution host.
- a800-2: reachable, but every GPU had another user's process and its root
  filesystem had only about 61.7 GiB free. Not selected.
- a800-3: SSH public-key authentication failed. Not selected.

On a800-1, Python 3.10.12, CUDA toolkit 12.4, ffmpeg 4.4.2, git-lfs
3.7.1, tmux, uv 0.8.22, GCC 11.4, and a working system PyTorch 2.6.0+cu124
were measured. Hugging Face timed out, while GitHub, PyPI, PyTorch downloads,
and ModelScope were reachable. The official ModelScope mirror is therefore
pinned for checkpoint transfer.

## Local machine

- OS/architecture: macOS 26.5.2, arm64.
- Python: 3.14.4 at /opt/homebrew/bin/python3.
- Git: 2.50.1 (Apple Git-155).
- Workspace volume: 460 GiB total, about 29 GiB free at audit time.
- Missing: NVIDIA/CUDA, Conda, ffmpeg, and git-lfs.
- Lightweight validation environment: repository-local .venv; no GPU packages.

The Mac is suitable for source editing and CPU tests, but not for this inference
job. The checkpoint alone is larger than the available space.

## Proposed GPU environment

The reproducible environment is defined by environment.yml,
requirements/wan-animate.txt, and scripts/bootstrap_environment.sh:

- Python 3.10.16
- PyTorch 2.6.0
- torchvision 0.21.0
- torchaudio 2.6.0
- pytorch-cuda 12.4
- ffmpeg 7.1.1
- git-lfs 3.6.1
- flash-attn 2.7.4.post1, installed last without build isolation
- Wan native dependencies pinned individually
- SAM 2 is pinned to commit 0e78a118995e66bb27d78518c4bd9a3e95b4e266
  and built without isolation against the selected CUDA/PyTorch runtime; the
  upstream animation preprocessor imports it even outside replacement mode

The server's existing PyTorch 2.6.0+cu124 runtime successfully initialized CUDA
under driver 535.161.07. The dedicated project environment reproduces that
measured combination rather than retaining the pre-audit 2.5.1 proposal.

## Model and storage

The selected upstream is the official Apache-2.0 Wan-Video/Wan2.2 repository,
pinned to commit 42bf4cfaa384bc21833865abc2f9e6c0e67233dc. The official model is
Wan-AI/Wan2.2-Animate-14B.
The model snapshot is pinned to revision
cb93a225fbaf1ca100f54e79da8f994995b689b3.
For the reachable official ModelScope mirror, the pinned revision is
bdcd76afebe1932ecb69916dd14ca255780f1d30.

- Hugging Face model tree: 72.4 GB decimal, or 67.41 GiB.
- Included process_checkpoint subtree: about 3.89 GiB.
- Source/environment/build/cache/intermediate/output allowance: at least 20-40
  GiB depending on package caching and video length.
- Enforced free-space threshold before download: 120 GiB.
- Recommended operational free space: 150 GiB when retaining multiple
  experiments.

No checkpoint was downloaded locally. The preparation script checks free space
before network transfer and pins the source revision.

Primary references:

- https://github.com/Wan-Video/Wan2.2
- https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md
- https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/tree/main
