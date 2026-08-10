# Third-party component policy

License review date: 2026-08-08.

- Wan2.2 / Wan2.2-Animate: Apache-2.0. Selected for the visual teacher.
- OSCAR: Apache-2.0 source, kept under ignored `external/` and pinned at
  `4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb`. The OSCAR-2B snapshot is pinned
  at `c9781ffa7dd8556d862d7d9f338a2ea008a58ca6`; its runtime additionally uses
  Cosmos-Reason1-7B revision
  `3210bec0495fdc7a8d3dbb8d58da5711eab4b423` and the Wan2.1 VAE from revision
  `37ec512624d61f7aa208f7ea8140a131f93afc9a`. Checkpoint and transitive model
  terms must be reviewed separately from the OSCAR code license.
- Segment Anything 2 (SAM2): Apache-2.0, optional morphology-segmentation tool
  pinned at `0e78a118995e66bb27d78518c4bd9a3e95b4e266`. The AC-WM canonical-hand
  run uses the `sam2_hiera_large.pt` checkpoint with SHA-256
  `7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b`.
  Source and weights remain under ignored external/checkpoint directories.
- Boundless World Model: optional external AC-WM backend pinned at source
  `44acfd1b06f35f365f02f7bb2fc5da6beafcd6bc` and model revision
  `738a8d3c008e637b8b1b18d5e98a82f6de9c04aa`. It also requires a separately
  obtained Wan2.2-TI2V-5B base model and task action statistics. Nothing is
  vendored; users must review all source and weight terms before download/use.
- Kinema4D: optional external geometry-conditioned backend pinned at source
  `716e80249376cb2843af41188a832d56a2d8d78d` and model revision
  `0c52ee34ee464e9a568e84945e431f62106c4270`. It also requires the external
  Wan2.1-I2V-14B base transformer and prepared robot RGB+pointmap data. Nothing
  is vendored; users must review all source, base-model, and weight terms.
- Wan-Animate-2: Apache-2.0, pinned at source commit
  `3ad2fef7d61d6200c9c653e0fe47be7616b323f3` and Hugging Face model revision
  `3c1a1ccd035b9997478d288040358891a06bf682` or official ModelScope mirror
  revision `7053fd05166cdd99a49896364d01c06c281a9d69`. It directly consumes a
  reference image and driving video, removing the original Animate pose/face
  preprocessing dependency. Its outputs remain an approximate proxy, not PhiZero
  unchanged-token decoding.
- DiffSynth-Studio: Apache-2.0, pinned at
  `b1c02ce76aabc989f6bf534756b2da84532249e5` for its documented
  Wan2.2-Animate-14B animation-mode LoRA example. Its reviewed configuration uses
  eight 80 GiB GPUs. It does not establish replacement-mode or
  identity-image-only LoRA compatibility.
- PhiZero: the public code repository contains only a coming-soon README at
  revision `6bc7428f2ad5282e0c1a7b122465957b6abb1edc`; no implementation,
  checkpoints, or license are published yet. The project-page video repository
  also has no license file at pinned revision
  `72fc49fb17b56fab6f7407239b38bdedf7c76546`. Reference videos are fetched on
  demand into `external/` and are not redistributed.
- Cosmos Framework and Cosmos3-Nano: OpenMDW-1.1. Selected for the primary
  trajectory-conditioned rendering path and pinned to source commit
  `4155d61d14b14e05a8cafe2bd796d090fcb5f145` and model revision
  `411f42a8fdfb8c5b2583cb8786e0938f49796eaa`. Keep the framework and weights
  external; users must review OpenMDW-1.1 and downloaded third-party terms.
- HaMeR: MIT. Suitable for an optional hand-reconstruction adapter, subject to
  the licenses of its detector/checkpoint dependencies.
- dex-retargeting: MIT. Preferred permissive retargeting baseline.
- MuJoCo: Apache-2.0. Selected first physics backend.
- Pexels flower-arranging video 5893642 by Anna Shvets: used under the Pexels
  License as the single continuous 27.5-second source for the README full-robot
  geometric flower demo. The transformed comparison includes attribution even
  though the license does not require it.
  https://www.pexels.com/video/woman-cutting-leaves-of-flower-stems-5893642/
- MuJoCo Menagerie: pinned externally at
  `c1a4eeb85694ae1dffe33ff1797d4e528928a133`. The Wonik Allegro model is
  BSD-2-Clause and the Shadow Hand model is Apache-2.0. Their upstream rendered
  PNGs may be used as identity-conditioning inputs, but generated transfer videos
  are visual proxies rather than vendor-validated robot executions.
- MuJoCo Menagerie Unitree G1: BSD-3-Clause, pinned at
  `c1a4eeb85694ae1dffe33ff1797d4e528928a133` as the full-body visualization
  whose real arm joints and wrist frames drive the clean articulated Sharpa and
  Allegro flower demo.
- Sharpa RL Lab: Apache-2.0, pinned at
  `95ccda3d948801bb5da4cb7ffea766e03067a63b`. Its official `real.gif` and
  `sim.gif` show cylinder rotation with SharpaWave; they may support identity
  and interaction analysis but are not paired human-to-robot training data.
- mini-ArtiCraft: Apache-2.0. Optional articulated-asset generator, pinned to
  commit `7d43e25b26e9459aabf53d77d1d9325805bc1ea3` and executed in its own
  environment. Provider API terms and generated-asset provenance remain the
  user's responsibility.
- SPIDER: Creative Commons Attribution-NonCommercial 4.0. It may be integrated
  as an optional non-commercial research teacher, but cannot be treated as a
  permissive production dependency.
- FoundationPose: NVIDIA Source Code License, not a standard permissive
  open-source license. Keep it optional and require users to accept its terms;
  provide a clean interface and a fallback that does not claim 6D pose.
- FLUX.1-Kontext-dev: separate model terms. Enhanced Wan pose retargeting stays
  opt-in until the intended use is checked against those terms.
- RoboMaster: the official GitHub repository does not publish a code license as
  of the review date, so PhiAgent does not copy, vendor, or execute it. The
  Hugging Face model card is tagged Apache-2.0, but the bundled
  CogVideoX-Fun-V1.5-5b-InP base checkpoint carries separate CogVideoX terms.
  Integration remains blocked until the code license and intended checkpoint
  use are resolved explicitly.

Primary project pages:

- https://github.com/Wan-Video/Wan2.2
- https://github.com/wuzy2115/oscar-public
- https://github.com/facebookresearch/sam2
- https://huggingface.co/zywu2115/OSCAR-2B
- https://github.com/boundless-large-model/boundless-world-model
- https://huggingface.co/BLM-Lab/Boundless-World-Model
- https://github.com/mutianxu/Kinema4D
- https://huggingface.co/Minoday/Kinema4D
- https://github.com/Wan-Video/Wan-Animate-2
- https://github.com/modelscope/DiffSynth-Studio
- https://phi-zero.github.io/
- https://github.com/yaoyao-jpg/PhiZero
- https://github.com/NVIDIA/cosmos-framework
- https://huggingface.co/nvidia/Cosmos3-Nano
- https://github.com/geopavlakos/hamer
- https://github.com/NVlabs/FoundationPose
- https://github.com/dexsuite/dex-retargeting
- Sharpa Wave URDF/USD/XML assets: Apache-2.0, pinned for the first simulation
  demo at revision `6eea427eb24189519f32b9f21674cd534d3f973c`. Keep the
  checkout under ignored `external/`; do not vendor the meshes.
  https://github.com/sharpa-robotics/sharpa-urdf-usd-xml
- MuJoCo Menagerie Wonik Allegro model: BSD-3-Clause model license, pinned at
  revision `c1a4eeb85694ae1dffe33ff1797d4e528928a133` for the robot-model
  sensitivity contrast. Assets remain under ignored `external/`.
  https://github.com/google-deepmind/mujoco_menagerie/tree/main/wonik_allegro
- Sunday Robotics Memo hero media: official promotional media from
  `https://www.sunday.ai/`. No redistributable media license was found; it is
  used only as an internal research reference and must not be redistributed or
  used commercially without permission.
- Wan-Animate-2: Apache-2.0, pinned at
  `3ad2fef7d61d6200c9c653e0fe47be7616b323f3`; distilled checkpoint marker
  `modelscope:7053fd05166cdd99a49896364d01c06c281a9d69`.
  https://github.com/Wan-Video/Wan-Animate-2
- https://github.com/facebookresearch/spider
- https://github.com/google-deepmind/mujoco
- https://github.com/google-deepmind/mujoco_menagerie
- https://www.pexels.com/license/
- https://github.com/sharpa-robotics/sharpa-rl-lab
- https://github.com/articraftresearch/Articraft
- https://github.com/KlingAIResearch/RoboMaster
- https://huggingface.co/KlingTeam/RoboMaster
