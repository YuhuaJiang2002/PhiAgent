# Ego RGB → fixed third-person replay

当前用户确认版本：双高白板背景、无显示器/柜子/置物架，保留手物动作，闹钟采用显式直立修正。

This is a **scene-specific research recipe**, not a one-command converter for arbitrary videos.
The real-input demo is 360 frames / 30 FPS / 12 seconds (source seconds 4–16).
The latest v21 renderer does **not** use a video diffusion model. Earlier generated
lab backgrounds were rejected because they introduced cabinets and a monitor.

## Data flow

```text
RGB crop → HaWoR camera-space MANO hands ───────────┐
         → SAM2 reviewed instance masks ──────────┤
         → VGGT-Omega camera/depth → metric mat ───┤
                                                 ↓
                  persistent table-world hand state + SAM3D meshes
                                                 ↓
           FoundationPose → layout fit → multi-view static-span fit
                                                 ↓
           explicit clock-upright prior → fixed camera + two whiteboards
                                                 ↓
                third-view MP4 + side-by-side MP4 + foreground mask
```

## What is actually reconstructed?

- Predicted: HaWoR hands, VGGT-Omega camera/depth, SAM2 masks, SAM3D geometry,
  FoundationPose trajectories. Invisible geometry is model-inferred, not measured.
- Reviewed: 0.70 × 0.45 m mat anchor, source image corner picks, segmentation clicks,
  held frame intervals: clock 0–82, small cylinder 105–218, tall cylinder 258–359.
- Imposed: resting-object persistence, hand table support, upright clock mesh Y axis,
  maximum 20° handheld clock tilt. These are not contact/pose ground truth.
- Synthetic: room, table legs, two 1.95 m whiteboards and soft projected shadows.
  Whiteboard dimensions/placement are approximate, not calibrated room geometry.
- Not provided: robot retargeting, force/contact validation, task success certification.

## Runtime and external assets

Python 3.11, CUDA 12.8 / PyTorch-compatible NVIDIA runtime, NumPy, SciPy,
OpenCV, Pillow, trimesh, nvdiffrast, and FFmpeg (libx264 and FFV1).
Model stages additionally require their upstream environments and licensed weights.
Keep these outside this repository; MANO requires its own license.
HaWoR inference is an external prerequisite: export per-hand camera-space JSON,
then use `export_hawor_mano_meshes`. This archive does not vendor the modified
HaWoR inference implementation or claim a clean-room end-to-end install.

Verified source revision: VGGT-Omega `39a0cb8af88554f15ddcb5354cd52bde588fa014`.
The original HaWoR directory was labelled `66c7d410`; other model sources were
archive installations without Git metadata. Exact upstream revisions for those
are **unresolved**, so fully pinned source-to-result reproduction is PARTIAL.
No bootstrap script silently installs moving `main` revisions.

Adapters with native rendering dependencies expect this external layout:

```text
ROOT/
  third_party/FoundationPose/       # Utils.py, weights, compiled dependencies
  envs/sam3d-container-cu128/
    phiagent-native/                # compatible compiled extensions
```

Install each model in its compatible environment. `pipeline.py` runs the stage
with the current Python interpreter, so invoke it separately in each environment.
Use `PYTHONPATH` for HaWoR, VGGT-Omega, SAM2 and SAM3D source packages as needed.

## Re-render the accepted result

Given the private existing run bundle below, run in the native rendering environment:

```bash
nvidia-smi
python examples/ego_to_third_view/pipeline.py render_lab_demo \
  --root /path/to/external-assets \
  --run /path/to/ego-boxes-thirdview-v1 \
  --output /path/to/new-publication-render \
  --gpu GPU-your-selected-physical-uuid \
  -- --clock-upright --whiteboards
```

Use a UUID in GPU-restricted containers: physical index 2 may appear as index 0
inside a container. The launcher validates the visible GPU inventory, sets
`CUDA_VISIBLE_DEVICES`, refuses an existing execution output directory, and records
command/config, package versions, hostname, Git state, seed, GPU snapshot and logs.
The seed field is a hash seed; model-specific RNG flags remain separate.
No GPU job is scheduled automatically. Inspect memory and other users' jobs first.

Expected input bundle (private, not checked in):

```text
RUN/
  input/ego_action_4s_16s.mp4
  reconstruction/hawor_mano_camera_meshes.npz
  fixed-third-view-v6-persistence/reconstruction/world_state_tracks.npz
  observed-table-texture-v2/table_texture.png
  vggt-omega-512-allframes-v1/table-aligned/vggt_omega_table_aligned.npz
  foundationpose-layout-v2/{alarm_clock,small_cylinder,tall_cylinder}/
    tracking_mesh_metric.obj          # retain associated MTL/textures too
  foundationpose-clock-upright-v4/{alarm_clock,small_cylinder,tall_cylinder}/
    foundationpose_tracks.npz
```

Output: `OUTPUT/render/{fixed_third_view.mp4,ego_vs_lab.mp4,foreground_protection.mkv,manifest.json}`.
Rendering preserves the supplied temporal indices and pose arrays. H.264 output is
lossy; the FFV1 mask is lossless. Do not describe this as lossless RGB preservation.

## Rebuilding the intermediate bundle

Each adapter is exposed as `pipeline.py STAGE --root ROOT --run RUN --output NEW_EXEC_DIR
[--gpu UUID] -- ADAPTER_ARGUMENTS`. Use a **new RUN bundle** for a new experiment;
legacy adapters write named intermediate directories within RUN and may overwrite
them if repeated. The execution directory does not redirect those adapter outputs.
All source-specific constants remain explicit in the adapters; review them for a new clip.

| Order | Adapter / prerequisite | Main output |
| --- | --- | --- |
| 1 | FFmpeg crop 4–16 s, 30 FPS; external HaWoR inference; `export_hawor_mano_meshes` | input clip/JPEGs, camera-space MANO NPZ |
| 2 | `run_sam2_video_point_track` per instance, manually review/revise prompts | `reconstruction/sam2/*/masks/0000.png` onward |
| 3 | `run_vggt_omega_ego`, `align_vggt_omega_to_table` | compact predictions and metric table alignment |
| 4 | `render_ego_persistent_objects` with aligned NPZ | v6 table-world hand state |
| 5 | `prepare_sam3d_object_inputs`, `run_sam3d_ego_metric` per instance | pointmap-conditioned GLB and metric state |
| 6 | `track_ego_foundationpose`, `refine_sam3d_layout`, re-track refined meshes | `foundationpose-layout-v2` tracks/metric meshes |
| 7 | `refine_static_multiview`, `fix_clock_upright` | persistent v3, clock-upright v4 tracks |
| 8 | `build_ego_observed_texture`, `render_lab_demo --clock-upright --whiteboards` | accepted v21 scene |
| 9 | `audit_fp_trajectories --tracks-root foundationpose-clock-upright-v4` | source-view reprojection/pose CSV diagnostics |

SAM2 reviewed mask IDs are `alarm_clock_v2`, `small_cylinder_v2`, `tall_cylinder`;
final mesh IDs drop `_v2`. Stage 5/6 require deliberate ID/path mapping (or copying
the reviewed masks to the corresponding final IDs). The small cylinder may retain
its unrefined mesh while clock/tall cylinder use refined layout assets. Review the
chosen asset before tracking; the recipe does not automate this selection.

For a stage's own CLI help, use the same launcher and append `-- --help` in its
model environment. Some archived fixed-scene stages have no CLI; consult source.
The renderer is deliberately fixed at 960×720, 360 frames, 30 FPS, camera eye
`[0.72,-0.66,0.52]`, target `[0.02,-0.01,0.07]` in table-world metres.

Frame convention: points are stored as row vectors; `world_from_object` is a
homogeneous column-vector transform, applied as `p_world = p_object @ R.T + t`.
`R_table_world_to_camera_m`, `t_table_world_to_camera_m` map table-world metres
into the source camera. Rendering uses `camera_from_world @ world_from_object`.
Do not substitute HaWoR camera coordinates directly for table-world coordinates.

## Checks and evidence

```bash
python -m unittest discover -s examples/ego_to_third_view/tests -v
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames,duration \
  -of json /path/to/new-publication-render/render/ego_vs_lab.mp4
```

Expected comparison: 1920×720, 30/1 FPS, 360 frames, 12 s. Inspect first/middle/last
frames for wrong background objects, pose drift, hand/object occlusion and source sync.
Zero resting drift is a modeling constraint, not an accuracy metric.
See `evidence/` for the accepted renderer's manifest and verification report.
No original RGB video, licensed hand assets, model repositories or weights are published.
