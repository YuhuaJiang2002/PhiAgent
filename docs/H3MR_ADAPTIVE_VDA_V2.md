# Stage-3-BIR + Adaptive VDA

Status: **WORKING on the frozen H2O Train20 development protocol**. This is a
GT-2D upper-bound evaluation, not a production end-to-end result or an external
generalization claim.

## Method

Adaptive VDA keeps the existing Stage-3-BIR hand pose and interaction result as
the protected baseline. It routes sequences using evidence computed without 3D
ground truth:

- **V1 hard recovery:** use the frozen dense recovery when
  `valid_keyframes < 4` or `scale_relative_mad > 0.20`.
- **V2 ordinary correction:** otherwise estimate a low-frequency hand-root
  depth residual from metric-calibrated VDA wrist patches.
- **Stage-3-BIR fallback:** preserve Stage-3-BIR when support, center, or penetration
  gates reject the candidate.

V2 uses an 11×11 wrist patch, its 25th depth percentile, at least four robust
temporal anchors, ±40 mm residual clipping, and a 0.5 fusion coefficient. The
applied correction is therefore bounded to ±20 mm. Around BIR interaction
frames, both hands blend to a shared shift. Any newly detected penetration or
increase in total penetration energy rejects the complete V2 candidate.

The implementation changes only:

- `transl`;
- `joints_3d_world` and `vertices_world`;
- `joints_3d_camera` and `vertices_camera`;
- `joints_2d` and `joints_in_frame`.

It preserves MANO pose and shape, camera rotation and translation, validity
masks, Stage-1/2/3 diagnostics, and BIR diagnostics. Camera-frame +Z shifts are
explicitly rotated by `camera_R_c2w` before being applied in the world frame.

## One-command pipeline

Install the lightweight numeric dependency:

```bash
python -m pip install -e ".[adaptive-vda]"
```

The runner expects numeric VDA archives named `<sequence>_depths.npz`, each
containing `depths` and `frame_indices`. The scale summary supplies the frozen
relative-to-metric mapping and the V1/V2 routing evidence. Model repositories,
MANO assets, and checkpoints remain external.

```bash
export H2O_ROOT=/path/to/H2O/raw

python scripts/run_h3mr_adaptive_vda.py all \
  --stage3-root /path/to/stage3_bir/runs \
  --vda-depth-root /path/to/vda_depths_sparse \
  --scale-summary /path/to/scale_summary.json \
  --v1-hard-root /path/to/frozen_v1_hard/runs \
  --h3mr-code-root /path/to/private_h3mr_code_copy \
  --output-root /path/to/new_adaptive_vda_experiment
```

`all` performs these phases in order:

1. GPU and input preflight, including the frozen branch classification;
2. GT-free V1/V2 candidate generation and per-sequence SHA manifests;
3. official H2O evaluation, only after the complete candidate manifest exists;
4. field, coordinate-frame, local-geometry, finite-value, and copy audits.

H2O is resolved from `--h2o-root`, then `H2O_ROOT`, then a self-contained
`input/h2o_gt` directory beside the supplied Stage-3-BIR copy. The runner never
writes to any input directory and refuses to overwrite an existing candidate
manifest or output NPZ.

Individual phases are also available through `preflight`, `generate`,
`evaluate`, and `audit`.

## Three-way demo renderer

The published demo shows Native HaWoR, Stage-3-BIR, and Stage-3-BIR + Adaptive
VDA on the same 300-frame timeline. The compositor reads two already-audited,
frame-aligned comparison videos and draws one metric table from the frozen
Native and Adaptive summary files:

```bash
python -m pip install -e ".[h3mr-demo]"

python scripts/render_h3mr_three_way_comparison.py \
  --native-comparison /path/to/native-vs-adaptive.mp4 \
  --stage3-comparison /path/to/stage3-vs-adaptive.mp4 \
  --native-summary /path/to/native-summary.json \
  --adaptive-summary /path/to/adaptive-summary.json \
  --output /path/to/comparison-three-way.mp4 \
  --manifest /path/to/comparison-three-way-manifest.json
```

The renderer validates both 1920×1080/30-FPS/300-frame inputs, refuses output
overwrites, produces H.264 with web fast-start, decodes all 300 output frames,
and records source/output SHA-256 values. It performs no model inference or
candidate selection.

## Frozen H2O Train20 result

The comparison uses 20 videos, 40 single-hand trajectories, and 25,352 shared
valid hand frames. All metrics are lower-is-better.

| Metric | Native HaWoR | Stage-3-BIR | + Adaptive VDA | vs. Native | vs. Stage-3 |
|---|---:|---:|---:|---:|---:|
| PA-MPJPE | 7.000234 mm | 6.904049 mm | **6.904049 mm** | **1.374%** | ≈0% |
| W-MPJPE | 26.083663 mm | 24.995810 mm | **23.941061 mm** | **8.214%** | **4.220%** |
| WA-MPJPE | 15.686847 mm | 15.135565 mm | **14.584753 mm** | **7.026%** | **3.639%** |
| RTE | 1.131513% | 1.107845% | **1.031538%** | **8.836%** | **6.888%** |
| Accel | 7.051547 m/s² | 6.442513 m/s² | **6.406733 m/s²** | **9.144%** | **0.555%** |

Nineteen ordinary sequences were routed to V2 and the single frozen hard
sequence `subject1_ego__k2__2` retained V1 dense recovery. Eleven ordinary
sequences passed all V2 gates; seven fell back after the exact penetration gate
and one fell back for insufficient depth evidence.

## Evidence and limitations

- Candidate generation did not read 3D GT; GT was first used after candidates
  and hashes were frozen.
- Stage-3-BIR baseline reproduction matched the previous report exactly.
- The final audit passed all 38 V2 candidate hashes and both exact frozen-V1
  copies, plus coordinate consistency, local hand geometry, BIR protection, and
  finite modified-field checks.
- V2 did not improve every active sequence on every metric. In particular,
  `subject1_ego__h2__4` regressed in W-MPJPE, `subject2_ego__h2__0` regressed in
  RTE, and `subject1_ego__k1__7` regressed slightly in Accel.
- Train20 participated in development. A pre-frozen external ordinary/hard
  test set is still required before claiming broad generalization.

The published three-way video uses the first 300 frames of the pre-frozen
ordinary pilot `subject2_ego__h2__3`; the clip was not selected using GT
improvement. Its three trajectory panels use GT only for evaluation
visualization, while the metric table reports the complete Train20 aggregate
above.
