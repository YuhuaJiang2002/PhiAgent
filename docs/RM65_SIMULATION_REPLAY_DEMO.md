# RM65-B + AG2F90-C simulation replay

> Hardware correction: this document preserves the C-labelled geometry used by
> the published immutable replay. The company's installed unit is CTAG2F90-D.
> PhiAgent-Bench targets D and records its mass/speed limits separately; the
> visual replay below is not evidence of D dynamics or high-speed operation.

## Result

The public v17 demo establishes that the reviewed source-conditioned folding
motion can be realized as a synchronized MuJoCo joint replay for two six-axis
RealMan RM65-B arms with AG2F90-C grippers.

The public comparison contains 192 frames at 24 FPS. Every saved state is finite
and the complete trajectory is IK-solvable in the configured model. The measured
left/right EEF forward-kinematics residual is 0.72/0.68 mm on average and
3.78/1.92 mm at maximum. The IK solver's pre-render maximum position error is
1.75/1.72 mm.

## Pipeline

```text
source RGB video
  -> reviewed left/right gripper points, wrist-to-tip axes and open/close events
  -> planar 3-D proposal plus camera-ray depth regularization
  -> event-conditioned tabletop approach and transport heights
  -> dual-arm base-pose and workspace layout
  -> axis-only wrist orientation constraints
  -> multistart, branch-continuous RM65-B URDF-constrained IK
  -> fixed flange-to-tool roll regularization
  -> AG2F90-C gripper-chain replay
  -> FK, temporal, height and media audit
  -> synchronized MuJoCo render
```

The base poses were iteratively adjusted on the near long edge of the table.
Relative to the previous visual proposal, the final shoulder centers move 6 cm
away from the table and 3 cm upward. The fingertip-center path uses explicit
height anchors so grasp events approach to approximately 21-32 mm above the
tabletop while transport phases lift clear.

## Wrist branch and tool-roll correction

A single monocular EEF point does not observe a complete SO(3) wrist pose.
Earlier full-orientation regularization therefore admitted a
position-equivalent wrist-flip solution: joint 4 reached approximately
`+/-3.106 rad` even though the EEF target remained close. V16 constrains only
the source-visible wrist-to-tip axis, leaving roll about that axis to a posture
prior. Ten reviewed direction anchors cover the visible direction change in the
left gripper during the second half of the clip.

Each reviewed anchor is solved from multiple IK initializations. A temporal
path cost then selects one continuous branch before dense per-frame refinement.
The final joint-4 ranges are `[-0.058, 0.365] rad` on the left and
`[1.046, 1.760] rad` on the right; the minimum margin to any joint limit is
0.270 rad.

The source installation also shows the left gripper opening plane rotated by
approximately 90 degrees relative to the raw RM65-flange/AG2F90-C mesh
composition. V16 records this as a fixed `+90 deg` left tool-roll offset. This
does not change the EEF path or the observed longitudinal axis. On real
hardware, the same quantity must be represented by a measured
flange-to-gripper extrinsic rather than blindly added to a joint command.

## Browser media compatibility

V17 changes only the web delivery encoding. The v16 comparison MP4 used
MPEG-4 Part 2 (`mp4v`), which is not accepted by several HTML5 browser video
decoders even though the container is MP4. The comparison is re-encoded as
H.264/AVC (`avc1`, High@4.1) with `yuv420p`, and both web MP4s place the `moov`
metadata before media data for progressive playback. Relative to the decoded
v16 comparison, the compatibility encode measures 47.42 dB PSNR and 0.9948
SSIM. The 192-frame trajectory, MuJoCo state and rendering content are
otherwise unchanged.

## Reproducibility assets

- `configs/rm65_reference_video_actual_eef_anchors.json`: reviewed image-space
  gripper anchors.
- `configs/rm65_reference_video_a_edge_mount.json`: table, camera and dual-base
  layout.
- `configs/rm65_reference_video_eef_height_anchors.json`: event-conditioned EEF
  heights.
- `configs/rm65_reference_video_gripper_axis_anchors.json`: reviewed, time-varying
  wrist-to-tip image directions and regularized camera-depth components.
- `configs/rm65_reference_video_joint_branch_anchors.json`: branch-continuous
  multistart IK posture anchors.
- `scripts/recover_rm65_synchronized_state.py`: source tracking and initial IK.
- `scripts/refine_rm65_state_from_candidates.py`: depth/height regularization and
  final IK.
- `scripts/render_realman_rm65_visual_replay.py`: RM65-B and AG2F90-C MuJoCo
  construction and rendering.
- `scripts/evaluate_rm65_synchronized_replay.py`: FK, timing, gripper and table
  clearance audit.
- `demo/showcase/rm65-ag2f90c-source-vs-simulation-v16-state.npz`: final 192-frame
  q, EEF, gripper and timing state.
- `demo/showcase/rm65-ag2f90c-source-vs-simulation-v17.mp4`: browser-compatible
  synchronized comparison.
- `demo/showcase/rm65-ag2f90c-simulation-only-v17.mp4`: browser-compatible
  simulation-only replay.
- `demo/showcase/rm65-ag2f90c-source-vs-simulation-v17-audit.json`: public
  measured summary and artifact hashes.

The robot and gripper mesh packages remain external assets and are not vendored
by this demo.

## Evidence boundary

`WORKING_SIMULATION` means the URDF-constrained kinematic replay is solvable,
audited and renderable in MuJoCo. The current demo does not claim deformable
cloth dynamics, collision-safe controller output, calibrated camera extrinsics,
force closure, or recorded real-robot execution.
