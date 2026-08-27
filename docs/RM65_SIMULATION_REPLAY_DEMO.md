# RM65-B + AG2F90-C simulation replay

## Result

The public v11 demo establishes that the reviewed source-conditioned folding
motion can be realized as a synchronized MuJoCo joint replay for two six-axis
RealMan RM65-B arms with AG2F90-C grippers.

The public comparison contains 192 frames at 24 FPS. Every saved state is finite
and the complete trajectory is IK-solvable in the configured model. The measured
left/right EEF forward-kinematics residual is 0.61/0.51 mm on average and
3.92/1.65 mm at maximum.

## Pipeline

```text
source RGB video
  -> reviewed left/right gripper points and open/close events
  -> planar 3-D proposal plus camera-ray depth regularization
  -> event-conditioned tabletop approach and transport heights
  -> dual-arm base-pose and workspace layout
  -> RM65-B URDF-constrained per-frame IK
  -> AG2F90-C gripper-chain replay
  -> FK, temporal, height and media audit
  -> synchronized MuJoCo render
```

The base poses were iteratively adjusted on the near long edge of the table.
Relative to the previous visual proposal, the final shoulder centers move 6 cm
away from the table and 3 cm upward. The fingertip-center path uses explicit
height anchors so grasp events approach to approximately 20-31 mm above the
tabletop while transport phases lift clear.

## Reproducibility assets

- `configs/rm65_reference_video_actual_eef_anchors.json`: reviewed image-space
  gripper anchors.
- `configs/rm65_reference_video_a_edge_mount.json`: table, camera and dual-base
  layout.
- `configs/rm65_reference_video_eef_height_anchors.json`: event-conditioned EEF
  heights.
- `scripts/recover_rm65_synchronized_state.py`: source tracking and initial IK.
- `scripts/refine_rm65_state_from_candidates.py`: depth/height regularization and
  final IK.
- `scripts/render_realman_rm65_visual_replay.py`: RM65-B and AG2F90-C MuJoCo
  construction and rendering.
- `scripts/evaluate_rm65_synchronized_replay.py`: FK, timing, gripper and table
  clearance audit.
- `demo/showcase/rm65-ag2f90c-source-vs-simulation-v11-state.npz`: final 192-frame
  q, EEF, gripper and timing state.
- `demo/showcase/rm65-ag2f90c-source-vs-simulation-v11-audit.json`: public
  measured summary and artifact hashes.

The robot and gripper mesh packages remain external assets and are not vendored
by this demo.

## Evidence boundary

`WORKING_SIMULATION` means the URDF-constrained kinematic replay is solvable,
audited and renderable in MuJoCo. The current demo does not claim deformable
cloth dynamics, collision-safe controller output, calibrated camera extrinsics,
force closure, or recorded real-robot execution.
