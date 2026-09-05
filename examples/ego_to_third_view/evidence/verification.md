# v21 publication check — 2026-09-05

- WORKING: packaged `pipeline.py render_lab_demo` re-rendered the private real-input
  bundle using `--clock-upright --whiteboards`, exit code 0.
- Comparison: 1920×720, 30/1 FPS, 360 frames, 12.000000 seconds, confirmed by FFprobe.
- Packaged and original renderer MP4s are **byte-identical** (both outputs).
  This verifies packaging preserved the accepted result, not reconstruction accuracy.
- Pure third view SHA256: `d792d0e6dfb59fb288456b295df895dbfdab3d382a3fa924780a1f28c952c90f`.
- Comparison SHA256: `a235c2b496aade28377708527b3878d627b9379736dc6a4d1b0b7f622737d256`.
- Original first/frame-330 comparisons visually reviewed: two tall plain whiteboards,
  no added monitor, cabinet or shelving; source actions remain synchronized.
- CPU acceptance: 6 unittest cases passed, including source compilation, GPU-stage
  classification, GPU UUID selection/container renumbering, explicit GPU requirement,
  refusal of an existing execution directory, and no private machine paths in adapters.
- PARTIAL: entire model frontend was not rerun from RGB during publication; it uses
  the previously generated private intermediate bundle. Upstream pinning is incomplete.

## Attempt history

The initial container command failed because the SSH user needed the existing sudo
Docker access. The next preflight rejected physical index 2 after container
renumbering. Passing the inspected physical GPU UUID succeeded. Neither failed
attempt rendered frames or modified the accepted v21 result. Use UUIDs in containers.

Legacy rendering emits a harmless invalid-matmul warning where static depth is
infinite outside scene coverage; those pixels are excluded by the finite-depth mask.
This warning was preserved to avoid altering the accepted renderer while packaging.
