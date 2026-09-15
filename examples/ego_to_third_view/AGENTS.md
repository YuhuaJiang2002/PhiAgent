# Ego-to-third-view pipeline work

- Read `AUTOMATION.md` for the active scene-independent contract. The earlier
  `stages/` archive and `recipes/whiteboard_v35/` retain scene-specific reproduction;
  never promote their coordinates, names, durations or appearance into defaults.
- For visual review/repair, read and use the repository skill
  `agentic_skills/ego-third-view-review/SKILL.md` (repository-root relative).
- Keep actor source-world origin independent of render camera, constant per-actor
  limb lengths, shared stable torso, original action clock, and same-NUMA policy.
  Do not accept a review without inspecting its bound visual artifacts.
- Keep heavyweight imports inside optional runtime stages. Run CPU tests under
  `examples/ego_to_third_view/tests`; real GPU evidence is a separate acceptance tier.
- Changes to upstream artifacts invalidate downstream reviews. Preserve old runs
  and report missing perception/calibration as incomplete, not an arbitrary-scene success.
