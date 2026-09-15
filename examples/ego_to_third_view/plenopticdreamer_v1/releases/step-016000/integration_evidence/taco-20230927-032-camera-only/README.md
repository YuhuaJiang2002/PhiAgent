# TACO camera-only target-blind check

The final step-016000 checkpoint generated 81 frames from the calibrated ego
video and requested fixed exo camera before the held-out target was introduced.
`generation-provenance.json`, `preflight.json`, and `inference.json` bind the
target-free request, inputs, checkpoint, runtime, and generated-video hash.
`heldout-evaluation.json` records the later offline comparison.
The nested `source_git` object is the frozen source-checkout lineage supplied
to the deployed copy; the top-level launcher, adapter, runner, and evaluator
hashes record the path-adapted files that actually executed on the H20 host.

The final weight improves substantially over the earlier step-010000 attempt:
generated-to-target PSNR rises by 1.052051 dB and luminance SSIM by 0.044916.
Against copying the ego source, however, the gain is only 0.054478 dB PSNR and
0.007506 SSIM. Generated/source motion-energy correlation is 0.671356, while
generated/target correlation is 0.115207.

Review covered every frame in chronological contact sheets. The output remains
a close, overhead, source-like composition with only a modest lateral change.
It never becomes the requested wide fixed view containing the complete seated
person, table, chair, and room. The decisive camera/composition gate therefore
fails even though the action is recognizable and the pixel metrics improve.

The generated and comparison MP4s are excluded from this lightweight package;
their hashes are retained in `review.json` and the raw provenance. This
camera-only diagnostic supplies one unique ego RGB view and repeats it across
the final checkpoint's four source-view slots. It is not a test of separate
SIM-derived control channels and does not accept the full ego-to-SIM-to-exo
route. The historical case identifier still contains `k1` because it denotes
one unique input view, while `inference.json` is authoritative for runtime
`k=4` and `context_policy=repeat_single_source`.
