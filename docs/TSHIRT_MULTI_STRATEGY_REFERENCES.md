# Multi-strategy T-shirt folding references

Status: `PARTIAL` visual generation evidence, user quality review passed on
2026-08-22.

## Result

PhiAgent retains three generated 8-second T-shirt-folding videos with the same
real-scene layout, dual-arm embodiment, 1024x768 resolution, 24 FPS cadence, and
192-frame length. Their visible choreography differs:

| Reference | Visible route | SHA-256 |
| --- | --- | --- |
| `fold-alternating` | alternating arm roles | `66edab0ba24ee441816694a373528628befe29c5136fd6d57f34fc53f09f4f1b` |
| `fold-staged` | staged sleeve handling | `b101b611aab37d3d87af90e291633e0f23e4ea8a627ae4d0fdd2bf5563a4cbca` |
| `fold-synchronized` | synchronized bilateral sleeve fold | `fe590fc0fbac1a83c325f131066e6df23c5de09dfa472960c4ccf659cb7b82d3` |

The user explicitly accepted the visual generation quality of all three input
files. This is a candidate-SHA-bound visual-quality review, not an automatic
physical-gate result. The first frames are highly similar but not pixel exact:
SSIM against `fold-alternating` is 0.936417 for `fold-staged` and 0.939120 for
`fold-synchronized`. Documentation and the website therefore say "same scene"
rather than "exact same first frame."

The public comparison is 1920x720, 24 FPS, 192 frames, and 8 seconds. Its
SHA-256 is
`36cd86d3b63ef094e35de3ff47b562371c656f65334a911a78b474765f32d05d`.
The three original full-resolution files remain beside it because the scaled
comparison is presentation media rather than evaluation evidence. Complete
media metadata and hashes are in
[`demo/showcase/tshirt-fold-strategies/manifest.json`](../demo/showcase/tshirt-fold-strategies/manifest.json).

## Generation lessons

1. Hold scene variables constant. A fixed camera, stable background, consistent
   dual-arm appearance, matched duration, and matched FPS make action differences
   readable instead of confounding them with style changes.
2. Expose visible phases. The successful examples separate approach/contact,
   sleeve motion, body compaction, and terminal hold instead of relying on one
   underspecified monolithic instruction.
3. Branch on choreography before sampling. Candidate allocation should cover
   left-first, right-first, and synchronized sleeve schedules instead of letting
   a best-of-N pool collapse onto one attractive motion pattern.
4. Preserve a terminal dwell. The compact in-place hold makes the task result
   inspectable and gives the temporal evaluator a stable terminal window.
5. Keep positive examples subordinate to gates. A visually preferred reference
   may guide cadence and arm-role separation, but it cannot supply or override a
   material, contact, continuity, task-order, background, or human-review result.

## Harness change

`phiagent.harness.tshirt_positive_reference` adds a dependency-light,
hash-bound positive-reference bank and compiles strategy-compatible reference
conditioning into a task plan. Each conditioning object binds the source video
SHA-256, reference-bank SHA-256, task-plan SHA-256, sleeve-order hypothesis,
terminal strategy, prompt addendum, every original non-overrideable gate ID, and
an explicit claim boundary.

The source examples end compacted in place. A reference-conditioned plan must
still execute its own declared terminal placement; the compiler explicitly
forbids copying the reference's terminal behavior when the task plan requests a
different final placement. Cross-strategy references fail closed, hash changes
fail closed, and the package remains importable without PyTorch, CUDA, a model,
or a simulator.

## Evidence boundary

The videos are generated camera-pixel proposals. They do not provide calibrated
3-D cloth geometry, force or tactile sensing, exact robot joint trajectories,
collision safety, executable commands, or real-robot task success. Automatic
hard gates and candidate-SHA-bound native-resolution review remain separate
acceptance authorities.
