# Completed TI2V paired generation and scoring

Generation run: `20260918T095225Z`. Scoring-only recovery: `20260919T021424Z`, completed September 19, 2026.

The scoring handoff and case-paired analysis are complete. All six raw/selected requests contain 60 scored outputs. The recovery generated no videos and made no new proposal or candidate-audit calls. It reused the existing 77 native outputs and the original baseline fallbacks.

## Result

The prespecified primary comparison does not establish a benefit from joint extended repair. Against its matched parent on raw outputs, dyn increases in mean, while BLEU, CLIP, hsd, and ndtw decrease. All five primary 99% intervals include zero. Higher proposal coverage and more candidate adoptions therefore do not establish better video quality.

## Raw candidates

| Arm | BLEU ↑ | CLIP ↑ | hsd ↑ | dyn ↑ | ndtw ↑ |
|---|---:|---:|---:|---:|---:|
| Parent | 0.188045 | 87.833703 | 0.283467 | 0.203400 | 0.306800 |
| Joint extended repair | 0.175930 | 87.810925 | 0.280100 | 0.209433 | 0.293467 |
| Factored extended repair | 0.180968 | 87.925381 | 0.278783 | 0.208433 | 0.294917 |

## Final selected outputs

| Arm | BLEU ↑ | CLIP ↑ | hsd ↑ | dyn ↑ | ndtw ↑ |
|---|---:|---:|---:|---:|---:|
| Parent | 0.203714 | 88.764598 | 0.299950 | 0.205600 | 0.299017 |
| Joint extended repair | 0.197063 | 88.822104 | 0.299083 | 0.213033 | 0.293500 |
| Factored extended repair | 0.196496 | 88.904426 | 0.297583 | 0.211700 | 0.292467 |

## Prespecified primary contrast

Joint extended repair minus matched parent, raw candidates. These differences and intervals are copied from the server-produced analysis; no statistics were recomputed on the workstation.

| Metric | Paired mean difference | 99% case-cluster interval |
|---|---:|---:|
| BLEUScore | -0.012115 | [-0.040648, +0.005843] |
| CLIPScore | -0.022778 | [-1.076726, +0.898188] |
| hsd | -0.003367 | [-0.028583, +0.020750] |
| dyn | +0.006033 | [-0.004933, +0.025183] |
| ndtw | -0.013333 | [-0.033717, +0.000267] |

The analysis retains all three seeds within each of 20 cases, using 10,000 bootstrap replicates and seed 20260918. The 99% intervals apply to the five-metric primary family. Other contrasts are exploratory.

## Secondary findings

Factored repair also decreases raw BLEU, hsd, and ndtw means while increasing CLIP and dyn. Its raw ndtw difference is −0.011883, with a 99% interval of [−0.028800, −0.000133]; this is an exploratory negative contrast, not independent confirmation.

After the unchanged selector, both repair arms increase CLIP and dyn means but decrease BLEU, hsd, and ndtw relative to the selected parent. All five repair-versus-parent intervals for each selected-output contrast include zero. The parent adopts 8/60 candidates; joint and factored repair each adopt 12/60. More accepted candidates did not produce an overall metric improvement.

## Provenance and cost

Full generation verification checks all video hashes, native receipts, 1,309 actual observer images, the inherited gate schemas, and replay of the original selector. Official receipt verification checks all six requests against their CSVs and the frozen case/seed/video bindings. All three verification stages passed.

The source run timed out after generation while waiting for an undelivered scoring receipt. Its failure and deadline remain preserved. The user-authorized scoring continuation uses separate directories and an eight-hour scoring window. The official evaluator and caption wrapper remain unchanged. There are 131 unique scoring-video hashes across the six manifests because final selections can reuse older baseline videos; only the already completed 77 videos were generated for this experiment.

Generation verification SHA-256: `1a6539d36eac2c20e495f966f83c45636b6e706614d7888bf2394cf7211648ee`.

Score verification SHA-256: `8e370d7965bcdee689f4c03cf5dd0e666717cd23e9a75467afc856a47a588685`.

Analysis SHA-256: `80da095bdd766698983c9dd90ccab8cf79674a7d7304cbd1913b3dfc4fef430c`.

The six scoring jobs used 2127.055 allocated GPU-seconds and 2305.003 summed job wall-seconds. Original generation cost was 77 native calls, 77 candidate-audit calls, 187,123 audit tokens, and 23,533.313 native model-seconds; startup and earlier proposal costs remain separate. The recovery adds zero native-generation and zero proposal/audit-model calls.

## Interpretation

This is a development negative result on repeatedly used cases. It does not support promoting this repair variant as an improvement over its matched parent or VideoWeaver. Its parent is the one frozen in the generation run; the experiment is not a test of a frozen SkillAdam skill followed by repair. The completed SkillAdam integration and its earlier crossed sampling ablation retain their separate identities. Human review and evaluator changes remain outside this work.

Public evidence: [paired analysis](evidence/paired-repair-analysis.json) and [scoring cost](evidence/paired-repair-scoring-cost.json).
