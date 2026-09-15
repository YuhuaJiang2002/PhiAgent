# Final step-016000 reproduction snapshot

This directory binds the exact custom source overlay used at the end of the
PlenopticDreamer V1 reproduction to the final training checkpoint identity and
the final fixed-validation records. It is a historical, lightweight source and
evidence package. Model weights, datasets, generated videos, caches, and Python
environments are intentionally not included.

## Reconstructing the source state

The public NVIDIA Cosmos Transfer 2.5 repository is the upstream base, pinned
to commit `2ff49d0717af02057ae79bc75c00fbff9da1b4e7`. Clone that commit, then copy
`source_overlay/` over the checkout root to restore the custom training and
evaluation layer captured from the completed run:

```bash
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
git -C cosmos-transfer2.5 checkout 2ff49d0717af02057ae79bc75c00fbff9da1b4e7
cp -a source_overlay/. cosmos-transfer2.5/
```

The upstream repository alone does not contain the reproduction-specific
multi-view model, progressive-context trainer, spatial context parallelism, or
final validation tooling. Those files are preserved byte-for-byte in
`source_overlay/`. Absolute cluster paths appearing in historical evidence or
in the captured path adapter are provenance, not portable defaults.

## Bound artifacts

| Artifact | Included | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Stage-one resume checkpoint, step 16,000 | No | 6,703,192,745 | `f7ad29d7e91e1e9674335b2368b079f63c8051e9d97a122b02bce97cac93d6b3` |
| Inference-only snapshot, 196 tensors | No | 2,240,098,149 | `a4f1f46305bf1c20075b9b23b607bf07880ebb414e780fffe5b92eda63a3ce20` |
| Captured source overlay | Yes, 36 files | — | Per-file hashes in `files.sha256` |
| PhiAgent target-free integration overlay | Yes, 4 files | — | Per-file hashes in `files.sha256` |
| Training and fixed-validation evidence | Yes, 15 files | — | Per-file hashes in `files.sha256` |
| Target-blind TACO integration evidence | Yes, 6 files | — | Per-file hashes in `files.sha256` |

`release.json` is the machine-readable contract. The checkpoint sidecar,
resolved training config, four progressive-context update checks, inference
snapshot metadata, source pin, fixed suite, run manifest, and final metrics are
under `evidence/`. `integration_overlay/` preserves the exact path-adapted
PhiAgent preparation, guarded launch, and held-out evaluation files that ran on
the H20 host. The final real-input target-blind check and its 81/81-frame review
are under `integration_evidence/`.

This is a post-run release binding. The original checkpoint sidecar did not
embed a source-tree digest, so the retained workspace and evidence let this
package preserve and bind the final state retrospectively, but cannot turn it
into an in-checkpoint build attestation. Future training should write the source
manifest hash into checkpoint metadata before the first optimizer update.

## Verification

From the V1 package root, verify the complete captured payload without PyTorch:

```bash
python tools/verify_final_release.py
```

If a separately transferred checkpoint is available, bind-check both its size
and its content hash:

```bash
python tools/verify_final_release.py --checkpoint /path/to/step-016000.pt
```

The archived evaluation tools include `evaluate_stage1.py`, the fixed-suite
runner and reporting modules, and custom target-withheld inference. The four
case fixed suite is a calibrated multiview monitor. The separate TACO check was
also target-blind during generation but failed its camera/composition gate; it
does not accept an arbitrary ego→SIM→exo pipeline.
