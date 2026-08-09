# Frozen PhiZero reproduction target

Evidence date: 2026-08-08.

## Authoritative sources

- Paper: `PhiZero: A World Model Built Around Physical Language`,
  arXiv:2607.28624v1, Figure 8(b), Appendix C.2.
- Project page: <https://phi-zero.github.io/>.
- Project-page revision:
  `72fc49fb17b56fab6f7407239b38bdedf7c76546`.
- Official code repository: <https://github.com/yaoyao-jpg/PhiZero>, pinned
  revision `6bc7428f2ad5282e0c1a7b122465957b6abb1edc`.

At the pinned revision the official repository says code and pretrained models
are being prepared for release. No local component may claim exact PhiZero
execution until released source and checkpoints are pinned and run.

## Target behavior

Reproduce Human Hand to Sharpa Dexterous Hand Transfer, not a two-arm object
handover:

1. Briefly adapt the Physical Language Tokenizer on human-hand videos from
   HRDexDB without using paired robot videos or cross-embodiment correspondence.
2. Encode a source human-hand video's state transitions into the learned discrete
   physical-language sequence.
3. Edit the first frame to replace the human hand with the Sharpa dexterous hand.
   The paper uses GPT-Image 2.0.
4. Decode the unchanged physical-language sequence conditioned on the edited
   first frame.

The tokenizer uses a Wan2.2 VAE-style spatiotemporal encoder, transition-level
Q-Former with 32 queries per adjacent latent-state pair, and
`FSQ(8, 5, 5, 5, 5, 5)` for a 25K-symbol vocabulary. The decoder is initialized
from Wan2.2-5B and adapted with rank-32 LoRA. A standard 33-frame clip maps to
nine latent temporal states and 256 physical-language symbols.

## Pinned public references

All six files are 3 seconds and 896x512. They are downloaded only on demand with
`python scripts/prepare_phizero_reference.py`.

| Case | Role | SHA-256 |
| --- | --- | --- |
| 1 | source | `d863b91c4f160d0b634a73a6b996b5362359245910e7402197b78a14e5ad9a03` |
| 1 | transferred | `014d3615ba0448e6500cbe76f63801603943373d7e47ae7562a1af75666dcd39` |
| 2 | source | `3245e585ad2351f1340b02ab37e211866960b89a4d24e4b3a78f5c041b064098` |
| 2 | transferred | `2fc8e9c3d48ed987fabc3e90bcc47cb2432433f8137346a0a3a641088c663327` |
| 3 | source | `ad33958a7e291aa2edde71f4437de2959bff2950e740a98e0ee87ee2c787a8c0` |
| 3 | transferred | `af57f9606c7d27696df573ccd086eee4e9f3481512e93d09f2a534ea28ccc834` |

## Acceptance

An accepted reproduction must use released PhiZero tokenizer/decoder code and
weights, preserve the encoded transition tokens unchanged during transfer, and
save the source video, edited first frame, tokens, adaptation configuration,
source/model revisions, environment, GPU selection, seed, logs, and output in a
new experiment directory. It must be evaluated against all three reference
pairs for motion preservation, Sharpa identity/geometry, object interaction, and
temporal consistency.
