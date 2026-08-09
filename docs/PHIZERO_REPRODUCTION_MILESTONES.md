# PhiZero Figure 8(b) reproduction milestones

## Milestone 0: freeze references - WORKING

Pin arXiv:2607.28624v1, the official project/code revisions, Appendix C.2
protocol, and all three public `hand2dex` source/transferred pairs with hashes.

Acceptance requires a machine-verifiable manifest and target specification.

## Milestone 1: official release intake - BLOCKED

When released, review licenses and pin the official PhiZero source, tokenizer,
decoder, source-domain adaptation checkpoint, and all dependent model revisions.
Do not vendor repositories or checkpoints.

Acceptance requires import and preflight without executing inference.

## Milestone 2: tokenizer execution - NOT STARTED

Run the official spatiotemporal encoder, transition-level Q-Former, and FSQ
quantizer on a pinned human-hand source. Save the discrete sequence and verify
its shape and vocabulary against the released implementation.

Acceptance requires real source-video tokens from the official checkpoint.

## Milestone 3: Sharpa first-frame condition - NOT STARTED

Reproduce or legally replace the paper's GPT-Image 2.0 first-frame edit while
recording prompt, provider/model revision, seed where supported, and output hash.

Acceptance requires a reviewed frame that changes the embodiment to Sharpa while
preserving the scene, object, viewpoint, and interaction state.

## Milestone 4: unchanged-token decoding - NOT STARTED

Decode the exact source token sequence with the edited Sharpa first frame using
the released PhiZero Wan2.2-5B decoder. Record evidence that no tokens were
changed between encoding and decoding.

Acceptance requires a decodable transferred video and complete run provenance.

## Milestone 5: reference evaluation - NOT STARTED

Evaluate all three cases for motion preservation, Sharpa identity and geometry,
object-interaction consistency, temporal artifacts, and similarity to the pinned
reference outputs. Report failures rather than selecting only a successful case.

Acceptance requires declared metrics and reviewed side-by-side artifacts for all
three cases.

## Milestone 6: end-to-end reproduction - NOT STARTED

Run source adaptation, encoding, first-frame conditioning, unchanged-token
decoding, and evaluation in immutable experiment directories.

Acceptance requires every prior gate to pass. EPL, explicit robot retargeting,
simulation, Cosmos, Wan2.2-Animate, or ArtiCraft results may be auxiliary
experiments but cannot replace the official PhiZero path.

## Parallel proxy track - PARTIAL

While Milestone 1 is blocked, run a clearly labelled agentic approximation:
generate Wan/Cosmos candidates from Sharpa first-frame variants, evaluate motion,
identity, object, and temporal consistency locally, and use failed dimensions to
drive bounded repair rounds. Proxy success informs engineering work but does not
advance exact-reproduction Milestones 1-6.
