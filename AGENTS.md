# PhiAgent-0 contributor instructions

PhiAgent-0 is a research system whose results must be reproducible and honestly
reported.

- Keep heavyweight dependencies optional and behind adapters. Importing the
  phiagent package must not require CUDA, PyTorch, simulators, or checkpoints.
- Never mix camera, world, and robot-base coordinates implicitly. New geometric
  data structures must name their frame and new transforms require tests.
- Every GPU entry point must inspect GPU state, select or validate a physical GPU,
  set CUDA_VISIBLE_DEVICES, and save the selection with the experiment.
- Every experiment must use a new directory and save its configuration, commands,
  Git state, hostname, package versions, seed, logs, and outputs.
- Every concluded experiment or engineering attempt must append its success,
  partial result, failure, blocker, or not-started decision to
  `experiences/ledger.jsonl` through `scripts/experience_ledger.py`. Never erase
  a failed attempt; append a correcting record with `supersedes` when evidence
  changes the conclusion.
- Do not claim a milestone is working until its acceptance test has run on real
  inputs. Use WORKING, PARTIAL, NOT STARTED, or BLOCKED in docs/STATUS.md.
- Do not vendor model repositories or checkpoints. Pin third-party revisions in
  scripts or requirements and store them under ignored directories.
- Unit tests must remain runnable without a GPU. GPU/model tests must be marked
  explicitly and fail with a useful preflight diagnosis when prerequisites are
  absent.
