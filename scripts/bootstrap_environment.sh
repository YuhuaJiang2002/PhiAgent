#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  printf 'conda is required but was not found\n' >&2
  exit 1
fi

conda env create --file environment.yml
conda run --no-capture-output -n phiagent \
  python -m pip install --requirement requirements/wan-animate.txt
conda run --no-capture-output -n phiagent \
  python -m pip install --no-build-isolation --requirement requirements/sam2.txt
conda run --no-capture-output -n phiagent \
  python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
conda run --no-capture-output -n phiagent python -m pytest
