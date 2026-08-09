#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/external/Wan-Animate-2"
VENV="$ROOT/.venv-wan-animate2"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is required but was not found\n' >&2
  exit 1
fi
if [[ ! -f "$REPO/requirements.txt" ]]; then
  printf 'prepare the pinned Wan-Animate-2 source before bootstrapping\n' >&2
  exit 1
fi

uv python install 3.11
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.11 "$VENV"
fi
uv pip install \
  --python "$VENV/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --requirement "$REPO/requirements.txt"
uv pip install \
  --python "$VENV/bin/python" \
  --no-build-isolation \
  flash-attn==2.7.4.post1
uv pip install --python "$VENV/bin/python" --no-deps --editable "$REPO"

"$VENV/bin/python" -c \
  'import flash_attn, torch; print(torch.__version__, torch.version.cuda, flash_attn.__version__)'
