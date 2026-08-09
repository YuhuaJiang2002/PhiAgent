#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="${1:-${project_root}/.venv-gpu}"
build_cache="${project_root}/.cache"

# Keep temporary wheels and installer caches on the same filesystem. Shared
# servers may place the default pip cache on /data1 while this project lives
# on /data0; flash-attn uses an atomic rename that fails across mounts.
mkdir -p "${build_cache}/pip" "${build_cache}/tmp" "${build_cache}/uv"
export PIP_CACHE_DIR="${build_cache}/pip"
export TMPDIR="${build_cache}/tmp"
export UV_CACHE_DIR="${build_cache}/uv"

if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is required but was not found\n' >&2
  exit 1
fi
if [[ -x "$venv_path/bin/python" ]]; then
  printf 'Reusing partial environment at %s\n' "$venv_path"
elif [[ -e "$venv_path" ]]; then
  printf 'environment path exists but is not a valid venv: %s\n' "$venv_path" >&2
  exit 1
else
  uv venv --python /usr/bin/python3 "$venv_path"
fi

uv pip install --python "$venv_path/bin/python" \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "$venv_path/bin/python" \
  pip setuptools wheel packaging ninja
uv pip install --python "$venv_path/bin/python" \
  --editable "${project_root}[dev,simulation]" \
  --requirement "$project_root/requirements/wan-animate.txt"
"$venv_path/bin/python" -m pip install \
  flash-attn==2.7.4.post1 --no-build-isolation
"$venv_path/bin/python" -m pytest "$project_root/tests"
