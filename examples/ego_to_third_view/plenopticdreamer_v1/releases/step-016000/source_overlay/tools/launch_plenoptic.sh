#!/usr/bin/env bash
set -euo pipefail
_pleno_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$_pleno_script_dir/activate_h20.sh"
cd "$PLENOPTIC_ROOT"
_pleno_config="${1:-configs/plenoptic/smoke_1gpu.json}"
if (( $# )); then shift; fi
_pleno_entry="$(python - "$_pleno_config" <<'PY'
import json, sys
from plenoptic_paths import rooted, ensure_migration_complete, conflicting_training, LEGACY_ROOT, handoff_source_mirror
ensure_migration_complete()
if any(item['cwd'] == str(LEGACY_ROOT) for item in conflicting_training()):
    raise RuntimeError('The original local training is still active')
handoff_source_mirror()
config = json.loads(rooted(sys.argv[1]).read_text())
print('tools/train_tabletop.py' if config.get('phase') == 'tabletop_sft' else 'tools/train_plenoptic.py')
PY
)"
if (( ${NNODES:-1} > 1 )); then
  : "${MASTER_ADDR:?Set MASTER_ADDR to a reachable address on the head node}"
  : "${NODE_RANK:?Set NODE_RANK to a unique node index starting at zero}"
  if [[ "$MASTER_ADDR" == 127.* || "$MASTER_ADDR" == localhost ]]; then
    echo 'MASTER_ADDR must be reachable by all nodes.' >&2
    exit 2
  fi
fi
exec python -m torch.distributed.run \
  --nnodes="${NNODES:-1}" --nproc_per_node="${NPROC_PER_NODE:-1}" \
  --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29670}" \
  "$_pleno_entry" --config "$_pleno_config" "$@"
