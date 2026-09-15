#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ "$(hostname -s)" != "h20-1" ]]; then
    echo "Validation GPU work is permitted only on h20-1." >&2
    exit 1
fi
source tools/activate_h20.sh
if [[ "${1:-}" == "--check-only" || ( "${1:-}" == "--refine" && "${2:-}" == "--check-only" ) ]]; then
    exec env CUDA_VISIBLE_DEVICES= python tools/evaluate_stage1.py "$@"
fi
export NCCL_IB_DISABLE=1
export GLOO_SOCKET_IFNAME=eth0
_validation_batch=1
_validation_previous=""
_validation_loss_only=false
for _validation_arg in "$@"; do
    if [[ "$_validation_previous" == "--video-batch-size" ]]; then
        _validation_batch="$_validation_arg"
    elif [[ "$_validation_arg" == --video-batch-size=* ]]; then
        _validation_batch="${_validation_arg#*=}"
    elif [[ "$_validation_arg" == "--loss-only" ]]; then
        _validation_loss_only=true
    fi
    _validation_previous="$_validation_arg"
done
if [[ "$_validation_loss_only" == true ]]; then _validation_batch=1; fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$(python tools/validation_resources.py --video-batch-size "$_validation_batch")"
    export CUDA_VISIBLE_DEVICES
fi
IFS=',' read -r -a _validation_devices <<< "$CUDA_VISIBLE_DEVICES"
_validation_cp="${#_validation_devices[@]}"
case "$_validation_cp" in
    4) ;;
    *) echo "Fixed validation requires exactly four selected GPUs." >&2; exit 1 ;;
esac
exec python -m torch.distributed.run --nnodes=1 --nproc_per_node="$_validation_cp" \
    --master_addr=127.0.0.1 --master_port=29691 tools/evaluate_stage1.py "$@"
