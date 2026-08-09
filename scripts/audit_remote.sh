#!/usr/bin/env bash
set -euo pipefail

host="${1:-phi-a800}"
output="${2:-docs/audits/${host}-$(date -u +%Y%m%dT%H%M%SZ).json}"
temporary="${output}.tmp.$$"

mkdir -p "$(dirname "$output")"
trap 'rm -f "$temporary"' EXIT
ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" python3 -s \
  < scripts/audit_environment.py > "$temporary"
mv "$temporary" "$output"
trap - EXIT
printf 'Wrote %s\n' "$output"
