#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  printf 'usage: %s SESSION COMMAND [ARG ...]\n' "$0" >&2
  exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required for long-running jobs\n' >&2
  exit 1
fi

session="$1"
shift
if tmux has-session -t "$session" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session" >&2
  exit 1
fi
printf -v quoted_command '%q ' "$@"
tmux new-session -d -s "$session" -c "$PWD" "$quoted_command"
printf 'Started tmux session %s; attach with: tmux attach -t %s\n' "$session" "$session"

