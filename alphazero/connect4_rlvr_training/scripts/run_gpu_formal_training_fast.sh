#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-runs/gpu_ablation_fast_logs}"
mkdir -p "$LOG_DIR"

if command -v tmux >/dev/null 2>&1; then
  SESSION="${SESSION:-ibrlvr_gpu_train_fast}"
  tmux new-session -d -s "$SESSION" "cd '$PWD' && bash scripts/run_gpu_ablation_matrix_fast.sh > '$LOG_DIR/launcher.out' 2> '$LOG_DIR/launcher.err'"
  tmux display-message -p -t "$SESSION" '#{pid}' > "$LOG_DIR/launcher.pid"
  echo "started tmux session $SESSION pid $(cat "$LOG_DIR/launcher.pid")"
  echo "attach: tmux attach -t $SESSION"
else
  setsid bash scripts/run_gpu_ablation_matrix_fast.sh > "$LOG_DIR/launcher.out" 2> "$LOG_DIR/launcher.err" < /dev/null &
  echo "$!" > "$LOG_DIR/launcher.pid"
  disown || true
  echo "started detached fast launcher pid $(cat "$LOG_DIR/launcher.pid")"
fi
echo "logs: $LOG_DIR"
