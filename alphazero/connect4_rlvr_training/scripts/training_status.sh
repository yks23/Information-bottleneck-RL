#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-runs/gpu_ablation_logs}"

echo "== launcher =="
if [[ -f "$LOG_DIR/launcher.pid" ]]; then
  pid="$(cat "$LOG_DIR/launcher.pid")"
  ps -p "$pid" -o pid,ppid,stat,etime,cmd || true
else
  echo "no launcher pid at $LOG_DIR/launcher.pid"
fi

echo
echo "== workers =="
if [[ -f "$LOG_DIR/worker_pids.txt" ]]; then
  for pid in $(cat "$LOG_DIR/worker_pids.txt"); do
    ps -p "$pid" -o pid,ppid,stat,etime,cmd || true
  done
else
  echo "workers not started yet"
fi

echo
echo "== gpu =="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw --format=csv

echo
echo "== latest metrics =="
find runs -path "*/metrics.jsonl" -type f -printf "%T@ %p\n" | sort -nr | head -8 | while read -r _ path; do
  printf "%s: " "$path"
  tail -n 1 "$path"
done
