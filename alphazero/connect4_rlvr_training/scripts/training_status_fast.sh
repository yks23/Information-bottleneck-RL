#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-runs/gpu_ablation_fast_logs}"

echo "== fast launcher =="
if [[ -f "$LOG_DIR/launcher.pid" ]]; then
  pid="$(cat "$LOG_DIR/launcher.pid")"
  ps -p "$pid" -o pid,ppid,stat,etime,cmd || true
else
  echo "no launcher pid at $LOG_DIR/launcher.pid"
fi

echo
echo "== fast workers =="
if [[ -f "$LOG_DIR/worker_pids.txt" ]]; then
  while read -r pid; do
    ps -p "$pid" -o pid,ppid,stat,etime,cmd || true
  done < "$LOG_DIR/worker_pids.txt"
else
  echo "workers not started yet"
fi

echo
echo "== train processes =="
ps -eo pid,ppid,stat,pcpu,pmem,etime,cmd --sort=pid | grep -E 'python -m ibrlvr.cli.train' | grep -v grep || true

echo
echo "== gpu =="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw --format=csv

echo
echo "== latest fast metrics =="
find runs -path "*/metrics.jsonl" -type f -printf "%T@ %p\n" \
  | sort -nr \
  | grep 'connect4_fast_' \
  | head -12 \
  | while read -r _ path; do
      printf "%s: " "$path"
      tail -n 1 "$path"
    done
