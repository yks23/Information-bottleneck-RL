#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"
export WANDB_MODE="${WANDB_MODE:-offline}"

STEPS="${STEPS:-3000}"
EPISODES_PER_STEP="${EPISODES_PER_STEP:-8}"
LOG_DIR="${LOG_DIR:-runs/gpu_ablation_logs}"
mkdir -p "$LOG_DIR"

run_train() {
  local gpu="$1"
  local config="$2"
  local seed="$3"
  local run_name="$4"
  local noise_c="$5"
  local shared="$6"
  local log_path="$LOG_DIR/${run_name}.log"

  echo "[$(date -Is)] gpu=${gpu} start ${run_name}" | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m ibrlvr.cli.train \
    --config "$config" \
    --override seed="$seed" \
    --override run_name="$run_name" \
    --override training.steps="$STEPS" \
    --override training.episodes_per_step="$EPISODES_PER_STEP" \
    --override reward.noise_c="$noise_c" \
    --override self_play.shared_params="$shared" \
    >"$log_path" 2>&1
  echo "[$(date -Is)] gpu=${gpu} done ${run_name}" | tee -a "$LOG_DIR/launcher.log"
}

worker0() {
  run_train 0 configs/connect4_gpu_rollout32.yaml 0 connect4_baseline_rollout32_seed0 0.0 true
  run_train 0 configs/connect4_gpu_rollout32.yaml 1 connect4_baseline_rollout32_seed1 0.0 true
  run_train 0 configs/connect4_gpu_rollout32.yaml 2 connect4_baseline_rollout32_seed2 0.0 true
}

worker1() {
  run_train 1 configs/connect4_gpu_noise.yaml 0 connect4_noise_c001_seed0 0.01 true
  run_train 1 configs/connect4_gpu_noise.yaml 1 connect4_noise_c001_seed1 0.01 true
  run_train 1 configs/connect4_gpu_noise.yaml 2 connect4_noise_c001_seed2 0.01 true
}

worker2() {
  run_train 2 configs/connect4_gpu_noise.yaml 0 connect4_noise_c005_seed0 0.05 true
  run_train 2 configs/connect4_gpu_noise.yaml 1 connect4_noise_c005_seed1 0.05 true
  run_train 2 configs/connect4_gpu_noise.yaml 2 connect4_noise_c005_seed2 0.05 true
  run_train 2 configs/connect4_gpu_noise.yaml 0 connect4_noise_c010_seed0 0.10 true
}

worker3() {
  run_train 3 configs/connect4_gpu_nonshared.yaml 0 connect4_nonshared_rollout32_seed0 0.0 false
  run_train 3 configs/connect4_gpu_nonshared.yaml 1 connect4_nonshared_rollout32_seed1 0.0 false
  run_train 3 configs/connect4_gpu_nonshared.yaml 2 connect4_nonshared_rollout32_seed2 0.0 false
  run_train 3 configs/connect4_gpu_noise.yaml 1 connect4_noise_c010_seed1 0.10 true
  run_train 3 configs/connect4_gpu_noise.yaml 2 connect4_noise_c010_seed2 0.10 true
}

worker0 &
pid0=$!
worker1 &
pid1=$!
worker2 &
pid2=$!
worker3 &
pid3=$!

echo "$pid0 $pid1 $pid2 $pid3" > "$LOG_DIR/worker_pids.txt"
wait "$pid0" "$pid1" "$pid2" "$pid3"
"$PYTHON_BIN" scripts/export_run_summary.py --runs-dir runs --out runs/summary.csv
