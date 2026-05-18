#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

COMMON_CONFIG="${COMMON_CONFIG:-configs/connect4_small_cpu_rollout32.yaml}"
NONSHARED_CONFIG="${NONSHARED_CONFIG:-configs/connect4_small_cpu_nonshared.yaml}"
STEPS="${STEPS:-300}"

for seed in 0 1 2; do
  "$PYTHON_BIN" -m ibrlvr.cli.train \
    --config "$COMMON_CONFIG" \
    --override seed="$seed" \
    --override run_name="connect4_baseline_rollout32_seed${seed}" \
    --override training.steps="$STEPS" \
    --override reward.noise_c=0.0 \
    --override self_play.shared_params=true

  for c in 0.01 0.05 0.1; do
    "$PYTHON_BIN" -m ibrlvr.cli.train \
      --config "$COMMON_CONFIG" \
      --override seed="$seed" \
      --override run_name="connect4_noise_c${c}_seed${seed}" \
      --override training.steps="$STEPS" \
      --override reward.noise_c="$c" \
      --override self_play.shared_params=true
  done

  "$PYTHON_BIN" -m ibrlvr.cli.train \
    --config "$NONSHARED_CONFIG" \
    --override seed="$seed" \
    --override run_name="connect4_nonshared_rollout32_seed${seed}" \
    --override training.steps="$STEPS" \
    --override reward.noise_c=0.0 \
    --override self_play.shared_params=false
done
