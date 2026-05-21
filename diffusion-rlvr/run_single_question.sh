#!/bin/bash
# ============================================================
# Launch script for H800 — run on a single 80GB H800
# ============================================================
#
# Usage:
#   bash run_single_question.sh          # default: 100 epochs
#   bash run_single_question.sh 200      # custom epochs
#
# What this does:
#   1. Installs dependencies (if needed)
#   2. Downloads SD1.5 (~4GB) on first run
#   3. Starts DDPO training on single prompt "a red square"
#   4. Logs to wandb project "diffusion-rlvr"
#
# Expected memory usage (LoRA rank=4):
#   - UNet with LoRA: ~2-3 GB VRAM
#   - VAE (fp16): ~1 GB
#   - Text encoder: ~1 GB
#   - Activations + rollouts: ~3-4 GB
#   - Total ≈ 8-10 GB (well within 80GB H800)
#
# ============================================================

EPOCHS=${1:-100}
LOG_NAME="sq_${EPOCHS}ep_$(date +%m%d_%H%M)"

echo "=============================================="
echo "  Diffusion RLVR — Single Question Experiment"
echo "  Epochs : $EPOCHS"
echo "  Run    : $LOG_NAME"
echo "=============================================="

# Install dependencies
echo "[1/3] Installing dependencies..."
pip install -q torch diffusers transformers accelerate peft wandb Pillow

# Login wandb (uncomment if needed)
# wandb login

# Launch training
echo "[2/3] Starting training..."
python3 /Users/oyly/Desktop/kaisen/single_question_diffusion_rlvr.py \
    --num_epochs $EPOCHS \
    --sample_batch_size 16 \
    --lora_rank 4 \
    --lr 1e-4 \
    --run_name "$LOG_NAME"

echo "[3/3] Done."
