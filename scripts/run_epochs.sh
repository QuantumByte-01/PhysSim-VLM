#!/usr/bin/env bash
# PhysSim-VLM - Auto-continue training across epochs
# Usage: bash scripts/run_epochs.sh [start_epoch] [end_epoch]
# e.g. bash scripts/run_epochs.sh 1 3 (epochs 1,2,3)
#
# Epoch 1 is assumed to be running already (PID in /tmp/sft_epoch1.pid if launched here).
# For epochs 2+, this script resumes from the previous epoch's final checkpoint.

set -euo pipefail

START_EPOCH=${1:-2}
END_EPOCH=${2:-2}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TRAIN_ENV=(
    USE_ROCM_AITER_ROPE_BACKEND=0
    TOKENIZERS_PARALLELISM=false
    MUJOCO_GL=osmesa
    PYTORCH_ALLOC_CONF=expandable_segments:True
    HIP_FORCE_DEV_KERNARG=1
    FLASH_ATTENTION_TRITON_AMD_ENABLE=1
    MIOPEN_FIND_MODE=2
)

# Wait for epoch 1 (PID 79493) to finish
EPOCH1_PID=79493
if kill -0 "$EPOCH1_PID" 2>/dev/null; then
    echo "[$(date -u '+%H:%M:%S UTC')] Waiting for epoch 1 (PID $EPOCH1_PID) to finish..."
    wait "$EPOCH1_PID" || true
    echo "[$(date -u '+%H:%M:%S UTC')] Epoch 1 complete."
fi

for EPOCH in $(seq "$START_EPOCH" "$END_EPOCH"); do
    PREV_EPOCH=$((EPOCH - 1))
    RESUME_DIR="$ROOT/checkpoints/lora_sft_epoch${PREV_EPOCH}/final"
    LOG="/tmp/sft_epoch${EPOCH}.log"

    echo ""
    echo "[$(date -u '+%H:%M:%S UTC')] ════════════════════════════════════"
    echo "[$(date -u '+%H:%M:%S UTC')] Starting Epoch $EPOCH"
    echo "[$(date -u '+%H:%M:%S UTC')] Resume from: $RESUME_DIR"
    echo "[$(date -u '+%H:%M:%S UTC')] Log: $LOG"
    echo "[$(date -u '+%H:%M:%S UTC')] ════════════════════════════════════"

    if [ ! -d "$RESUME_DIR" ]; then
        echo "[ERROR] Checkpoint not found: $RESUME_DIR"
        echo " Epoch $PREV_EPOCH may have failed. Aborting."
        exit 1
    fi

    env "${TRAIN_ENV[@]}" python "$ROOT/scripts/train_lora_sft.py" \
        --resume "$RESUME_DIR" \
        --epoch_num "$EPOCH" \
        > "$LOG" 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date -u '+%H:%M:%S UTC')] Epoch $EPOCH FAILED (exit $EXIT_CODE). Check $LOG"
        exit $EXIT_CODE
    fi

    echo "[$(date -u '+%H:%M:%S UTC')] Epoch $EPOCH complete. Results: $ROOT/results/sft_epoch${EPOCH}/results.md"
done

echo ""
echo "[$(date -u '+%H:%M:%S UTC')] All epochs $START_EPOCH - $END_EPOCH complete."
