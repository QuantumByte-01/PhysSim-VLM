#!/bin/bash
# Watchdog: auto-restarts the GRPO eval if predictions.json goes stale >8 min

PRED_FILE="results/physbench_grpo_r3_test/predictions.json"
LOG_FILE="results/physbench_grpo_r3_test/eval_resume.log"
MODEL_PATH="tinker://<run-id>:train:0/sampler_weights/final"
CONCURRENCY=12
STALE_THRESHOLD=480 # 8 minutes in seconds
TOTAL=9786

cd "/c/Users/Swastik R/Documents/Personal_Projects/VLM and Physics"

echo "[watchdog] Started at $(date '+%H:%M:%S')"

while true; do
    # Check if done
    DONE=$(python -c "import json; print(len(json.load(open('$PRED_FILE'))))" 2>/dev/null)
    if [ "$DONE" -ge "$TOTAL" ] 2>/dev/null; then
        echo "[watchdog] Eval complete! $DONE/$TOTAL samples done. Exiting."
        exit 0
    fi

    # Check staleness
    LAST_MOD=$(python -c "import os,time; print(int(time.time() - os.path.getmtime('$PRED_FILE')))" 2>/dev/null)

    if [ "$LAST_MOD" -gt "$STALE_THRESHOLD" ] 2>/dev/null; then
        echo "[watchdog] $(date '+%H:%M:%S') STALLED ($LAST_MOD sec) - killing and restarting from $DONE..."
        # Kill any running eval
        ps aux | grep "eval_physbench_tinker" | grep -v grep | awk '{print $1}' | xargs kill 2>/dev/null
        sleep 3
        # Restart
        nohup python -u scripts/eval_physbench_tinker.py \
            --model-path "$MODEL_PATH" \
            --out-tag grpo_r3_test \
            --split test \
            --concurrency $CONCURRENCY >> "$LOG_FILE" 2>&1 &
        NEW_PID=$!
        echo "[watchdog] Restarted with PID $NEW_PID"
        # Wait a bit before next check to let it start up
        sleep 60
    else
        echo "[watchdog] $(date '+%H:%M:%S') OK - $DONE/$TOTAL done, last write $LAST_MOD sec ago"
        sleep 60
    fi
done
