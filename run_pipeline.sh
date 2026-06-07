#!/bin/bash
# PhysSim-VLM: Full pipeline runner
# Waits for data generation to finish, then starts training

cd /workspace/PhysSim-VLM

export WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE
export WANDB_PROJECT=PhysSim-VLM
export HF_TOKEN=YOUR_HF_TOKEN_HERE
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false

echo "============================================"
echo " PhysSim-VLM Pipeline"
echo " $(date)"
echo "============================================"

# Wait for data generation process to finish
DATA_GEN_PID=6661
echo "[1/2] Waiting for data generation (PID $DATA_GEN_PID)..."
while kill -0 $DATA_GEN_PID 2>/dev/null; do
    TTC=$(ls data/generated/ttc/ 2>/dev/null | wc -l)
    STAB=$(ls data/generated/stability/ 2>/dev/null | wc -l)
    TRAJ=$(ls data/generated/trajectory/ 2>/dev/null | wc -l)
    echo " Progress: TTC=$TTC, Stability=$STAB, Trajectory=$TRAJ [$(date +%H:%M:%S)]"
    sleep 60
done

echo ""
TTC_COUNT=$(ls data/generated/ttc/ 2>/dev/null | wc -l)
STAB_COUNT=$(ls data/generated/stability/ 2>/dev/null | wc -l)
TRAJ_COUNT=$(ls data/generated/trajectory/ 2>/dev/null | wc -l)
echo "Data generation complete!"
echo "Scenes: TTC=$TTC_COUNT, Stability=$STAB_COUNT, Trajectory=$TRAJ_COUNT"
echo ""

# Start training
echo "[2/2] Starting LoRA SFT training at $(date)..."
echo " WandB project: PhysSim-VLM"
echo " Output: checkpoints/lora_sft_epoch1/"
echo ""

python3 scripts/train_lora_sft.py 2>&1 | tee /tmp/training.log

echo ""
echo "============================================"
echo " Training complete! $(date)"
echo "============================================"
