@echo off
cd /d "C:\Users\Swastik R\Documents\Personal_Projects\VLM and Physics"
if not exist "results\grpo_tinker\grpo-r3-from-r2redo" mkdir "results\grpo_tinker\grpo-r3-from-r2redo"
python -u scripts/train_grpo_tinker.py --run-name grpo-r3-from-r2redo >> results\grpo_tinker\grpo-r3-from-r2redo\train_final.log 2>&1
