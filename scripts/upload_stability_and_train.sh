#!/bin/bash
set -e
cd /workspace/PhysSim-VLM

export HF_TOKEN=YOUR_HF_TOKEN_HERE
export WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE
export WANDB_PROJECT=PhysSim-VLM
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false

STAB_GEN_PID=21916

echo "[1/3] Waiting for stability generation (PID $STAB_GEN_PID)..."
while kill -0 $STAB_GEN_PID 2>/dev/null; do
    COUNT=$(ls data/generated/stability/ 2>/dev/null | wc -l)
    echo " Stability scenes: $COUNT/5000 [$(date +%H:%M:%S)]"
    sleep 30
done

COUNT=$(ls data/generated/stability/ 2>/dev/null | wc -l)
echo "Generation done: $COUNT stability scenes"
echo ""

echo "[2/3] Uploading stability scenes to HuggingFace..."
python3 - <<'PYEOF'
import os, json, base64, random
from pathlib import Path
from PIL import Image
import io
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
ROOT = Path("/workspace/PhysSim-VLM")
STAB_DIR = ROOT / "data" / "generated" / "stability"

def gt_answer(gt):
    return "stable" if gt.get("is_stable", False) else "unstable"

def gt_reasoning(gt):
    stable = gt.get("is_stable", False)
    disp = gt.get("max_displacement_m", 0.0)
    if stable:
        return f"The stack is balanced; base supports layers above. Max displacement = {disp:.4f}m - no collapse. Stable."
    else:
        tc = gt.get("collapse_time_s", 0.0)
        return f"Top-heavy or misaligned. Displacement reaches {disp:.4f}m; collapses at {tc:.2f}s. Unstable."

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

print("Loading stability scenes...")
records = []
for scene_dir in sorted(STAB_DIR.iterdir()):
    if not scene_dir.is_dir():
        continue
    gt_path = scene_dir / "ground_truth.json"
    prompt_path = scene_dir / "prompt.txt"
    if not gt_path.exists() or not prompt_path.exists():
        continue

    with open(gt_path) as f:
        gt = json.load(f)
    with open(prompt_path) as f:
        prompt = f.read().strip()

    # Find image
    img_path = None
    for candidate in ["scene.png", "frame_000.png", "frames/frame_000.png"]:
        p = scene_dir / candidate
        if p.exists():
            img_path = p
            break
    if img_path is None:
        continue

    answer = gt_answer(gt)
    reasoning = gt_reasoning(gt)
    assistant_text = f"<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>"
    config_path = scene_dir / "config.json"
    config_str = open(config_path).read() if config_path.exists() else "{}"

    records.append({
        "scene_id": scene_dir.name,
        "task": "stability",
        "difficulty": gt.get("difficulty", "unknown"),
        "prompt": prompt,
        "answer": answer,
        "reasoning": reasoning,
        "assistant_text": assistant_text,
        "n_frames": 1,
        "frames_b64": [encode_image(img_path)],
        "ground_truth": json.dumps(gt),
        "config": config_str,
    })

print(f"Loaded {len(records)} stability records")

# Split 80/10/10
random.seed(42)
random.shuffle(records)
n = len(records)
train_end = int(n * 0.80)
val_end = int(n * 0.90)
train_recs = [{**r, "split": "train"} for r in records[:train_end]]
val_recs = [{**r, "split": "val"} for r in records[train_end:val_end]]
test_recs = [{**r, "split": "test"} for r in records[val_end:]]
print(f"Split: train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")

# Load existing HF dataset and merge
print("Loading existing HF dataset...")
existing = load_dataset("Swastikr/PhysSim-VLM-Dataset", token=HF_TOKEN)

new_train = Dataset.from_list(train_recs)
new_val = Dataset.from_list(val_recs)
new_test = Dataset.from_list(test_recs)

merged = DatasetDict({
    "train": concatenate_datasets([existing["train"], new_train]),
    "val": concatenate_datasets([existing["val"], new_val]),
    "test": concatenate_datasets([existing["test"], new_test]),
})

print(f"Merged dataset: train={len(merged['train'])} val={len(merged['val'])} test={len(merged['test'])}")

print("Pushing to HuggingFace...")
merged.push_to_hub("Swastikr/PhysSim-VLM-Dataset", token=HF_TOKEN)
print("Upload complete!")
PYEOF

echo ""
echo "[3/3] Starting full training on 15k dataset..."
# Remove local data so HF dataset is used
rm -rf data/generated

python3 scripts/train_lora_sft.py > /tmp/training.log 2>&1
echo "Training complete!"
