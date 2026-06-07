#!/usr/bin/env python3
"""Upload SFT R2 dataset to HuggingFace Hub."""

import os
import json
from pathlib import Path
from collections import Counter

# Load from local data/sft_r2/
DATA_DIR = Path(__file__).parent.parent / "data" / "sft_r2"
HF_TOKEN = os.getenv("HF_TOKEN")

print("Loading SFT R2 scenes...")
records = []
for task_dir in sorted(DATA_DIR.glob("*")):
    if not task_dir.is_dir():
        continue

    task = task_dir.name
    scene_dirs = list(task_dir.glob("*"))
    print(f" {task}: {len(scene_dirs)} scenes")

    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            continue

        gt_path = scene_dir / "ground_truth.json"
        prompt_path = scene_dir / "prompt.txt"
        asst_path = scene_dir / "assistant_text.txt"

        if not all([gt_path.exists(), prompt_path.exists(), asst_path.exists()]):
            continue

        with open(gt_path) as f:
            gt = json.load(f)
        with open(prompt_path) as f:
            prompt = f.read().strip()
        with open(asst_path) as f:
            asst = f.read().strip()

        # Count frames
        frames_dir = scene_dir / "frames"
        if frames_dir.exists():
            n_frames = len(list(frames_dir.glob("frame_*.png")))
        else:
            n_frames = 1 if (scene_dir / "scene.png").exists() else 0

        if n_frames == 0:
            continue

        records.append({
            "scene_id": scene_dir.name,
            "task": task,
            "split": "train",
            "prompt": prompt,
            "assistant_text": asst,
            "n_frames": n_frames,
            "ground_truth": json.dumps(gt),
        })

print(f"\nTotal records: {len(records)}")
print(f"Task distribution:")
for task, count in sorted(Counter(r["task"] for r in records).items()):
    print(f" {task}: {count}")

if records and HF_TOKEN:
    print(f"\nUploading to HuggingFace...")
    from datasets import Dataset

    ds = Dataset.from_list(records)
    HF_DATASET = "Swastikr/PhysSim-VLM-SFT-R2-Data"

    try:
        ds.push_to_hub(HF_DATASET, split="train", token=HF_TOKEN, private=True)
        print(f"\n[OK] Success! {len(records)} samples uploaded to {HF_DATASET}")
    except Exception as e:
        print(f"\n[ERR] Upload failed: {e}")
else:
    print("ERROR: No HF_TOKEN")
