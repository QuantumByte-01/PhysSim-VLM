#!/usr/bin/env python3
"""
PhysSim-VLM: Pre-tokenize Dataset
========================================
Runs the Qwen3-VL processor over all training/val scenes ONCE and saves
input_ids, attention_mask, labels, pixel_values, image_grid_thw to disk.

Training then does zero image I/O per step - just loads pre-computed tensors.
Expected speedup: 5-10× over on-the-fly tokenization.

Usage:
  python scripts/pretokenize_dataset.py # all splits
  python scripts/pretokenize_dataset.py --split train
  python scripts/pretokenize_dataset.py --workers 16
"""

import os, sys, json, argparse
from pathlib import Path
from typing import Optional
import torch
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "osmesa")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "generated"
CACHE_DIR = ROOT / "data" / "tokenized"
HF_TOKEN = os.getenv("HF_TOKEN")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
MAX_SEQ_LEN = 4096
TASKS = ["ttc", "stability", "trajectory"]


def format_answer(task, gt):
    if task == "ttc":
        return f"{gt['time_to_collision']:.2f}"
    elif task == "stability":
        return "stable" if gt["is_stable"] else "unstable"
    elif task == "trajectory":
        lp = gt["landing_position"]
        return f"x={lp['x']:.2f}, y={lp['y']:.2f}"
    return ""


def format_reasoning(task, gt):
    if task == "ttc":
        ttc = gt.get("time_to_collision", 0.0)
        v = gt.get("video_info", {})
        return (f"The video covers {v.get('duration_s',0):.2f}s of motion. "
                f"Across {v.get('n_frames',8)} frames the objects converge. "
                f"~{v.get('time_remaining_s',0):.2f}s remain → TTC = {ttc:.2f}s.")
    elif task == "stability":
        stable = gt.get("is_stable", False)
        disp = gt.get("max_displacement_m", 0.0)
        if stable:
            return f"Stack balanced. Max displacement = {disp:.4f}m. Stable."
        tc = gt.get("collapse_time_s", 0.0)
        return f"Top-heavy. Displacement = {disp:.4f}m, collapses at {tc:.2f}s. Unstable."
    elif task == "trajectory":
        lp = gt.get("landing_position", {})
        x, y = lp.get("x", 0.0), lp.get("y", 0.0)
        return (f"Parabolic arc. Peak {gt.get('max_height_m',0):.2f}m. "
                f"Flight {gt.get('flight_time_s',0):.2f}s. Lands x={x:.2f}m, y={y:.2f}m.")
    return "Based on visual analysis."


def load_frames(scene_dir, task):
    if task == "stability":
        for c in [scene_dir/"scene.png", scene_dir/"frame_000.png",
                  scene_dir/"frames"/"frame_000.png", scene_dir/"thumbnail.png"]:
            if c.exists():
                return [Image.open(c).convert("RGB")]
        raise FileNotFoundError(f"No image in {scene_dir}")
    return [Image.open(f).convert("RGB")
            for f in sorted((scene_dir/"frames").glob("frame_*.png"))]


def tokenize_one(args):
    """Worker function: tokenize a single scene and save to cache."""
    scene_id, task, scene_dir_str, cache_dir_str, processor_id = args
    scene_dir = Path(scene_dir_str)
    out_path = Path(cache_dir_str) / f"{scene_id}.pt"

    if out_path.exists():
        return scene_id, True, "skipped"

    try:
        from transformers import AutoProcessor
        # Load processor (each worker loads it once - cached by process)
        if not hasattr(tokenize_one, "_processor"):
            tokenize_one._processor = AutoProcessor.from_pretrained(
                processor_id,
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True,
            )
        proc = tokenize_one._processor

        with open(scene_dir / "ground_truth.json") as f:
            gt = json.load(f)
        with open(scene_dir / "prompt.txt") as f:
            prompt_text = f.read().strip()

        frames = load_frames(scene_dir, task)
        answer = format_answer(task, gt)
        reason = format_reasoning(task, gt)
        assistant = f"<reasoning>{reason}</reasoning>\n<answer>{answer}</answer>"

        user_content = [{"type": "image"} for _ in frames]
        user_content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant}]},
        ]
        prompt_messages = [{"role": "user", "content": user_content}]

        full_text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text_only = proc.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        inputs = proc(text=full_text, images=frames, return_tensors="pt",
                      max_length=MAX_SEQ_LEN, truncation=True, padding=False)
        prompt_inputs = proc(text=prompt_text_only, images=frames, return_tensors="pt",
                             max_length=MAX_SEQ_LEN, truncation=True, padding=False)

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[: prompt_inputs["input_ids"].shape[1]] = -100

        record = {
            "scene_id": scene_id,
            "task": task,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "answer": answer,
        }
        if "pixel_values" in inputs: record["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs: record["image_grid_thw"] = inputs["image_grid_thw"]

        torch.save(record, out_path)
        return scene_id, True, "ok"

    except Exception as e:
        return scene_id, False, str(e)


def pretokenize_split(split: str, n_workers: int = 8):
    split_file = DATA_DIR.parent / f"{split}.json"
    if not split_file.exists():
        print(f" No {split}.json - skipping")
        return

    with open(split_file) as f:
        index = json.load(f)

    cache_dir = CACHE_DIR / split
    cache_dir.mkdir(parents=True, exist_ok=True)

    already_done = sum(1 for item in index
                       if (cache_dir / f"{item['scene_id']}.pt").exists())
    print(f" {split}: {len(index)} samples ({already_done} already cached)")

    tasks_list = [
        (item["scene_id"], item["task"],
         str(DATA_DIR / item["task"] / item["scene_id"]),
         str(cache_dir), BASE_MODEL)
        for item in index
        if not (cache_dir / f"{item['scene_id']}.pt").exists()
    ]

    if not tasks_list:
        print(f" {split}: fully cached, nothing to do.")
        return

    print(f" Tokenizing {len(tasks_list)} samples with {n_workers} workers...")
    errors = 0

    # Use single process if n_workers=1 (easier debugging)
    if n_workers == 1:
        for args in tqdm(tasks_list, desc=f" {split}"):
            sid, ok, msg = tokenize_one(args)
            if not ok:
                errors += 1
                if errors <= 5:
                    print(f" [ERR] {sid}: {msg}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(tokenize_one, a): a[0] for a in tasks_list}
            pbar = tqdm(total=len(futures), desc=f" {split}")
            for fut in as_completed(futures):
                sid, ok, msg = fut.result()
                if not ok:
                    errors += 1
                    if errors <= 5:
                        tqdm.write(f" [ERR] {sid}: {msg}")
                pbar.update(1)
            pbar.close()

    total = sum(1 for item in index if (cache_dir / f"{item['scene_id']}.pt").exists())
    print(f" {split}: {total}/{len(index)} cached. Errors: {errors}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=["all","train","val","test"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    print(f"\n PhysSim-VLM · Pre-tokenize Dataset")
    print(f" Cache dir : {CACHE_DIR}")
    print(f" Workers : {args.workers}\n")

    splits = ["train","val","test"] if args.split == "all" else [args.split]
    for s in splits:
        pretokenize_split(s, args.workers)

    print("\n Done. Training will use cached tensors - no per-step image I/O.\n")


if __name__ == "__main__":
    main()
