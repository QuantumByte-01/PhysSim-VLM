#!/usr/bin/env python3
"""
PhysSim-VLM: Dataset Preparation & HuggingFace Upload
============================================================
Converts generated physics scenes into a HF Dataset and pushes
to a private HuggingFace repository.

Usage:
  python scripts/prepare_dataset.py --dry_run # stats only, no upload
  python scripts/prepare_dataset.py --upload # build + upload to HF
  python scripts/prepare_dataset.py --upload --push_images # include raw frames

Repo: Swastikr/PhysSim-VLM-Dataset (private)
"""

import os
import json
import base64
import argparse
from pathlib import Path
from io import BytesIO
from typing import Optional

from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "Swastikr/PhysSim-VLM-Dataset")
DATA_DIR = Path(__file__).parent.parent / "data" / "generated"

TASKS = ["ttc", "stability", "trajectory"]

# SFT R2 tasks (from generate_sft_r2_data.py + generate_taichi_fluid.py)
SFT_R2_DIR = Path(__file__).parent.parent / "data" / "sft_r2"
SFT_R2_TASKS = [
    "motion_comparison", "object_comparison", "counting",
    "viewpoint", "manipulation",
    "fluid_direction", "fluid_viscosity", "fluid_level",
]

# ── Answer formatter ──────────────────────────────────────────────────────────

def format_answer(task: str, gt: dict) -> str:
    """Convert ground truth dict to model target string."""
    if task == "ttc":
        ttc = gt.get("time_to_collision", 0.0)
        return f"{ttc:.2f}"

    elif task == "stability":
        return "stable" if gt.get("is_stable", False) else "unstable"

    elif task == "trajectory":
        lp = gt.get("landing_position", {})
        x = lp.get("x", 0.0)
        y = lp.get("y", 0.0)
        return f"x={x:.2f}, y={y:.2f}"

    # SFT R2 tasks: answer is stored directly in ground_truth
    elif "answer" in gt:
        return str(gt["answer"])

    return ""


def format_reasoning(task: str, gt: dict) -> str:
    """Build a ground-truth reasoning trace the model should learn to produce."""
    if task == "ttc":
        ttc = gt.get("time_to_collision", 0.0)
        vinfo = gt.get("video_info", {})
        dur = vinfo.get("duration_s", 0.0)
        rem = vinfo.get("time_remaining_s", 0.0)
        return (
            f"The video covers {dur:.2f}s of motion before collision. "
            f"Based on the closing speed observed across the {vinfo.get('n_frames',8)} frames, "
            f"the objects are converging steadily. "
            f"There are approximately {rem:.2f}s remaining after the clip ends, "
            f"giving a total time-to-collision from the video start of {ttc:.2f}s."
        )

    elif task == "stability":
        stable = gt.get("is_stable", False)
        disp = gt.get("max_displacement_m", 0.0)
        verdict = "stable" if stable else "unstable"
        if stable:
            return (
                f"The stack appears balanced. The base is wide enough to support the layers above. "
                f"Maximum displacement observed was only {disp:.4f}m, indicating no collapse. "
                f"The arrangement will remain stable."
            )
        else:
            t_col = gt.get("collapse_time_s", 0.0)
            return (
                f"The stack is top-heavy or misaligned. "
                f"Maximum displacement reaches {disp:.4f}m and the structure collapses at {t_col:.2f}s. "
                f"The arrangement is unstable."
            )

    elif task == "trajectory":
        lp = gt.get("landing_position", {})
        x, y = lp.get("x", 0.0), lp.get("y", 0.0)
        ht = gt.get("max_height_m", 0.0)
        ft = gt.get("flight_time_s", 0.0)
        vinfo = gt.get("video_info", {})
        return (
            f"From the {vinfo.get('n_frames',5)}-frame clip, the object follows a parabolic arc. "
            f"Peak height is approximately {ht:.2f}m. "
            f"Extrapolating the trajectory gives a total flight time of ~{ft:.2f}s. "
            f"The object lands at x={x:.2f}m forward, y={y:.2f}m sideways from the launch point."
        )

    return "Based on visual analysis of the scene."


# ── Image loading ─────────────────────────────────────────────────────────────

def load_frames(scene_dir: Path, task: str) -> list[Image.Image]:
    """Load all frames for a scene as PIL Images."""
    # Single-image tasks
    single_image_tasks = {"stability", "counting", "viewpoint"}
    if task in single_image_tasks:
        # Try scene.png first (SFT R2 format), then frame_000.png, then frames/
        for candidate in [
            scene_dir / "scene.png",
            scene_dir / "frame_000.png",
            scene_dir / "frames" / "frame_000.png",
            scene_dir / "thumbnail.png",
        ]:
            if candidate.exists():
                return [Image.open(candidate).convert("RGB")]
        raise FileNotFoundError(f"No image found in {scene_dir}")

    # Multi-frame tasks
    frames_dir = scene_dir / "frames"
    if frames_dir.exists():
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if frame_files:
            return [Image.open(f).convert("RGB") for f in frame_files]

    raise FileNotFoundError(f"No frames found in {scene_dir}")


def pil_to_base64(img: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ── Scene builder ─────────────────────────────────────────────────────────────

def build_record_sft_r2(task: str, scene_dir: Path, split: str = "train") -> Optional[dict]:
    """Build one dataset record from an SFT R2 scene directory.

    SFT R2 scenes store assistant_text.txt directly (pre-generated with
    reasoning + answer tags), so we don't need format_reasoning().
    """
    gt_path = scene_dir / "ground_truth.json"
    cfg_path = scene_dir / "config.json"
    prompt_path = scene_dir / "prompt.txt"
    asst_path = scene_dir / "assistant_text.txt"

    if not gt_path.exists() or not prompt_path.exists():
        return None

    with open(gt_path) as f: gt = json.load(f)
    with open(cfg_path) as f: cfg = json.load(f)
    with open(prompt_path) as f: prompt = f.read().strip()

    # Use pre-generated assistant text if available, otherwise build it
    if asst_path.exists():
        with open(asst_path) as f:
            assistant_text = f.read().strip()
    else:
        answer = format_answer(task, gt)
        assistant_text = (
            f"<reasoning>Based on visual analysis of the scene.</reasoning>\n"
            f"<answer>{answer}</answer>"
        )

    try:
        frames = load_frames(scene_dir, task)
    except Exception:
        return None

    frames_b64 = [pil_to_base64(f) for f in frames]

    return {
        "scene_id": scene_dir.name,
        "task": task,
        "split": split,
        "difficulty": cfg.get("difficulty", "unknown"),
        "prompt": prompt,
        "answer": format_answer(task, gt),
        "reasoning": "", # stored in assistant_text directly
        "assistant_text": assistant_text,
        "n_frames": len(frames),
        "frames_b64": frames_b64,
        "ground_truth": json.dumps(gt),
        "config": json.dumps(cfg),
    }


def build_record(task: str, scene_dir: Path, split: str) -> Optional[dict]:
    """Build one dataset record from a scene directory."""
    gt_path = scene_dir / "ground_truth.json"
    cfg_path = scene_dir / "config.json"
    prompt_path = scene_dir / "prompt.txt"

    if not gt_path.exists() or not prompt_path.exists():
        return None

    with open(gt_path) as f: gt = json.load(f)
    with open(cfg_path) as f: cfg = json.load(f)
    with open(prompt_path) as f: prompt = f.read().strip()

    try:
        frames = load_frames(scene_dir, task)
    except Exception:
        return None

    answer = format_answer(task, gt)
    reasoning = format_reasoning(task, gt)

    # Build the full assistant turn
    assistant_text = (
        f"<reasoning>{reasoning}</reasoning>\n"
        f"<answer>{answer}</answer>"
    )

    # Encode frames to base64 JPEG for storage
    frames_b64 = [pil_to_base64(f) for f in frames]

    return {
        "scene_id": scene_dir.name,
        "task": task,
        "split": split,
        "difficulty": cfg.get("difficulty", "unknown"),
        "prompt": prompt,
        "answer": answer,
        "reasoning": reasoning,
        "assistant_text": assistant_text,
        "n_frames": len(frames),
        "frames_b64": frames_b64, # list of base64-encoded JPEG strings
        "ground_truth": json.dumps(gt),
        "config": json.dumps(cfg),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def load_splits() -> dict[str, str]:
    """Load scene_id → split mapping from train/val/test JSON files."""
    split_map = {}
    for split_name in ["train", "val", "test"]:
        split_file = DATA_DIR.parent / f"{split_name}.json"
        if split_file.exists():
            with open(split_file) as f:
                items = json.load(f)
            for item in items:
                split_map[item["scene_id"]] = split_name
    return split_map


def collect_records(verbose: bool = True, include_sft_r2: bool = True) -> list[dict]:
    split_map = load_splits()
    records = []
    errors = 0

    # ── Original tasks (data/generated/) ───────────────────────────────
    for task in TASKS:
        task_dir = DATA_DIR / task
        if not task_dir.exists():
            continue

        scene_dirs = sorted(task_dir.iterdir())
        if verbose:
            print(f"\n {task}: {len(scene_dirs)} scenes")

        for scene_dir in tqdm(scene_dirs, desc=f" {task}", disable=not verbose):
            if not scene_dir.is_dir():
                continue
            split = split_map.get(scene_dir.name, "train")
            record = build_record(task, scene_dir, split)
            if record:
                records.append(record)
            else:
                errors += 1

    # ── SFT R2 tasks (data/sft_r2/) ───────────────────────────────────
    if include_sft_r2 and SFT_R2_DIR.exists():
        for task in SFT_R2_TASKS:
            task_dir = SFT_R2_DIR / task
            if not task_dir.exists():
                continue

            scene_dirs = sorted(task_dir.iterdir())
            if verbose:
                print(f"\n {task} (R2): {len(scene_dirs)} scenes")

            for scene_dir in tqdm(scene_dirs, desc=f" {task}", disable=not verbose):
                if not scene_dir.is_dir():
                    continue
                record = build_record_sft_r2(task, scene_dir, split="train")
                if record:
                    records.append(record)
                else:
                    errors += 1

    if verbose:
        print(f"\n Total records: {len(records)} | Errors: {errors}")
    return records


def print_stats(records: list[dict]):
    from collections import Counter
    tasks = Counter(r["task"] for r in records)
    splits = Counter(r["split"] for r in records)
    diffs = Counter(r["difficulty"] for r in records)
    print("\n ── Dataset Statistics ──────────────────────")
    print(f" Total samples : {len(records):,}")
    for t, n in tasks.items():
        print(f" {t:<12}: {n:,}")
    print(f" Splits: {dict(splits)}")
    print(f" Difficulties: {dict(diffs)}")


def upload_to_hf(records: list[dict]):
    import datasets
    from huggingface_hub import HfApi

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set in .env")

    api = HfApi(token=HF_TOKEN)

    # Ensure repo exists (private)
    api.create_repo(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    print(f"\n Repo ready: https://huggingface.co/datasets/{HF_DATASET_REPO}")

    # Split into train/val/test
    for split_name in ["train", "val", "test"]:
        subset = [r for r in records if r["split"] == split_name]
        if not subset:
            continue

        print(f" Pushing {split_name}: {len(subset):,} samples...")
        ds = datasets.Dataset.from_list(subset)
        ds.push_to_hub(
            HF_DATASET_REPO,
            split=split_name,
            token=HF_TOKEN,
            private=True,
        )
        print(f" ✓ {split_name} pushed")

    print(f"\n Dataset live at: https://huggingface.co/datasets/{HF_DATASET_REPO}")


def main():
    parser = argparse.ArgumentParser(description="Prepare and upload PhysSim-VLM dataset")
    parser.add_argument("--dry_run", action="store_true", help="Stats only, no upload")
    parser.add_argument("--upload", action="store_true", help="Push to HuggingFace")
    parser.add_argument("--task", choices=TASKS, help="Process single task only")
    parser.add_argument("--no-sft-r2", action="store_true", help="Exclude SFT R2 data")
    args = parser.parse_args()

    print("\n PhysSim-VLM · Dataset Preparation")
    print(f" Data dir : {DATA_DIR}")
    if not args.no_sft_r2:
        print(f" SFT R2 : {SFT_R2_DIR}")
    print(f" HF repo : {HF_DATASET_REPO}\n")

    records = collect_records(verbose=True, include_sft_r2=not args.no_sft_r2)
    print_stats(records)

    if args.dry_run:
        print("\n [dry_run] Skipping upload.")
        return

    if args.upload:
        upload_to_hf(records)
    else:
        print("\n Run with --upload to push to HuggingFace.")


if __name__ == "__main__":
    main()
