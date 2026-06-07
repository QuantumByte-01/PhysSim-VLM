#!/usr/bin/env python3
"""
PhysSim-VLM: PhysBench Evaluation
========================================
Runs PhysBench evaluation with:
  1. Base model only (zero-shot) - or uses existing baseline if available
  2. LoRA fine-tuned model (SFT epoch 1)

Saves results to:
  results/physbench/zero_shot/results.md
  results/physbench/sft_epoch1/results.md
  results/physbench/comparison.md

Usage:
  python scripts/eval_physbench.py # both models
  python scripts/eval_physbench.py --zero_shot_only # base only
  python scripts/eval_physbench.py --finetuned_only # LoRA only
  python scripts/eval_physbench.py --limit 500 # subset for speed
"""

import os, re, json, argparse, random, base64, io, torch
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
ROOT = Path(__file__).parent.parent
PHYSBENCH = ROOT / "physbench"
IMAGE_DIR = PHYSBENCH / "image"
TEST_JSON = Path("/root/.cache/huggingface/hub/datasets--USC-GVL--PhysBench/snapshots/478fd93da8ec8d6f5252b9586b1fa10f335c5a95/test.json")
RESULTS = ROOT / "results" / "physbench"
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
CKPT_E1 = ROOT / "checkpoints" / "lora_sft_epoch1" / "final"
BASELINE_PREDS = ROOT / "results" / "baselines" / "predictions.json"

ANSWER_SUFFIX = (
    "\nAnswer with the option's letter from the given choices directly. "
    "You can only answer one letter from A, B, C, or D."
)


def load_physbench_samples(limit=None):
    """Load image-only PhysBench samples that we have images + answers for."""
    # Build answer map from existing baseline predictions
    answer_map = {}
    task_map = {}
    sub_map = {}
    ability_map = {}
    if BASELINE_PREDS.exists():
        for pred in json.load(open(BASELINE_PREDS)):
            idx = str(pred["idx"])
            answer_map[idx] = pred["gt"]
            task_map[idx] = pred.get("task_type", "unknown")
            sub_map[idx] = pred.get("sub_type", "unknown")
            ability_map[idx]= pred.get("ability_type", "unknown")

    # Load test.json
    all_samples = json.load(open(TEST_JSON))

    samples = []
    for s in all_samples:
        idx = str(s["idx"])
        mode = s.get("mode", "")
        if mode != "image-only":
            continue
        if idx not in answer_map:
            continue # no ground truth available

        # Resolve image path
        file_names = s.get("file_name", [])
        if isinstance(file_names, str):
            file_names = json.loads(file_names.replace("'", '"')) if file_names.startswith("[") else [file_names]
        img_path = None
        for fn in file_names:
            p = IMAGE_DIR / fn
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        samples.append({
            "idx": idx,
            "image_path": img_path,
            "question": s["question"].replace("<image>\n", "").strip() + ANSWER_SUFFIX,
            "gt": answer_map[idx],
            "task_type": task_map.get(idx, "unknown"),
            "sub_type": sub_map.get(idx, "unknown"),
            "ability_type": ability_map.get(idx, "unknown"),
            "source": s.get("source", "unknown"),
            "mode": mode,
        })

    random.seed(42)
    random.shuffle(samples)
    if limit:
        samples = samples[:limit]
    return samples


def extract_answer(text):
    text = text.strip()
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else text[:1].upper()


def run_eval(model, processor, samples, desc="eval"):
    """Run inference and return predictions list."""
    from PIL import Image
    from tqdm import tqdm
    predictions = []

    for s in tqdm(samples, desc=desc):
        img = Image.open(s["image_path"]).convert("RGB")
        user_content = [{"type": "image"}, {"type": "text", "text": s["question"]}]
        messages = [{"role": "user", "content": user_content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=text, images=[img],
            return_tensors="pt", max_length=2048,
            truncation=True, max_pixels=401408,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        pred_text = processor.decode(new_tokens, skip_special_tokens=True)
        pred = extract_answer(pred_text)

        predictions.append({
            "idx": s["idx"],
            "task_type": s["task_type"],
            "sub_type": s["sub_type"],
            "ability_type": s["ability_type"],
            "source": s["source"],
            "mode": s["mode"],
            "gt": s["gt"],
            "predicted": pred,
            "correct": pred == s["gt"],
            "raw_output": pred_text,
        })
    return predictions


def compute_scores(predictions):
    total = len(predictions)
    correct = sum(p["correct"] for p in predictions)
    overall = correct / total if total else 0

    task_scores = defaultdict(lambda: [0, 0])
    for p in predictions:
        task_scores[p["task_type"]][1] += 1
        if p["correct"]:
            task_scores[p["task_type"]][0] += 1

    return {"overall": overall, "total": total, "correct": correct, "task_scores": dict(task_scores)}


def write_results_md(out_dir, scores, label, ckpt_desc, n_samples):
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# PhysBench Evaluation - {label}\n\n",
        f"**Date:** {now}\n",
        f"**Model:** {ckpt_desc}\n",
        f"**Samples:** {n_samples} (image-only)\n\n",
        "## Overall\n\n",
        f"**Accuracy: {scores['correct']}/{scores['total']} ({scores['overall']*100:.1f}%)**\n\n",
        "## Task Type Breakdown\n\n",
        "| Task Type | Correct | Total | Accuracy |\n",
        "|-----------|---------|-------|----------|\n",
    ]
    for task, (c, t) in sorted(scores["task_scores"].items()):
        lines.append(f"| {task} | {c} | {t} | {c/t*100:.1f}% |\n")
    with open(out_dir / "results.md", "w") as f:
        f.writelines(lines)
    print(f" Saved: {out_dir}/results.md")


def write_comparison_md(zero_scores, sft_scores):
    out = RESULTS / "comparison.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# PhysBench: Zero-Shot vs SFT Epoch 1\n\n",
        f"**Date:** {now}\n\n",
        "## Overall Comparison\n\n",
        "| | Zero-Shot (baseline) | SFT Epoch 1 | Delta |\n",
        "|--|---------------------|-------------|-------|\n",
    ]
    # overall
    z_acc = zero_scores["overall"] * 100
    s_acc = sft_scores["overall"] * 100
    delta = s_acc - z_acc
    sign = "+" if delta >= 0 else ""
    lines.append(f"| **Overall** | {z_acc:.1f}% | {s_acc:.1f}% | **{sign}{delta:.1f}%** |\n")

    # per task
    all_tasks = sorted(set(list(zero_scores["task_scores"].keys()) + list(sft_scores["task_scores"].keys())))
    lines += ["\n## Per-Task Comparison\n\n",
              "| Task Type | Zero-Shot | SFT Epoch 1 | Delta |\n",
              "|-----------|-----------|-------------|-------|\n"]
    for task in all_tasks:
        z_c, z_t = zero_scores["task_scores"].get(task, [0, 0])
        s_c, s_t = sft_scores["task_scores"].get(task, [0, 0])
        z_a = z_c/z_t*100 if z_t else 0
        s_a = s_c/s_t*100 if s_t else 0
        d = s_a - z_a
        sign = "+" if d >= 0 else ""
        lines.append(f"| {task} | {z_a:.1f}% ({z_t}) | {s_a:.1f}% ({s_t}) | {sign}{d:.1f}% |\n")

    with open(out, "w") as f:
        f.writelines(lines)
    print(f" Comparison saved: {out}")


def patch_rocm():
    if torch.cuda.is_available() and hasattr(torch.version, "hip"):
        if hasattr(torch, "_grouped_mm"):
            del torch._grouped_mm
        if hasattr(torch.nn.functional, "grouped_mm"):
            del torch.nn.functional.grouped_mm


def load_base_model():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"Loading base model: {BASE_MODEL}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, token=HF_TOKEN, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    patch_rocm()
    return model, processor


def load_finetuned_model():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    print(f"Loading fine-tuned model from {CKPT_E1}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, token=HF_TOKEN, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(model, str(CKPT_E1))
    model.eval()
    patch_rocm()
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero_shot_only", action="store_true")
    parser.add_argument("--finetuned_only", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per model (default: all available)")
    parser.add_argument("--use_existing_baseline", action="store_true",
                        help="Skip zero-shot inference, use existing baseline results instead")
    args = parser.parse_args()

    print("="*60)
    print(" PhysSim-VLM - PhysBench Evaluation")
    print("="*60)

    samples = load_physbench_samples(limit=args.limit)
    print(f"Loaded {len(samples)} image-only PhysBench samples with ground truth")

    if not samples:
        print("ERROR: No samples found. Ensure PhysBench images are downloaded.")
        print(" Run: python -c \"from huggingface_hub import hf_hub_download; ...\"")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    zero_scores = sft_scores = None

    # ── Zero-shot ─────────────────────────────────────────────────
    run_zero = not args.finetuned_only
    if run_zero:
        if args.use_existing_baseline and BASELINE_PREDS.exists():
            print("\n[Zero-Shot] Using existing baseline predictions...")
            preds_all = json.load(open(BASELINE_PREDS))
            # Filter to image-only matching our sample set
            idx_set = {s["idx"] for s in samples}
            preds = [p for p in preds_all if str(p["idx"]) in idx_set and p.get("mode") == "image-only"]
            zero_scores = compute_scores(preds)
        else:
            print("\n[Zero-Shot] Running base model inference...")
            model, processor = load_base_model()
            preds = run_eval(model, processor, samples, desc="Zero-shot")
            zero_scores = compute_scores(preds)
            (RESULTS / "zero_shot").mkdir(parents=True, exist_ok=True)
            with open(RESULTS / "zero_shot" / "predictions.json", "w") as f:
                json.dump(preds, f, indent=2)
            del model
            torch.cuda.empty_cache()

        write_results_md(
            RESULTS / "zero_shot", zero_scores,
            "Zero-Shot (Base Model)",
            f"{BASE_MODEL} - no fine-tuning",
            len(samples)
        )
        print(f" Zero-shot overall: {zero_scores['overall']*100:.1f}%")
        for task, (c, t) in sorted(zero_scores["task_scores"].items()):
            print(f" {task}: {c/t*100:.1f}% ({t} samples)")

    # ── Fine-tuned ────────────────────────────────────────────────
    run_sft = not args.zero_shot_only
    if run_sft:
        if not CKPT_E1.exists():
            print(f"\n[SFT] Checkpoint not found at {CKPT_E1} - skipping.")
        else:
            print("\n[SFT Epoch 1] Running fine-tuned model inference...")
            model, processor = load_finetuned_model()
            preds = run_eval(model, processor, samples, desc="SFT Epoch 1")
            sft_scores = compute_scores(preds)
            (RESULTS / "sft_epoch1").mkdir(parents=True, exist_ok=True)
            with open(RESULTS / "sft_epoch1" / "predictions.json", "w") as f:
                json.dump(preds, f, indent=2)
            del model
            torch.cuda.empty_cache()

            write_results_md(
                RESULTS / "sft_epoch1", sft_scores,
                "SFT Epoch 1 (LoRA rank=128)",
                f"{BASE_MODEL} + LoRA SFT (15k physics scenes, rank=128)",
                len(samples)
            )
            print(f" SFT overall: {sft_scores['overall']*100:.1f}%")
            for task, (c, t) in sorted(sft_scores["task_scores"].items()):
                print(f" {task}: {c/t*100:.1f}% ({t} samples)")

    # ── Comparison ───────────────────────────────────────────────
    if zero_scores and sft_scores:
        write_comparison_md(zero_scores, sft_scores)

    print("\nDone! Results in results/physbench/")


if __name__ == "__main__":
    main()
