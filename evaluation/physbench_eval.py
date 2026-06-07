#!/usr/bin/env python3
"""
Zero-shot PhysBench Baseline
Runs Qwen3-VL-30B-A3B-Instruct on PhysBench via the Tinker SDK.

Usage:
  python baseline.py # run 100 image-only samples
  python baseline.py --limit 500 # run 500 samples
  python baseline.py --all # run all image-only samples
  python baseline.py --score_only # re-score existing predictions
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import io
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm

MAX_IMAGE_BYTES = 2_000_000 # Tinker limit: 2MB

# ── Config ───────────────────────────────────────────────────────

load_dotenv()
TINKER_API_KEY = os.getenv("TINKER_API_KEY")
if not TINKER_API_KEY:
    raise RuntimeError("TINKER_API_KEY missing in .env")
os.environ["TINKER_API_KEY"] = TINKER_API_KEY

MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

DATA_ROOT = Path("physbench")
TEST_JSON = DATA_ROOT / "test.json"
ANSWER_JSON = DATA_ROOT / "test_answer.json"
IMAGE_DIR = DATA_ROOT / "image"

NUM_SAMPLES = 100

PREDICTIONS_JSON = "predictions.json"
OUT_CSV = "predictions.csv"
RESULTS_MD = "results.md"

ANSWER_SUFFIX = (
    "\nAnswer with the option's letter from the given choices directly. "
    "You can only answer one letter from A, B, C, or D."
)


# ── Helpers ──────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """Pull A/B/C/D from model output."""
    if not text:
        return "X"
    text = text.strip()

    if text in ("A", "B", "C", "D"):
        return text

    if text and text[0] in "ABCD" and (len(text) == 1 or not text[1].isalpha()):
        return text[0]

    m = re.search(r"(?:answer|option)\s*(?:is|:)\s*([A-D])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1)

    return "X"


def build_question_text(raw_question: str) -> str:
    """Clean up placeholders in question text and add answer suffix."""
    q = raw_question.replace("<video>", "[video]").replace("<image>", "[image]")
    return q + ANSWER_SUFFIX


def load_image_bytes(path: Path) -> tuple[bytes, str]:
    """Load image, resize if over 2MB limit. Returns (bytes, format)."""
    ext = path.suffix.lower().lstrip(".")
    fmt = "jpeg" if ext in ("jpg", "jpeg") else "png"

    raw = path.read_bytes()
    if len(raw) <= MAX_IMAGE_BYTES:
        return raw, fmt

    # resize down until under limit
    img = Image.open(path).convert("RGB")
    for scale in [0.75, 0.5, 0.35, 0.25]:
        w, h = int(img.width * scale), int(img.height * scale)
        resized = img.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
        if len(data) <= MAX_IMAGE_BYTES:
            return data, "jpeg"

    # last resort: very small
    resized = img.resize((320, 240), Image.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=70)
    return buf.getvalue(), "jpeg"


# ── Data Loading ─────────────────────────────────────────────────

def load_data():
    """Load test questions, merge answers, filter to samples with available images."""
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        questions = json.load(f)

    answer_map = {}
    if ANSWER_JSON.exists():
        with open(ANSWER_JSON, "r", encoding="utf-8") as f:
            for item in json.load(f):
                answer_map[item["idx"]] = item

    merged = []
    for q in questions:
        idx = q["idx"]
        ans_info = answer_map.get(idx, {})

        files = q.get("file_name", [])
        if not isinstance(files, list):
            files = [files]

        image_file = None
        for fname in files:
            if str(fname).lower().endswith((".png", ".jpg", ".jpeg")):
                if (IMAGE_DIR / fname).exists():
                    image_file = fname
                    break

        if not image_file:
            continue
        if not ans_info.get("answer"):
            continue

        merged.append({
            "idx": idx,
            "question": q["question"],
            "image_file": image_file,
            "answer": ans_info["answer"],
            "task_type": ans_info.get("task_type", "unknown"),
            "sub_type": ans_info.get("sub_type", "unknown"),
            "ability_type": ans_info.get("ability_type", "unknown"),
            "mode": ans_info.get("mode", q.get("mode", "unknown")),
            "source": q.get("source", "unknown"),
        })

    print(f" Loaded {len(merged)} samples with images + answers")
    return merged


# ── Tinker Inference ─────────────────────────────────────────────

def create_sampler():
    """Create a Tinker SamplingClient for the model."""
    import tinker

    print(f" Connecting to Tinker ({MODEL})...")
    client = tinker.ServiceClient()
    sampler = client.create_sampling_client(base_model=MODEL)
    tokenizer = sampler.get_tokenizer()
    print(f" Connected. Tokenizer: {type(tokenizer).__name__}")
    return sampler, tokenizer


def run_inference(sampler, tokenizer, image_path: Path, question: str) -> str:
    """Send one image + question to Tinker and return the model's text response."""
    import tinker

    # build chat-template prompt:
    # <|im_start|>user\n [IMAGE] question<|im_end|>\n<|im_start|>assistant\n
    before_tokens = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    after_tokens = tokenizer.encode(
        question + "<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )

    image_data, fmt = load_image_bytes(image_path)

    prompt = tinker.ModelInput(chunks=[
        tinker.EncodedTextChunk(tokens=before_tokens),
        tinker.types.ImageChunk(data=image_data, format=fmt),
        tinker.EncodedTextChunk(tokens=after_tokens),
    ])

    params = tinker.SamplingParams(max_tokens=64, temperature=0.0)
    future = sampler.sample(prompt, num_samples=1, sampling_params=params)
    response = future.result()

    output = ""
    for seq in response.sequences:
        output = tokenizer.decode(seq.tokens, skip_special_tokens=True)
    return output.strip()


# ── Evaluation Loop ──────────────────────────────────────────────

def run_eval(data: list, limit: int, sampler, tokenizer):
    """Run inference on samples. Saves predictions incrementally."""

    predictions = []
    done_idxs = set()
    if Path(PREDICTIONS_JSON).exists():
        with open(PREDICTIONS_JSON, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        done_idxs = {p["idx"] for p in predictions}
        print(f" Resuming: {len(predictions)} already done")

    samples = data[:limit]
    remaining = [s for s in samples if s["idx"] not in done_idxs]

    if not remaining:
        print(" All samples already processed.")
        return predictions

    print(f" Running {len(remaining)} new samples...\n")
    pbar = tqdm(total=len(remaining))

    for item in remaining:
        image_path = IMAGE_DIR / item["image_file"]
        question_text = build_question_text(item["question"])

        try:
            model_text = run_inference(sampler, tokenizer, image_path, question_text)
            predicted = extract_answer(model_text)
        except Exception as e:
            model_text = f"ERROR: {e}"
            predicted = "X"

        gt = item["answer"]
        correct = predicted == gt

        pred = {
            "idx": item["idx"],
            "predicted": predicted,
            "gt": gt,
            "correct": correct,
            "task_type": item["task_type"],
            "sub_type": item["sub_type"],
            "ability_type": item["ability_type"],
            "mode": item["mode"],
            "source": item["source"],
            "question": item["question"][:200],
            "image": item["image_file"],
            "response": model_text,
        }
        predictions.append(pred)
        done_idxs.add(item["idx"])

        status = "ok" if correct else "XX"
        pbar.set_postfix_str(f"pred={predicted} gt={gt} [{status}]")
        pbar.update(1)

        # checkpoint every 25 samples
        if len(predictions) % 25 == 0:
            with open(PREDICTIONS_JSON, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2)

    pbar.close()

    with open(PREDICTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    return predictions


# ── Scoring ──────────────────────────────────────────────────────

def compute_scores(predictions: list) -> dict:
    def acc(items):
        if not items:
            return 0.0, 0, 0
        c = sum(1 for p in items if p["correct"])
        return c / len(items) * 100, c, len(items)

    scores = {}
    pct, c, t = acc(predictions)
    scores["overall"] = {"accuracy": pct, "correct": c, "total": t}

    for key in ["task_type", "sub_type", "ability_type", "source", "mode"]:
        groups = defaultdict(list)
        for p in predictions:
            groups[p.get(key, "unknown")].append(p)
        scores[f"by_{key}"] = {}
        for k in sorted(groups):
            pct, c, t = acc(groups[k])
            scores[f"by_{key}"][k] = {"accuracy": pct, "correct": c, "total": t}

    return scores


def save_results(scores: dict, predictions: list):
    """Write results.md and predictions.csv."""

    def table(heading, key):
        lines = [
            f"\n## {heading}\n",
            f"| {heading} | Correct | Total | Accuracy |",
            f"|{'---'*5}|---------|-------|----------|",
        ]
        for k, v in scores[key].items():
            lines.append(
                f"| {k} | {v['correct']} | {v['total']} | {v['accuracy']:.1f}% |"
            )
        return lines

    o = scores["overall"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# PhysBench Zero-Shot Baseline Results\n",
        f"**Model:** {MODEL}",
        f"**Date:** {ts}",
        f"**Samples:** {o['total']}\n",
        "## Overall\n",
        f"**Accuracy: {o['correct']}/{o['total']} ({o['accuracy']:.1f}%)**\n",
    ]
    lines += table("Task Type", "by_task_type")
    lines += table("Sub Type", "by_sub_type")
    lines += table("Ability Type", "by_ability_type")
    lines += table("Data Source", "by_source")
    lines += table("Input Mode", "by_mode")

    errors = [p for p in predictions if not p["correct"]]
    err_by_task = defaultdict(int)
    for e in errors:
        err_by_task[e["task_type"]] += 1

    lines += ["\n## Error Summary\n"]
    lines.append(f"Total errors: {len(errors)}/{o['total']}\n")
    lines.append("| Task Type | Errors |")
    lines.append("|-----------|--------|")
    for k in sorted(err_by_task, key=err_by_task.get, reverse=True):
        lines.append(f"| {k} | {err_by_task[k]} |")

    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f" Saved {RESULTS_MD}")

    pd.DataFrame(predictions).to_csv(OUT_CSV, index=False)
    print(f" Saved {OUT_CSV}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Zero-shot PhysBench baseline")
    parser.add_argument("--limit", type=int, default=NUM_SAMPLES)
    parser.add_argument("--all", action="store_true", help="Run all available samples")
    parser.add_argument("--score_only", action="store_true")
    args = parser.parse_args()

    print(f"\n PhysBench Zero-Shot Baseline")
    print(f" Model: {MODEL}")
    print(f" {'='*40}\n")

    data = load_data()

    if args.score_only:
        if not Path(PREDICTIONS_JSON).exists():
            print(" No predictions.json found.")
            return
        with open(PREDICTIONS_JSON, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    else:
        sampler, tokenizer = create_sampler()
        limit = len(data) if args.all else args.limit
        predictions = run_eval(data, limit, sampler, tokenizer)

    scores = compute_scores(predictions)

    o = scores["overall"]
    print(f"\n Overall: {o['correct']}/{o['total']} ({o['accuracy']:.1f}%)\n")
    for k, v in scores["by_task_type"].items():
        bar = "#" * int(v["accuracy"] / 10) + "-" * (10 - int(v["accuracy"] / 10))
        print(f" {k:20s} {bar} {v['accuracy']:5.1f}% ({v['correct']}/{v['total']})")
    print()

    save_results(scores, predictions)
    print("\n Done.\n")


if __name__ == "__main__":
    main()
