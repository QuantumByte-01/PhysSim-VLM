#!/usr/bin/env python3
"""
PhysSim-VLM: PhysBench Eval via Thinking Machines Tinker SDK
===================================================================
Evaluates Qwen3-VL-30B on the PhysBench val set (200 samples) covering
all input modes: image-only, image&video, and general (video + choice images).

Setup:
  pip install tinker tinker-cookbook av python-dotenv
  export TINKER_API_KEY=<from https://tinker-console.thinkingmachines.ai>

  # Extract PhysBench videos (one-time):
  cd data/raw/physbench && unzip video.zip -d video/

Usage:
  python scripts/eval_physbench_tinker.py # full baseline (200 samples)
  python scripts/eval_physbench_tinker.py --max-samples 20 # quick smoke test
  python scripts/eval_physbench_tinker.py \
      --model-path "tinker://<job-id>:train:0/sampler_weights/<step>" --compare
"""

import os, re, json, asyncio, argparse, random, sys
from pathlib import Path
from io import BytesIO
from collections import defaultdict
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Propagate HF_TOKEN so huggingface_hub picks it up
if os.environ.get("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])

from PIL import Image
import tinker
from tinker import types
from tinker_cookbook import tokenizer_utils

ROOT = Path(__file__).parent.parent
PHYSBENCH = ROOT / "data" / "raw" / "physbench"
IMG_DIR = PHYSBENCH / "image"
VID_DIR = PHYSBENCH / "video"
RESULTS_DIR = ROOT / "results"

BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
VIDEO_FRAMES = 8

GREEN = ""
RED = ""
YELLOW = ""
BOLD = ""
RESET = ""

# Qwen3-VL special token IDs
IM_START = 151644 # <|im_start|>
IM_END = 151645 # <|im_end|>
VISION_START = 151652 # <|vision_start|>
VISION_END = 151653 # <|vision_end|>


# ── Video frame extraction ────────────────────────────────────────────────────

MAX_ASSET_BYTES = 1_800_000 # Tinker limit is 2MB; stay under with margin
MAX_FRAME_DIM = 1120 # max side length before downscaling

def _compress_image(img) -> bytes:
    """Resize if needed and encode as JPEG under MAX_ASSET_BYTES."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_FRAME_DIM:
        scale = MAX_FRAME_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    for quality in (85, 70, 55, 40):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= MAX_ASSET_BYTES:
            return buf.getvalue()
    return buf.getvalue() # best effort


def extract_video_frames(video_path: Path, n_frames: int = VIDEO_FRAMES) -> list[bytes]:
    """Sample n_frames evenly from mp4. Returns list of JPEG bytes."""
    import av
    with av.open(str(video_path)) as container:
        all_frames = [f.to_image() for f in container.decode(video=0)]
    if not all_frames:
        return []
    indices = [int(i * len(all_frames) / n_frames) for i in range(n_frames)]
    indices = [min(i, len(all_frames) - 1) for i in indices]
    sampled = [all_frames[i] for i in dict.fromkeys(indices)]
    return [_compress_image(img) for img in sampled]


def image_to_bytes(path: Path) -> bytes:
    return _compress_image(Image.open(path))


# ── Prompt building (low-level ModelInput) ────────────────────────────────────

def build_model_input(tokenizer, image_bytes_list: list[bytes], text: str) -> types.ModelInput:
    """
    Build Qwen3-VL ModelInput:
      <|im_start|>user\n
      <|vision_start|>[img]<|vision_end|> × N images
      text<|im_end|>\n<|im_start|>assistant\n
    """
    chunks = []

    # User turn start
    prefix = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    chunks.append(types.EncodedTextChunk(tokens=prefix))

    # Each image wrapped in vision tokens
    vs = tokenizer.encode("<|vision_start|>", add_special_tokens=False)
    ve = tokenizer.encode("<|vision_end|>", add_special_tokens=False)
    for img_bytes in image_bytes_list:
        chunks.append(types.EncodedTextChunk(tokens=vs))
        chunks.append(types.ImageChunk(data=img_bytes, format="jpeg"))
        chunks.append(types.EncodedTextChunk(tokens=ve))

    # Text + end of user turn + assistant turn start
    suffix_str = text + "<|im_end|>\n<|im_start|>assistant\n"
    suffix = tokenizer.encode(suffix_str, add_special_tokens=False)
    chunks.append(types.EncodedTextChunk(tokens=suffix))

    return types.ModelInput(chunks=chunks)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_val_split(max_samples: int | None = None, seed: int = 42,
                   split: str = "val") -> list[dict]:
    answer_file = "val_answer.json" if split == "val" else "test_answer.json"
    questions = {q["idx"]: q
                 for q in json.load(open(PHYSBENCH / "test.json"))
                 if q["split"] == split}
    answers = {a["idx"]: a
                 for a in json.load(open(PHYSBENCH / answer_file))
                 if a.get("idx") in questions}

    records = []
    for idx, q in questions.items():
        ans = answers.get(idx)
        if not ans or not ans.get("answer"):
            continue
        records.append({
            "idx": idx,
            "mode": q["mode"],
            "task_type": ans.get("task_type", "unknown"),
            "sub_type": ans.get("sub_type", "unknown"),
            "ability_type": ans.get("ability_type", "unknown"),
            "question": q["question"],
            "file_names": q["file_name"],
            "answer": ans["answer"].strip().upper(),
        })

    if max_samples:
        random.Random(seed).shuffle(records)
        records = records[:max_samples]
    return records


def collect_image_bytes(record: dict) -> list[bytes] | None:
    """Collect all image/video bytes for a record in display order."""
    result = []
    for fname in record["file_names"]:
        if fname.endswith(".mp4"):
            p = VID_DIR / fname
            if not p.exists():
                return None
            try:
                result.extend(extract_video_frames(p))
            except Exception:
                return None
        else:
            found = False
            for d in [IMG_DIR, PHYSBENCH]:
                p = d / fname
                if p.exists():
                    try:
                        result.append(image_to_bytes(p))
                        found = True
                        break
                    except Exception:
                        return None
            if not found:
                return None
    return result


MCQ_SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def clean_question(question: str, mcq_suffix: bool = False) -> str:
    """Strip <video>/<image> placeholders - images sent separately."""
    text = re.sub(r"<video>|<image>", "", question).strip()
    if mcq_suffix:
        text += MCQ_SUFFIX
    return text


# ── Inference ─────────────────────────────────────────────────────────────────

async def predict_async(sampling_client, tokenizer, record: dict,
                        mcq_suffix: bool = False) -> str | None:
    loop = asyncio.get_event_loop()
    img_bytes = await loop.run_in_executor(None, collect_image_bytes, record)
    if img_bytes is None:
        return None

    text = clean_question(record["question"], mcq_suffix=mcq_suffix)
    prompt = build_model_input(tokenizer, img_bytes, text)

    try:
        result = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=512,
                temperature=0.0,
                stop=["<|im_end|>"],
            ),
        )
        tokens = result.sequences[0].tokens
        return tokenizer.decode(tokens, skip_special_tokens=True).strip()
    except Exception as e:
        return f"ERROR: {e}"


def extract_choice(text: str) -> str:
    if not text or text.startswith("ERROR"):
        return ""
    t = text.upper()
    # 1. Explicit answer patterns (highest priority)
    for pat in [r"ANSWER[:\s]+([A-D])", r"THEREFORE[,\s]+([A-D])\b",
                r"OPTION\s+([A-D])\b", r"CORRECT.*?([A-D])\b",
                r"\bANSWER IS\s+([A-D])\b"]:
        m = re.search(pat, t)
        if m:
            return m.group(1)
    # 2. Last standalone A/B/C/D in text
    matches = list(re.finditer(r"\b([A-D])\b", t))
    if matches:
        return matches[-1].group(1)
    return ""


# ── Eval loop ─────────────────────────────────────────────────────────────────

async def run_eval(sampling_client, tokenizer, records: list[dict],
                   label: str, out_dir: Path,
                   concurrency: int = 8,
                   mcq_suffix: bool = False) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.json"

    done = {}
    if pred_file.exists():
        for p in json.load(open(pred_file)):
            done[p["idx"]] = p
        print(f" Resuming: {len(done)} already done")

    predictions = list(done.values())
    remaining = [r for r in records if r["idx"] not in done]
    print(f" Running {len(remaining)} samples [{label}] (concurrency={concurrency})...")

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    done_count = 0

    async def process(record):
        nonlocal done_count
        async with sem:
            raw = await predict_async(sampling_client, tokenizer, record,
                                            mcq_suffix=mcq_suffix)
            choice = extract_choice(raw or "")
            correct = choice == record["answer"]
            result = {
                "idx": record["idx"],
                "mode": record["mode"],
                "task_type": record["task_type"],
                "sub_type": record["sub_type"],
                "ability_type": record["ability_type"],
                "answer": record["answer"],
                "predicted": choice,
                "raw": (raw or "")[:800],
                "correct": correct,
            }
            tick = "OK" if correct else "XX"
            async with lock:
                done_count += 1
                predictions.append(result)
                print(f" [{done_count:4}/{len(remaining)}] idx={record['idx']:5} "
                      f"mode={record['mode']:<12} task={record['task_type']:<14} "
                      f"GT={record['answer']} pred={choice} {tick}")
                if done_count % 50 == 0:
                    with open(pred_file, "w") as f:
                        json.dump(predictions, f, indent=2)

    await asyncio.gather(*[process(r) for r in remaining])

    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=2)
    return predictions


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_scores(predictions: list[dict]) -> dict:
    def acc(items):
        if not items: return {"acc": 0.0, "n": 0, "correct": 0}
        c = sum(p["correct"] for p in items)
        return {"acc": c/len(items)*100, "n": len(items), "correct": c}

    scores = {"overall": acc(predictions)}
    for key in ["mode", "task_type", "sub_type"]:
        groups = defaultdict(list)
        for p in predictions: groups[p[key]].append(p)
        scores[f"by_{key}"] = {k: acc(v) for k, v in sorted(groups.items())}
    # Cross-tabulation: (mode, task_type) for per-mode per-task tables
    mode_task = defaultdict(list)
    for p in predictions:
        mode_task[(p["mode"], p["task_type"])].append(p)
    scores["by_mode_task"] = {k: acc(v) for k, v in sorted(mode_task.items())}
    return scores


def print_summary(scores: dict, label: str):
    o = scores["overall"]
    sep = "-" * 40
    print(f"\n == {label} ==")
    print(f" Overall: {o['correct']}/{o['n']} = {o['acc']:.1f}%")
    print(f"\n {sep}")
    print(f" {'Mode':<14} {'Acc':>8} {'N':>5}")
    print(f" {sep}")
    for m, v in scores["by_mode"].items():
        print(f" {m:<14} {v['acc']:>7.1f}% {v['n']:>5}")
    print(f"\n {sep}")
    print(f" {'Task Type':<16} {'Acc':>8} {'N':>5}")
    print(f" {sep}")
    for t, v in scores["by_task_type"].items():
        print(f" {t:<16} {v['acc']:>7.1f}% {v['n']:>5}")


def save_results_md(scores: dict, out_dir: Path, label: str,
                    baseline_scores: dict | None = None):
    o = scores["overall"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# PhysBench Eval -- {label}\n",
        f"**Date:** {ts} | **Samples:** {o['n']} | **Split:** PhysBench test set\n",
        f"## Overall: {o['correct']}/{o['n']} ({o['acc']:.1f}%)\n",
    ]
    # Per-mode tables (image-only, image+video, general)
    mode_labels = {"image-only": "Image-Only", "image&video": "Image+Video",
                   "general": "Video+Choices"}
    for mode_key, mode_name in mode_labels.items():
        mv = scores["by_mode"].get(mode_key)
        if not mv or mv["n"] == 0:
            continue
        lines.append(f"## {mode_name} ({mv['n']} samples)\n")
        if baseline_scores:
            lines.append("| Task | Correct | N | Acc | Baseline | Delta |")
            lines.append("|------|---------|---|-----|----------|-------|")
        else:
            lines.append("| Task | Correct | N | Accuracy |")
            lines.append("|------|---------|---|----------|")
        for t, tv in scores["by_task_type"].items():
            # Filter to this mode: use by_mode_task if available
            mt = scores.get("by_mode_task", {}).get((mode_key, t))
            if not mt or mt["n"] == 0:
                continue
            if baseline_scores:
                bt = baseline_scores.get("by_mode_task", {}).get((mode_key, t))
                bacc = f"{bt['acc']:.1f}%" if bt and bt["n"] > 0 else "-"
                delta = f"{mt['acc']-bt['acc']:+.1f}" if bt and bt["n"] > 0 else "-"
                lines.append(f"| {t} | {mt['correct']} | {mt['n']} | "
                             f"{mt['acc']:.1f}% | {bacc} | {delta} |")
            else:
                lines.append(f"| {t} | {mt['correct']} | {mt['n']} | {mt['acc']:.1f}% |")
        # Mode total row
        if baseline_scores:
            bmv = baseline_scores["by_mode"].get(mode_key)
            bacc = f"{bmv['acc']:.1f}%" if bmv and bmv["n"] > 0 else "-"
            delta = f"{mv['acc']-bmv['acc']:+.1f}" if bmv and bmv["n"] > 0 else "-"
            lines.append(f"| **Total** | **{mv['correct']}** | **{mv['n']}** | "
                         f"**{mv['acc']:.1f}%** | **{bacc}** | **{delta}** |")
        else:
            lines.append(f"| **Total** | **{mv['correct']}** | **{mv['n']}** | "
                         f"**{mv['acc']:.1f}%** |")
        lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f" Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def async_main(args):
    print(f"\n{BOLD}PhysBench Eval via Tinker - {BASE_MODEL}{RESET}")
    vid_status = "OK" if VID_DIR.exists() else f"{YELLOW}NOT FOUND{RESET}"
    print(f" Video dir : {VID_DIR} [{vid_status}]")

    if args.anchor:
        # Stratified 80-sample anchor set: 10 per (task_type × mode) combination
        # Used for GRPO regression monitoring every N steps
        all_records = load_val_split(max_samples=None, seed=args.seed, split="test")
        from collections import defaultdict as _dd
        _buckets = _dd(list)
        for r in all_records:
            _buckets[(r["task_type"], r["mode"])].append(r)
        rng80 = random.Random(args.seed + 999)
        records = []
        for bucket in sorted(_buckets.keys()):
            pool = _buckets[bucket]
            rng80.shuffle(pool)
            records.extend(pool[:10])
        records = records[:80]
        print(f" [ANCHOR] Stratified 80-sample set: "
              f"{len(set(r['task_type'] for r in records))} task types, "
              f"{len(set(r['mode'] for r in records))} modes")
    else:
        records = load_val_split(max_samples=args.max_samples, seed=args.seed,
                                 split=args.split)
    if not VID_DIR.exists():
        before = len(records)
        records = [r for r in records if r["mode"] == "image-only"]
        print(f" {YELLOW}[WARN]{RESET} No video dir - filtered to image-only: {len(records)}/{before}")

    mode_counts = defaultdict(int)
    for r in records: mode_counts[r["mode"]] += 1
    print(f" Loaded {len(records)} samples: " +
          " ".join(f"{m}={c}" for m, c in sorted(mode_counts.items())))

    print(f"\n Loading tokenizer...")
    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    service_client = tinker.ServiceClient()

    baseline_scores = None

    split_tag = args.split # "val" or "test"

    if args.compare or not args.model_path:
        print(f"\n {BOLD}[Baseline] Zero-shot {BASE_MODEL}{RESET}")
        base_client = service_client.create_sampling_client(base_model=BASE_MODEL)
        out_dir = RESULTS_DIR / f"physbench_baseline_{split_tag}"
        base_preds = await run_eval(base_client, tokenizer, records,
                                         "Zero-shot", out_dir, args.concurrency,
                                         mcq_suffix=args.mcq_suffix)
        baseline_scores = compute_scores(base_preds)
        print_summary(baseline_scores, "Zero-shot Baseline")
        save_results_md(baseline_scores, out_dir, "Zero-shot Baseline")

    if args.model_path:
        print(f"\n {BOLD}[SFT] {args.model_path}{RESET}")
        sft_client = service_client.create_sampling_client(model_path=args.model_path)
        out_tag = args.out_tag if args.out_tag else f"sft_{split_tag}"
        out_dir = RESULTS_DIR / f"physbench_{out_tag}"
        sft_preds = await run_eval(sft_client, tokenizer, records,
                                    "SFT", out_dir, args.concurrency,
                                    mcq_suffix=args.mcq_suffix)
        sft_scores = compute_scores(sft_preds)
        print_summary(sft_scores, "SFT Checkpoint")
        save_results_md(sft_scores, out_dir, "SFT Checkpoint",
                        baseline_scores=baseline_scores)

        if baseline_scores:
            b = baseline_scores["overall"]
            s = sft_scores["overall"]
            col = GREEN if s["acc"] >= b["acc"] else RED
            print(f"\n {BOLD}-- Delta (SFT vs Baseline) ------------------{RESET}")
            print(f" Overall: {b['acc']:.1f}% -> {s['acc']:.1f}% {col}{s['acc']-b['acc']:+.1f}%{RESET}")
            print(f"\n {'Mode':<14} {'Base':>8} {'SFT':>8} {'Delta':>8}")
            print(f" {'-'*40}")
            for m in sorted(baseline_scores["by_mode"]):
                bv = baseline_scores["by_mode"][m]["acc"]
                sv = sft_scores["by_mode"].get(m, {}).get("acc", 0.0)
                col = GREEN if sv >= bv else RED
                print(f" {m:<14} {bv:>7.1f}% {sv:>6.1f}% {col}{sv-bv:>+6.1f}%{RESET}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None,
                        help="Tinker checkpoint: tinker://<job-id>:train:0/sampler_weights/<step>")
    parser.add_argument("--compare", action="store_true",
                        help="Run baseline alongside fine-tuned checkpoint")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap val samples (default: all 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="PhysBench split to evaluate on (default: val)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent requests to Tinker (default: 8)")
    parser.add_argument("--out-tag", type=str, default=None,
                        help="Override output dir suffix (default: sft_<split>). E.g. 'sft_fluid_val'")
    parser.add_argument("--mcq-suffix", action="store_true", default=False,
                        help="Append 'Answer with the option\\'s letter...' to each question")
    parser.add_argument("--anchor", action="store_true", default=False,
                        help="Run stratified 80-sample anchor eval (10/task) for GRPO regression monitoring")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
