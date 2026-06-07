#!/usr/bin/env python3
"""
PhysSim-VLM: Quick model tinker via Thinking Machines Tinker SDK
=======================================================================
Runs inference on Qwen3-VL-30B using Tinker's managed GPU infrastructure.
No local model loading - runs from any machine with a TINKER_API_KEY.

Setup:
  pip install tinker tinker-cookbook
  export TINKER_API_KEY=<from https://tinker-console.thinkingmachines.ai>

Usage:
  # Test base model (zero-shot)
  python scripts/tinker.py

  # Test a fine-tuned Tinker checkpoint
  python scripts/tinker.py --model-path "tinker://<job-id>:train:0/sampler_weights/<step>"

  # Compare base vs fine-tuned
  python scripts/tinker.py --model-path "tinker://<run-id>" --compare

  # More samples
  python scripts/tinker.py --n 5
"""

import os, re, json, asyncio, argparse, random
from pathlib import Path
from io import BytesIO

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from PIL import Image

import tinker
from tinker import types
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.image_processing_utils import get_image_processor

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "generated"
TASKS = ["ttc", "stability", "trajectory"]

BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ── Image helpers ─────────────────────────────────────────────────────────────

def img_to_bytes(path: Path) -> bytes:
    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_samples(n_per_task: int = 3, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for task in TASKS:
        task_dir = DATA_DIR / task
        if not task_dir.exists():
            print(f" {YELLOW}[SKIP]{RESET} {task_dir} not found")
            continue
        scenes = sorted(task_dir.iterdir())
        rng.shuffle(scenes)
        for scene_dir in scenes[:n_per_task]:
            gt_path = scene_dir / "ground_truth.json"
            prompt_path = scene_dir / "prompt.txt"
            if not gt_path.exists() or not prompt_path.exists():
                continue

            with open(gt_path) as f: gt = json.load(f)
            with open(prompt_path) as f: prompt = f.read().strip()

            if task == "stability":
                img_paths = []
                for c in [scene_dir / "scene.png",
                           scene_dir / "frame_000.png",
                           scene_dir / "frames" / "frame_000.png"]:
                    if c.exists():
                        img_paths = [c]
                        break
            else:
                img_paths = sorted((scene_dir / "frames").glob("frame_*.png"))

            if not img_paths:
                continue

            samples.append({
                "scene_id": scene_dir.name,
                "task": task,
                "prompt": prompt,
                "img_paths": list(img_paths),
                "gt": gt,
            })
    return samples


def gt_str(task: str, gt: dict) -> str:
    if task == "ttc":
        return f"{gt['time_to_collision']:.2f}s"
    elif task == "stability":
        return "stable" if gt["is_stable"] else "unstable"
    elif task == "trajectory":
        lp = gt["landing_position"]
        return f"x={lp['x']:.2f}, y={lp['y']:.2f}"
    return "?"


# ── Renderer setup ────────────────────────────────────────────────────────────

def build_renderer():
    print(f" Loading tokenizer + image processor for {BASE_MODEL}...")
    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    image_processor = get_image_processor(BASE_MODEL)
    return renderers.Qwen3VLInstructRenderer(tokenizer, image_processor)


# ── Inference ─────────────────────────────────────────────────────────────────

async def predict_async(sampling_client, renderer, sample: dict) -> str:
    content = []
    for p in sample["img_paths"]:
        content.append({"type": "image", "image": img_to_bytes(p)})
    content.append({"type": "text", "text": sample["prompt"]})

    messages = [{"role": "user", "content": content}]
    prompt = renderer.build_generation_prompt(messages)

    result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=256, temperature=0.0),
    )
    seq = result.sequences[0]
    response = renderer.parse_response(seq.tokens)
    return response.content if hasattr(response, "content") else str(response)


# ── Scoring ───────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def score(pred: str, truth: str, task: str) -> tuple[bool, float]:
    import math
    pred = pred.strip().lower()
    truth = truth.strip().lower()
    if task == "stability":
        ok = pred == truth or truth in pred
        return ok, 1.0 if ok else 0.0
    elif task == "ttc":
        try:
            p = float(re.search(r"[\d.]+", pred).group())
            t = float(re.search(r"[\d.]+", truth).group())
            err = abs(p - t) / max(t, 0.1)
            return err < 0.10, math.exp(-3.0 * err)
        except Exception:
            return False, 0.0
    elif task == "trajectory":
        try:
            def xy(s):
                x = float(re.search(r"x\s*=\s*([-\d.]+)", s).group(1))
                y = float(re.search(r"y\s*=\s*([-\d.]+)", s).group(1))
                return x, y
            px, py = xy(pred)
            tx, ty = xy(truth)
            dist = ((px - tx)**2 + (py - ty)**2)**0.5
            return dist < 0.5, math.exp(-0.5 * dist)
        except Exception:
            return False, 0.0
    return False, 0.0


# ── Display ───────────────────────────────────────────────────────────────────

def print_result(scene_id, task, label, raw, answer, truth_str, exact, sc):
    tick = f"{GREEN}✓{RESET}" if exact else f"{RED}✗{RESET}"
    print(f"\n {BOLD}[{scene_id}]{RESET} task={task}")
    print(f" {BOLD}Ground truth:{RESET} {YELLOW}{truth_str}{RESET}")
    print(f" {BOLD}{label}:{RESET} {answer} {tick} score={sc:.3f}")
    if not exact:
        m = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL)
        if m:
            snippet = m.group(1).strip()[:200].replace("\n", " ")
            print(f" {BOLD}Reasoning:{RESET} {snippet}…")


async def run_on_samples(sampling_client, renderer, samples, label) -> list[dict]:
    results = []
    for s in samples:
        truth = gt_str(s["task"], s["gt"])
        try:
            raw = await predict_async(sampling_client, renderer, s)
        except Exception as e:
            raw = f"ERROR: {e}"
        ans = extract_answer(raw)
        exact, sc = score(ans, truth, s["task"])
        print_result(s["scene_id"], s["task"], label, raw, ans, truth, exact, sc)
        results.append({"task": s["task"], "exact": exact, "score": sc})
    return results


def print_summary(results, label) -> tuple[int, float]:
    from collections import defaultdict
    by_task = defaultdict(list)
    for r in results: by_task[r["task"]].append(r)

    print(f"\n {BOLD}── {label} ──────────────────────────────────{RESET}")
    n = len(results)
    exact = sum(r["exact"] for r in results)
    avg = sum(r["score"] for r in results) / n if n else 0.0
    print(f" Overall: {exact}/{n} exact | avg score={avg:.3f}")
    print(f" {'Task':<14} {'Exact':>8} {'Avg Score':>12}")
    print(f" {'-'*36}")
    for task, rs in sorted(by_task.items()):
        ex = sum(r["exact"] for r in rs)
        sc = sum(r["score"] for r in rs) / len(rs)
        print(f" {task:<14} {ex}/{len(rs):>4} {sc:.3f}")
    return exact, avg


# ── Main ──────────────────────────────────────────────────────────────────────

async def async_main(args):
    print(f"\n{BOLD}PhysSim-VLM Tinker{RESET}")
    print(f" Base model : {BASE_MODEL}")
    if args.model_path:
        print(f" Checkpoint : {args.model_path}")
    print(f" Samples : {args.n} per task ({args.n * len(TASKS)} total)\n")

    samples = load_samples(n_per_task=args.n, seed=args.seed)
    print(f" Loaded {len(samples)} samples")

    renderer = build_renderer()
    service_client = tinker.ServiceClient()

    if args.compare:
        base_client = service_client.create_sampling_client(base_model=BASE_MODEL)
        base_results = await run_on_samples(base_client, renderer, samples, "Zero-shot base")
        base_exact, base_avg = print_summary(base_results, "Zero-shot base")

    sft_client = service_client.create_sampling_client(
        model_path=args.model_path) if args.model_path else \
        service_client.create_sampling_client(base_model=BASE_MODEL)

    label = "SFT checkpoint" if args.model_path else "Zero-shot base"
    sft_results = await run_on_samples(sft_client, renderer, samples, label)
    sft_exact, sft_avg = print_summary(sft_results, label)

    if args.compare and args.model_path:
        from collections import defaultdict
        def by_task_avg(rs):
            d = defaultdict(list)
            for r in rs: d[r["task"]].append(r)
            return {t: (sum(x["exact"] for x in v)/len(v)*100,
                        sum(x["score"] for x in v)/len(v))
                    for t, v in d.items()}
        bt_b = by_task_avg(base_results)
        bt_s = by_task_avg(sft_results)
        print(f"\n {BOLD}── Delta (SFT − Base) ──────────────────────────{RESET}")
        print(f" {'Task':<14} {'Δ Exact%':>10} {'Δ Score':>10}")
        print(f" {'-'*36}")
        for task in sorted(bt_b):
            de = bt_s[task][0] - bt_b[task][0]
            ds = bt_s[task][1] - bt_b[task][1]
            col = GREEN if de >= 0 else RED
            print(f" {task:<14} {col}{de:+.1f}%{RESET} {col}{ds:+.3f}{RESET}")
        print(f"\n Overall Δ exact={sft_exact-base_exact:+d} Δ score={sft_avg-base_avg:+.3f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None,
                        help="Tinker checkpoint: tinker://<job-id>:train:0/sampler_weights/<step>")
    parser.add_argument("--compare", action="store_true",
                        help="Also run base model for side-by-side delta")
    parser.add_argument("--n", type=int, default=3,
                        help="Samples per task (default: 3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
