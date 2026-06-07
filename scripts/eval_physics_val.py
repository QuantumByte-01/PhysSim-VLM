#!/usr/bin/env python3
"""
PhysSim-VLM: Post-Epoch Evaluation on Physics Val Set
============================================================
Runs the trained model on the generated val/test split and scores
per-task accuracy. Called automatically by the training callback,
or manually after training.

Usage:
  # Evaluate a checkpoint
  python scripts/eval_physics_val.py --checkpoint checkpoints/lora_sft_epoch1/final

  # Evaluate base model (zero-shot)
  python scripts/eval_physics_val.py --zero_shot

  # Score existing predictions JSON
  python scripts/eval_physics_val.py --score_only --predictions results/sft_epoch1/predictions.json
"""

import os, re, json, argparse, random, base64, io
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "generated"
RESULTS = ROOT / "results"

HF_TOKEN = os.getenv("HF_TOKEN")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")

TASKS = ["ttc", "stability", "trajectory"]


# ── Ground truth answer extractor ────────────────────────────────────────────

def gt_answer(task: str, gt: dict) -> str:
    if task == "ttc":
        return f"{gt['time_to_collision']:.2f}"
    elif task == "stability":
        return "stable" if gt["is_stable"] else "unstable"
    elif task == "trajectory":
        lp = gt["landing_position"]
        return f"x={lp['x']:.2f}, y={lp['y']:.2f}"
    return ""


# ── Answer parser ─────────────────────────────────────────────────────────────

def parse_answer(text: str, task: str) -> str:
    """Extract the answer from model output."""
    if not text:
        return ""

    # Try <answer>...</answer> tag first
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def score_answer(pred: str, truth: str, task: str) -> tuple[bool, float]:
    """
    Returns (exact_match, continuous_score).
    continuous_score: 1.0 = perfect, 0.0 = completely wrong.
    """
    pred = pred.strip().lower()
    truth = truth.strip().lower()

    if task == "stability":
        match = pred == truth or truth in pred
        return match, 1.0 if match else 0.0

    elif task == "ttc":
        # Extract float from both
        try:
            p_val = float(re.search(r"[\d.]+", pred).group())
            t_val = float(re.search(r"[\d.]+", truth).group())
            err = abs(p_val - t_val) / max(t_val, 0.1)
            score = float(torch.exp(torch.tensor(-3.0 * err)).item())
            exact = err < 0.10 # within 10%
            return exact, score
        except Exception:
            return False, 0.0

    elif task == "trajectory":
        # Parse x=X.XX, y=X.XX
        try:
            def parse_xy(s):
                xm = re.search(r"x\s*=\s*([-\d.]+)", s)
                ym = re.search(r"y\s*=\s*([-\d.]+)", s)
                return float(xm.group(1)), float(ym.group(1))
            px, py = parse_xy(pred)
            tx, ty = parse_xy(truth)
            dist = ((px - tx)**2 + (py - ty)**2)**0.5
            score = float(torch.exp(torch.tensor(-0.5 * dist)).item())
            exact = dist < 0.5 # within 0.5m
            return exact, score
        except Exception:
            return False, 0.0

    return False, 0.0


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(split: str = "val", max_samples: int = None) -> list[dict]:
    split_file = DATA_DIR.parent / f"{split}.json"
    if not split_file.exists():
        return _load_split_from_hf(split, max_samples)


def _load_split_from_hf(split: str = "val", max_samples: int = None) -> list[dict]:
    """Fallback: load val/test records from Swastikr/PhysSim-VLM-Dataset on HuggingFace."""
    from datasets import load_dataset as hf_load_dataset
    from PIL import Image
    hf_token = os.getenv("HF_TOKEN")
    hf_ds = hf_load_dataset("Swastikr/PhysSim-VLM-Dataset", token=hf_token, split=split)
    if max_samples:
        import random as _rnd
        indices = list(range(len(hf_ds)))
        _rnd.seed(42)
        _rnd.shuffle(indices)
        hf_ds = hf_ds.select(indices[:max_samples])

    records = []
    for item in hf_ds:
        gt = json.loads(item["ground_truth"])
        imgs = []
        for b64_str in item["frames_b64"]:
            img_bytes = base64.b64decode(b64_str)
            imgs.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        records.append({
            "scene_id": item["scene_id"],
            "task": item["task"],
            "difficulty": item["difficulty"],
            "prompt": item["prompt"],
            "images": imgs,
            "gt_answer": item["answer"],
            "gt_raw": gt,
        })
    return records


def _load_split_from_file(split: str = "val", max_samples: int = None) -> list[dict]:
    split_file = DATA_DIR.parent / f"{split}.json"
    if not split_file.exists():
        raise FileNotFoundError(f"No {split}.json - run generate_training_data.py --splits_only first")

    with open(split_file) as f:
        index = json.load(f)

    records = []
    for item in index:
        task = item["task"]
        scene_dir = DATA_DIR / task / item["scene_id"]
        gt_path = scene_dir / "ground_truth.json"
        prompt_p = scene_dir / "prompt.txt"
        if not gt_path.exists() or not prompt_p.exists():
            continue

        with open(gt_path) as f: gt = json.load(f)
        with open(prompt_p) as f: prompt = f.read().strip()

        # Load frames
        if task == "stability":
            imgs = []
            for c in [scene_dir/"scene.png", scene_dir/"frame_000.png", scene_dir/"frames"/"frame_000.png", scene_dir/"thumbnail.png"]:
                if c.exists():
                    imgs = [Image.open(c).convert("RGB")]
                    break
        else:
            imgs = [Image.open(f).convert("RGB")
                    for f in sorted((scene_dir/"frames").glob("frame_*.png"))]

        if not imgs:
            continue

        records.append({
            "scene_id": item["scene_id"],
            "task": task,
            "difficulty": gt.get("difficulty", item.get("difficulty", "unknown")),
            "prompt": prompt,
            "images": imgs,
            "gt_answer": gt_answer(task, gt),
            "gt_raw": gt,
        })

    if max_samples:
        random.seed(42)
        random.shuffle(records)
        records = records[:max_samples]

    return records


# ── Model inference ───────────────────────────────────────────────────────────

def merge_lora_if_needed(checkpoint: str, merged_dir: Path) -> str:
    """Merge LoRA adapter into base weights and save to merged_dir for vLLM."""
    if merged_dir.exists() and (merged_dir / "config.json").exists():
        print(f" ✓ Merged model already exists at {merged_dir}")
        return str(merged_dir)

    print(f" Merging LoRA adapter → {merged_dir} (needed for vLLM)...")
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, token=HF_TOKEN, torch_dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_dir))

    proc = AutoProcessor.from_pretrained(BASE_MODEL, token=HF_TOKEN, trust_remote_code=True)
    proc.save_pretrained(str(merged_dir))

    print(f" ✓ Merged model saved to {merged_dir}")
    return str(merged_dir)


def load_vllm_engine(model_path: str):
    """
    Load model via vLLM for fast batched inference.
    vLLM gives ~4-8× throughput vs HF generate on MI300X.
    """
    from vllm import LLM, SamplingParams

    print(f" Loading vLLM engine: {model_path}")
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85, # leave headroom for image tokens
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 10}, # up to 10 frames per sample
    )
    print(" ✓ vLLM engine ready")
    return llm


def build_vllm_prompt(images: list[Image.Image], prompt_text: str,
                      processor) -> tuple[str, list]:
    """Build the chat-formatted prompt string + image list for vLLM."""
    user_content = [{"type": "image"} for _ in images]
    user_content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": user_content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text, images


def run_inference_batch_vllm(llm, processor, batch: list[dict]) -> list[str]:
    """
    Run a batch of samples through vLLM.
    Returns list of raw text outputs in same order as input.
    """
    from vllm import SamplingParams
    from vllm.multimodal.image import ImagePixelData

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        stop=["<|im_end|>", "</answer>"],
    )

    prompts = []
    for rec in batch:
        text, imgs = build_vllm_prompt(rec["images"], rec["prompt"], processor)
        prompts.append({
            "prompt": text,
            "multi_modal_data": {"image": imgs},
        })

    outputs = llm.generate(prompts, sampling_params=sampling)
    return [o.outputs[0].text.strip() for o in outputs]


# ── Evaluation loop ───────────────────────────────────────────────────────────

EVAL_BATCH_SIZE = 16 # vLLM batches - tune based on image token count


def run_eval(llm, processor, records: list[dict], out_dir: Path) -> list[dict]:
    """
    Batched evaluation using vLLM engine.
    llm: vLLM LLM instance (or None to skip inference and score_only).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.json"

    # Resume
    done = {}
    if pred_file.exists():
        with open(pred_file) as f:
            existing = json.load(f)
        done = {p["scene_id"]: p for p in existing}
        print(f" Resuming: {len(done)} already done")

    predictions = list(done.values())
    remaining = [r for r in records if r["scene_id"] not in done]

    if not remaining:
        print(" All samples already evaluated.")
        return predictions

    print(f" Running {len(remaining)} samples with vLLM (batch={EVAL_BATCH_SIZE})...")

    for i in tqdm(range(0, len(remaining), EVAL_BATCH_SIZE), desc=" vLLM inference"):
        batch = remaining[i : i + EVAL_BATCH_SIZE]
        try:
            raw_outputs = run_inference_batch_vllm(llm, processor, batch)
        except Exception as e:
            raw_outputs = [f"ERROR: {e}"] * len(batch)

        for rec, raw_output in zip(batch, raw_outputs):
            predicted = parse_answer(raw_output, rec["task"])
            exact, score = score_answer(predicted, rec["gt_answer"], rec["task"])
            predictions.append({
                "scene_id": rec["scene_id"],
                "task": rec["task"],
                "difficulty": rec["difficulty"],
                "gt_answer": rec["gt_answer"],
                "predicted": predicted,
                "exact_match": exact,
                "score": round(score, 4),
                "raw_output": raw_output[:500],
            })

        # Checkpoint every 5 batches
        if (i // EVAL_BATCH_SIZE + 1) % 5 == 0:
            with open(pred_file, "w") as f:
                json.dump(predictions, f, indent=2)

    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=2)

    return predictions


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_scores(predictions: list[dict]) -> dict:
    def agg(items):
        if not items: return {"exact": 0.0, "score": 0.0, "n": 0}
        return {
            "exact": sum(p["exact_match"] for p in items) / len(items) * 100,
            "score": sum(p["score"] for p in items) / len(items),
            "n": len(items),
        }

    scores = {"overall": agg(predictions)}
    for key in ["task", "difficulty"]:
        groups = defaultdict(list)
        for p in predictions:
            groups[p[key]].append(p)
        scores[f"by_{key}"] = {k: agg(v) for k, v in sorted(groups.items())}

    return scores


# ── Results writer ────────────────────────────────────────────────────────────

def save_results_md(scores: dict, out_dir: Path, epoch: int,
                    checkpoint: str, n_samples: int, split: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    o = scores["overall"]

    lines = [
        f"# PhysSim-VLM - Epoch {epoch} Evaluation\n",
        f"**Date:** {ts}",
        f"**Checkpoint:** `{checkpoint}`",
        f"**Split:** {split} | **Samples:** {n_samples}\n",
        "## Overall\n",
        f"**Exact Match: {o['exact']:.1f}%** | "
        f"**Physics Score (continuous): {o['score']:.4f}** "
        f"({o['n']} samples)\n",
        "## By Task\n",
        "| Task | Exact Match | Physics Score | N |",
        "|------|-------------|---------------|---|",
    ]
    for task, v in scores["by_task"].items():
        lines.append(f"| {task} | {v['exact']:.1f}% | {v['score']:.4f} | {v['n']} |")

    lines += [
        "\n## By Difficulty\n",
        "| Difficulty | Exact Match | Physics Score | N |",
        "|------------|-------------|---------------|---|",
    ]
    for diff, v in scores["by_difficulty"].items():
        lines.append(f"| {diff} | {v['exact']:.1f}% | {v['score']:.4f} | {v['n']} |")

    # Comparison vs baseline
    lines += [
        "\n## vs Zero-Shot Baseline\n",
        "| Metric | Zero-shot | This checkpoint | Δ |",
        "|--------|-----------|-----------------|---|",
        f"| Overall exact match | - | {o['exact']:.1f}% | - |",
    ]
    for task, v in scores["by_task"].items():
        lines.append(f"| {task} exact match | - | {v['exact']:.1f}% | - |")

    md = "\n".join(lines) + "\n"
    out_path = out_dir / "results.md"
    out_path.write_text(md)
    print(f" ✓ Results saved → {out_path}")
    return md


# ── Trainer callback (used by train_lora_sft.py) ─────────────────────────────

from transformers import TrainerCallback

class PhysicsEvalCallback(TrainerCallback):
    """
    HuggingFace TrainerCallback that runs physics val evaluation after each epoch.
    Import and attach to Trainer in train_lora_sft.py.
    """

    def __init__(self, processor, val_records: list[dict],
                 results_dir: Path, max_eval_samples: int = 300,
                 epoch_offset: int = 0):
        self.processor = processor
        self.val_records = val_records
        self.results_dir = results_dir
        self.max_eval_samples = max_eval_samples
        self._epoch = epoch_offset

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self._epoch += 1
        epoch = self._epoch

        print(f"\n{'='*60}")
        print(f" Post-epoch {epoch} evaluation on physics val set (vLLM)")
        print(f"{'='*60}")

        out_dir = self.results_dir / f"sft_epoch{epoch}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint so vLLM can load it
        ckpt_path = Path(args.output_dir) / f"epoch{epoch}_eval_tmp"
        model.save_pretrained(str(ckpt_path))
        self.processor.save_pretrained(str(ckpt_path))

        # Merge LoRA + load via vLLM
        try:
            import vllm # noqa: F401
        except ImportError:
            print(" vLLM not installed - skipping post-epoch eval. Install vllm to enable.")
            return control

        merged_path = Path(args.output_dir) / f"epoch{epoch}_merged"
        merged_str = merge_lora_if_needed(str(ckpt_path), merged_path)
        llm = load_vllm_engine(merged_str)

        # Use a random subset for speed (full val = 1500, default 300)
        rng = random.Random(42)
        records = self.val_records.copy()
        rng.shuffle(records)
        records = records[:self.max_eval_samples]

        predictions = run_eval(llm, self.processor, records, out_dir)
        del llm # free VRAM before resuming training
        torch.cuda.empty_cache()
        scores = compute_scores(predictions)

        checkpoint_path = str(args.output_dir)
        save_results_md(scores, out_dir, epoch,
                        checkpoint=checkpoint_path,
                        n_samples=len(predictions),
                        split="val")

        # Print summary
        o = scores["overall"]
        print(f"\n Epoch {epoch} val results:")
        print(f" Overall exact={o['exact']:.1f}% score={o['score']:.4f}")
        for task, v in scores["by_task"].items():
            print(f" {task:<12} exact={v['exact']:.1f}% score={v['score']:.4f}")

        # Log to trainer state
        if state is not None:
            state.log_history.append({
                "epoch": epoch,
                "physics_val_exact": o["exact"],
                "physics_val_score": o["score"],
                **{f"physics_val_{t}_exact": v["exact"]
                   for t, v in scores["by_task"].items()},
            })

        model.train()
        print(f"{'='*60}\n")


# ── Main (standalone) ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PhysSim-VLM val evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="LoRA checkpoint dir (or 'final')")
    parser.add_argument("--zero_shot", action="store_true",
                        help="Evaluate base model without LoRA")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"])
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--score_only", action="store_true",
                        help="Re-score existing predictions.json without inference")
    args = parser.parse_args()

    label = "zero_shot" if args.zero_shot else f"sft_epoch{args.epoch}"
    out_dir = RESULTS / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n PhysSim-VLM · Physics Eval · {label}")
    print(f" Split: {args.split} | Max samples: {args.max_samples}")

    if args.score_only:
        pred_file = out_dir / "predictions.json"
        with open(pred_file) as f:
            predictions = json.load(f)
    else:
        records = load_split(args.split, max_samples=args.max_samples)
        print(f" Loaded {len(records)} samples")

        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(BASE_MODEL, token=HF_TOKEN, trust_remote_code=True)

        if args.zero_shot:
            model_path = BASE_MODEL
        else:
            ckpt = args.checkpoint or str(ROOT / "checkpoints" / "lora_sft_epoch1" / "final")
            merged_path = Path(ckpt).parent / "merged_for_vllm"
            model_path = merge_lora_if_needed(ckpt, merged_path)

        llm = load_vllm_engine(model_path)
        predictions = run_eval(llm, processor, records, out_dir)

    scores = compute_scores(predictions)
    ckpt_label = args.checkpoint or ("base model" if args.zero_shot else "sft_epoch1/final")
    save_results_md(scores, out_dir, args.epoch, ckpt_label, len(predictions), args.split)

    o = scores["overall"]
    print(f"\n ── Results ──────────────────────────────────")
    print(f" Exact match : {o['exact']:.1f}%")
    print(f" Physics score: {o['score']:.4f}")
    for task, v in scores["by_task"].items():
        print(f" {task:<12} exact={v['exact']:.1f}% score={v['score']:.4f}")
    print()


if __name__ == "__main__":
    main()
