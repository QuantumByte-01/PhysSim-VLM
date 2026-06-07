#!/usr/bin/env python3
"""
External physics-benchmark transfer eval: ScienceQA-Physics + MMMU-Physics.

Tests whether physics knowledge from synthetic MuJoCo/PhiFlow training transfers
to held-out physics QA benchmarks (zero training-data overlap).

Usage:
  # Smoke test (20 samples each, both ckpts):
  python scripts/eval_external_physics.py --smoke
  # Full ScienceQA-Physics-vision (425 samples):
  python scripts/eval_external_physics.py --bench scienceqa --max 425
  # Full MMMU-Physics test (408 samples):
  python scripts/eval_external_physics.py --bench mmmu --max 408
"""
import os, re, json, asyncio, argparse, sys
from pathlib import Path
from io import BytesIO
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
if os.environ.get("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])

from PIL import Image
from datasets import load_dataset
import tinker
from tinker import types
from tinker_cookbook import tokenizer_utils

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
R2_REDO = "tinker://<run-id>:train:0/sampler_weights/final"

MAX_BYTES = 1_800_000
MAX_DIM = 1120

def compress(img: Image.Image) -> bytes:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_DIM:
        s = MAX_DIM / max(w, h)
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    for q in (85, 70, 55, 40):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q)
        if buf.tell() <= MAX_BYTES:
            return buf.getvalue()
    return buf.getvalue()

def build_input(tokenizer, image_bytes_list, text):
    chunks = []
    chunks.append(types.EncodedTextChunk(
        tokens=tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)))
    vs = tokenizer.encode("<|vision_start|>", add_special_tokens=False)
    ve = tokenizer.encode("<|vision_end|>", add_special_tokens=False)
    for b in image_bytes_list:
        chunks.append(types.EncodedTextChunk(tokens=vs))
        chunks.append(types.ImageChunk(data=b, format="jpeg"))
        chunks.append(types.EncodedTextChunk(tokens=ve))
    suffix = text + "<|im_end|>\n<|im_start|>assistant\n"
    chunks.append(types.EncodedTextChunk(
        tokens=tokenizer.encode(suffix, add_special_tokens=False)))
    return types.ModelInput(chunks=chunks)

def fmt_choices(choices):
    return "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))

def load_scienceqa(max_n=None):
    sq = load_dataset("lmms-lab/ScienceQA", "ScienceQA-FULL", split="test")
    phys = sq.filter(lambda x: x.get("topic") == "physics")
    records = []
    for i, r in enumerate(phys):
        if r["image"] is None:
            continue
        gt_letter = chr(65 + r["answer"])
        prompt = (f"{r['question']}\n\n{fmt_choices(r['choices'])}\n\n"
                  f"Answer with the option's letter from the given choices directly.")
        records.append({
            "idx": f"sciqa_{i}",
            "images": [r["image"]],
            "prompt": prompt,
            "answer": gt_letter,
            "n_choices": len(r["choices"]),
        })
        if max_n and len(records) >= max_n:
            break
    return records

def load_mmmu(split="validation", max_n=None):
    ds = load_dataset("MMMU/MMMU", "Physics", split=split)
    records = []
    for i, r in enumerate(ds):
        if r.get("question_type") != "multiple-choice":
            continue
        try:
            opts = eval(r["options"]) if isinstance(r["options"], str) else r["options"]
        except Exception:
            continue
        if not opts or not isinstance(opts, list):
            continue
        imgs = [r[f"image_{j}"] for j in range(1, 8) if r.get(f"image_{j}") is not None]
        if not imgs:
            continue
        q = re.sub(r"<image \d+>", "", r["question"]).strip()
        prompt = (f"{q}\n\n{fmt_choices(opts)}\n\n"
                  f"Answer with the option's letter from the given choices directly.")
        records.append({
            "idx": f"mmmu_{r['id']}",
            "images": imgs,
            "prompt": prompt,
            "answer": r["answer"].strip().upper(),
            "n_choices": len(opts),
        })
        if max_n and len(records) >= max_n:
            break
    return records

VALID_LETTERS = set("ABCDEFGH")
def extract(text):
    if not text:
        return ""
    t = text.upper()
    for pat in [r"ANSWER[:\s]+([A-H])", r"\bANSWER IS\s+([A-H])\b",
                r"OPTION\s+([A-H])\b", r"^([A-H])[\.\)\s]?$"]:
        m = re.search(pat, t, re.M)
        if m:
            return m.group(1)
    matches = list(re.finditer(r"\b([A-H])\b", t))
    if matches:
        return matches[-1].group(1)
    return ""

async def predict(client, tokenizer, record):
    loop = asyncio.get_event_loop()
    img_bytes = await loop.run_in_executor(None,
        lambda: [compress(im) for im in record["images"]])
    prompt = build_input(tokenizer, img_bytes, record["prompt"])
    try:
        result = await client.sample_async(
            prompt=prompt, num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=512, temperature=0.0, stop=["<|im_end|>"]))
        return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True).strip()
    except Exception as e:
        return f"ERROR: {e}"

async def run_eval(client, tokenizer, records, label, out_dir, conc=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.json"
    done = {}
    if pred_file.exists():
        for p in json.load(open(pred_file)):
            done[p["idx"]] = p
        print(f" Resume: {len(done)} done")
    preds = list(done.values())
    remaining = [r for r in records if r["idx"] not in done]
    print(f" Running {len(remaining)} samples [{label}] (conc={conc})...")
    sem = asyncio.Semaphore(conc)
    lock = asyncio.Lock()
    n_done = 0
    async def proc(r):
        nonlocal n_done
        async with sem:
            raw = await predict(client, tokenizer, r)
            choice = extract(raw or "")
            correct = choice == r["answer"]
            res = {"idx": r["idx"], "answer": r["answer"], "predicted": choice,
                   "raw": (raw or "")[:600], "correct": correct,
                   "n_choices": r["n_choices"]}
            tick = "OK" if correct else "XX"
            async with lock:
                n_done += 1
                preds.append(res)
                print(f" [{n_done:3}/{len(remaining)}] {r['idx']:<20} GT={r['answer']} pred={choice} {tick}")
                if n_done % 25 == 0:
                    json.dump(preds, open(pred_file, "w"), indent=2)
    await asyncio.gather(*[proc(r) for r in remaining])
    json.dump(preds, open(pred_file, "w"), indent=2)
    return preds

def score(preds):
    n = len(preds); c = sum(p["correct"] for p in preds)
    return {"n": n, "correct": c, "acc": 100*c/n if n else 0}

async def async_main(args):
    print(f"\nExternal Physics Benchmark Eval - {BASE_MODEL}")
    if args.bench in ("scienceqa", "both"):
        sq_records = load_scienceqa(max_n=args.max)
        print(f" ScienceQA-Physics-vision: {len(sq_records)} records")
    if args.bench in ("mmmu", "both"):
        mm_records = load_mmmu(split=args.mmmu_split, max_n=args.max)
        print(f" MMMU-Physics-{args.mmmu_split}: {len(mm_records)} records")

    print(f"\n Loading tokenizer...")
    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    sc = tinker.ServiceClient()

    results = {}
    for ckpt_label, ckpt_path in [("baseline", None), ("r2_redo", R2_REDO)]:
        if ckpt_path:
            client = sc.create_sampling_client(model_path=ckpt_path)
        else:
            client = sc.create_sampling_client(base_model=BASE_MODEL)
        print(f"\n === {ckpt_label.upper()} ===")
        if args.bench in ("scienceqa", "both"):
            out = RESULTS / f"external_scienceqa_{ckpt_label}{'_smoke' if args.smoke else ''}"
            sq_preds = await run_eval(client, tokenizer, sq_records, ckpt_label, out, args.conc)
            results[(ckpt_label, "scienceqa")] = score(sq_preds)
        if args.bench in ("mmmu", "both"):
            out = RESULTS / f"external_mmmu_{ckpt_label}{'_smoke' if args.smoke else ''}"
            mm_preds = await run_eval(client, tokenizer, mm_records, ckpt_label, out, args.conc)
            results[(ckpt_label, "mmmu")] = score(mm_preds)

    print("\n" + "="*60)
    print(f" {'Bench':<14} {'Baseline':>14} {'R2-redo':>14} {'Delta':>10}")
    print("="*60)
    for bench in ["scienceqa", "mmmu"]:
        b = results.get(("baseline", bench))
        r = results.get(("r2_redo", bench))
        if b and r:
            d = r["acc"] - b["acc"]
            print(f" {bench:<14} {b['correct']:>3}/{b['n']:<3} ({b['acc']:>5.1f}%) "
                  f"{r['correct']:>3}/{r['n']:<3} ({r['acc']:>5.1f}%) {d:+.1f}pp")

    summary_file = RESULTS / f"external_summary{'_smoke' if args.smoke else ''}.json"
    json.dump({f"{c}_{b}": v for (c,b), v in results.items()},
              open(summary_file, "w"), indent=2)
    print(f"\n Summary -> {summary_file}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["scienceqa", "mmmu", "both"], default="both")
    p.add_argument("--mmmu-split", default="validation", choices=["validation", "test"])
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 20 samples each bench")
    p.add_argument("--conc", type=int, default=6)
    args = p.parse_args()
    if args.smoke:
        args.max = 20
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()
