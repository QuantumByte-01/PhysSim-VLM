#!/usr/bin/env python3
"""
SeePhys eval (vision-essential subset, n=1500): baseline vs R2-redo.

Free-form numerical/symbolic answers. Scoring: normalized numerical match
within 5% tolerance, OR exact string match after stripping units/whitespace/LaTeX.

Usage:
  python scripts/eval_seephys.py --smoke # 25 samples each ckpt
  python scripts/eval_seephys.py --max 1500 --conc 16 # full
"""
import os, re, json, asyncio, argparse, sys
from pathlib import Path
from io import BytesIO

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

def load_seephys(max_n=None, vision_essential_only=True):
    ds = load_dataset("SeePhys/SeePhys", split="train")
    records = []
    for i, r in enumerate(ds):
        if vision_essential_only and r.get("vision_relevance") != "essential":
            continue
        if not r.get("images"):
            continue
        prompt = (
            f"{r['question']}\n\n"
            f"Solve this physics problem step by step. "
            f"After your reasoning, give the final numerical answer "
            f"on a new line in the form: Final answer: <value>"
        )
        records.append({
            "idx": f"seephys_{r['index']}",
            "images": list(r["images"]),
            "prompt": prompt,
            "answer": str(r["answer"]).strip(),
            "subject": r.get("subject", ""),
            "level": r.get("level", 0),
            "img_category": r.get("img_category", ""),
            "sig_figs": r.get("sig_figs", 0),
        })
        if max_n and len(records) >= max_n:
            break
    return records

# --- Scoring helpers ---
_NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")

def _strip_latex(s):
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\times", "*", s)
    s = re.sub(r"\^\{?([^}\s]+)\}?", r"e\1", s) # 10^4 -> 10e4 (rough)
    s = re.sub(r"[\\${},~]", "", s)
    return s.strip()

def _extract_numbers(s):
    s = _strip_latex(s)
    return [float(m) for m in _NUM_RE.findall(s) if m not in ("", ".", "-", "-.")]

def extract_final_answer(text):
    if not text:
        return ""
    m = re.search(r"final\s+answer[:\s]+(.+?)(?:\n|$)", text, re.I)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""

def score_answer(pred_text, gt):
    """Lenient: normalized numerical match (within 5%) OR substring match."""
    if not pred_text:
        return False
    pred_final = extract_final_answer(pred_text)
    if not pred_final:
        return False
    gt_nums = _extract_numbers(gt)
    pred_nums = _extract_numbers(pred_final)
    if gt_nums and pred_nums:
        gt_v = gt_nums[0]
        for pv in pred_nums:
            if gt_v == 0:
                if abs(pv) < 1e-6:
                    return True
            elif abs(pv - gt_v) / max(abs(gt_v), 1e-9) <= 0.05:
                return True
    gt_norm = _strip_latex(gt).replace(" ", "").lower()
    pred_norm = _strip_latex(pred_final).replace(" ", "").lower()
    if gt_norm and gt_norm in pred_norm:
        return True
    return False

async def predict(client, tokenizer, record):
    loop = asyncio.get_event_loop()
    img_bytes = await loop.run_in_executor(None,
        lambda: [compress(im) for im in record["images"]])
    prompt = build_input(tokenizer, img_bytes, record["prompt"])
    try:
        result = await client.sample_async(
            prompt=prompt, num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=1024, temperature=0.0, stop=["<|im_end|>"]))
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
            correct = score_answer(raw, r["answer"])
            res = {"idx": r["idx"], "answer": r["answer"],
                   "predicted_final": extract_final_answer(raw or ""),
                   "raw": (raw or "")[:1200], "correct": correct,
                   "subject": r["subject"], "level": r["level"],
                   "img_category": r["img_category"]}
            tick = "OK" if correct else "XX"
            async with lock:
                n_done += 1
                preds.append(res)
                if n_done % 5 == 0 or n_done <= 10:
                    print(f" [{n_done:4}/{len(remaining)}] {r['idx']:<18} GT={r['answer'][:30]:<30} pred={res['predicted_final'][:30]:<30} {tick}")
                if n_done % 50 == 0:
                    json.dump(preds, open(pred_file, "w"), indent=2)
    await asyncio.gather(*[proc(r) for r in remaining])
    json.dump(preds, open(pred_file, "w"), indent=2)
    return preds

def score(preds):
    n = len(preds); c = sum(p["correct"] for p in preds)
    return {"n": n, "correct": c, "acc": 100*c/n if n else 0}

def by_subject(preds):
    from collections import defaultdict
    s = defaultdict(lambda: [0, 0])
    for p in preds:
        s[p["subject"]][1] += 1
        if p["correct"]:
            s[p["subject"]][0] += 1
    return {k: (c, t, 100*c/t) for k, (c, t) in s.items()}

async def async_main(args):
    print(f"\nSeePhys (vision-essential) Eval - {BASE_MODEL}")
    records = load_seephys(max_n=args.max)
    print(f" Records: {len(records)}")
    print(f"\n Loading tokenizer...")
    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    sc = tinker.ServiceClient()

    results = {}
    ckpts = [("baseline", None), ("r2_redo", R2_REDO)]
    if args.ckpt:
        ckpts = [c for c in ckpts if c[0] == args.ckpt]
    for ckpt_label, ckpt_path in ckpts:
        if ckpt_path:
            client = sc.create_sampling_client(model_path=ckpt_path)
        else:
            client = sc.create_sampling_client(base_model=BASE_MODEL)
        print(f"\n === {ckpt_label.upper()} ===")
        out = RESULTS / f"external_seephys_{ckpt_label}{'_smoke' if args.smoke else ''}"
        preds = await run_eval(client, tokenizer, records, ckpt_label, out, args.conc)
        results[ckpt_label] = score(preds)
        print(f"\n {ckpt_label}: {results[ckpt_label]}")
        print(f" By subject:")
        for subj, (c, t, a) in sorted(by_subject(preds).items()):
            print(f" {subj:<8} {a:>5.1f}% ({c}/{t})")

    if "baseline" in results and "r2_redo" in results:
        b, r = results["baseline"], results["r2_redo"]
        d = r["acc"] - b["acc"]
        print("\n" + "="*60)
        print(f" SeePhys: baseline {b['acc']:.1f}% ({b['correct']}/{b['n']}) "
              f"r2_redo {r['acc']:.1f}% ({r['correct']}/{r['n']}) delta={d:+.1f}pp")

    summary_file = RESULTS / f"external_seephys_summary{'_smoke' if args.smoke else ''}.json"
    json.dump(results, open(summary_file, "w"), indent=2)
    print(f"\n Summary -> {summary_file}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="25 samples per checkpoint")
    p.add_argument("--conc", type=int, default=8)
    p.add_argument("--ckpt", choices=["baseline", "r2_redo"], default=None,
                   help="run only one ckpt (default: both)")
    args = p.parse_args()
    if args.smoke:
        args.max = 25
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()
