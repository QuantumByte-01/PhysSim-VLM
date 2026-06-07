"""
Calibration / confidence proxies - without access to model logprobs we use:
  1) Letter-prior bias: how often does each letter (A/B/C/D) get predicted?
     Compared against ground-truth distribution.
  2) Hedging-word frequency: explicit uncertainty markers in raw response.
  3) Per-subtask accuracy vs N: does accuracy vary smoothly with task type
     (calibrated knowledge) or spikily (overfit on specific patterns)?
  4) Response decisiveness: ratio of decisive answers (just letter or
     "answer: X" pattern) vs explanatory answers.
  5) Subtask coverage: how many subtasks improve / regress over baseline.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"

CHECKPOINTS = [
    ("Baseline", "results/physbench_baseline_test/predictions.json"),
    ("SFT R1", "results/physbench_sft_test/predictions.json"),
    ("SFT R2-redo", "results/physbench_sft_r2_redo_test/predictions.json"),
    ("GRPO R3", "results/physbench_grpo_r3_test/predictions.json"),
]

HEDGE_WORDS = ("might", "maybe", "perhaps", "possibly", "could be",
               "i think", "i believe", "not sure", "unclear", "hard to tell",
               "appears to", "seems to", "likely", "probably")
DECISIVE_PATTERNS = (re.compile(r'\banswer\s*[:=]\s*[ABCD]\b', re.I),
                     re.compile(r'^[ABCD]\.?\s*$', re.M),
                     re.compile(r'\bthe answer is\s*[ABCD]\b', re.I))

data = {name: json.load(open(ROOT / p, encoding="utf-8")) for name, p in CHECKPOINTS}

# Ground-truth letter distribution (same across all checkpoints)
gt_dist = Counter(r["answer"] for r in data["Baseline"] if r["answer"] in "ABCD")
gt_total = sum(gt_dist.values())

md = ["# Calibration & Confidence Analysis - PhysBench Test\n",
      "Without model logprobs, we use response-text proxies for calibration.\n"]

# ── 1) Letter-prior bias ──────────────────────────────────────────────────
md.append("## 1. Letter-Prior Bias\n")
md.append("Does the model prefer certain letters? Compare prediction distribution to ground-truth distribution.\n")
md.append("**Ground truth distribution**:")
for L in "ABCD":
    pct = 100 * gt_dist[L] / gt_total
    md.append(f" - {L}: {gt_dist[L]} ({pct:.1f}%)")
md.append("")
md.append("| Checkpoint | A | B | C | D | Bias entropy* | Letter-bias score† |")
md.append("|---|---|---|---|---|---|---|")
import math

def entropy(dist):
    total = sum(dist.values())
    if total == 0:
        return 0.0
    h = 0.0
    for v in dist.values():
        if v > 0:
            p = v / total
            h -= p * math.log2(p)
    return h

def bias_score(pred_dist, gt_dist):
    # KL(pred || gt) higher = more biased
    pred_total = sum(pred_dist.values()) or 1
    gt_total_local = sum(gt_dist.values()) or 1
    s = 0.0
    for L in "ABCD":
        p = pred_dist[L] / pred_total
        q = gt_dist[L] / gt_total_local
        if p > 0 and q > 0:
            s += p * math.log2(p / q)
    return s

for name, _ in CHECKPOINTS:
    pred_dist = Counter()
    for r in data[name]:
        p = (r.get("predicted") or "").strip().upper()
        if p in "ABCD":
            pred_dist[p] += 1
    total = sum(pred_dist.values()) or 1
    cells = [f"{100*pred_dist[L]/total:.1f}%" for L in "ABCD"]
    md.append(f"| {name} | " + " | ".join(cells) +
              f" | {entropy(pred_dist):.3f} | {bias_score(pred_dist, gt_dist):.4f} |")
md.append("")
md.append("*Entropy: 2.0 = uniform across A-D (unbiased), <2.0 = letter-preference bias.")
md.append("†Bias score: KL(prediction || ground-truth). 0 = perfectly calibrated to GT distribution.\n")

# ── 2) Hedging-word frequency ─────────────────────────────────────────────
md.append("## 2. Hedging Language (Confidence Proxy)\n")
md.append("Frequency of uncertainty markers (\"might\", \"perhaps\", \"unclear\", etc.).\n")
md.append("| Checkpoint | % responses with hedging | Hedge-on-correct | Hedge-on-wrong | Calibration gap* |")
md.append("|---|---|---|---|---|")
for name, _ in CHECKPOINTS:
    rows = data[name]
    hedge = sum(1 for r in rows if any(w in (r.get("raw") or "").lower() for w in HEDGE_WORDS))
    correct = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    h_correct = sum(1 for r in correct if any(w in (r.get("raw") or "").lower() for w in HEDGE_WORDS))
    h_wrong = sum(1 for r in wrong if any(w in (r.get("raw") or "").lower() for w in HEDGE_WORDS))
    h_correct_pct = 100 * h_correct / len(correct) if correct else 0
    h_wrong_pct = 100 * h_wrong / len(wrong) if wrong else 0
    cal_gap = h_wrong_pct - h_correct_pct
    md.append(f"| {name} | {100*hedge/len(rows):.2f}% | "
              f"{h_correct_pct:.2f}% | {h_wrong_pct:.2f}% | {cal_gap:+.2f}pp |")
md.append("\n*Calibration gap: positive = model hedges more on wrong answers (well-calibrated). "
          "Negative = model overconfident on wrong answers.\n")

# ── 3) Response decisiveness ──────────────────────────────────────────────
md.append("## 3. Response Decisiveness\n")
md.append("Fraction of responses matching a decisive answer pattern (just letter, \"Answer: X\", etc.).\n")
md.append("| Checkpoint | Decisive % | Decisive accuracy | Non-decisive accuracy |")
md.append("|---|---|---|---|")
for name, _ in CHECKPOINTS:
    rows = data[name]
    decisive = []
    nondec = []
    for r in rows:
        raw = r.get("raw") or ""
        is_dec = any(p.search(raw) for p in DECISIVE_PATTERNS) or len(raw.strip()) <= 5
        (decisive if is_dec else nondec).append(r)
    dec_acc = 100 * sum(1 for r in decisive if r["correct"]) / len(decisive) if decisive else 0
    nd_acc = 100 * sum(1 for r in nondec if r["correct"]) / len(nondec) if nondec else 0
    md.append(f"| {name} | {100*len(decisive)/len(rows):.1f}% | {dec_acc:.1f}% | {nd_acc:.1f}% |")
md.append("")

# ── 4) Subtask coverage (improvement breadth) ─────────────────────────────
md.append("## 4. Subtask Improvement Breadth (Generalization Proxy)\n")
md.append("Across all 39 subtasks, how many improve / regress vs baseline?\n")
md.append("Broader improvement = stronger generalization (vs narrow overfit).\n")
md.append("| Checkpoint | Subtasks improved | Subtasks regressed | No change | Mean Δ | Median Δ |")
md.append("|---|---|---|---|---|---|")
baseline_per_st = defaultdict(lambda: [0, 0])
for r in data["Baseline"]:
    k = (r["mode"], r["task_type"], r["sub_type"])
    baseline_per_st[k][1] += 1
    if r["correct"]:
        baseline_per_st[k][0] += 1

for name, _ in CHECKPOINTS:
    if name == "Baseline":
        md.append(f"| {name} | - | - | 39 | 0.0pp | 0.0pp |")
        continue
    per_st = defaultdict(lambda: [0, 0])
    for r in data[name]:
        k = (r["mode"], r["task_type"], r["sub_type"])
        per_st[k][1] += 1
        if r["correct"]:
            per_st[k][0] += 1
    deltas = []
    for k, (c, t) in per_st.items():
        if t == 0:
            continue
        b_c, b_t = baseline_per_st.get(k, [0, 0])
        if b_t == 0:
            continue
        d = (100 * c / t) - (100 * b_c / b_t)
        deltas.append(d)
    deltas.sort()
    impr = sum(1 for d in deltas if d > 0.5)
    regr = sum(1 for d in deltas if d < -0.5)
    nochg = len(deltas) - impr - regr
    mean_d = sum(deltas) / len(deltas)
    median_d = deltas[len(deltas) // 2]
    md.append(f"| {name} | {impr} | {regr} | {nochg} | {mean_d:+.2f}pp | {median_d:+.2f}pp |")
md.append("")

# ── 5) Variance across modes (mode-consistency) ───────────────────────────
md.append("## 5. Cross-Mode Consistency\n")
md.append("If a model has \"learned physics\" (vs format), accuracy should be roughly similar across modes. ")
md.append("Large gaps = mode-specific exploitation.\n")
md.append("| Checkpoint | image-only | image&video | general | Std (across 3 modes) |")
md.append("|---|---|---|---|---|")
for name, _ in CHECKPOINTS:
    rows = data[name]
    by_mode = defaultdict(lambda: [0, 0])
    for r in rows:
        by_mode[r["mode"]][1] += 1
        if r["correct"]:
            by_mode[r["mode"]][0] += 1
    accs = []
    for m in ["image-only", "image&video", "general"]:
        if by_mode[m][1] > 0:
            accs.append(100 * by_mode[m][0] / by_mode[m][1])
    mean_a = sum(accs) / len(accs)
    std = math.sqrt(sum((a - mean_a) ** 2 for a in accs) / len(accs))
    cells = [f"{a:.1f}%" for a in accs]
    md.append(f"| {name} | " + " | ".join(cells) + f" | {std:.2f} |")

(ANALYSIS / "calibration.md").write_text("\n".join(md), encoding="utf-8")
print(f"Wrote {ANALYSIS / 'calibration.md'}")
