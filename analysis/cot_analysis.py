"""
Reasoning chain (CoT) analysis across 4 checkpoints:
  - Token length distribution (correct vs incorrect)
  - Length-vs-accuracy curve
  - Format-tag presence (<reasoning>...</reasoning>) - does SFT teach format?
  - Physics-vocab density (proxy for "physics grounding")

Outputs:
  analysis/cot_summary.csv
  analysis/cot_summary.md
  analysis/cot_length_buckets.csv
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"

CHECKPOINTS = [
    ("Baseline", "results/physbench_baseline_test/predictions.json"),
    ("SFT_R1", "results/physbench_sft_test/predictions.json"),
    ("SFT_R2redo", "results/physbench_sft_r2_redo_test/predictions.json"),
    ("GRPO_R3", "results/physbench_grpo_r3_test/predictions.json"),
]

# Physics-vocab proxy: words that suggest physical reasoning is happening
PHYSICS_VOCAB = {
    "gravity", "force", "velocity", "acceleration", "momentum", "friction",
    "mass", "weight", "energy", "kinetic", "potential", "viscosity", "density",
    "pressure", "buoyancy", "tension", "torque", "inertia", "elastic",
    "rigid", "fluid", "trajectory", "collision", "stable", "unstable",
    "equilibrium", "rotation", "translation", "newton", "law", "physics",
    "motion", "moving", "falling", "sliding", "rolling", "spinning",
    "balance", "support", "magnitude", "direction", "axis", "vector",
    "horizontal", "vertical", "lateral", "downward", "upward", "parallel",
    "perpendicular", "reflection", "refraction", "specular", "diffuse",
    "thermal", "temperature", "heat",
}

def tokenize_simple(text: str) -> list[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())

def has_format_tags(text: str) -> bool:
    return ("<reasoning>" in text and "</reasoning>" in text and
            "<answer>" in text and "</answer>" in text)

def physics_density(text: str) -> float:
    toks = tokenize_simple(text)
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in PHYSICS_VOCAB)
    return hits / len(toks)

def analyze(preds):
    rows = []
    for r in preds:
        raw = r.get("raw", "")
        toks = tokenize_simple(raw)
        rows.append({
            "len_words": len(toks),
            "len_chars": len(raw),
            "correct": r["correct"],
            "mode": r["mode"],
            "task": r["task_type"],
            "fmt_tags": has_format_tags(raw),
            "phys_density": physics_density(raw),
        })
    return rows

# ── Run analysis ──────────────────────────────────────────────────────────
all_data = {}
for name, path in CHECKPOINTS:
    print(f" Analyzing {name}...")
    preds = json.load(open(ROOT / path, encoding="utf-8"))
    all_data[name] = analyze(preds)

# ── Summary stats ─────────────────────────────────────────────────────────
def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0

def median(xs):
    xs = sorted(x for x in xs if x is not None)
    n = len(xs)
    return xs[n // 2] if n else 0.0

summary = []
for name, _ in CHECKPOINTS:
    rows = all_data[name]
    correct = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    summary.append({
        "ckpt": name,
        "n_total": len(rows),
        "mean_len_words": mean(r["len_words"] for r in rows),
        "median_len_words": median(r["len_words"] for r in rows),
        "mean_len_correct": mean(r["len_words"] for r in correct),
        "mean_len_wrong": mean(r["len_words"] for r in wrong),
        "fmt_tag_pct": 100 * sum(1 for r in rows if r["fmt_tags"]) / len(rows),
        "phys_density_correct": 100 * mean(r["phys_density"] for r in correct),
        "phys_density_wrong": 100 * mean(r["phys_density"] for r in wrong),
        "phys_density_overall": 100 * mean(r["phys_density"] for r in rows),
    })

# Markdown summary
md = ["# Chain-of-Thought (CoT) Analysis - PhysBench Test\n",
      "Word counts use whitespace+regex tokenizer; format tags = `<reasoning>...<answer>` block; ",
      "physics density = fraction of words in a curated physics-vocabulary set (gravity, force, viscosity, ...).\n",
      "## Length Statistics (words per response)\n",
      "| Checkpoint | Mean | Median | Mean (correct) | Mean (wrong) | Δ correct−wrong |",
      "|---|---|---|---|---|---|"]
for s in summary:
    delta = s["mean_len_correct"] - s["mean_len_wrong"]
    md.append(f"| {s['ckpt']} | {s['mean_len_words']:.0f} | {s['median_len_words']:.0f} | "
              f"{s['mean_len_correct']:.0f} | {s['mean_len_wrong']:.0f} | {delta:+.0f} |")

md.append("\n## Format Tag Adoption (`<reasoning>...<answer>` schema)\n")
md.append("| Checkpoint | % responses with full format tags |")
md.append("|---|---|")
for s in summary:
    md.append(f"| {s['ckpt']} | {s['fmt_tag_pct']:.1f}% |")

md.append("\n## Physics-Vocabulary Density (× 100 = words per 100 tokens)\n")
md.append("| Checkpoint | Overall | Correct responses | Wrong responses | Δ correct−wrong |")
md.append("|---|---|---|---|---|")
for s in summary:
    delta = s["phys_density_correct"] - s["phys_density_wrong"]
    md.append(f"| {s['ckpt']} | {s['phys_density_overall']:.2f} | "
              f"{s['phys_density_correct']:.2f} | {s['phys_density_wrong']:.2f} | {delta:+.2f} |")

# ── Length buckets vs accuracy ────────────────────────────────────────────
md.append("\n## Accuracy by Response-Length Bucket\n")
buckets = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 10_000)]
md.append("| Bucket (words) | " + " | ".join(name for name, _ in CHECKPOINTS) + " |")
md.append("|---|" + "---|" * len(CHECKPOINTS))
csv_buckets = ["bucket," + ",".join(name for name, _ in CHECKPOINTS)]
for lo, hi in buckets:
    label = f"{lo}-{hi if hi < 10_000 else '∞'}"
    cells = []
    for name, _ in CHECKPOINTS:
        rows = [r for r in all_data[name] if lo <= r["len_words"] < hi]
        if rows:
            acc = 100 * sum(1 for r in rows if r["correct"]) / len(rows)
            cells.append(f"{acc:.1f}% (n={len(rows)})")
        else:
            cells.append(" - ")
    md.append(f"| {label} | " + " | ".join(cells) + " |")
    csv_buckets.append(f"{label}," + ",".join(c.split('%')[0] for c in cells))

(ANALYSIS / "cot_summary.md").write_text("\n".join(md), encoding="utf-8")
(ANALYSIS / "cot_length_buckets.csv").write_text("\n".join(csv_buckets), encoding="utf-8")

# CSV summary
csv_lines = ["ckpt,n,mean_len,median_len,mean_len_correct,mean_len_wrong,fmt_pct,phys_density_overall,phys_density_correct,phys_density_wrong"]
for s in summary:
    csv_lines.append(f"{s['ckpt']},{s['n_total']},{s['mean_len_words']:.1f},{s['median_len_words']},"
                     f"{s['mean_len_correct']:.1f},{s['mean_len_wrong']:.1f},{s['fmt_tag_pct']:.2f},"
                     f"{s['phys_density_overall']:.3f},{s['phys_density_correct']:.3f},{s['phys_density_wrong']:.3f}")
(ANALYSIS / "cot_summary.csv").write_text("\n".join(csv_lines), encoding="utf-8")

print("Wrote:")
print(f" {ANALYSIS / 'cot_summary.md'}")
print(f" {ANALYSIS / 'cot_summary.csv'}")
print(f" {ANALYSIS / 'cot_length_buckets.csv'}")
