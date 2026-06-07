"""
Build paper assets in parallel with GRPO run:
  1. ablation.md - overall + per-mode + per-task across 4 checkpoints
  2. per_subtask.csv - 39-subtask × 4-checkpoint accuracy matrix
  3. per_subtask.md - same in markdown for paper
  4. compute_receipt.md - $ + step counts per training stage
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
ANALYSIS.mkdir(exist_ok=True)

CHECKPOINTS = [
    ("Baseline", "results/physbench_baseline_test/predictions.json", "results/physbench_baseline_test/results.md"),
    ("SFT R1", "results/physbench_sft_test/predictions.json", "results/physbench_sft_test/results.md"),
    ("SFT R2-redo","results/physbench_sft_r2_redo_test/predictions.json", "results/physbench_sft_r2_redo_test/results.md"),
    ("GRPO R3", "results/physbench_grpo_r3_test/predictions.json", "results/physbench_grpo_r3_test/results.md"),
]

def load_predictions(rel):
    return json.load(open(ROOT / rel, encoding="utf-8"))

# ── 1) Per-subtask matrix ───────────────────────────────────────────────────
data = {name: load_predictions(p) for name, p, _ in CHECKPOINTS}

# group key = (mode, task_type, sub_type)
def by_subtask(preds):
    counts = defaultdict(lambda: [0, 0]) # [correct, total]
    for r in preds:
        key = (r["mode"], r["task_type"], r["sub_type"])
        counts[key][1] += 1
        if r["correct"]:
            counts[key][0] += 1
    return counts

per_ckpt = {name: by_subtask(p) for name, p in data.items()}
all_keys = sorted(set().union(*[set(d.keys()) for d in per_ckpt.values()]))

# CSV
csv_lines = ["mode,task_type,sub_type,N," + ",".join(name for name, _, _ in CHECKPOINTS)]
for k in all_keys:
    mode, task, sub = k
    n = per_ckpt["Baseline"].get(k, [0, 0])[1] or per_ckpt["SFT R1"].get(k, [0, 0])[1]
    accs = []
    for name, _, _ in CHECKPOINTS:
        c, t = per_ckpt[name].get(k, [0, 0])
        accs.append(f"{100*c/t:.1f}" if t else "")
    csv_lines.append(f"{mode},{task},{sub},{n}," + ",".join(accs))

(ANALYSIS / "per_subtask.csv").write_text("\n".join(csv_lines), encoding="utf-8")

# Markdown - sorted by N desc within each mode
md = ["# Per-Subtask Accuracy Matrix (PhysBench Test, 39 subtasks)\n",
      "Generated for paper. Bold = best per row; * = > Baseline.\n"]

for mode in ["image-only", "image&video", "general"]:
    keys_mode = [k for k in all_keys if k[0] == mode]
    keys_mode.sort(key=lambda k: -(per_ckpt["Baseline"].get(k, [0, 0])[1]))
    md.append(f"\n## Mode: `{mode}`\n")
    md.append("| Task | Subtask | N | " + " | ".join(name for name, _, _ in CHECKPOINTS) + " | Δ vs Baseline |")
    md.append("|---|---|---|" + "|".join(["---"] * (len(CHECKPOINTS) + 1)) + "|")
    for k in keys_mode:
        mode_, task, sub = k
        n = per_ckpt["Baseline"].get(k, [0, 0])[1]
        row = []
        accs_num = []
        for name, _, _ in CHECKPOINTS:
            c, t = per_ckpt[name].get(k, [0, 0])
            if t:
                acc = 100 * c / t
                accs_num.append(acc)
                row.append(f"{acc:.1f}%")
            else:
                accs_num.append(None)
                row.append(" - ")
        # bold best
        best = max(a for a in accs_num if a is not None)
        row = [f"**{r}**" if accs_num[i] == best else r for i, r in enumerate(row)]
        # delta vs baseline
        b = accs_num[0]
        last_useful = next((a for a in [accs_num[2], accs_num[1]] if a is not None), b)
        delta = last_useful - b if (b is not None and last_useful is not None) else 0
        delta_s = f"{delta:+.1f}"
        md.append(f"| {task} | {sub} | {n} | " + " | ".join(row) + f" | {delta_s} |")

(ANALYSIS / "per_subtask.md").write_text("\n".join(md), encoding="utf-8")

# ── 2) Ablation table ──────────────────────────────────────────────────────
def overall_stats(preds):
    by_mode = defaultdict(lambda: [0, 0])
    by_task = defaultdict(lambda: [0, 0])
    overall = [0, 0]
    for r in preds:
        by_mode[r["mode"]][1] += 1
        by_task[r["task_type"]][1] += 1
        overall[1] += 1
        if r["correct"]:
            by_mode[r["mode"]][0] += 1
            by_task[r["task_type"]][0] += 1
            overall[0] += 1
    return overall, by_mode, by_task

stats = {name: overall_stats(p) for name, p in data.items()}

ab = ["# Ablation Table - PhysBench Test (9,786 samples)\n",
      "| Stage | Overall | image-only | image&video | general (text-only) | Δ vs Baseline |",
      "|---|---|---|---|---|---|"]
b_overall = stats["Baseline"][0]
for name, _, _ in CHECKPOINTS:
    ov, bm, _ = stats[name]
    overall_acc = 100 * ov[0] / ov[1]
    delta = overall_acc - 100 * b_overall[0] / b_overall[1]
    ab.append(f"| {name} | **{overall_acc:.1f}%** ({ov[0]}/{ov[1]}) | "
              f"{100*bm['image-only'][0]/bm['image-only'][1]:.1f}% | "
              f"{100*bm['image&video'][0]/bm['image&video'][1]:.1f}% | "
              f"{100*bm['general'][0]/bm['general'][1]:.1f}% | "
              f"{delta:+.1f}pp |")

ab.append("\n## Per-Task-Type Decomposition\n")
ab.append("| Stage | Dynamics | Property | Relationships | Scene |")
ab.append("|---|---|---|---|---|")
for name, _, _ in CHECKPOINTS:
    _, _, bt = stats[name]
    ab.append(f"| {name} | " + " | ".join(
        f"{100*bt[t][0]/bt[t][1]:.1f}%" if bt[t][1] else " - "
        for t in ["dynamics", "property", "relationships", "scene"]
    ) + " |")

(ANALYSIS / "ablation.md").write_text("\n".join(ab), encoding="utf-8")

print("Wrote:")
print(f" {ANALYSIS / 'per_subtask.csv'}")
print(f" {ANALYSIS / 'per_subtask.md'}")
print(f" {ANALYSIS / 'ablation.md'}")
