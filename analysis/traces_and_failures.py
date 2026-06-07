"""
Two analyses:
  A) qualitative_traces.md - 4 example questions per task type showing
     all 4 checkpoint responses side-by-side (paper qualitative section)
  B) failure_taxonomy.md - categorize wrong predictions by failure mode
     (empty, off-by-one MCQ, plausible-but-wrong, refusal, etc.)
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
    ("SFT R2-redo","results/physbench_sft_r2_redo_test/predictions.json"),
    ("GRPO R3", "results/physbench_grpo_r3_test/predictions.json"),
]

print("Loading...")
data = {name: json.load(open(ROOT / p, encoding="utf-8")) for name, p in CHECKPOINTS}
# index by idx
by_idx = {name: {r["idx"]: r for r in rows} for name, rows in data.items()}

# ── A) Qualitative traces ─────────────────────────────────────────────────
# Pick "improvement examples": baseline WRONG, SFT-R2-redo CORRECT
# (most interesting pedagogically)
improvement_idxs = defaultdict(list)
all_idxs = list(by_idx["Baseline"].keys())
for idx in all_idxs:
    b = by_idx["Baseline"].get(idx)
    r2 = by_idx["SFT R2-redo"].get(idx)
    if not b or not r2:
        continue
    if not b["correct"] and r2["correct"]:
        improvement_idxs[r2["task_type"]].append(idx)

# Pick 4 per task, prioritizing diverse modes
N_PER_TASK = 4
selected = []
for task in ["dynamics", "property", "relationships", "scene"]:
    candidates = improvement_idxs[task]
    seen_modes = set()
    picks = []
    for idx in candidates:
        mode = by_idx["Baseline"][idx]["mode"]
        if mode not in seen_modes:
            picks.append(idx)
            seen_modes.add(mode)
        if len(picks) >= N_PER_TASK:
            break
    while len(picks) < N_PER_TASK and candidates:
        for idx in candidates:
            if idx not in picks:
                picks.append(idx)
                if len(picks) >= N_PER_TASK:
                    break
    selected.append((task, picks))

# Truncate raw responses for readability
def truncate(s, n=400):
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + " […truncated]"

md = ["# Qualitative Reasoning Traces - PhysBench Test\n",
      "Each block shows one PhysBench question that the **Baseline got wrong** and **SFT R2-redo got right**. ",
      "Responses are truncated at 400 chars. ✅ = correct, ❌ = wrong.\n"]

for task, idxs in selected:
    md.append(f"\n## Task: `{task}`\n")
    for n, idx in enumerate(idxs, 1):
        meta = by_idx["Baseline"][idx]
        md.append(f"### Example {n}: `{meta['mode']}` / `{task}` / `{meta['sub_type']}` (idx={idx})\n")
        md.append(f"**Ground truth**: `{meta['answer']}`\n")
        for name, _ in CHECKPOINTS:
            r = by_idx[name].get(idx)
            if r is None:
                continue
            mark = "✅" if r["correct"] else "❌"
            md.append(f"**{name}** {mark} pred=`{r.get('predicted', '?')}`")
            md.append(f"> {truncate(r['raw'])}\n")

(ANALYSIS / "qualitative_traces.md").write_text("\n".join(md), encoding="utf-8")
print(f" qualitative_traces.md ({sum(len(idxs) for _, idxs in selected)} examples)")

# ── B) Failure taxonomy ───────────────────────────────────────────────────
def classify_failure(r) -> str:
    raw = (r.get("raw") or "").strip()
    pred = (r.get("predicted") or "").strip().upper()
    gt = (r.get("answer") or "").strip().upper()
    if r["correct"]:
        return "correct"

    # Truly empty response (no content at all)
    if not raw:
        return "empty_response"

    # Unparseable: predicted is empty / not a letter
    if not pred or pred not in "ABCDEFG":
        # Did the response refuse or hedge?
        low = raw.lower()
        if any(w in low for w in ("i cannot", "i can't", "unable to", "i'm sorry",
                                   "i apologize", "as an ai", "no answer")):
            return "refusal"
        return "unparseable_predicted"

    # Terse answer (just a letter, but wrong)
    if len(raw) <= 3:
        return "terse_wrong_letter"

    # Off-by-one MCQ (adjacent letter)
    if pred in "ABCD" and gt in "ABCD":
        if abs(ord(pred) - ord(gt)) == 1:
            return "off_by_one_mcq"

    # Confident wrong: response contains physics reasoning + wrong letter
    phys_words = ("force", "gravity", "physics", "velocity", "viscosity",
                  "fluid", "mass", "momentum", "trajectory", "stable",
                  "collision", "rotation", "balance")
    low = raw.lower()
    if any(w in low for w in phys_words) and len(raw) > 30:
        return "wrong_reasoning_with_physics"

    # Short wrong answer (terse miss)
    if len(raw) < 30:
        return "terse_wrong"

    return "other_wrong"

tax_per_ckpt = {}
for name, _ in CHECKPOINTS:
    counts = Counter()
    for r in data[name]:
        counts[classify_failure(r)] += 1
    tax_per_ckpt[name] = counts

categories = ["empty_response", "refusal", "unparseable_predicted",
              "off_by_one_mcq", "wrong_reasoning_with_physics",
              "terse_wrong_letter", "terse_wrong", "other_wrong", "correct"]

md2 = ["# Failure Mode Taxonomy - PhysBench Test\n",
       "Classification heuristic applied to all 9,786 predictions per checkpoint. ",
       "Each response is bucketed into exactly one mode. Numbers are absolute counts.\n"]

md2.append("| Mode | " + " | ".join(name for name, _ in CHECKPOINTS) + " |")
md2.append("|---|" + "---|" * len(CHECKPOINTS))
for cat in categories:
    cells = []
    for name, _ in CHECKPOINTS:
        c = tax_per_ckpt[name][cat]
        cells.append(f"{c} ({100*c/9786:.1f}%)")
    md2.append(f"| {cat} | " + " | ".join(cells) + " |")

# ── Failure breakdown vs success ───────────────────────────────────────────
md2.append("\n## Wrong-only Distribution (excludes correct)\n")
md2.append("| Mode | " + " | ".join(name for name, _ in CHECKPOINTS) + " |")
md2.append("|---|" + "---|" * len(CHECKPOINTS))
for cat in [c for c in categories if c != "correct"]:
    cells = []
    for name, _ in CHECKPOINTS:
        c = tax_per_ckpt[name][cat]
        wrong_total = sum(v for k, v in tax_per_ckpt[name].items() if k != "correct")
        cells.append(f"{100*c/wrong_total:.1f}%")
    md2.append(f"| {cat} | " + " | ".join(cells) + " |")

# ── Salient observations ──────────────────────────────────────────────────
md2.append("\n## Notable Patterns\n")
notes = []
for name, _ in CHECKPOINTS:
    t = tax_per_ckpt[name]
    wrong_total = sum(v for k, v in t.items() if k != "correct")
    obo = 100 * t["off_by_one_mcq"] / wrong_total if wrong_total else 0
    refusal = t["refusal"]
    empty = t["empty_response"]
    unparse = t["unparseable_predicted"]
    notes.append(f"- **{name}**: {wrong_total} wrong, off-by-one MCQ rate {obo:.1f}% of failures, "
                 f"refusals={refusal}, empty={empty}, unparseable={unparse}")
md2.extend(notes)

(ANALYSIS / "failure_taxonomy.md").write_text("\n".join(md2), encoding="utf-8")
print(f" failure_taxonomy.md")

# CSV
csv = ["mode," + ",".join(name for name, _ in CHECKPOINTS)]
for cat in categories:
    csv.append(f"{cat}," + ",".join(str(tax_per_ckpt[name][cat]) for name, _ in CHECKPOINTS))
(ANALYSIS / "failure_taxonomy.csv").write_text("\n".join(csv), encoding="utf-8")
print(f" failure_taxonomy.csv")
