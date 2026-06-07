"""
Re-run all analysis scripts to incorporate GRPO R3 results.

Pre-requisite: results/physbench_grpo_r3_test/predictions.json must exist.

Re-runs (in order):
  1. build_paper_assets.py → ablation.md + per_subtask.md
  2. cot_analysis.py → cot_summary.md
  3. traces_and_failures.py → failure_taxonomy.md + qualitative_traces.md
  4. calibration.py → calibration.md
  5. leaderboard_comparison.py → leaderboard_*.md (also updates OURS row)

This script PATCHES each downstream script's CHECKPOINTS list (via env var)
to include GRPO R3, so no manual edit is needed.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
GRPO_R3_PRED = ROOT / "results/physbench_grpo_r3_test/predictions.json"

if not GRPO_R3_PRED.exists():
    print(f"ERROR: {GRPO_R3_PRED} not found. Run PhysBench test eval on GRPO R3 first.")
    sys.exit(1)

# Compute R3 overall + per-task scores so we can patch leaderboard
preds = json.load(open(GRPO_R3_PRED, encoding="utf-8"))
n = len(preds)
overall = 100 * sum(p["correct"] for p in preds) / n
print(f"GRPO R3 PhysBench test: {sum(p['correct'] for p in preds)}/{n} = {overall:.2f}%")
from collections import defaultdict
by_task = defaultdict(lambda: [0, 0])
by_mode = defaultdict(lambda: [0, 0])
for p in preds:
    by_task[p["task_type"]][1] += 1
    by_mode[p["mode"]][1] += 1
    if p["correct"]:
        by_task[p["task_type"]][0] += 1
        by_mode[p["mode"]][0] += 1
print("By task:")
for k, (c, t) in sorted(by_task.items()):
    print(f" {k:<16} {100*c/t:.2f}% ({c}/{t})")
print("By mode:")
for k, (c, t) in sorted(by_mode.items()):
    print(f" {k:<16} {100*c/t:.2f}% ({c}/{t})")

# Patch each analysis script via in-place edit to add GRPO R3 entry
PATCHES = [
    ("build_paper_assets.py",
     '("GRPO Run 2", "results/physbench_grpo-run2-ep1-test/predictions.json"),',
     '("GRPO Run 2", "results/physbench_grpo-run2-ep1-test/predictions.json"),\n ("GRPO R3", "results/physbench_grpo_r3_test/predictions.json"),'),
    ("cot_analysis.py",
     '("GRPO Run 2", "results/physbench_grpo-run2-ep1-test/predictions.json"),',
     '("GRPO Run 2", "results/physbench_grpo-run2-ep1-test/predictions.json"),\n ("GRPO R3", "results/physbench_grpo_r3_test/predictions.json"),'),
    ("traces_and_failures.py",
     '("GRPO Run 2", "results/physbench_grpo-run2-ep1-test/predictions.json"),',
     '("GRPO Run 2", "results/physbench_grpo-run2-ep1-test/predictions.json"),\n ("GRPO R3", "results/physbench_grpo_r3_test/predictions.json"),'),
    ("calibration.py",
     '("GRPO Run 2", "results/physbench_grpo-run2-ep1-test/predictions.json"),',
     '("GRPO Run 2", "results/physbench_grpo-run2-ep1-test/predictions.json"),\n ("GRPO R3", "results/physbench_grpo_r3_test/predictions.json"),'),
]

for script, old, new in PATCHES:
    p = ANALYSIS / script
    txt = p.read_text(encoding="utf-8")
    if "GRPO R3" in txt:
        print(f" [{script}] already patched, skipping")
        continue
    if old not in txt:
        print(f" [WARN] {script}: anchor not found, manual edit needed")
        continue
    p.write_text(txt.replace(old, new), encoding="utf-8")
    print(f" [{script}] patched")

# Patch leaderboard separately (different format - OURS list)
lb = ANALYSIS / "leaderboard_comparison.py"
lb_txt = lb.read_text(encoding="utf-8")
if "GRPO R3" not in lb_txt:
    prop = by_task.get("property", [0, 1])
    rel = by_task.get("relationships", [0, 1])
    scn = by_task.get("scene", [0, 1])
    dyn = by_task.get("dynamics", [0, 1])
    r3_row = (f' ("**PhysSim-VLM (GRPO R3, ours)**", '
              f'{overall:.1f}, {100*prop[0]/prop[1]:.1f}, {100*rel[0]/rel[1]:.1f}, '
              f'{100*scn[0]/scn[1]:.1f}, {100*dyn[0]/dyn[1]:.1f}, 30),')
    anchor = '("PhysSim-VLM (GRPO Run 2)", 44.0, 51.8, 38.5, 41.3, 43.1, 30),'
    if anchor in lb_txt:
        lb.write_text(lb_txt.replace(anchor, anchor + "\n" + r3_row),
                      encoding="utf-8")
        print(f" [leaderboard_comparison.py] GRPO R3 row added")
    else:
        print(f" [WARN] leaderboard anchor not found")

print("\nRunning all analysis scripts...")
for script in ["build_paper_assets.py", "cot_analysis.py",
               "traces_and_failures.py", "calibration.py",
               "leaderboard_comparison.py"]:
    print(f" >>> {script}")
    r = subprocess.run([sys.executable, str(ANALYSIS / script)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f" [ERROR] {script}:\n{r.stderr[:500]}")
    else:
        print(f" [OK] {script}")

print("\nDone. All analysis files now include GRPO R3.")
