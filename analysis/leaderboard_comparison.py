"""
Build PhysBench leaderboard comparison + multi-leaderboard view.

Sources:
  - PhysBench mini-leaderboard (data/raw/physbench/README.md, 25 models)
  - PhysSim-VLM from results/physbench_*_test/results.md

Generates:
  analysis/leaderboard_overall.md - Overall ranking
  analysis/leaderboard_per_task.md - 4 sub-leaderboards (Property, Relationships, Scene, Dynamics)
  analysis/leaderboard_efficiency.md - Score vs model size (efficiency frontier)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"

# ── PhysBench mini-leaderboard (parsed from data/raw/physbench/README.md) ─
# Columns: rank, model, ALL, Property, Relationships, Scene, Dynamics, [size in B]
LB = [
    ("InternVL2.5-38B", 51.94, 58.77, 67.51, 39.04, 45.00, 38),
    ("InternVL2.5-78B", 51.16, 60.32, 62.13, 37.32, 46.11, 78),
    ("GPT-4o", 49.49, 56.91, 64.80, 30.15, 46.99, None),
    ("Gemini-1.5-pro", 49.11, 57.26, 63.61, 36.52, 41.56, None),
    ("InternVL2.5-26B", 48.56, 59.08, 58.33, 36.61, 41.79, 26),
    ("NVILA-15B", 46.91, 59.16, 42.34, 38.78, 45.72, 15),
    ("InternVL2-76B", 46.77, 57.65, 52.43, 38.07, 40.12, 76),
    ("Gemini-1.5-flash", 46.07, 57.41, 52.24, 34.32, 40.93, None),
    ("InternVL2-40B", 45.66, 55.79, 50.05, 35.86, 41.33, 40),
    ("NVILA-Lite-15B", 44.93, 55.44, 40.15, 38.11, 44.38, 15),
    ("InternVL2.5-8B", 43.88, 55.87, 48.67, 29.35, 41.20, 8),
    ("NVILA-8B", 43.82, 55.79, 40.29, 33.95, 43.43, 8),
    ("InternVL2-26B", 43.50, 51.92, 45.20, 37.94, 39.34, 26),
    ("GPT-4o-mini", 43.15, 53.54, 44.24, 30.59, 42.90, None),
    ("mPLUG-Owl3-7B", 42.83, 49.25, 45.62, 35.90, 40.61, 7),
    ("NVILA-Lite-8B", 42.55, 53.81, 39.25, 34.62, 41.17, 8),
    ("InternVL2.5-4B", 42.44, 51.03, 44.77, 31.34, 41.79, 4),
    ("GPT-4V", 41.26, 49.59, 45.77, 26.34, 42.15, None),
    ("LLaVA-interleave", 41.00, 47.23, 44.62, 35.64, 37.21, 8),
    ("LLaVA-il-dpo", 40.83, 47.97, 42.67, 33.73, 38.78, 8),
    ("InternVL2-8B", 40.00, 49.05, 43.58, 27.05, 39.47, 8),
    ("Phi-3.5V", 39.75, 45.72, 40.15, 33.02, 39.40, 4),
    ("InternVL2-4B", 39.71, 47.12, 39.96, 30.94, 39.76, 4),
    ("InternVL2.5-2B", 39.22, 49.63, 38.15, 29.44, 38.39, 2),
    ("Phi-3V", 38.42, 43.67, 37.92, 34.93, 36.92, 4),
]

# ── Our results (R2-redo, the current best - GRPO R3 will be added later) ─
OURS = [
    ("Qwen3-VL-30B (Baseline)", 40.6, 56.4, 41.9, 26.0, 37.2, 30),
    ("**PhysSim-VLM (R1, ours)**", 46.9, 56.3, 42.6, 44.6, 43.6, 30),
    ("**PhysSim-VLM (R2-redo, ours)**", 47.2, 57.7, 41.2, 45.9, 43.4, 30),
    ("PhysSim-VLM (GRPO Run 2)", 44.0, 51.8, 38.5, 41.3, 43.1, 30),
]

# ── Combined and ranked ────────────────────────────────────────────────────
combined = LB + OURS
def rank_by(idx, label_col=0):
    sorted_models = sorted(combined, key=lambda x: -x[idx])
    return sorted_models

def fmt_size(s):
    return f"{s}B" if s else " - "

def fmt_row(model_name, all_score, *, ours_marker=False):
    return model_name

# ── 1. Overall leaderboard ────────────────────────────────────────────────
md = ["# PhysBench Test Leaderboard - Overall Ranking\n",
      "PhysSim-VLM (R2-redo, **30B with LoRA, $76 budget**) vs published baselines.\n",
      "Bold = our models. Human performance = 95.87%.\n",
      "| Rank | Model | Size | **Overall** | Property | Relationships | Scene | Dynamics |",
      "|---|---|---|---|---|---|---|---|"]

for rank, row in enumerate(rank_by(1), 1):
    name, all_, prop, rel, scn, dyn, size = row
    md.append(f"| {rank} | {name} | {fmt_size(size)} | **{all_:.2f}** | {prop:.2f} | {rel:.2f} | {scn:.2f} | {dyn:.2f} |")

(ANALYSIS / "leaderboard_overall.md").write_text("\n".join(md), encoding="utf-8")

# ── 2. Per-task sub-leaderboards (FOUR separate physics-domain rankings) ──
md2 = ["# PhysBench Sub-Leaderboards - Per Physics Domain\n",
       "PhysBench tests four independent physics dimensions. We treat each as a separate leaderboard ",
       "to demonstrate that PhysSim-VLM's gains are concentrated in **physics-grounded** dimensions ",
       "(Scene, Dynamics) rather than relational reasoning (Relationships).\n"]

task_cols = [("Property", 2, "Object physical properties (mass, color, attribute, number)"),
             ("Relationships", 3, "Spatial/depth/motion relations between objects"),
             ("Scene", 4, "Scene-level physics: light, viewpoint, temperature, air, fluid"),
             ("Dynamics", 5, "Physics-based motion: collision, throwing, manipulation, fluid")]

for task_name, idx, desc in task_cols:
    md2.append(f"\n## Sub-Leaderboard: **{task_name}**")
    md2.append(f"*{desc}*\n")
    md2.append("| Rank | Model | Size | **{0}** |".format(task_name))
    md2.append("|---|---|---|---|")
    for rank, row in enumerate(rank_by(idx), 1):
        name, _, prop, rel, scn, dyn, size = row
        score = row[idx]
        md2.append(f"| {rank} | {name} | {fmt_size(size)} | **{score:.2f}** |")

(ANALYSIS / "leaderboard_per_task.md").write_text("\n".join(md2), encoding="utf-8")

# ── 3. Efficiency frontier (size vs score) ────────────────────────────────
md3 = ["# Efficiency Frontier - Score per Parameter\n",
       "Models ranked by overall PhysBench Test accuracy with parameter count.\n",
       "Dashes = closed model (size unknown).\n",
       "| Model | Size | Overall | Score-per-B (×100) |",
       "|---|---|---|---|"]
combined_sorted = sorted([r for r in combined if r[6] is not None], key=lambda x: -x[1] / x[6])
for row in combined_sorted:
    name, all_, *_, size = row
    spp = (all_ / size) * 100
    md3.append(f"| {name} | {size}B | {all_:.2f} | {spp:.2f} |")

(ANALYSIS / "leaderboard_efficiency.md").write_text("\n".join(md3), encoding="utf-8")

# ── 4. Headline summary ────────────────────────────────────────────────────
md4 = ["# Leaderboard Headline - PhysSim-VLM vs Published Baselines\n",
       "## Overall Rank\n"]

# Find our R2-redo rank
sorted_overall = rank_by(1)
our_rank = next((i+1 for i, r in enumerate(sorted_overall) if "R2-redo" in r[0]), None)
beats = [r for r in LB if r[1] < 47.2]
md4.append(f"- **PhysSim-VLM (R2-redo)**: 47.20% → **Rank #{our_rank}** of {len(sorted_overall)} models")
md4.append(f"- **Beats**: {', '.join(r[0] for r in beats[:8])}{'...' if len(beats) > 8 else ''}")
md4.append(f"- Notably outperforms: **InternVL2-76B (76B params)** by +0.43pp, **GPT-4V** by +5.94pp, **Gemini-1.5-flash** by +1.13pp")

md4.append("\n## Per-Domain Rankings\n")
for task_name, idx, _ in task_cols:
    sorted_t = rank_by(idx)
    our_rank_t = next((i+1 for i, r in enumerate(sorted_t) if "R2-redo" in r[0]), None)
    our_score = next((r[idx] for r in OURS if "R2-redo" in r[0]), None)
    top = sorted_t[0]
    md4.append(f"- **{task_name}**: Our score {our_score:.2f} → **Rank #{our_rank_t}** "
               f"(top: {top[0]} at {top[idx]:.2f})")

# Scene-specific spotlight
scene_sorted = rank_by(4)
md4.append("\n## Spotlight: Scene Understanding (#1 Rank)\n")
md4.append("PhysSim-VLM (R2-redo) achieves **#1 on Scene** with synthetic data + LoRA on a 30B model:\n")
md4.append("| Rank | Model | Scene Score |")
md4.append("|---|---|---|")
for rank, row in enumerate(scene_sorted[:10], 1):
    md4.append(f"| {rank} | {row[0]} | {row[4]:.2f} |")
md4.append(f"\n**Margin over GPT-4o**: +{45.9 - 30.15:.2f}pp ")
md4.append(f"**Margin over best leaderboard model (InternVL2.5-38B)**: +{45.9 - 39.04:.2f}pp")

(ANALYSIS / "leaderboard_headline.md").write_text("\n".join(md4), encoding="utf-8")

print("Wrote:")
for f in ["leaderboard_overall.md", "leaderboard_per_task.md",
          "leaderboard_efficiency.md", "leaderboard_headline.md"]:
    print(f" {ANALYSIS / f}")
