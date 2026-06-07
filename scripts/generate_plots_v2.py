#!/usr/bin/env python3
"""Generate all plots for results.md v2 -- includes GRPO Run 1 & 2"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
C_BASE = '#90CAF9'
C_SFT1 = '#1565C0'
C_SFT2 = '#0D47A1'
C_GRPO1 = '#FF7043'
C_GRPO2 = '#BF360C'
C_GPT4O = '#FFB74D'
C_GREEN = '#43A047'
C_GREY = '#9E9E9E'

# ── 1. PhysBench comparison: all checkpoints ───────────────────────────────
categories = ['Overall', 'Dynamics', 'Property', 'Scene', 'Relationships']

# Val set results (199 samples, same val split for comparable checkpoints)
baseline = [54.3, 40.5, 70.3, 48.6, 69.6] # cached baseline
sft_r1_600 = [64.8, 58.2, 73.0, 67.6, 67.4]
sft_r2_flu = [63.8, 64.6, 70.3, 62.2, 60.9]
grpo_run1 = [65.3, 65.8, 75.7, 64.9, 56.5]
grpo_run2 = [65.8, 63.3, 75.7, 70.3, 58.7] # Ep1 - best checkpoint
grpo_run2e2 = [64.3, 62.0, 70.3, 64.9, 63.0] # Ep2 - regression
gpt4o_test = [49.5, 47.0, 56.9, 30.2, 64.8] # test set reference

x = np.arange(len(categories))
width = 0.11

fig, ax = plt.subplots(figsize=(16, 6))
b0 = ax.bar(x - 3*width, baseline, width, label='Baseline (zero-shot)', color=C_BASE, edgecolor='#1565C0', linewidth=0.5)
b1 = ax.bar(x - 2*width, sft_r1_600, width, label='SFT R1 Step 600', color=C_SFT1, edgecolor='#0D47A1', linewidth=0.5)
b2 = ax.bar(x - 1*width, sft_r2_flu, width, label='SFT R2 Fluid Final', color='#42A5F5', edgecolor='#1565C0', linewidth=0.5)
b3 = ax.bar(x + 0*width, grpo_run1, width, label='GRPO Run 1 (497 steps)', color=C_GRPO1, edgecolor='#BF360C', linewidth=0.5)
b4 = ax.bar(x + 1*width, grpo_run2, width, label='GRPO Run 2 Ep1 (best)', color=C_GRPO2, edgecolor='#7F0000', linewidth=0.5)
b5 = ax.bar(x + 2*width, grpo_run2e2, width, label='GRPO Run 2 Ep2 (overfit)', color='#78909C', edgecolor='#37474F', linewidth=0.5)
b6 = ax.bar(x + 3*width, gpt4o_test, width, label='GPT-4o (test, reference)', color=C_GPT4O, edgecolor='#E65100', linewidth=0.5)

for bars in [b0, b1, b2, b3, b4, b5, b6]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f'{h:.1f}', ha='center', va='bottom', fontsize=6.5, rotation=90)

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('PhysBench Val Set: All Checkpoints', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=8.5, loc='upper right')
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig(FIG_DIR / 'physbench_all_checkpoints.png', dpi=150, bbox_inches='tight')
print('Saved physbench_all_checkpoints.png')
plt.close()

# ── 2. GRPO reward trajectory (Run 1 vs Run 2) ──────────────────────────────

def load_grpo_metrics(path):
    steps_data = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                if 'grpo/reward_mean' in d:
                    steps_data.append(d)
            except:
                pass
    by_step = {}
    for s in steps_data:
        by_step[s['step']] = s
    return [by_step[k] for k in sorted(by_step.keys())]

run1_data = load_grpo_metrics(ROOT / 'results/grpo_tinker/grpo-epoch1/metrics.jsonl')
run2_data = load_grpo_metrics(ROOT / 'results/grpo_tinker/grpo-run2-balanced/metrics.jsonl')

r1_steps = [d['step'] for d in run1_data]
r1_avg50 = [d.get('grpo/reward_avg_50', d['grpo/reward_mean']) for d in run1_data]
r1_raw = [d['grpo/reward_mean'] for d in run1_data]

r2_steps = [d['step'] for d in run2_data]
r2_avg50 = [d.get('grpo/reward_avg_50', d['grpo/reward_mean']) for d in run2_data]
r2_raw = [d['grpo/reward_mean'] for d in run2_data]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(r1_steps, r1_raw, alpha=0.15, color=C_GRPO1, linewidth=0.6)
ax.plot(r1_steps, r1_avg50, color=C_GRPO1, linewidth=2.0, label='GRPO Run 1 (avg50) - VCRL, 195 scenes')

ax.plot(r2_steps, r2_raw, alpha=0.15, color=C_GRPO2, linewidth=0.6)
ax.plot(r2_steps, r2_avg50, color=C_GRPO2, linewidth=2.0, label='GRPO Run 2 (avg50) - Uniform, 951 scenes')

# Mark peak of run 2
ax.axvline(x=355, color=C_GRPO2, linestyle='--', alpha=0.5, linewidth=1)
ax.text(360, 0.87, 'Peak\n0.854', fontsize=9, color=C_GRPO2)

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Reward (avg50)', fontsize=12)
ax.set_title('GRPO Reward Trajectory: Run 1 vs Run 2', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0.4, 1.0)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / 'grpo_reward_trajectory.png', dpi=150, bbox_inches='tight')
print('Saved grpo_reward_trajectory.png')
plt.close()

# ── 3. Checkpoint progression (val accuracy over time) ───────────────────────
checkpoints = [
    ('Baseline', 54.3, 40.5, 70.3, 48.6, 69.6),
    ('SFT R1\nStep 300', 63.3, 59.5, 70.3, 64.9, 63.0),
    ('SFT R1\nStep 600', 64.8, 58.2, 73.0, 67.6, 67.4),
    ('SFT R1\nFinal', 64.3, 62.0, 70.3, 64.9, 63.0),
    ('SFT R2\nFluid', 63.8, 64.6, 70.3, 62.2, 60.9),
    ('GRPO\nRun 1', 65.3, 65.8, 75.7, 64.9, 56.5),
    ('GRPO R2\nEp1', 65.8, 63.3, 75.7, 70.3, 58.7),
    ('GRPO R2\nEp2', 64.3, 62.0, 70.3, 64.9, 63.0),
]
labels = [c[0] for c in checkpoints]
overall = [c[1] for c in checkpoints]
dyn = [c[2] for c in checkpoints]
prop = [c[3] for c in checkpoints]
scene = [c[4] for c in checkpoints]
rel = [c[5] for c in checkpoints]

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(x, overall, 'ko-', linewidth=2.5, markersize=8, label='Overall', zorder=5)
ax.plot(x, dyn, 'r^-', linewidth=1.5, markersize=7, label='Dynamics', alpha=0.85)
ax.plot(x, prop, 'b^-', linewidth=1.5, markersize=7, label='Property', alpha=0.85)
ax.plot(x, scene, 'g^-', linewidth=1.5, markersize=7, label='Scene', alpha=0.85)
ax.plot(x, rel, 'm^-', linewidth=1.5, markersize=7, label='Relationships', alpha=0.85)

for i, v in enumerate(overall):
    ax.text(i, v + 1.2, f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Val Set Accuracy Progression Across Checkpoints', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.set_ylim(35, 85)
ax.grid(True, alpha=0.3)
ax.axvspan(4.5, 7.5, alpha=0.07, color='orange', label='GRPO phase')
ax.text(6.0, 82, 'GRPO phase', ha='center', fontsize=9, color='darkorange')
# Mark Ep2 regression
ax.annotate('Ep2\noverfit', xy=(7, 64.3), xytext=(6.6, 67.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=8, color='red', ha='center')
fig.tight_layout()
fig.savefig(FIG_DIR / 'checkpoint_progression.png', dpi=150, bbox_inches='tight')
print('Saved checkpoint_progression.png')
plt.close()

# ── 4. GRPO coverage comparison (Run 1 vs Run 2) ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Run 1: 195/15714 scenes = 1.2%
ax = axes[0]
seen = 195
total = 15714
ax.pie([seen, total - seen], labels=[f'Seen\n{seen} (1.2%)', f'Unseen\n{total-seen} (98.8%)'],
       colors=[C_GRPO1, C_GREY], startangle=90,
       wedgeprops={'edgecolor': 'white', 'linewidth': 2},
       autopct='', textprops={'fontsize': 11})
ax.set_title('GRPO Run 1 Coverage\n(VCRL α=2.0, aggressive)', fontsize=11, fontweight='bold')

# Run 2: 951/951 scenes = 100%
ax = axes[1]
ax.pie([951, 0.0001], labels=['Seen\n951 (100%)', ''],
       colors=[C_GRPO2, C_GREY], startangle=90,
       wedgeprops={'edgecolor': 'white', 'linewidth': 2},
       autopct='', textprops={'fontsize': 11})
ax.set_title('GRPO Run 2 Coverage\n(Uniform, balanced dataset)', fontsize=11, fontweight='bold')

fig.suptitle('Scene Coverage: GRPO Run 1 vs Run 2', fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / 'grpo_coverage_comparison.png', dpi=150, bbox_inches='tight')
print('Saved grpo_coverage_comparison.png')
plt.close()

# ── 5. Per-task result breakdown for GRPO Run 2 (val) ───────────────────────
tasks = ['Dynamics', 'Property', 'Scene', 'Relationships']
base_v = [43.0, 62.2, 43.2, 65.2] # baseline from this eval run
run2_v = [63.3, 75.7, 70.3, 58.7] # GRPO Run 2

x = np.arange(len(tasks))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - width/2, base_v, width, label='Baseline (zero-shot)', color=C_BASE, edgecolor='#1565C0')
b2 = ax.bar(x + width/2, run2_v, width, label='GRPO Run 2 Final', color=C_GRPO2, edgecolor='#7F0000')

for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

deltas = [v2 - v1 for v1, v2 in zip(base_v, run2_v)]
for i, (d, xpos) in enumerate(zip(deltas, x)):
    color = C_GREEN if d > 0 else '#C62828'
    sign = '+' if d >= 0 else ''
    ax.text(xpos, max(base_v[i], run2_v[i]) + 4.5, f'{sign}{d:.1f}pp',
            ha='center', fontsize=10, color=color, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('GRPO Run 2 Final: Val Set by Task (vs Baseline)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 92)
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig(FIG_DIR / 'grpo_run2_per_task.png', dpi=150, bbox_inches='tight')
print('Saved grpo_run2_per_task.png')
plt.close()

# ── 5b. PhysBench TEST SET comparison (updated with real test data) ──────────
# Test set results: Baseline / SFT / GRPO / GPT-4o / NVILA-15B / InternVL2.5-38B
categories = ['Overall', 'Dynamics', 'Property', 'Scene', 'Relationships']
test_base = [40.7, 37.3, 56.6, 26.0, 41.9] # Baseline (Qwen3-VL-30B zero-shot)
test_sft = [47.6, 43.9, 57.9, 46.1, 42.0] # SFT R2-redo
test_grpo = [47.7, 44.3, 56.1, 45.5, 44.7] # GRPO R3
test_gpt4o = [49.5, 47.0, 56.9, 30.2, 64.8]
test_nvila = [46.9, 45.7, 59.2, 38.8, 42.3] # NVILA-15B
test_intern = [51.9, 45.0, 58.8, 39.0, 67.5] # InternVL2.5-38B

x = np.arange(len(categories))
width = 0.13

fig, ax = plt.subplots(figsize=(14, 6))
b0 = ax.bar(x - 2.5*width, test_base, width, label='Baseline (Qwen3-VL-30B zero-shot)', color=C_BASE, edgecolor='#1565C0', linewidth=0.5)
b1 = ax.bar(x - 1.5*width, test_sft, width, label='Ours: SFT R2-redo', color=C_SFT1, edgecolor='#0D47A1', linewidth=0.5)
b2 = ax.bar(x - 0.5*width, test_grpo, width, label='Ours: SFT+GRPO R3', color=C_GRPO2, edgecolor='#7F0000', linewidth=0.5)
b3 = ax.bar(x + 0.5*width, test_gpt4o, width, label='GPT-4o', color=C_GPT4O, edgecolor='#E65100', linewidth=0.5)
b4 = ax.bar(x + 1.5*width, test_nvila, width, label='NVILA-15B', color='#AB47BC', edgecolor='#6A1B9A', linewidth=0.5)
b5 = ax.bar(x + 2.5*width, test_intern, width, label='InternVL2.5-38B', color='#26A69A', edgecolor='#004D40', linewidth=0.5)

for bars in [b0, b1, b2, b3, b4, b5]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.4,
                f'{h:.1f}', ha='center', va='bottom', fontsize=7, rotation=90)

# Highlight Scene column where we are competitive on the listed snapshot
ax.axvspan(3 - 0.5, 3 + 0.5, alpha=0.08, color='green')
ax.text(3, 68, 'best of listed', ha='center', fontsize=10, color='green', fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('PhysBench TEST SET: Our Models vs Baselines (9,786 samples)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=8.5, loc='upper left')
ax.set_ylim(0, 75)
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig(FIG_DIR / 'physbench_comparison.png', dpi=150, bbox_inches='tight')
print('Saved physbench_comparison.png (UPDATED with test data)')
plt.close()

# ── 5c. Dynamics subtype breakdown (TEST SET) ────────────────────────────────
subtypes = ['manipulation\n(818)', 'collision\n(650)', 'fluid\n(642)',
             'throwing\n(410)', 'others\n(345)', 'chemistry\n(67)']
dyn_base_s = [29.1, 38.8, 41.0, 35.4, 48.7, 40.3] # Baseline
dyn_sft_s = [24.0, 43.4, 48.1, 44.4, 78.8, 70.1] # SFT R2-redo
dyn_grpo_s = [24.7, 45.8, 51.7, 37.1, 78.6, 64.2] # GRPO R3

x = np.arange(len(subtypes))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - width, dyn_base_s, width, label='Baseline', color=C_BASE, edgecolor='#1565C0', linewidth=0.5)
b2 = ax.bar(x, dyn_sft_s, width, label='SFT R2-redo', color=C_SFT1, edgecolor='#0D47A1', linewidth=0.5)
b3 = ax.bar(x + width, dyn_grpo_s, width, label='SFT+GRPO R3', color=C_GRPO2, edgecolor='#7F0000', linewidth=0.5)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f'{h:.0f}', ha='center', va='bottom', fontsize=8)

# Annotate SFT delta vs baseline
for i, (base, sft) in enumerate(zip(dyn_base_s, dyn_sft_s)):
    d = sft - base
    color = C_GREEN if d > 0 else '#C62828'
    sign = '+' if d >= 0 else ''
    ax.text(i, max(dyn_sft_s[i], dyn_grpo_s[i]) + 4,
            f'{sign}{d:.1f}pp', ha='center', fontsize=8.5, color=color, fontweight='bold')

# GPT-4o dynamics reference line at 47.0%
ax.axhline(47.0, color=C_GPT4O, linestyle='--', linewidth=1.5, alpha=0.8, label='GPT-4o dynamics (47.0%)')

# Highlight manipulation as problem area
ax.axvspan(-0.5, 0.5, alpha=0.07, color='red')
ax.text(0, 82, ' untrained', ha='center', fontsize=8.5, color='#C62828')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Dynamics Subtypes: Test Set Breakdown (2,932 samples total)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(subtypes, fontsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.set_ylim(0, 90)
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig(FIG_DIR / 'dynamics_improvement.png', dpi=150, bbox_inches='tight')
print('Saved dynamics_improvement.png (UPDATED with subtype breakdown)')
plt.close()

# ── 6. GRPO step time distribution (Run 2) ──────────────────────────────────
step_times = [d.get('grpo/step_time_s', 0) for d in run2_data if d.get('grpo/step_time_s')]
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(step_times, bins=40, color=C_GRPO2, edgecolor='#7F0000', alpha=0.8)
ax.axvline(np.mean(step_times), color='white', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(step_times):.1f}s')
ax.axvline(np.median(step_times), color='#FFB74D', linestyle='--', linewidth=2,
           label=f'Median: {np.median(step_times):.1f}s')
ax.set_xlabel('Step Time (seconds)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('GRPO Run 2: Step Time Distribution (475 steps)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig(FIG_DIR / 'grpo_step_time.png', dpi=150, bbox_inches='tight')
print('Saved grpo_step_time.png')
plt.close()

# ── 7. LR schedule (Run 2) ──────────────────────────────────────────────────
lr_vals = [d.get('grpo/learning_rate', 0) for d in run2_data if 'grpo/learning_rate' in d]
lr_steps = [d['step'] for d in run2_data if 'grpo/learning_rate' in d]
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(lr_steps, [v * 1e6 for v in lr_vals], color=C_SFT1, linewidth=2)
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Learning Rate (×10⁻⁶)', fontsize=12)
ax.set_title('GRPO Run 2: Learning Rate Schedule (Warmup 50 + Cosine Decay)', fontsize=12, fontweight='bold')
ax.axvline(50, color='grey', linestyle='--', alpha=0.7, label='End of warmup (step 50)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / 'grpo_lr_schedule.png', dpi=150, bbox_inches='tight')
print('Saved grpo_lr_schedule.png')
plt.close()

print(f'\nAll figures saved to {FIG_DIR}')
