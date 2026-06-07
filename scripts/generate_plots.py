#!/usr/bin/env python3
"""Generate training and eval plots for results.md"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Load metrics
lines = open(ROOT / "results/sft_tinker/sft-epoch1-v3/metrics.jsonl").readlines()
metrics = [json.loads(l) for l in lines if l.strip()]

steps = [m['step'] for m in metrics if 'train/loss' in m]
losses = [m['train/loss'] for m in metrics if 'train/loss' in m]
avg_losses = [m['train/loss_avg'] for m in metrics if 'train/loss_avg' in m]

# ── 1. Training loss curve ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps, losses, alpha=0.3, color='#2196F3', linewidth=0.8, label='Per-step loss')
ax.plot(steps, avg_losses, color='#1565C0', linewidth=2.0, label='Running average')
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Loss (Weighted NLL)', fontsize=12)
ax.set_title('SFT Training Loss (Epoch 1) -- Qwen3-VL-30B + LoRA r=64', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, max(steps))
ax.set_ylim(0, max(losses[:20]))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / 'training_loss.png', dpi=150, bbox_inches='tight')
print('Saved training_loss.png')
plt.close()

# ── 2. Step time ────────────────────────────────────────────────────────────
step_times = [m['train/step_time_s'] for m in metrics if 'train/step_time_s' in m]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(steps, step_times, color='#FF9800', alpha=0.6, linewidth=0.8)
window = 20
if len(step_times) > window:
    rolling = np.convolve(step_times, np.ones(window)/window, mode='valid')
    ax.plot(steps[window-1:], rolling, color='#E65100', linewidth=2.0, label=f'{window}-step avg')
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Seconds per Step', fontsize=12)
ax.set_title('Training Step Time', fontsize=13)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / 'step_time.png', dpi=150, bbox_inches='tight')
print('Saved step_time.png')
plt.close()

# ── 3. PhysBench comparison bar chart ───────────────────────────────────────
categories = ['Overall', 'Dynamics', 'Property', 'Scene', 'Relationships']
baseline_vals = [54.3, 40.5, 70.3, 48.6, 69.6]
sft_300_vals = [63.3, 59.5, 70.3, 64.9, 63.0]
gpt4o_vals = [49.5, 47.0, None, None, None]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width, baseline_vals, width, label='Baseline (Zero-shot)',
               color='#90CAF9', edgecolor='#1565C0')
bars2 = ax.bar(x, sft_300_vals, width, label='SFT Step 300',
               color='#1565C0', edgecolor='#0D47A1')
gpt_x = [i for i, v in enumerate(gpt4o_vals) if v is not None]
gpt_v = [v for v in gpt4o_vals if v is not None]
bars3 = ax.bar([xi + width for xi in gpt_x], gpt_v, width, label='GPT-4o',
               color='#FFB74D', edgecolor='#E65100')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('PhysBench Results: Baseline vs SFT vs GPT-4o', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 85)
ax.grid(True, alpha=0.2, axis='y')

for bar_group in [bars1, bars2, bars3]:
    for bar in bar_group:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

fig.tight_layout()
fig.savefig(FIG_DIR / 'physbench_comparison.png', dpi=150, bbox_inches='tight')
print('Saved physbench_comparison.png')
plt.close()

# ── 4. Dynamics improvement breakdown ───────────────────────────────────────
modes = ['Image-Only', 'Image+Video', 'General']
dyn_base = [20.0, 47.2, 31.2]
dyn_sft = [30.0, 71.7, 37.5]

x = np.arange(len(modes))
width = 0.3

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, dyn_base, width, label='Baseline',
               color='#EF9A9A', edgecolor='#C62828')
bars2 = ax.bar(x + width/2, dyn_sft, width, label='SFT Step 300',
               color='#C62828', edgecolor='#B71C1C')
ax.axhline(y=47.0, color='#FF9800', linestyle='--', linewidth=2, alpha=0.7,
           label='GPT-4o Overall Dynamics (47%)')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Dynamics Task Improvement by Input Mode', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(modes, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 85)
ax.grid(True, alpha=0.2, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

fig.tight_layout()
fig.savefig(FIG_DIR / 'dynamics_improvement.png', dpi=150, bbox_inches='tight')
print('Saved dynamics_improvement.png')
plt.close()

print(f'\nAll figures saved to {FIG_DIR}')
