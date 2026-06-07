"""Generate publication-quality plots from SFT training metrics."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
METRICS_PATH = Path(r"C:\Users\Swastik R\Documents\Personal_Projects\VLM and Physics\results\sft_tinker\sft-epoch1-v3\metrics.jsonl")
FIGURES_DIR = Path(r"C:\Users\Swastik R\Documents\Personal_Projects\VLM and Physics\results\figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150

# ── Load metrics ───────────────────────────────────────────────────────
steps, losses, loss_avgs, step_times = [], [], [], []
val_steps, val_losses = [], []
ckpt_steps = []

with open(METRICS_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)

        if "train/loss" in rec:
            steps.append(rec["step"])
            losses.append(rec["train/loss"])
            loss_avgs.append(rec["train/loss_avg"])
            step_times.append(rec["train/step_time_s"])

        if "val/loss" in rec:
            val_steps.append(rec["step"])
            val_losses.append(rec["val/loss"])

        if "checkpoint_step" in rec:
            ckpt_steps.append(rec["checkpoint_step"])

steps = np.array(steps)
losses = np.array(losses)
loss_avgs = np.array(loss_avgs)
step_times = np.array(step_times)

# ── Style defaults ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": DPI,
})

# ======================================================================
# 1. Training Loss Curve
# ======================================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(steps, losses, color="#93c5fd", alpha=0.45, linewidth=0.7,
        label="Per-step loss")
ax.plot(steps, loss_avgs, color="#2563eb", linewidth=2.0,
        label="Running average")

# Mark validation-loss points
if val_steps:
    ax.scatter(val_steps, val_losses, color="#dc2626", zorder=5, s=50,
               marker="D", label="Val loss")
    for vs, vl in zip(val_steps, val_losses):
        ax.annotate(f"{vl:.4f}", (vs, vl), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, color="#dc2626")

# Mark checkpoints with vertical lines
for cs in ckpt_steps:
    ax.axvline(cs, color="#9ca3af", linestyle="--", linewidth=0.6, alpha=0.5)

ax.set_title("SFT Training Loss (Epoch 1)")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.legend(loc="upper right")
ax.set_xlim(0, steps[-1] + 5)
ax.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "training_loss.png", dpi=DPI)
plt.close(fig)
print(f"Saved training_loss.png ({len(steps)} data points)")

# ======================================================================
# 2. Step Time
# ======================================================================
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(steps, step_times, color="#6366f1", linewidth=0.8, alpha=0.7)

# Add a smoothed line (rolling median, window=21)
window = 21
if len(step_times) >= window:
    pad = window // 2
    smoothed = np.array([np.median(step_times[max(0, i-pad):i+pad+1])
                         for i in range(len(step_times))])
    ax.plot(steps, smoothed, color="#312e81", linewidth=2.0,
            label=f"Median (w={window})")
    ax.legend(loc="upper right")

ax.set_title("Training Step Time")
ax.set_xlabel("Step")
ax.set_ylabel("Seconds per step")
ax.set_xlim(0, steps[-1] + 5)
ax.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "step_time.png", dpi=DPI)
plt.close(fig)
print(f"Saved step_time.png")

# ======================================================================
# 3. Accuracy Summary Bar Chart
# ======================================================================
fig, ax = plt.subplots(figsize=(7, 5))

labels = ["Baseline\n(Qwen3-VL)", "SFT Step 300"]
values = [54.3, 63.3]
colors = ["#94a3b8", "#3b82f6"]

bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white",
              linewidth=1.2)

# Target line
target = 70.0
ax.axhline(target, color="#16a34a", linestyle="--", linewidth=1.5, zorder=3)
ax.text(len(labels) - 0.5, target + 1.0, f"Target: {target}%",
        color="#16a34a", fontsize=11, fontweight="bold",
        ha="right", va="bottom")

# Annotate bars
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
            f"{val}%", ha="center", va="bottom", fontsize=13,
            fontweight="bold")

# Improvement arrow
mid_x = (bars[0].get_x() + bars[0].get_width() / 2 +
         bars[1].get_x() + bars[1].get_width() / 2) / 2
ax.annotate("", xy=(mid_x, 63.3), xytext=(mid_x, 54.3),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=2))
ax.text(mid_x + 0.08, 58.8, "+9.0 pp", fontsize=11, color="#059669",
        fontweight="bold", ha="left")

ax.set_title("PhysBench Accuracy: Baseline vs SFT")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 80)
ax.set_yticks(range(0, 81, 10))

fig.tight_layout()
fig.savefig(FIGURES_DIR / "loss_summary.png", dpi=DPI)
plt.close(fig)
print(f"Saved loss_summary.png")

print("\nAll figures saved to:", FIGURES_DIR)
