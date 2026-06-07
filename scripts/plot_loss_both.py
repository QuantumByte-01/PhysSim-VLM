"""Combined R1 + R2-redo training-loss plot."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
R1 = ROOT / "results/sft_tinker/sft-epoch1-v5/metrics.jsonl"
R2 = ROOT / "results/sft_tinker/sft-r2-redo/metrics.jsonl"
OUT = ROOT / "results/figures/training_loss_both.png"

def load(p):
    steps, losses = [], []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "loss" in r and "step" in r:
            steps.append(r["step"]); losses.append(r["loss"])
        elif "train/loss" in r:
            steps.append(r.get("step", len(steps))); losses.append(r["train/loss"])
    return steps, losses

def smooth(y, k=15):
    if len(y) < k: return y
    out = []
    for i in range(len(y)):
        a = max(0, i - k // 2); b = min(len(y), i + k // 2 + 1)
        out.append(sum(y[a:b]) / (b - a))
    return out

s1, l1 = load(R1)
s2, l2 = load(R2)
print(f"R1: {len(s1)} steps, start={l1[0]:.3f} end={l1[-1]:.3f}")
print(f"R2-redo: {len(s2)} steps, start={l2[0]:.3f} end={l2[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)

axes[0].plot(s1, l1, color="#ffb27a", alpha=0.35, lw=0.8)
axes[0].plot(s1, smooth(l1), color="#e8590c", lw=1.8, label="SFT R1 (smoothed)")
axes[0].set_title("SFT R1 (12,023 MuJoCo scenes)")
axes[0].set_xlabel("Step"); axes[0].set_ylabel("Train loss")
axes[0].grid(alpha=0.3); axes[0].legend(loc="upper right", fontsize=9)

axes[1].plot(s2, l2, color="#74b9ff", alpha=0.35, lw=0.8)
axes[1].plot(s2, smooth(l2), color="#0a6cbe", lw=1.8, label="SFT R2-redo (smoothed)")
axes[1].set_title("SFT R2-redo (resumed from R1; 2,574 scenes)")
axes[1].set_xlabel("Step"); axes[1].set_ylabel("Train loss")
axes[1].grid(alpha=0.3); axes[1].legend(loc="upper right", fontsize=9)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"Saved {OUT}")
