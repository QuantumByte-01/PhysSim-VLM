#!/usr/bin/env python3
"""
Plot GRPO training metrics from a run's metrics.jsonl.

Usage:
  python scripts/plot_grpo_metrics.py # latest run
  python scripts/plot_grpo_metrics.py --run grpo-epoch1 # specific run
  python scripts/plot_grpo_metrics.py --run grpo-epoch1 --show # also display
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
GRPO_DIR = ROOT / "results" / "grpo_tinker"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No metrics.jsonl in {run_dir}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def smooth(vals: list[float], w: int = 20) -> np.ndarray:
    if len(vals) < w:
        return np.array(vals)
    return np.convolve(vals, np.ones(w) / w, mode="valid")


def get(metrics: list[dict], key: str) -> tuple[list, list]:
    steps, vals = [], []
    for m in metrics:
        if key in m:
            steps.append(m["step"])
            vals.append(m[key])
    return steps, vals


def plot_grpo(run_name: str, show: bool = False):
    run_dir = GRPO_DIR / run_name
    metrics = load_metrics(run_dir)
    prefix = run_name.replace("/", "_")

    print(f" Loaded {len(metrics)} steps from {run_dir}")

    # ── 1. Reward curve ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"GRPO Training - {run_name}", fontsize=14, fontweight="bold")

    # Panel 1: Reward mean + avg50
    ax = axes[0, 0]
    steps_r, r_mean = get(metrics, "grpo/reward_mean")
    _, r_avg50 = get(metrics, "grpo/reward_avg_50")
    if steps_r:
        ax.plot(steps_r, r_mean, alpha=0.25, color="#2196F3", linewidth=0.7,
                label="Step reward")
        ax.plot(steps_r, r_avg50, color="#1565C0", linewidth=2.0,
                label="50-step avg")
        # Physics vs format breakdown
        _, phys_r = get(metrics, "grpo/physics_reward_mean")
        _, fmt_r = get(metrics, "grpo/format_reward_mean")
        if phys_r:
            ax.plot(steps_r[:len(phys_r)], phys_r, color="#E91E63", linewidth=1.2,
                    alpha=0.7, linestyle="--", label="Physics reward")
        if fmt_r:
            ax.plot(steps_r[:len(fmt_r)], fmt_r, color="#4CAF50", linewidth=1.2,
                    alpha=0.7, linestyle="--", label="Format reward")
    ax.set_xlabel("Step"); ax.set_ylabel("Reward")
    ax.set_title("Reward Curve"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 2: Advantage std + skipped groups
    ax = axes[0, 1]
    steps_a, adv_std = get(metrics, "grpo/advantage_std")
    _, skipped = get(metrics, "grpo/skipped_groups")
    _, n_groups = get(metrics, "grpo/n_groups")
    if steps_a:
        ax2 = ax.twinx()
        ax.plot(steps_a, adv_std, color="#FF9800", linewidth=1.5,
                label="Advantage std")
        if skipped and n_groups:
            skip_ratio = [s / max(g, 1) for s, g in zip(skipped, n_groups)]
            ax2.plot(steps_a[:len(skip_ratio)], skip_ratio, color="#9C27B0",
                     linewidth=1.0, alpha=0.6, linestyle=":", label="Skipped ratio")
            ax2.set_ylabel("Skipped group ratio", color="#9C27B0")
            ax2.tick_params(axis="y", labelcolor="#9C27B0")
            ax2.set_ylim(0, 1)
    ax.set_xlabel("Step"); ax.set_ylabel("Advantage std", color="#FF9800")
    ax.tick_params(axis="y", labelcolor="#FF9800")
    ax.set_title("Advantage Std + Skipped Groups"); ax.grid(True, alpha=0.3)

    # Panel 3: Tokens per step + cumulative
    ax = axes[1, 0]
    steps_t, tok_step = get(metrics, "grpo/tokens_step")
    _, tok_total = get(metrics, "grpo/tokens_total")
    if steps_t and tok_step:
        ax.bar(steps_t, tok_step, color="#00BCD4", alpha=0.5, width=0.8,
               label="Tokens/step")
        w = min(20, len(tok_step))
        roll = smooth(tok_step, w)
        ax.plot(steps_t[w-1:], roll, color="#006064", linewidth=2.0,
                label=f"{w}-step avg")
        if tok_total:
            ax2 = ax.twinx()
            ax2.plot(steps_t[:len(tok_total)],
                     [t / 1e6 for t in tok_total],
                     color="#E91E63", linewidth=1.5, label="Cumulative (M)")
            ax2.set_ylabel("Cumulative tokens (M)", color="#E91E63")
            ax2.tick_params(axis="y", labelcolor="#E91E63")
    ax.set_xlabel("Step"); ax.set_ylabel("Tokens generated")
    ax.set_title("Token Usage per Step"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 4: VCRL coverage + SSR buffer size
    ax = axes[1, 1]
    steps_v, vcrl_frac = get(metrics, "grpo/vcrl_seen_frac")
    _, ssr_size = get(metrics, "grpo/ssr_buffer_size")
    plotted = False
    if steps_v and vcrl_frac:
        ax.plot(steps_v, [f * 100 for f in vcrl_frac], color="#4CAF50",
                linewidth=2.0, label="VCRL scene coverage (%)")
        plotted = True
    if ssr_size:
        ax2 = ax.twinx()
        ax2.plot(steps_v[:len(ssr_size)], ssr_size, color="#FF5722",
                 linewidth=1.5, alpha=0.8, label="SSR buffer size")
        ax2.set_ylabel("SSR buffer size", color="#FF5722")
        ax2.tick_params(axis="y", labelcolor="#FF5722")
    if plotted:
        ax.set_ylim(0, 105)
    ax.set_xlabel("Step"); ax.set_ylabel("VCRL coverage (%)", color="#4CAF50")
    ax.tick_params(axis="y", labelcolor="#4CAF50")
    ax.set_title("VCRL Coverage + SSR Buffer"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / f"grpo_{prefix}_training.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f" Saved {out}")
    if show:
        plt.show()
    plt.close()

    # ── 2. Step time ────────────────────────────────────────────────────────
    steps_s, step_times = get(metrics, "grpo/step_time_s")
    if steps_s:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(steps_s, step_times, color="#FF9800", alpha=0.4, linewidth=0.8)
        w = min(20, len(step_times))
        roll = smooth(step_times, w)
        ax.plot(steps_s[w-1:], roll, color="#E65100", linewidth=2.0,
                label=f"{w}-step avg")
        ax.axhline(np.median(step_times), color="#9C27B0", linestyle="--",
                   linewidth=1.5, label=f"Median {np.median(step_times):.1f}s")
        ax.set_xlabel("Step"); ax.set_ylabel("Seconds")
        ax.set_title(f"GRPO Step Time - {run_name}")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out2 = FIG_DIR / f"grpo_{prefix}_step_time.png"
        fig.savefig(out2, dpi=150, bbox_inches="tight")
        print(f" Saved {out2}")
        plt.close()

    # ── Summary stats ───────────────────────────────────────────────────────
    if r_mean:
        print(f"\n Summary for {run_name}:")
        print(f" Steps : {len(r_mean)}")
        print(f" Reward mean : {np.mean(r_mean):.4f}")
        print(f" Reward last 50 : {np.mean(r_mean[-50:]):.4f}")
        print(f" Reward max : {max(r_mean):.4f}")
    tok_all = [m.get("grpo/tokens_total", 0) for m in metrics if "grpo/tokens_total" in m]
    if tok_all:
        print(f" Total tokens : {tok_all[-1]:,}")
        print(f" Avg tokens/step: {tok_all[-1] / max(len(r_mean), 1):.0f}")


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO training metrics")
    parser.add_argument("--run", type=str, default=None,
                        help="Run name under results/grpo_tinker/ (default: latest)")
    parser.add_argument("--show", action="store_true", help="Display plots")
    args = parser.parse_args()

    if args.run:
        run_name = args.run
    else:
        # Use latest run by mtime
        runs = sorted(GRPO_DIR.glob("*/metrics.jsonl"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            print(f"No GRPO runs found in {GRPO_DIR}")
            return
        run_name = runs[0].parent.name
        print(f" Using latest run: {run_name}")

    plot_grpo(run_name, show=args.show)
    print(f"\n Done. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
