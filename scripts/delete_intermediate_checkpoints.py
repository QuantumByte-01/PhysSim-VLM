#!/usr/bin/env python3
"""
Delete intermediate Tinker checkpoints to reduce storage costs.

Policy:
  KEEP: all final/early_stop checkpoints
  KEEP: last periodic checkpoint of every run (most recent intermediate)
  KEEP: all grpo-run2-ep2 (recently finished, highest reward)
  KEEP: grpo-run1/step_200 (peak reward 0.8594)
  KEEP: grpo-run2-ep1/step_400 (best intermediate, reward 0.7937) -- also last periodic
  KEEP: sft-r2-resume/step_600 (corresponds to known good SFT val checkpoint)
  DELETE: all other periodic intermediate checkpoints

Usage:
    python scripts/delete_intermediate_checkpoints.py # dry run (default)
    python scripts/delete_intermediate_checkpoints.py --confirm # actually delete
"""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / '.env')

import tinker

# ── Runs to process ────────────────────────────────────────────────────────────
RUNS = [
    # SFT R1
    ("sft_tinker", "sft-epoch1-v3", "sft-r1-v3"), # bb32ae94, steps 100-600, no final
    ("sft_tinker", "sft-epoch1-v5", "sft-r1-v5"), # 7942b72f, steps 100-751 + final
    # SFT R2
    ("sft_tinker", "sft-r2-resume", "sft-r2-resume"), # 63d1e1f7+ac5d030f, steps 50-787
    ("sft_tinker", "sft-r2-fluid", "sft-r2-fluid"), # 515a8ba5, steps 50-269
    ("sft_tinker", "sft-r2-redo", "sft-r2-redo"), # steps 50-600 + final 644
    ("sft_tinker", "sft-r3-replay", "sft-r3-replay"), # 0855f97e
    # GRPO
    ("grpo_tinker", "grpo-epoch1", "grpo-run1"),
    ("grpo_tinker", "grpo-run2-balanced", "grpo-run2-ep1"),
    ("grpo_tinker", "grpo-run2-ep2", "grpo-run2-ep2"),
]

# Intermediate checkpoints to KEEP (in addition to final/early_stop + last periodic)
# Format: {(run_dir_name, step): reason}
KEEP_INTERMEDIATES = {
    ("sft-epoch1-v3", 600): "best val score 64.8% (SFT R1 step 600) -- also last periodic",
    ("sft-r2-resume", 600): "corresponds to known good SFT val checkpoint",
    ("grpo-epoch1", 200): "peak reward 0.8594",
    # Keep ALL grpo-run2-ep2 steps (handled separately below)
}

# Runs where ALL checkpoints (including intermediates) are kept
KEEP_ALL_RUNS = {"grpo-run2-ep2"}


def load_checkpoints(results_subdir: str, run_name: str):
    ckpt_file = ROOT / "results" / results_subdir / run_name / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return []
    entries = []
    seen = set()
    for line in ckpt_file.read_text().strip().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        step = d.get("step")
        if step not in seen:
            seen.add(step)
            entries.append(d)
    return entries


def get_paths(entry: dict) -> list[str]:
    """Return all tinker:// paths for this checkpoint entry."""
    paths = []
    for key in ("sampler_path", "tinker_path", "state_path"):
        v = entry.get(key)
        if v and v.startswith("tinker://"):
            paths.append(v)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true",
                        help="Actually delete (default is dry-run)")
    args = parser.parse_args()

    if not args.confirm:
        print("DRY RUN -- pass --confirm to actually delete")
        print()

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()

    to_delete = [] # (label, step, tag, path, reason)
    to_keep = [] # (label, step, tag, reason)

    for results_subdir, run_name, label in RUNS:
        entries = load_checkpoints(results_subdir, run_name)
        if not entries:
            print(f"[SKIP] {label}: no checkpoints.jsonl found")
            continue

        # Find last periodic step (highest step that is NOT final/early_stop)
        periodic_steps = [e["step"] for e in entries if e.get("tag", "") not in ("final", "early_stop")]
        last_periodic = max(periodic_steps) if periodic_steps else None

        for entry in entries:
            step = entry["step"]
            tag = entry.get("tag", "")
            paths = get_paths(entry)

            # Always keep final and early_stop
            if tag in ("final", "early_stop"):
                to_keep.append((label, step, tag, f"tagged {tag}"))
                continue

            # Keep all checkpoints for protected runs
            if run_name in KEEP_ALL_RUNS:
                to_keep.append((label, step, tag, "recently finished run (keep all)"))
                continue

            # Keep last periodic step of every run
            if step == last_periodic:
                to_keep.append((label, step, tag, f"last periodic checkpoint"))
                continue

            # Keep specifically nominated intermediates
            keep_key = (run_name, step)
            if keep_key in KEEP_INTERMEDIATES:
                to_keep.append((label, step, tag, KEEP_INTERMEDIATES[keep_key]))
                continue

            # Everything else: delete
            for path in paths:
                ckpt_type = "sampler_weights" if "sampler_weights" in path else "weights"
                to_delete.append((label, step, tag, path, ckpt_type))

    # ── Report plan ────────────────────────────────────────────────────────────
    print(f"{'='*65}")
    print(f"KEEP ({len(to_keep)} checkpoints):")
    for label, step, tag, reason in sorted(to_keep, key=lambda x: (x[0], x[1])):
        print(f" {label:22s} step {step:4d} [{tag:12s}] ({reason})")

    size_gb = {"sampler_weights": 3.93, "weights": 11.78}
    total_gb = sum(size_gb.get(t[4], 0) for t in to_delete)
    print()
    print(f"DELETE ({len(to_delete)} checkpoint archives, ~{total_gb:.0f} GB):")
    for label, step, tag, path, ckpt_type in sorted(to_delete, key=lambda x: (x[0], x[1])):
        gb = size_gb.get(ckpt_type, 0)
        print(f" {label:22s} step {step:4d} {ckpt_type:16s} ~{gb:.1f} GB")
    print(f"{'='*65}")
    print()

    if not args.confirm:
        print("Re-run with --confirm to execute deletion.")
        return

    # ── Actually delete ────────────────────────────────────────────────────────
    done, failed = 0, 0
    for label, step, tag, path, ckpt_type in to_delete:
        print(f" Deleting {label}/step_{step}/{ckpt_type} ...", end=" ", flush=True)
        try:
            rc.delete_checkpoint_from_tinker_path(path).result()
            print("OK")
            done += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print()
    print(f"Deleted: {done}/{len(to_delete)} Failed: {failed} Freed: ~{done * 7.85:.0f} GB")


if __name__ == "__main__":
    main()
