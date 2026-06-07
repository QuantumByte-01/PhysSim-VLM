#!/usr/bin/env python3
"""
Prepare combined SFT dataset (R1 + R2) for training from baseline.
=====================================================================
Merges:
  1. HuggingFace R1 dataset (Swastikr/PhysSim-VLM-Dataset)
     - Tasks: ttc, trajectory, stability (~12,023 scenes)
  2. Local R2 data (data/sft_r2/)
     - Tasks: fluid_*, counting, manipulation, motion_comparison,
              object_comparison, viewpoint (~4,594 scenes)

Applies per-task caps (default 2000) and additive MCQ wrapping on
categorical R2 tasks (default 0.5 fraction).

Writes to data/combined_sft_manifest/:
  - manifest.json (counts, config, mcq report)
  - scene_index.json (every scene's id/task/source/mcq flag)
  - samples.md (human-readable prompt/answer samples)

Usage:
  python scripts/prepare_combined_sft.py
  python scripts/prepare_combined_sft.py --max-per-task 2000 --mcq-frac 0.5
  python scripts/prepare_combined_sft.py --skip-hf # dry-run without HF download
"""

import os, sys, json, random, argparse
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Reuse MCQ logic + R2 loader from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_sft_r2_tinker import (
    MCQ_TASKS,
    mcq_wrap_scene,
    load_sft_r2_scenes,
    _extract_answer,
)

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "combined_sft_manifest"
HF_DATASET = "Swastikr/PhysSim-VLM-Dataset"
R1_TASKS = {"ttc", "trajectory", "stability"}


# ── Source loaders ────────────────────────────────────────────────────────────

def load_hf_r1_metadata(hf_token: str) -> list[dict]:
    """Load R1 scene metadata from HF cache (no frame bytes)."""
    from datasets import load_dataset

    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    print(f" Loading {HF_DATASET} (train split)...")
    ds = load_dataset(HF_DATASET, split="train", token=hf_token, streaming=False)

    keep_cols = {"scene_id", "task", "prompt", "assistant_text"}
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    print(f" Loaded {len(ds)} HF rows (metadata only)")

    scenes = []
    for row in ds:
        scenes.append({
            "scene_id": row["scene_id"],
            "task": row["task"],
            "prompt_text": row["prompt"],
            "assistant_text": row["assistant_text"],
            "frames": None, # lazy-loaded from HF at training time
            "source": "hf_r1",
        })
    return scenes


def load_r2_local_metadata() -> list[dict]:
    """Load R2 local scenes via existing loader, stringify frame paths, filter R1 tasks."""
    scenes = load_sft_r2_scenes()
    out = []
    for s in scenes:
        if s["task"] in R1_TASKS:
            continue # R1 overlap - use HF authoritative pool instead
        s["source"] = "local_r2"
        s["frames"] = [str(p) for p in s["frames"]]
        out.append(s)
    return out


# ── Transforms ────────────────────────────────────────────────────────────────

def apply_caps(scenes: list[dict], max_per_task: int,
               task_overrides: dict[str, int] | None = None
               ) -> tuple[list[dict], dict]:
    by_task = defaultdict(list)
    for s in scenes:
        by_task[s["task"]].append(s)
    out = []
    report = {}
    for task, slist in sorted(by_task.items()):
        limit = (task_overrides or {}).get(task, max_per_task)
        cap = min(limit, len(slist))
        out.extend(slist[:cap])
        report[task] = {"original": len(slist), "capped": cap}
    return out, report


def apply_mcq(scenes: list[dict], mcq_frac: float, seed: int
              ) -> tuple[list[dict], dict]:
    """Additive MCQ wrapping. Applied to any scene whose task is in MCQ_TASKS,
    regardless of source - R1 stability uses the same answer format and wraps
    cleanly."""
    rng = random.Random(seed)
    extras = []
    report: dict = defaultdict(
        lambda: {"attempted": 0, "added": 0, "failed": 0})
    for s in scenes:
        if s["task"] not in MCQ_TASKS:
            continue
        if rng.random() >= mcq_frac:
            continue
        report[s["task"]]["attempted"] += 1
        wrapped = mcq_wrap_scene(s, rng)
        if wrapped is not None:
            extras.append(wrapped)
            report[s["task"]]["added"] += 1
        else:
            report[s["task"]]["failed"] += 1
    return extras, dict(report)


# ── Verification helpers ──────────────────────────────────────────────────────

def compute_letter_distribution(scenes: list[dict]) -> dict:
    dist = defaultdict(int)
    for s in scenes:
        if not s.get("_mcq_wrapped"):
            continue
        ans = _extract_answer(s["assistant_text"])
        if ans and len(ans) == 1 and ans in "ABCD":
            dist[ans] += 1
    return dict(dist)


def write_samples(all_scenes: list[dict], out_path: Path,
                  n_per_task: int = 3) -> None:
    by_task = defaultdict(list)
    for s in all_scenes:
        by_task[s["task"]].append(s)
    lines = ["# Combined SFT Dataset - Sample Dump", ""]
    rng = random.Random(123)
    for task in sorted(by_task):
        pool = by_task[task]
        originals = [s for s in pool if not s.get("_mcq_wrapped")]
        mcq_variants = [s for s in pool if s.get("_mcq_wrapped")]
        lines.append(f"\n## Task: `{task}`")
        lines.append(f"- total: **{len(pool)}** "
                     f"(originals: {len(originals)}, MCQ: {len(mcq_variants)})")

        picks = rng.sample(originals, min(n_per_task, len(originals)))
        for i, s in enumerate(picks, 1):
            lines.append(f"\n### {task} - original {i}")
            lines.append(f"- source: `{s['source']}`")
            lines.append(f"- scene_id: `{s['scene_id']}`")
            lines.append(f"- frames: {len(s['frames']) if s.get('frames') else 'HF (lazy)'}")
            lines.append("\n**Prompt:**\n")
            lines.append("```")
            lines.append(s["prompt_text"])
            lines.append("```")
            lines.append("\n**Assistant:**\n")
            lines.append("```")
            lines.append(s["assistant_text"])
            lines.append("```")

        if mcq_variants:
            picks = rng.sample(mcq_variants, min(n_per_task, len(mcq_variants)))
            for i, s in enumerate(picks, 1):
                lines.append(f"\n### {task} - MCQ-wrapped {i}")
                lines.append(f"- source: `{s['source']}`")
                lines.append(f"- scene_id: `{s['scene_id']}`")
                lines.append("\n**Prompt:**\n")
                lines.append("```")
                lines.append(s["prompt_text"])
                lines.append("```")
                lines.append("\n**Assistant:**\n")
                lines.append("```")
                lines.append(s["assistant_text"])
                lines.append("```")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-task", type=int, default=2000)
    parser.add_argument("--task-cap", type=str, default=None,
                        help="Per-task cap overrides, e.g. 'motion_comparison=1000,counting=100'")
    parser.add_argument("--mcq-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HF load (for quick dry-run of R2 only)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Preparing combined SFT dataset (R1 + R2)")
    print("=" * 60)
    print(f" max_per_task = {args.max_per_task}")
    print(f" mcq_frac = {args.mcq_frac}")
    print(f" seed = {args.seed}")
    print(f" out_dir = {OUT_DIR}")
    print()

    # Step 1: Load sources
    all_scenes: list[dict] = []
    if not args.skip_hf:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN not set in environment")
        all_scenes.extend(load_hf_r1_metadata(hf_token))
    print("\n Loading local R2 data...")
    r2 = load_r2_local_metadata()
    all_scenes.extend(r2)
    print(f" Total raw scenes: {len(all_scenes)}")

    # Step 2: Dedup by (task, scene_id)
    seen, deduped, n_dupes = set(), [], 0
    for s in all_scenes:
        key = (s["task"], s["scene_id"])
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)
        deduped.append(s)
    print(f" After dedup: {len(deduped)} ({n_dupes} duplicate (task,id) pairs removed)")

    # Step 3: Pre-cap breakdown
    pre_cap: dict = defaultdict(lambda: defaultdict(int))
    for s in deduped:
        pre_cap[s["task"]][s["source"]] += 1
    print("\n Pre-cap task breakdown:")
    for task in sorted(pre_cap):
        breakdown = dict(pre_cap[task])
        total = sum(breakdown.values())
        print(f" {task:20s} {total:5d} {breakdown}")

    # Step 4: Apply per-task cap
    task_overrides = {}
    if args.task_cap:
        for pair in args.task_cap.split(","):
            k, v = pair.split("=")
            task_overrides[k.strip()] = int(v.strip())
    capped, cap_report = apply_caps(deduped, args.max_per_task, task_overrides)
    print(f"\n After cap={args.max_per_task}/task: {len(capped)}")
    for task in sorted(cap_report):
        r = cap_report[task]
        print(f" {task:20s} {r['capped']:5d} / {r['original']:5d}")

    # Step 5: MCQ wrapping (on local_r2 categorical only)
    mcq_extras, mcq_report = apply_mcq(capped, args.mcq_frac, args.seed)
    final = capped + mcq_extras
    print(f"\n MCQ wrapping (frac={args.mcq_frac}, all MCQ_TASKS): "
          f"+{len(mcq_extras)}")
    for task in sorted(mcq_report):
        r = mcq_report[task]
        rate = r["added"] / max(r["attempted"], 1) * 100
        print(f" {task:20s} +{r['added']:4d} / {r['attempted']:4d} "
              f"({rate:.0f}% wrap rate, {r['failed']} failed)")

    # Step 6: Final counts
    final_counts = defaultdict(int)
    mcq_counts = defaultdict(int)
    source_counts = defaultdict(int)
    for s in final:
        final_counts[s["task"]] += 1
        source_counts[s["source"]] += 1
        if s.get("_mcq_wrapped"):
            mcq_counts[s["task"]] += 1
    print(f"\n FINAL TOTAL: {len(final)} scenes")
    print(f" By source: {dict(source_counts)}")
    print(f" Per-task:")
    for task in sorted(final_counts):
        mcq_part = f" (+{mcq_counts[task]} MCQ)" if mcq_counts[task] else ""
        print(f" {task:20s} {final_counts[task]:5d}{mcq_part}")

    # Step 7: MCQ letter distribution (sanity: should be ~25% each letter)
    letter_dist = compute_letter_distribution(final)
    total_mcq = sum(letter_dist.values())
    print(f"\n MCQ letter distribution (n={total_mcq}):")
    for letter in "ABCD":
        pct = letter_dist.get(letter, 0) / max(total_mcq, 1) * 100
        print(f" {letter}: {letter_dist.get(letter, 0):4d} ({pct:.1f}%)")

    # Step 8: Write manifest
    manifest = {
        "config": {
            "max_per_task": args.max_per_task,
            "mcq_frac": args.mcq_frac,
            "seed": args.seed,
            "hf_dataset": HF_DATASET,
            "r1_tasks": sorted(R1_TASKS),
            "mcq_tasks": sorted(MCQ_TASKS.keys()),
        },
        "totals": {
            "pre_dedup": len(all_scenes),
            "post_dedup": len(deduped),
            "duplicates": n_dupes,
            "post_cap": len(capped),
            "mcq_extras": len(mcq_extras),
            "final": len(final),
        },
        "pre_cap_counts": {t: dict(srcs) for t, srcs in pre_cap.items()},
        "cap_report": cap_report,
        "mcq_report": mcq_report,
        "final_counts": dict(final_counts),
        "mcq_counts_by_task": dict(mcq_counts),
        "source_counts": dict(source_counts),
        "mcq_letter_dist": letter_dist,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\n Manifest -> {OUT_DIR / 'manifest.json'}")

    # Step 9: Scene index (lightweight; no frame bytes)
    index = [
        {
            "scene_id": s["scene_id"],
            "task": s["task"],
            "source": s["source"],
            "mcq_wrapped": bool(s.get("_mcq_wrapped")),
            "n_frames": len(s["frames"]) if s.get("frames") else None,
        }
        for s in final
    ]
    (OUT_DIR / "scene_index.json").write_text(json.dumps(index, indent=2))
    print(f" Scene index -> {OUT_DIR / 'scene_index.json'}")

    # Step 10: Sample dump
    samples_path = OUT_DIR / "samples.md"
    write_samples(final, samples_path, n_per_task=3)
    print(f" Samples -> {samples_path}")

    # Approximate training cost
    batch_size = 16
    n_steps_est = -(-len(final) // batch_size) # ceil
    print(f"\n Estimated training: batch={batch_size} -> {n_steps_est} steps")

    print("\nDone. Review the manifest and samples before launching training.")


if __name__ == "__main__":
    main()
