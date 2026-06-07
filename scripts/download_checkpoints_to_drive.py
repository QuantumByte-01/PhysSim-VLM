#!/usr/bin/env python3
"""
Download intermediate Tinker checkpoints to Google Drive via streaming.
Reads verified tinker:// paths from checkpoints.jsonl - no path guessing.
Streams directly: Tinker signed URL → curl → rclone rcat (no local staging needed).

Usage:
    python scripts/download_checkpoints_to_drive.py [--dry-run] [--run RUN_NAME] [--type weights|sampler_weights]

Examples:
    # Dry run to see what would be downloaded
    python scripts/download_checkpoints_to_drive.py --dry-run

    # Download only sampler_weights (small, inference-ready) for a specific run
    python scripts/download_checkpoints_to_drive.py --run grpo-run2-balanced --type sampler_weights

    # Download all intermediate checkpoints for all runs
    python scripts/download_checkpoints_to_drive.py
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / '.env')

import tinker

# ── Runs to back up ────────────────────────────────────────────────────────────
# Format: (results_subdir, run_name, drive_label)
# We skip final checkpoints (already backed up to HF) - only intermediate steps.
RUNS = [
    # SFT runs
    ("sft_tinker", "sft-r2-fluid", "sft-r2-fluid"),
    ("sft_tinker", "sft-r3-replay", "sft-r3-replay"),

    # GRPO runs
    ("grpo_tinker", "grpo-epoch1", "grpo-run1"),
    ("grpo_tinker", "grpo-run2-balanced", "grpo-run2-ep1"),
    ("grpo_tinker", "grpo-run2-ep2", "grpo-run2-ep2"),
]

# Google Drive destination folder
DRIVE_BASE = "gdrive:PhysSim-VLM/checkpoints"


def load_checkpoints(results_subdir: str, run_name: str) -> list[dict]:
    """Load checkpoints.jsonl for a run, returning only intermediate (non-final) steps."""
    ckpt_file = ROOT / "results" / results_subdir / run_name / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return []
    entries = []
    seen_steps = set()
    for line in ckpt_file.read_text().strip().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        step = d.get("step")
        tag = d.get("tag", "")
        # Skip final/early_stop checkpoints (already on HF or intended for HF)
        if tag in ("final", "early_stop"):
            continue
        # Deduplicate (GRPO epoch1 had double entries)
        if step in seen_steps:
            continue
        seen_steps.add(step)
        entries.append(d)
    return entries


def get_tinker_paths(entry: dict) -> dict[str, str]:
    """Extract {ckpt_type: tinker_path} from a checkpoints.jsonl entry."""
    paths = {}
    # sampler_weights (inference-ready, smaller ~3.9 GB)
    sampler = entry.get("sampler_path") or entry.get("tinker_path")
    if sampler:
        paths["sampler_weights"] = sampler
    # weights/state (full optimizer state, larger ~11.8 GB)
    state = entry.get("state_path")
    if state:
        paths["weights"] = state
    return paths


def get_signed_url(rc, tinker_path: str) -> str:
    """Get signed download URL for a Tinker checkpoint."""
    resp = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()
    return resp.url


def stream_to_drive(url: str, drive_path: str) -> bool:
    """Stream from signed URL directly to Google Drive via curl | rclone rcat."""
    print(f" -> {drive_path}")
    curl_cmd = ["curl", "-L", "--silent", "--show-error", url]
    rclone_cmd = ["rclone", "rcat", drive_path]

    try:
        t0 = time.time()
        curl_proc = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        rclone_proc = subprocess.Popen(rclone_cmd, stdin=curl_proc.stdout,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        curl_proc.stdout.close()
        rclone_out, rclone_err = rclone_proc.communicate()
        _, curl_err = curl_proc.communicate()

        elapsed = time.time() - t0
        if rclone_proc.returncode == 0 and curl_proc.returncode == 0:
            print(f" OK {elapsed:.0f}s")
            return True
        else:
            print(f" FAILED (curl={curl_proc.returncode}, rclone={rclone_proc.returncode})")
            if curl_err:
                print(f" curl: {curl_err.decode(errors='replace')[:200]}")
            if rclone_err:
                print(f" rclone: {rclone_err.decode(errors='replace')[:200]}")
            return False
    except Exception as e:
        print(f" ERROR: {e}")
        return False


def check_already_uploaded(rc_client, drive_path: str) -> bool:
    """Check if file already exists on Drive via rclone ls."""
    result = subprocess.run(
        ["rclone", "ls", drive_path],
        capture_output=True, text=True
    )
    return result.returncode == 0 and result.stdout.strip() != ""


def main():
    parser = argparse.ArgumentParser(description="Download Tinker checkpoints to Google Drive")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without doing it")
    parser.add_argument("--run", default=None, help="Only process this run label (e.g. grpo-run2-ep1)")
    parser.add_argument("--type", default=None, choices=["weights", "sampler_weights"],
                        help="Only download this checkpoint type (default: both)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip files that already exist on Drive (default: True)")
    args = parser.parse_args()

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()

    # Build task list
    tasks = [] # (drive_label, step, ckpt_type, tinker_path, drive_path)
    for results_subdir, run_name, drive_label in RUNS:
        if args.run and drive_label != args.run:
            continue
        entries = load_checkpoints(results_subdir, run_name)
        if not entries:
            print(f"[SKIP] {drive_label}: no intermediate checkpoints found")
            continue
        for entry in entries:
            step = entry["step"]
            paths = get_tinker_paths(entry)
            for ckpt_type, tinker_path in paths.items():
                if args.type and ckpt_type != args.type:
                    continue
                drive_path = f"{DRIVE_BASE}/{drive_label}/{ckpt_type}/step_{step}.tar.gz"
                tasks.append((drive_label, step, ckpt_type, tinker_path, drive_path))

    # Size estimates
    size_gb = {"sampler_weights": 3.93, "weights": 11.78}
    total_gb = sum(size_gb.get(t[2], 0) for t in tasks)

    print(f"PhysSim-VLM: Tinker -> Google Drive")
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Files: {len(tasks)} Est. size: {total_gb:.0f} GB")
    print(f"Destination: {DRIVE_BASE}/")
    print()

    done, failed, skipped = 0, 0, 0
    for drive_label, step, ckpt_type, tinker_path, drive_path in tasks:
        label = f"{drive_label}/step_{step}/{ckpt_type} ({size_gb.get(ckpt_type, 0):.1f} GB)"
        print(f"[{done+failed+skipped+1}/{len(tasks)}] {label}")

        # Check if already on Drive
        if args.skip_existing and not args.dry_run:
            if check_already_uploaded(rc, drive_path):
                print(f" (skip) Already exists on Drive")
                skipped += 1
                continue

        # In dry-run mode, just print plan without hitting Tinker API
        if args.dry_run:
            print(f" [DRY RUN] -> {drive_path}")
            print(f" from {tinker_path}")
            done += 1
            continue

        # Get signed URL
        try:
            url = get_signed_url(rc, tinker_path)
        except Exception as e:
            print(f" SKIP: Could not get signed URL for {tinker_path}: {e}")
            failed += 1
            continue

        success = stream_to_drive(url, drive_path)
        if success:
            done += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Done: {done} Skipped: {skipped} Failed: {failed} Total: {len(tasks)}")
    if failed:
        print("Re-run with --run <label> --type <type> to retry failed items.")


if __name__ == "__main__":
    main()
