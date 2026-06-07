#!/usr/bin/env python3
"""
PhysSim-VLM: HuggingFace Upload Utility
==============================================
Uploads LoRA checkpoints from Tinker to HF Hub, one checkpoint at a time.

Usage:
  python scripts/upload_to_hf.py --list # show all available checkpoints
  python scripts/upload_to_hf.py --ckpt sft-final # upload SFT final (best test)
  python scripts/upload_to_hf.py --ckpt grpo-ep1-final # upload GRPO Ep1 final (best val)
  python scripts/upload_to_hf.py --ckpt grpo-ep1-step300 # upload GRPO Ep1 step 300
  python scripts/upload_to_hf.py --all # upload all in sequence
  python scripts/upload_to_hf.py --dataset # upload dataset only
"""

import os, json, sys, argparse, tarfile, tempfile, shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import requests
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
HF_USER = "Swastikr"
ROOT = Path(__file__).parent.parent
MODEL_CARD = ROOT / "docs" / "model_card.md"

# ── Checkpoint Registry ────────────────────────────────────────────────────────
# Each entry: name, tinker_path, hf_repo, description, test_acc, val_acc
CHECKPOINTS = [
    {
        "name": "sft-final",
        "tinker_path": "tinker://<run-id>:train:0/sampler_weights/final",
        "hf_repo": f"{HF_USER}/PhysSim-VLM-SFT",
        "description": "SFT R2 Fluid final (step 269) - BEST TEST MODEL",
        "test_acc": "46.9%",
        "val_acc": "63.8%",
        "commit_msg": "SFT R2 Fluid final - best test checkpoint (46.9% PhysBench, #1 Scene 44.6%)",
    },
    {
        "name": "sft-step200",
        "tinker_path": "tinker://<run-id>:train:0/sampler_weights/step_200",
        "hf_repo": f"{HF_USER}/PhysSim-VLM-SFT",
        "description": "SFT R2 step 200 (intermediate)",
        "test_acc": "~45%",
        "val_acc": "~62%",
        "commit_msg": "SFT R2 step 200 intermediate checkpoint",
    },
    {
        "name": "grpo-ep1-step300",
        "tinker_path": "tinker://<run-id>:train:0/sampler_weights/step_300",
        "hf_repo": f"{HF_USER}/PhysSim-VLM-GRPO",
        "description": "GRPO Run2 Ep1 step 300 (near reward peak)",
        "test_acc": "~44%",
        "val_acc": "~65%",
        "commit_msg": "GRPO Run2 Ep1 step 300 - near reward peak",
    },
    {
        "name": "grpo-ep1-step400",
        "tinker_path": "tinker://<run-id>:train:0/sampler_weights/step_400",
        "hf_repo": f"{HF_USER}/PhysSim-VLM-GRPO",
        "description": "GRPO Run2 Ep1 step 400",
        "test_acc": "~44%",
        "val_acc": "~65%",
        "commit_msg": "GRPO Run2 Ep1 step 400",
    },
    {
        "name": "grpo-ep1-final",
        "tinker_path": "tinker://<run-id>:train:0/sampler_weights/final",
        "hf_repo": f"{HF_USER}/PhysSim-VLM-GRPO",
        "description": "GRPO Run2 Ep1 final (step 475) - BEST VAL MODEL",
        "test_acc": "44.0%",
        "val_acc": "65.8%",
        "commit_msg": "GRPO Run2 Ep1 final - best val checkpoint (65.8% val, 44.0% test)",
    },
]

CKPT_BY_NAME = {c["name"]: c for c in CHECKPOINTS}


# ── Core upload logic ──────────────────────────────────────────────────────────

def upload_checkpoint(ckpt: dict):
    """Download one checkpoint from Tinker and push every file to HF Hub."""
    from tinker import ServiceClient

    api_key = os.environ.get("TINKER_API_KEY", "")
    hf_token = HF_TOKEN
    if not api_key:
        raise RuntimeError("TINKER_API_KEY not set in .env")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set in .env")

    print(f"\n{'='*60}")
    print(f" Checkpoint : {ckpt['name']}")
    print(f" Description: {ckpt['description']}")
    print(f" Tinker : {ckpt['tinker_path']}")
    print(f" HF repo : {ckpt['hf_repo']}")
    print(f" Test acc : {ckpt['test_acc']} Val acc: {ckpt['val_acc']}")
    print(f"{'='*60}")

    # Persistent temp dir - survives retries between runs
    tmp_base = Path(tempfile.gettempdir()) / "physsim-vlm_hf"
    tmp = tmp_base / ckpt["name"]
    tmp.mkdir(parents=True, exist_ok=True)
    archive = tmp / "checkpoint.tar"
    extract_dir = tmp / "extract"
    extract_dir.mkdir(exist_ok=True)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    cached_adapter = extract_dir / "adapter_model.safetensors"
    if cached_adapter.exists() and cached_adapter.stat().st_size > 1e9:
        print("[1/4] Skipping download - cached extract already present.")
        print("[2/4] Skipping download (cached).")
        print("[3/4] Skipping extraction (cached).")
        rc = None
    else:
        sc = ServiceClient(api_key=api_key)
        rc = sc.create_rest_client()
        print("[1/4] Fetching signed URL...")
        url_resp = rc.get_checkpoint_archive_url_from_tinker_path(ckpt["tinker_path"]).result()
        url = url_resp.url
        print(f" URL: {url[:80]}...")

        print("[2/4] Downloading archive (with resume)...")
        _download_with_resume(rc, ckpt["tinker_path"], url, archive)

        print("[3/4] Extracting...")
        with tarfile.open(archive, "r") as tar:
            members = [m for m in tar.getmembers()
                       if not m.name.startswith("/") and ".." not in m.name]
            tar.extractall(extract_dir, members=members)
        files = [f.name for f in extract_dir.iterdir()]
        print(f" Extracted: {files}")

        # Patch adapter_config base model
        adapter_cfg = extract_dir / "adapter_config.json"
        if adapter_cfg.exists():
            cfg = json.loads(adapter_cfg.read_text())
            if not cfg.get("base_model_name_or_path") or cfg["base_model_name_or_path"] == "unknown":
                cfg["base_model_name_or_path"] = BASE_MODEL
                adapter_cfg.write_text(json.dumps(cfg, indent=2) + "\n")
                print(" Patched adapter_config.base_model_name_or_path")

    # Copy model card
    if MODEL_CARD.exists():
        shutil.copy(MODEL_CARD, extract_dir / "README.md")
        print(" Copied model card -> README.md")

    # ── Step 4: Upload file by file ───────────────────────────────────────────
    print("[4/4] Uploading to HuggingFace Hub (file by file)...")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=ckpt["hf_repo"], repo_type="model", private=True, exist_ok=True)

    upload_files = sorted(
        [f for f in extract_dir.iterdir() if f.name != "checkpoint_complete"],
        key=lambda f: f.stat().st_size # smallest first - fail fast on config files
    )

    for i, f in enumerate(upload_files, 1):
        size_mb = f.stat().st_size / 1e6
        print(f" [{i}/{len(upload_files)}] {f.name} ({size_mb:.0f} MB) ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=ckpt["hf_repo"],
            repo_type="model",
            commit_message=f"{ckpt['commit_msg']} - {f.name}",
        )
        print("done")

    print(f"\n OK {ckpt['name']} uploaded -> https://huggingface.co/{ckpt['hf_repo']}")

    # Clean up archive (keep extract dir for retry safety)
    if archive.exists():
        archive.unlink()
        print(" (archive cleaned up)")


def _download_with_resume(rc, tinker_path: str, url: str, archive: Path):
    """Download with resume + URL refresh on every retry attempt."""
    max_retries = 20
    for attempt in range(max_retries):
        done = archive.stat().st_size if archive.exists() else 0
        if attempt > 0:
            print(f"\n [Retry {attempt}] Refreshing signed URL...")
            url_resp = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()
            url = url_resp.url
        headers = {"Range": f"bytes={done}-"} if done > 0 else {}
        try:
            with requests.get(url, stream=True, timeout=600, headers=headers) as r:
                if r.status_code == 416:
                    print(f"\n 416 Range error - restarting from 0...")
                    archive.unlink(missing_ok=True)
                    done = 0
                    headers = {}
                    r = requests.get(url, stream=True, timeout=600)
                total = done + int(r.headers.get("content-length", 0))
                r.raise_for_status()
                mode = "ab" if done > 0 else "wb"
                with open(archive, mode) as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f" {done/1e6:.0f}/{total/1e6:.0f} MB", end="\r")
            break
        except Exception as e:
            print(f"\n Attempt {attempt+1} failed at {done/1e6:.0f} MB: {e}")
            if attempt >= max_retries - 1:
                raise
    print(f"\n Downloaded {archive.stat().st_size/1e6:.1f} MB total")


# ── Dataset upload ─────────────────────────────────────────────────────────────

def upload_dataset():
    import subprocess
    print("\n── Uploading Dataset ──")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prepare_dataset.py"), "--upload"],
        check=True,
    )
    print("OK Dataset upload complete")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload PhysSim-VLM checkpoints to HF")
    parser.add_argument("--list", action="store_true", help="List all available checkpoints")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Upload a specific checkpoint by name (see --list)")
    parser.add_argument("--all", action="store_true", help="Upload all checkpoints in sequence")
    parser.add_argument("--dataset", action="store_true", help="Upload dataset")
    args = parser.parse_args()

    if not any([args.list, args.ckpt, args.all, args.dataset]):
        parser.print_help()
        return

    if args.list:
        print(f"\n{'Name':<25} {'HF Repo':<45} {'Test':>6} {'Val':>6} Description")
        print("-" * 110)
        for c in CHECKPOINTS:
            print(f" {c['name']:<23} {c['hf_repo']:<45} {c['test_acc']:>6} {c['val_acc']:>6} {c['description']}")
        return

    if args.dataset:
        upload_dataset()

    if args.ckpt:
        if args.ckpt not in CKPT_BY_NAME:
            print(f"ERROR: Unknown checkpoint '{args.ckpt}'. Run --list to see options.")
            sys.exit(1)
        upload_checkpoint(CKPT_BY_NAME[args.ckpt])

    if args.all:
        print(f"\nUploading {len(CHECKPOINTS)} checkpoints in sequence...")
        for i, ckpt in enumerate(CHECKPOINTS, 1):
            print(f"\n[{i}/{len(CHECKPOINTS)}] Starting: {ckpt['name']}")
            try:
                upload_checkpoint(ckpt)
            except Exception as e:
                print(f"\nERROR on {ckpt['name']}: {e}")
                print("Continuing with next checkpoint...")

    print("\nOK Done.")


if __name__ == "__main__":
    main()
