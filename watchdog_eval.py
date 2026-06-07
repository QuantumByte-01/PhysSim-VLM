#!/usr/bin/env python3
"""Watchdog: auto-restarts GRPO eval if predictions.json goes stale."""
import json, os, time, subprocess, sys
from datetime import datetime

PRED_FILE = "results/physbench_grpo_r3_test/predictions.json"
LOG_FILE = "results/physbench_grpo_r3_test/eval_resume.log"
MODEL_PATH = "tinker://<run-id>:train:0/sampler_weights/final"
CONCURRENCY = 42
HF_TOKEN = "YOUR_HF_TOKEN_HERE"
TOTAL = 9786
STALE_SECS = 480 # 8 minutes
CHECK_EVERY = 60 # seconds

PYTHON = sys.executable
CMD = [
    PYTHON, "-u", "scripts/eval_physbench_tinker.py",
    "--model-path", MODEL_PATH,
    "--out-tag", "grpo_r3_test",
    "--split", "test",
    "--concurrency", str(CONCURRENCY),
]

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[watchdog] {ts} {msg}"
    print(line, flush=True)

def done_count():
    try:
        return len(json.load(open(PRED_FILE)))
    except Exception:
        return 0

def stale_secs():
    try:
        return time.time() - os.path.getmtime(PRED_FILE)
    except Exception:
        return 9999

def restart():
    log("Killing old process...")
    os.system("ps aux | grep eval_physbench_tinker | grep -v grep | awk '{print $1}' | xargs kill 2>/dev/null")
    time.sleep(3)
    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN
    env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    with open(LOG_FILE, "a") as lf:
        proc = subprocess.Popen(CMD, stdout=lf, stderr=lf, env=env)
    log(f"Restarted - PID {proc.pid}")
    time.sleep(60) # let it start before next check

log("Started watchdog")
while True:
    done = done_count()
    if done >= TOTAL:
        log(f"DONE! {done}/{TOTAL} - exiting watchdog.")
        break

    stale = stale_secs()
    if stale > STALE_SECS:
        log(f"STALLED ({stale:.0f}s stale, {done}/{TOTAL} done) - restarting...")
        restart()
    else:
        log(f"OK - {done}/{TOTAL} ({100*done/TOTAL:.1f}%), last write {stale:.0f}s ago")
        time.sleep(CHECK_EVERY)
