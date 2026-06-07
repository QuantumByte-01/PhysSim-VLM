#!/usr/bin/env python3
"""
PhysSim-VLM: Supervised Fine-Tuning on Tinker
====================================================
Trains Qwen3-VL-30B-A3B-Instruct on MuJoCo-generated physics scenes
(TTC, Stability, Trajectory) using the Thinking Machines Tinker SDK.

Data pipeline:
  Local data/generated/{task}/{scene_id}/ → ImageChunk + EncodedTextChunk
  → Tinker TrainingClient.forward_backward(loss_fn="cross_entropy")
  → optim_step → checkpoint every N steps

Training loop (single epoch):
  1. Scan all local scene directories
  2. Shuffle + split 90/10 train/val
  3. Stream batches to Tinker (data never leaves local machine, only
     image bytes + token IDs are sent per API call)
  4. Log loss, val loss, lr to WandB + metrics.jsonl
  5. Save tinker:// checkpoint every --save-every steps
  6. Final checkpoint saved as "final"

Usage:
  # Full run (all local scenes, 1 epoch)
  python scripts/train_sft_tinker.py

  # Smoke test (first 16 samples only)
  python scripts/train_sft_tinker.py --max-samples 16 --run-name smoke-test

  # Resume from checkpoint
  python scripts/train_sft_tinker.py --resume-path "tinker://<id>/state/step_100"

  # Custom config
  python scripts/train_sft_tinker.py \\
      --lora-rank 64 --batch-size 4 --lr 1e-4 \\
      --wandb-project PhysSim-VLM --run-name sft-epoch1-tinker

Output:
  results/sft_tinker/<run-name>/
    metrics.jsonl per-step metrics
    checkpoints.jsonl tinker:// checkpoint paths + steps
    config.json full run config snapshot
"""

import os, re, sys, json, math, argparse, random, time
import numpy as np
from pathlib import Path
from io import BytesIO
from datetime import datetime
from collections import defaultdict

# ── Force unbuffered stdout ───────────────────────────────────────────────────
sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Propagate tokens
if os.environ.get("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
if not os.environ.get("TINKER_API_KEY"):
    raise RuntimeError("TINKER_API_KEY not set. Check .env")

import base64
from PIL import Image
import tinker
from tinker import types
from tinker_cookbook import tokenizer_utils
from datasets import load_dataset

# Optional WandB - gracefully disabled if not installed/authed
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "generated"
RESULTS_DIR = ROOT / "results" / "sft_tinker"

BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
HF_DATASET = "Swastikr/PhysSim-VLM-Dataset"
TASKS = ["ttc", "stability", "trajectory"]

# Tinker enforces 2 MB per image asset
MAX_IMG_BYTES = 1_900_000 # 1.9 MB with safety margin
MAX_IMG_SIDE = 1024 # max pixel dimension before JPEG compression

# Qwen3-VL special token IDs (from tokenizer vocab)
# These are inserted manually around ImageChunks
_VISION_START_STR = "<|vision_start|>"
_VISION_END_STR = "<|vision_end|>"
_IM_START_STR = "<|im_start|>"
_IM_END_STR = "<|im_end|>"


# ── Qwen3-VL image token computation ─────────────────────────────────

def _smart_resize(height: int, width: int, factor: int = 28,
                  min_pixels: int = 3136, max_pixels: int = 12845056
                  ) -> tuple[int, int]:
    """Qwen3-VL smart_resize: round to multiples of factor, clamp to pixel budget."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)
    return h_bar, w_bar


def compute_qwen3vl_image_tokens(height: int, width: int,
                                  patch_size: int = 14,
                                  merge_size: int = 2,
                                  min_pixels: int = 3136,
                                  max_pixels: int = 235200) -> int:
    """
    Compute number of visual tokens for a single image in Qwen3-VL on Tinker.
    Tinker uses max_pixels=235200 (empirically determined: 640x480 -> 300 tokens).
    """
    factor = patch_size * merge_size # 28
    resized_h, resized_w = _smart_resize(height, width, factor, min_pixels, max_pixels)
    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    return (grid_h // merge_size) * (grid_w // merge_size)


# ── Image utilities ───────────────────────────────────────────────────────────

def compress_image(img_path: Path, quality: int = 85) -> bytes:
    """
    Load image, resize if needed (max MAX_IMG_SIDE px on longest side),
    encode as JPEG. Retries with lower quality until under MAX_IMG_BYTES.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMG_SIDE:
        scale = MAX_IMG_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    for q in [quality, 75, 60, 45]:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q)
        data = buf.getvalue()
        if len(data) <= MAX_IMG_BYTES:
            return data
    raise ValueError(f"Cannot compress {img_path} below {MAX_IMG_BYTES} bytes")


def image_bytes_from_pil(img: Image.Image, quality: int = 85) -> bytes:
    """PIL Image → compressed JPEG bytes."""
    w, h = img.size
    if max(w, h) > MAX_IMG_SIDE:
        scale = MAX_IMG_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    for q in [quality, 75, 60, 45]:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q)
        data = buf.getvalue()
        if len(data) <= MAX_IMG_BYTES:
            return data
    raise ValueError("Cannot compress image below size limit")


# ── Data loading from HuggingFace Dataset ─────────────────────────────────────

def load_hf_scenes(hf_dataset: str, hf_token: str, split: str = "train",
                   max_samples: int | None = None, seed: int = 42) -> list[dict]:
    """
    Load PhysSim-VLM-Dataset from HuggingFace.
    Downloads to local HF cache on first call (~1.5 GB); subsequent calls load
    from cache instantly. Uses streaming=False after download for robustness.

    Returns list of scene dicts with keys:
      scene_id, task, prompt_text, assistant_text, frames_b64
    """
    import os as _os
    # Increase download timeout for large parquet files
    _os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

    print(f" Loading {hf_dataset} ({split}) - downloading to HF cache if needed...")
    ds = load_dataset(
        hf_dataset,
        split=split,
        token=hf_token,
        streaming=False, # download to cache first, then load locally
    )

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    print(f" Loaded {len(ds)} scenes (cached locally)")

    scenes = []
    for row in ds:
        scenes.append({
            "scene_id": row["scene_id"],
            "task": row["task"],
            "prompt_text": row["prompt"],
            "assistant_text": row["assistant_text"],
            "frames_b64": row["frames_b64"],
        })
    return scenes


def decode_frames_b64(frames_b64: list[str]) -> list[bytes]:
    """Decode base64 frame strings to JPEG bytes.
    Always re-encodes through PIL to guarantee JPEG format
    (some dataset frames may be PNG) and respect size limits.
    """
    result = []
    for b64str in frames_b64:
        raw = base64.b64decode(b64str)
        img = Image.open(BytesIO(raw)).convert("RGB")
        result.append(image_bytes_from_pil(img))
    return result


# ── Prompt / Datum building ───────────────────────────────────────────────────

def build_training_datum(tokenizer, scene: dict) -> "tinker.Datum | None":
    """
    Build a Tinker Datum for cross-entropy SFT from a HF dataset scene.

    Full-sequence shifted approach (matches tinker_cookbook/supervised/common.py):
      model_input = full_sequence[:-1] (prompt images+text + response, minus last token)
      target_tokens = full_sequence[1:] (shifted by 1)
      weights = 0.0 for prompt positions, 1.0 for response positions

    Image tokens are expanded server-side; we set expected_tokens on each
    ImageChunk so client-side length matches server-side token_count.
    """
    # Decode base64 frames -> JPEG bytes
    try:
        img_bytes_list = decode_frames_b64(scene["frames_b64"])
    except Exception as e:
        print(f" [WARN] {scene['scene_id']} frame decode: {e}")
        return None

    if not img_bytes_list:
        print(f" [WARN] {scene['scene_id']}: no frames")
        return None

    # Compute expected visual tokens per image
    img_infos = []
    for img_bytes in img_bytes_list:
        img = Image.open(BytesIO(img_bytes))
        w, h = img.size
        n_tok = compute_qwen3vl_image_tokens(h, w)
        img_infos.append((img_bytes, n_tok))

    # ── Build chunks + flat token/flag arrays simultaneously ──────────────────
    chunks = []
    flat_tokens = [] # token IDs (0 for image placeholder positions)
    flat_is_response = [] # True only for assistant response positions

    def _add_text(token_ids, is_resp=False):
        chunks.append(types.EncodedTextChunk(tokens=token_ids))
        flat_tokens.extend(token_ids)
        flat_is_response.extend([is_resp] * len(token_ids))

    def _add_image(img_bytes, n_tokens):
        chunks.append(types.ImageChunk(
            data=img_bytes, format="jpeg", expected_tokens=n_tokens))
        flat_tokens.extend([0] * n_tokens) # placeholder
        flat_is_response.extend([False] * n_tokens) # images = prompt

    # -- User turn --
    prefix_ids = tokenizer.encode(
        f"{_IM_START_STR}user\n", add_special_tokens=False)
    _add_text(prefix_ids)

    vs_ids = tokenizer.encode(_VISION_START_STR, add_special_tokens=False)
    ve_ids = tokenizer.encode(_VISION_END_STR, add_special_tokens=False)
    for img_bytes, n_tok in img_infos:
        _add_text(vs_ids)
        _add_image(img_bytes, n_tok)
        _add_text(ve_ids)

    # Prompt text + transition to assistant
    suffix_str = (f"{scene['prompt_text']}"
                  f"{_IM_END_STR}\n{_IM_START_STR}assistant\n")
    suffix_ids = tokenizer.encode(suffix_str, add_special_tokens=False)
    _add_text(suffix_ids)

    # -- Assistant response (train on this) --
    assistant_text = scene["assistant_text"]
    if not assistant_text.endswith(_IM_END_STR):
        assistant_text += _IM_END_STR
    asst_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
    _add_text(asst_ids, is_resp=True)

    # ── Next-token-prediction shift ───────────────────────────────────────────
    N = len(flat_tokens)
    # target_tokens[i] = flat_tokens[i+1] (what the model should predict)
    # weights[i] = 1.0 if flat_is_response[i+1] else 0.0
    target_flat = flat_tokens[1:] # length N-1
    weights_flat = [1.0 if flat_is_response[i] else 0.0
                    for i in range(1, N)] # length N-1

    # Remove last token from last text chunk so model_input.length = N-1
    last_chunk = chunks[-1]
    if not isinstance(last_chunk, types.EncodedTextChunk) or len(last_chunk.tokens) < 2:
        print(f" [WARN] {scene['scene_id']}: last chunk too short, skipping")
        return None
    chunks[-1] = types.EncodedTextChunk(tokens=list(last_chunk.tokens)[:-1])

    model_input = types.ModelInput(chunks=chunks)

    # Sanity check
    mi_len = model_input.length
    if mi_len != N - 1:
        print(f" [WARN] {scene['scene_id']}: length mismatch "
              f"model_input={mi_len} expected={N-1}, skipping")
        return None

    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=[int(x) for x in target_flat],
                dtype="int64",
                shape=[N - 1],
            ),
            "weights": tinker.TensorData(
                data=weights_flat,
                dtype="float32",
                shape=[N - 1],
            ),
        },
    )


# ── Metrics & logging ─────────────────────────────────────────────────────────

class MetricsLogger:
    """Logs metrics to jsonl file + optionally WandB."""

    def __init__(self, out_dir: Path, use_wandb: bool,
                 wandb_project: str, run_name: str, config: dict):
        self.out_dir = out_dir
        self.jsonl_path = out_dir / "metrics.jsonl"
        self.use_wandb = use_wandb and _WANDB_AVAILABLE

        if self.use_wandb:
            wandb.init(project=wandb_project, name=run_name, config=config,
                       resume="allow")
            print(f" WandB run: {wandb.run.url}")
        elif use_wandb:
            print(" [WARN] WandB not installed - logging to jsonl only")

    def log(self, metrics: dict, step: int):
        record = {"step": step, "ts": datetime.now().isoformat(), **metrics}
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def finish(self):
        if self.use_wandb:
            wandb.finish()


class CheckpointTracker:
    """Tracks tinker:// checkpoint paths + step numbers."""

    def __init__(self, out_dir: Path):
        self.path = out_dir / "checkpoints.jsonl"

    def record(self, step: int, tinker_path: str, tag: str = "",
               state_path: str = ""):
        record = {"step": step, "tinker_path": tinker_path,
                  "state_path": state_path,
                  "tag": tag, "ts": datetime.now().isoformat()}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f" [CKPT] step={step} -> {tinker_path}")
        if state_path:
            print(f" [CKPT] step={step} state -> {state_path}")

    def latest(self) -> dict | None:
        if not self.path.exists():
            return None
        lines = [l for l in self.path.read_text().splitlines() if l.strip()]
        if not lines:
            return None
        return json.loads(lines[-1])


# ── Loss computation ──────────────────────────────────────────────────────────

def compute_weighted_nll(fwd_result, datums: list) -> float:
    """
    Compute weighted mean NLL from forward/forward_backward result.
    Uses per-token logprobs from loss_fn_outputs and weights from datums.
    Matches tinker_cookbook/supervised/common.py::compute_mean_nll.
    """
    nll_sum = 0.0
    w_sum = 0.0
    for lfo, d in zip(fwd_result.loss_fn_outputs, datums):
        lp = np.array(lfo["logprobs"].data)
        w = np.array(d.loss_fn_inputs["weights"].data)
        nll_sum += float((-lp * w).sum())
        w_sum += float(w.sum())
    return nll_sum / max(w_sum, 1.0)


# ── Validation ────────────────────────────────────────────────────────────────

def compute_val_loss(training_client, tokenizer, val_scenes: list[dict]) -> float:
    """Compute average cross-entropy forward loss on val set (no weight update)."""
    nll_sum = 0.0
    w_sum = 0.0
    for scene in val_scenes:
        datum = build_training_datum(tokenizer, scene)
        if datum is None:
            continue
        try:
            out = training_client.forward(data=[datum], loss_fn="cross_entropy").result()
            lp = np.array(out.loss_fn_outputs[0]["logprobs"].data)
            w = np.array(datum.loss_fn_inputs["weights"].data)
            nll_sum += float((-lp * w).sum())
            w_sum += float(w.sum())
        except Exception as e:
            print(f" [WARN] val loss {scene['scene_id']}: {e}")
    return nll_sum / max(w_sum, 1.0) if w_sum > 0 else float("nan")


# ── Main training loop (synchronous - Tinker uses APIFuture.result()) ─────────

def train(args):
    run_name = args.run_name or f"sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPhysSim-VLM SFT on Tinker")
    print(f" Run : {run_name}")
    print(f" Model : {BASE_MODEL}")
    print(f" Out dir : {out_dir}")

    # ── Load tokenizer FIRST (heavy download; must finish before connecting) ──
    print(f"\n Loading tokenizer ({BASE_MODEL})...")
    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    print(f" Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # ── Load dataset from HuggingFace ─────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set. Check .env")

    train_scenes = load_hf_scenes(
        HF_DATASET, hf_token, split="train",
        max_samples=args.max_samples, seed=args.seed)
    val_scenes = load_hf_scenes(
        HF_DATASET, hf_token, split="val",
        max_samples=min(200, args.max_samples or 200), seed=args.seed)

    if not train_scenes:
        raise RuntimeError(f"No training scenes loaded from {HF_DATASET}")

    random.Random(args.seed).shuffle(train_scenes)
    print(f" Dataset : {len(train_scenes)} train | {len(val_scenes)} val")
    task_counts: dict[str, int] = defaultdict(int)
    for s in train_scenes:
        task_counts[s["task"]] += 1
    print(f" Tasks : {dict(task_counts)}")

    # ── Config snapshot ───────────────────────────────────────────────────────
    config = {
        "base_model": BASE_MODEL,
        "hf_dataset": HF_DATASET,
        "lora_rank": args.lora_rank,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "eps": args.eps,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "n_train": len(train_scenes),
        "n_val": len(val_scenes),
        "task_counts": dict(task_counts),
        "save_every": args.save_every,
        "val_every": args.val_every,
        "run_name": run_name,
        "started_at": datetime.now().isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f" Config -> {out_dir}/config.json")

    # ── Metrics & checkpoint trackers ─────────────────────────────────────────
    metrics_logger = MetricsLogger(
        out_dir, use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project, run_name=run_name, config=config)
    ckpt_tracker = CheckpointTracker(out_dir)

    # ── Tinker clients (sync - created after data/tokenizer are ready) ────────
    print(f"\n Connecting to Tinker...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=args.lora_rank,
    )
    print(f" Training client ready (LoRA rank={args.lora_rank})")

    # Resume from checkpoint (must be a state_with_optimizer path, NOT sampler_weights)
    if args.resume_path:
        print(f" Resuming from: {args.resume_path}")
        if "sampler_weights" in args.resume_path:
            print(f" [ERROR] Cannot resume from sampler_weights path!")
            print(f" Use a state_with_optimizer path from checkpoints.jsonl instead.")
            print(f" Example: --resume-path 'tinker://<id>/state/<step>'")
            return
        training_client.load_state_with_optimizer(args.resume_path).result()
        print(f" Checkpoint loaded (weights + optimizer state)")

    # ── Training loop ─────────────────────────────────────────────────────────
    n_batches = math.ceil(len(train_scenes) / args.batch_size)
    start_batch = args.start_step if args.start_step else 0
    global_step = start_batch
    epoch_loss = []
    adam_params = types.AdamParams(
        learning_rate=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
    )

    remaining = n_batches - start_batch
    print(f"\n Training: {n_batches} steps | batch={args.batch_size} "
          f"| lr={args.lr} | rank={args.lora_rank}")
    if start_batch > 0:
        print(f" Resuming from step {start_batch} ({remaining} steps remaining)")
    print(f" Save every {args.save_every} steps | Val every {args.val_every} steps\n")

    for batch_idx in range(start_batch, n_batches):
        t0 = time.time()
        start = batch_idx * args.batch_size
        batch = train_scenes[start: start + args.batch_size]

        datums = [build_training_datum(tokenizer, s) for s in batch]
        datums = [d for d in datums if d is not None]

        if not datums:
            print(f" [WARN] step {batch_idx+1} - all datums failed, skipping")
            continue

        # Forward-backward + optimizer step (pipelined per tinker_cookbook)
        try:
            fwd_bwd_future = training_client.forward_backward(
                data=datums, loss_fn="cross_entropy")
            optim_future = training_client.optim_step(adam_params=adam_params)
            fwd_out = fwd_bwd_future.result()
            optim_future.result()
            step_loss = compute_weighted_nll(fwd_out, datums)
        except Exception as e:
            print(f" [ERROR] step {batch_idx+1}: {e}")
            continue

        global_step += 1
        epoch_loss.append(step_loss)
        elapsed = time.time() - t0

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        metrics_logger.log({
            "train/loss": step_loss,
            "train/loss_avg": avg_loss,
            "train/lr": args.lr,
            "train/batch_size": len(datums),
            "train/step_time_s": elapsed,
            "train/samples_seen": global_step * args.batch_size,
        }, step=global_step)

        print(f" step {global_step:4d}/{n_batches} "
              f"loss={step_loss:.4f} avg={avg_loss:.4f} "
              f"({elapsed:.1f}s) [{len(datums)}/{len(batch)} ok]")

        # Validation
        if global_step % args.val_every == 0 and val_scenes:
            print(f" [VAL] {len(val_scenes)} samples...")
            val_loss = compute_val_loss(training_client, tokenizer, val_scenes)
            print(f" [VAL] step={global_step} val_loss={val_loss:.4f}")
            metrics_logger.log({"val/loss": val_loss}, step=global_step)

        # Checkpoint (save both sampler weights + optimizer state for resumability)
        if global_step % args.save_every == 0:
            ckpt_name = f"step_{global_step}"
            try:
                sampler_future = training_client.save_weights_for_sampler(ckpt_name)
                state_future = training_client.save_state(ckpt_name)
                sampler_res = sampler_future.result()
                state_res = state_future.result()
                ckpt_tracker.record(global_step, sampler_res.path,
                                    tag="periodic", state_path=state_res.path)
                metrics_logger.log({"checkpoint_step": global_step}, step=global_step)
            except Exception as e:
                print(f" [WARN] Checkpoint step={global_step}: {e}")

    # ── Final checkpoint (both sampler + optimizer state) ─────────────────────
    print(f"\n Saving final checkpoint...")
    try:
        sampler_future = training_client.save_weights_for_sampler("final")
        state_future = training_client.save_state("final")
        sampler_res = sampler_future.result()
        state_res = state_future.result()
        ckpt_tracker.record(global_step, sampler_res.path,
                            tag="final", state_path=state_res.path)
        print(f" Final checkpoint: {sampler_res.path}")
        print(f" Resumable state: {state_res.path}")
    except Exception as e:
        print(f" [WARN] Final checkpoint failed: {e}")

    # Final val loss
    if val_scenes:
        val_loss = compute_val_loss(training_client, tokenizer, val_scenes)
        print(f" Final val loss : {val_loss:.4f}")
        metrics_logger.log({
            "val/loss_final": val_loss,
            "train/loss_final": sum(epoch_loss) / len(epoch_loss) if epoch_loss else float("nan"),
        }, step=global_step)

    # ── Summary ───────────────────────────────────────────────────────────────
    final_avg = sum(epoch_loss) / len(epoch_loss) if epoch_loss else float("nan")
    (out_dir / "summary.json").write_text(json.dumps({
        "run_name": run_name,
        "total_steps": global_step,
        "total_samples": len(train_scenes),
        "final_train_loss": final_avg,
        "finished_at": datetime.now().isoformat(),
    }, indent=2))
    metrics_logger.finish()

    latest = ckpt_tracker.latest()
    print(f"\n Done! steps={global_step} avg_loss={final_avg:.4f}")
    print(f" Results : {out_dir}")
    if latest:
        print(f" Checkpoint : {latest['tinker_path']}")
        print(f"\n Eval SFT vs baseline:")
        print(f" python scripts/eval_physbench_tinker.py "
              f"--model-path \"{latest['tinker_path']}\" --compare")
        print(f"\n Download LoRA weights:")
        print(f" python scripts/download_tinker_checkpoint.py "
              f"--path \"{latest['tinker_path']}\"")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PhysSim-VLM SFT on Tinker - 1 epoch over MuJoCo physics dataset")

    # Data
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap total train samples (default: all 12,023)")
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--lora-rank", type=int, default=64,
                        help="LoRA rank (default: 64)")

    # Optimizer
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Datums per forward_backward call")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95,
                        help="Tinker default beta2 (0.95 vs PyTorch 0.999)")
    parser.add_argument("--eps", type=float, default=1e-12,
                        help="Tinker default eps (1e-12)")
    parser.add_argument("--weight-decay",type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm (0 = disabled)")

    # Checkpointing
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save tinker:// checkpoint every N steps")
    parser.add_argument("--val-every", type=int, default=50,
                        help="Compute val loss every N steps")
    parser.add_argument("--resume-path", type=str, default=None,
                        help="tinker:// path to resume from")
    parser.add_argument("--start-step", type=int, default=0,
                        help="Skip to this step when resuming (data batches are deterministic)")

    # Logging
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="PhysSim-VLM")
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
