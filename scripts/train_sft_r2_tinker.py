#!/usr/bin/env python3
"""
PhysSim-VLM: SFT Training on Tinker
==========================================
Supports two modes:
  1. R2-only: Resume from SFT R1 checkpoint, train on local R2 data
  2. Combined: Train from baseline on HF R1 + local R2 merged data

Usage:
  # R2-only (original mode)
  python scripts/train_sft_r2_tinker.py --run-name sft-r2-redo

  # Combined from baseline (recommended)
  python scripts/train_sft_r2_tinker.py --from-baseline --include-hf \
      --task-cap motion_comparison=1000 --mcq-frac 0.5 \
      --batch-size 16 --lr 1e-4 --run-name sft-combined
"""

import os, sys, json, math, argparse, random, re, time
import numpy as np
from pathlib import Path
from io import BytesIO
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if os.environ.get("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
if not os.environ.get("TINKER_API_KEY"):
    raise RuntimeError("TINKER_API_KEY not set. Check .env")

import base64
from PIL import Image
import tinker
from tinker import types
from tinker_cookbook import tokenizer_utils

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "sft_r2" # SFT R2 local data
RESULTS_DIR = ROOT / "results" / "sft_tinker"

BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
HF_DATASET = "Swastikr/PhysSim-VLM-Dataset"
R1_TASKS = {"ttc", "trajectory", "stability"}

SFT_R1_CHECKPOINT_STATE = "tinker://<run-id>:train:0/weights/final"
SFT_R2_FLUID_CHECKPOINT_STATE = "tinker://<run-id>:train:0/weights/final"

MAX_IMG_BYTES = 1_900_000
MAX_IMG_SIDE = 1024

_VISION_START_STR = "<|vision_start|>"
_VISION_END_STR = "<|vision_end|>"
_IM_START_STR = "<|im_start|>"
_IM_END_STR = "<|im_end|>"


# ── Image utilities ───────────────────────────────────────────────────────────

def _smart_resize(height: int, width: int, factor: int = 28,
                  min_pixels: int = 3136, max_pixels: int = 12845056
                  ) -> tuple[int, int]:
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
    factor = patch_size * merge_size
    resized_h, resized_w = _smart_resize(height, width, factor, min_pixels, max_pixels)
    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    return (grid_h // merge_size) * (grid_w // merge_size)


def image_bytes_from_pil(img: Image.Image, quality: int = 85) -> bytes:
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


# ── MCQ format wrapping ───────────────────────────────────────────────────────
#
# PhysBench evaluates with multiple-choice prompts (A/B/C/D). Plain SFT teaches
# the model to emit free-text answers like "left" or "object b", which costs
# ~1.7pp on the benchmark because the parser cannot map free-text to letters
# reliably. mcq_wrap_scene() rewrites a categorical scene as an MCQ variant
# (shuffled options + letter answer) so the model learns both formats.

MCQ_TASKS: dict[str, list[str]] = {
    # "up" removed - R2 ground truth is only left/right/down (235 each).
    # Distractor "no clear flow" added to keep it 4-option like PhysBench.
    "fluid_direction": ["left", "right", "down", "no clear flow"],
    "fluid_level": ["low (below 20%)", "medium (20-50%)",
                          "high (50-80%)", "very high (above 80%)"],
    # 2-option tasks padded to 4 options to match PhysBench MCQ format.
    # Distractors are semantically valid but never-correct in training - 
    # the model learns to pick any of 4 letters based on physics content.
    "fluid_viscosity": ["fluid a", "fluid b",
                          "both fluids similar",
                          "cannot tell from video"],
    "viewpoint": ["left", "right", "above", "below"],
    "motion_comparison": ["object a", "object b",
                          "both move at similar speed",
                          "cannot tell from video"],
    "object_comparison": ["object a", "object b",
                          "both objects similar",
                          "cannot tell from video"],
    "manipulation": ["barely moves",
                          "barely lifts off the surface",
                          "slides a short distance",
                          "slides far across the surface"],
    "stability": ["stable", "unstable",
                          "marginally stable",
                          "cannot determine"],
}

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>",
                           re.IGNORECASE | re.DOTALL)
_REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>",
                           re.IGNORECASE | re.DOTALL)


def _extract_answer(text: str) -> str | None:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def _strip_prompt_template(prompt: str) -> str:
    out = []
    for ln in prompt.splitlines():
        s = ln.strip()
        if (s.startswith("<answer>") or s.startswith("<reasoning>")
                or s.startswith("<confidence>")):
            continue
        if s.lower().startswith("options:"):
            continue
        out.append(ln)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def mcq_wrap_scene(scene: dict, rng: random.Random) -> dict | None:
    """Return an MCQ-format variant of `scene`, or None if it cannot be wrapped.

    The variant shares frames but rewrites prompt_text and assistant_text so the
    model emits a single letter A-D inside <answer> tags.
    """
    options = MCQ_TASKS.get(scene["task"])
    if not options:
        return None

    answer = _extract_answer(scene["assistant_text"])
    if not answer:
        return None

    norm_ans = answer.strip().lower()
    correct = next((o for o in options if o.lower() == norm_ans), None)
    if correct is None:
        # fuzzy: option text contained in answer or vice versa
        for o in options:
            ol = o.lower()
            if ol in norm_ans or norm_ans in ol:
                correct = o
                break
    if correct is None:
        return None

    shuffled = list(options)
    rng.shuffle(shuffled)
    correct_letter = "ABCD"[shuffled.index(correct)]

    base_prompt = _strip_prompt_template(scene["prompt_text"])
    mcq_block = "\n".join(f"{l}. {opt}"
                          for l, opt in zip("ABCD", shuffled))
    new_prompt = (
        f"{base_prompt}\n\n"
        f"Choose the best answer:\n{mcq_block}\n\n"
        f"<reasoning>Brief physics analysis</reasoning>\n"
        f"<answer>letter</answer>"
    )

    rmatch = _REASONING_RE.search(scene["assistant_text"])
    reasoning = (rmatch.group(1).strip() if rmatch
                 else "Based on the physics analysis above.")
    new_assistant = (
        f"<reasoning>{reasoning}</reasoning>\n"
        f"<answer>{correct_letter}</answer>"
    )

    return {
        **scene,
        "scene_id": scene["scene_id"] + "_mcq",
        "prompt_text": new_prompt,
        "assistant_text": new_assistant,
        "_mcq_wrapped": True,
    }


# ── Load SFT R2 scenes from local data ────────────────────────────────────────

def load_sft_r2_scenes(max_samples: int | None = None, seed: int = 42,
                       tasks: list[str] | None = None) -> list[dict]:
    """Load SFT R2 scenes from local data directories.
    Searches both data/sft_r2/ and data/generated/ for task directories.
    Pass tasks= to filter to specific task names (e.g. ['fluid_direction','fluid_level']).
    """
    data_dirs = [DATA_DIR, ROOT / "data" / "generated"]
    for d in data_dirs:
        print(f" Scanning {d}...")
    if tasks:
        print(f" Filtering to tasks: {tasks}")

    scenes = []
    errors = 0
    seen_tasks = set()

    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for task_dir in sorted(data_dir.glob("*")):
            if not task_dir.is_dir():
                continue

            task = task_dir.name
            if task in seen_tasks:
                continue # avoid loading same task from both dirs
            if tasks and task not in tasks:
                continue
            seen_tasks.add(task)
            scene_dirs = sorted(task_dir.glob("*"))
            print(f" {task}: {len(scene_dirs)} scenes (from {data_dir.name})")

            for scene_dir in scene_dirs:
                if not scene_dir.is_dir():
                    continue

                try:
                    gt_path = scene_dir / "ground_truth.json"
                    prompt_path = scene_dir / "prompt.txt"
                    asst_path = scene_dir / "assistant_text.txt"

                    if not all([gt_path.exists(), prompt_path.exists(), asst_path.exists()]):
                        continue

                    with open(gt_path, encoding="utf-8") as f:
                        gt = json.load(f)
                    with open(prompt_path, encoding="utf-8") as f:
                        prompt = f.read().strip()
                    with open(asst_path, encoding="utf-8") as f:
                        assistant_text = f.read().strip()

                    # Load frames
                    frames = []
                    frames_dir = scene_dir / "frames"
                    if frames_dir.exists():
                        frames = sorted(frames_dir.glob("frame_*.png"))
                    else:
                        scene_png = scene_dir / "scene.png"
                        if scene_png.exists():
                            frames = [scene_png]

                    if not frames:
                        errors += 1
                        continue

                    scenes.append({
                        "scene_id": scene_dir.name,
                        "task": task,
                        "prompt_text": prompt,
                        "assistant_text": assistant_text,
                        "frames": frames,
                    })
                except Exception as e:
                    errors += 1
                    continue

    if max_samples and len(scenes) > max_samples:
        scenes = scenes[:max_samples]

    print(f" Loaded {len(scenes)} scenes (errors: {errors})")
    return scenes


# ── HF R1 data loading ──────────────────────────────────────────────────────

def load_hf_r1_scenes(max_per_task: int | None = None,
                      seed: int = 42) -> list[dict]:
    """Load R1 scenes (ttc/trajectory/stability) from the HF dataset cache."""
    from datasets import load_dataset

    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set - needed for HF dataset")

    print(f" Loading {HF_DATASET} (train split)...")
    ds = load_dataset(HF_DATASET, split="train", token=hf_token, streaming=False)
    print(f" HF dataset: {len(ds)} rows")

    by_task: dict[str, list] = defaultdict(list)
    for row in ds:
        by_task[row["task"]].append(row)

    scenes = []
    for task in sorted(by_task):
        rows = by_task[task]
        cap = min(max_per_task, len(rows)) if max_per_task else len(rows)
        for row in rows[:cap]:
            scenes.append({
                "scene_id": row["scene_id"],
                "task": row["task"],
                "prompt_text": row["prompt"],
                "assistant_text": row["assistant_text"],
                "frames_b64": row["frames_b64"],
                "frames": None,
                "source": "hf_r1",
            })
        print(f" {task}: {cap}/{len(rows)}")

    print(f" HF R1 scenes loaded: {len(scenes)}")
    return scenes


def decode_frames_b64(frames_b64: list[str]) -> list[bytes]:
    """Decode base64 frame strings → compressed JPEG bytes."""
    result = []
    for b64str in frames_b64:
        raw = base64.b64decode(b64str)
        img = Image.open(BytesIO(raw)).convert("RGB")
        result.append(image_bytes_from_pil(img))
    return result


# ── Datum building ────────────────────────────────────────────────────────────

def build_training_datum(tokenizer, scene: dict) -> "tinker.Datum | None":
    """Build a Tinker Datum for cross-entropy SFT. Handles both HF (frames_b64)
    and local (file path) frame sources."""
    try:
        # Load and compress frames from whichever source
        img_bytes_list = []
        if scene.get("frames_b64"):
            img_bytes_list = decode_frames_b64(scene["frames_b64"])
        elif scene.get("frames"):
            for frame_path in scene["frames"]:
                try:
                    img_data = image_bytes_from_pil(
                        Image.open(frame_path).convert("RGB"))
                    img_bytes_list.append(img_data)
                except:
                    pass

        if not img_bytes_list:
            return None

        # Compute image tokens
        img_infos = []
        for img_bytes in img_bytes_list:
            img = Image.open(BytesIO(img_bytes))
            w, h = img.size
            n_tok = compute_qwen3vl_image_tokens(h, w)
            img_infos.append((img_bytes, n_tok))

        # Build chunks + token arrays
        chunks = []
        flat_tokens = []
        flat_is_response = []

        def _add_text(token_ids, is_resp=False):
            chunks.append(types.EncodedTextChunk(tokens=token_ids))
            flat_tokens.extend(token_ids)
            flat_is_response.extend([is_resp] * len(token_ids))

        def _add_image(img_bytes, n_tokens):
            chunks.append(types.ImageChunk(
                data=img_bytes, format="jpeg", expected_tokens=n_tokens))
            flat_tokens.extend([0] * n_tokens)
            flat_is_response.extend([False] * n_tokens)

        # User turn
        prefix_ids = tokenizer.encode(f"{_IM_START_STR}user\n", add_special_tokens=False)
        _add_text(prefix_ids)

        vs_ids = tokenizer.encode(_VISION_START_STR, add_special_tokens=False)
        ve_ids = tokenizer.encode(_VISION_END_STR, add_special_tokens=False)
        for img_bytes, n_tok in img_infos:
            _add_text(vs_ids)
            _add_image(img_bytes, n_tok)
            _add_text(ve_ids)

        suffix_str = (f"{scene['prompt_text']}"
                      f"{_IM_END_STR}\n{_IM_START_STR}assistant\n")
        suffix_ids = tokenizer.encode(suffix_str, add_special_tokens=False)
        _add_text(suffix_ids)

        # Assistant response
        assistant_text = scene["assistant_text"]
        if not assistant_text.endswith(_IM_END_STR):
            assistant_text += _IM_END_STR
        asst_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
        _add_text(asst_ids, is_resp=True)

        # Next-token-prediction shift
        N = len(flat_tokens)
        target_flat = flat_tokens[1:]
        weights_flat = [1.0 if flat_is_response[i] else 0.0
                        for i in range(1, N)]

        last_chunk = chunks[-1]
        if not isinstance(last_chunk, types.EncodedTextChunk) or len(last_chunk.tokens) < 2:
            return None
        chunks[-1] = types.EncodedTextChunk(tokens=list(last_chunk.tokens)[:-1])

        model_input = types.ModelInput(chunks=chunks)

        mi_len = model_input.length
        if mi_len != N - 1:
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
    except Exception as e:
        return None


# ── Metrics & logging ─────────────────────────────────────────────────────────

class MetricsLogger:
    def __init__(self, out_dir: Path, use_wandb: bool,
                 wandb_project: str, run_name: str, config: dict):
        self.out_dir = out_dir
        self.jsonl_path = out_dir / "metrics.jsonl"
        self.use_wandb = use_wandb and _WANDB_AVAILABLE

        if self.use_wandb:
            wandb.init(project=wandb_project, name=run_name, config=config,
                       resume="allow")
            print(f" WandB run: {wandb.run.url}")

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
    def __init__(self, out_dir: Path):
        self.path = out_dir / "checkpoints.jsonl"

    def record(self, step: int, tinker_path: str, tag: str = "", state_path: str = ""):
        record = {"step": step, "tinker_path": tinker_path, "state_path": state_path,
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
    nll_sum = 0.0
    w_sum = 0.0
    for lfo, d in zip(fwd_result.loss_fn_outputs, datums):
        lp = np.array(lfo["logprobs"].data)
        w = np.array(d.loss_fn_inputs["weights"].data)
        nll_sum += float((-lp * w).sum())
        w_sum += float(w.sum())
    return nll_sum / max(w_sum, 1.0)


def compute_val_loss(training_client, tokenizer, val_scenes: list[dict]) -> float:
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
            pass
    return nll_sum / max(w_sum, 1.0) if w_sum > 0 else float("nan")


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    mode = "combined" if args.include_hf else "r2-only"
    run_name = args.run_name or f"sft-{mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_str = "from baseline" if args.from_baseline else "resume from checkpoint"
    print(f"\nPhysSim-VLM SFT on Tinker ({mode}, {baseline_str})")
    print(f" Run : {run_name}")
    print(f" Model : {BASE_MODEL}")
    print(f" Out dir : {out_dir}")

    # Load tokenizer
    print(f"\n Loading tokenizer ({BASE_MODEL})...")
    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    print(f" Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # ── Load scenes ──────────────────────────────────────────────────────────
    tasks_filter = args.tasks.split(",") if args.tasks else None

    # Local R2 data (always loaded)
    r2_scenes = load_sft_r2_scenes(
        max_samples=args.max_samples, seed=args.seed, tasks=tasks_filter)

    if args.include_hf:
        # Filter R1 tasks from local to avoid overlap with HF
        r2_scenes = [s for s in r2_scenes if s["task"] not in R1_TASKS]
        for s in r2_scenes:
            s["source"] = "local_r2"
        # Load HF R1 data
        hf_scenes = load_hf_r1_scenes(
            max_per_task=args.max_per_task, seed=args.seed)
        train_scenes = hf_scenes + r2_scenes
        print(f" Combined: {len(hf_scenes)} HF + {len(r2_scenes)} local "
              f"= {len(train_scenes)}")
    else:
        train_scenes = r2_scenes

    # Parse per-task cap overrides
    task_overrides = {}
    if args.task_cap:
        for pair in args.task_cap.split(","):
            k, v = pair.split("=")
            task_overrides[k.strip()] = int(v.strip())

    # Cap per-task for balanced training
    if args.max_per_task or task_overrides:
        by_task = defaultdict(list)
        for s in train_scenes:
            by_task[s["task"]].append(s)
        capped = []
        for task, scenes_list in sorted(by_task.items()):
            limit = task_overrides.get(task, args.max_per_task or len(scenes_list))
            cap = min(limit, len(scenes_list))
            capped.extend(scenes_list[:cap])
            print(f" {task}: {cap}/{len(scenes_list)}")
        train_scenes = capped
        print(f" Capped total: {len(train_scenes)}")

    # MCQ augmentation (additive - emits letter-format variants alongside originals)
    if args.mcq_frac > 0:
        rng = random.Random(args.seed + 1)
        n_added, n_attempted, by_task_added = 0, 0, defaultdict(int)
        extras = []
        for s in train_scenes:
            if s["task"] not in MCQ_TASKS:
                continue
            if rng.random() >= args.mcq_frac:
                continue
            n_attempted += 1
            wrapped = mcq_wrap_scene(s, rng)
            if wrapped is not None:
                extras.append(wrapped)
                n_added += 1
                by_task_added[s["task"]] += 1
        train_scenes = train_scenes + extras
        print(f" MCQ-wrapped: +{n_added}/{n_attempted} variants "
              f"(frac={args.mcq_frac}) -> {len(train_scenes)} total")
        for t, n in sorted(by_task_added.items()):
            print(f" +{n:4d} {t}")

    if not train_scenes:
        raise RuntimeError(f"No training scenes loaded from {DATA_DIR}")

    # Held-out validation split: stratified per task, removed from train.
    # Takes up to 5 scenes/task (min 1), capped at 50 total, then shuffles rest.
    val_rng = random.Random(args.seed + 7)
    _by_task = defaultdict(list)
    for s in train_scenes:
        _by_task[s["task"]].append(s)
    val_scenes: list[dict] = []
    for task, pool in sorted(_by_task.items()):
        val_rng.shuffle(pool)
        take = min(5, max(1, len(pool)//50))
        val_scenes.extend(pool[:take])
    val_scenes = val_scenes[:50]
    _val_ids = {(s["task"], s["scene_id"], bool(s.get("_mcq_wrapped"))) for s in val_scenes}
    train_scenes = [s for s in train_scenes
                    if (s["task"], s["scene_id"], bool(s.get("_mcq_wrapped"))) not in _val_ids]

    random.Random(args.seed).shuffle(train_scenes)
    print(f" Dataset : {len(train_scenes)} train | {len(val_scenes)} val")
    task_counts = defaultdict(int)
    for s in train_scenes:
        task_counts[s["task"]] += 1
    print(f" Tasks : {dict(task_counts)}")

    # Config snapshot
    config = {
        "base_model": BASE_MODEL,
        "mode": mode,
        "from_baseline": args.from_baseline,
        "include_hf": args.include_hf,
        "resume_checkpoint": None if args.from_baseline else (
            args.resume_ckpt or SFT_R1_CHECKPOINT_STATE),
        "lora_rank": args.lora_rank,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "n_train": len(train_scenes),
        "n_val": len(val_scenes),
        "task_counts": dict(task_counts),
        "task_overrides": task_overrides,
        "save_every": args.save_every,
        "val_every": args.val_every,
        "mcq_frac": args.mcq_frac,
        "run_name": run_name,
        "started_at": datetime.now().isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f" Config -> {out_dir}/config.json")

    # Metrics trackers
    metrics_logger = MetricsLogger(
        out_dir, use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project, run_name=run_name, config=config)
    ckpt_tracker = CheckpointTracker(out_dir)

    # Tinker clients
    print(f"\n Connecting to Tinker...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=args.lora_rank,
    )
    print(f" Training client ready (LoRA rank={args.lora_rank})")

    # Checkpoint handling
    if args.from_baseline:
        print(f"\n Training from baseline (no checkpoint)")
    else:
        ckpt_to_load = args.resume_ckpt if args.resume_ckpt else SFT_R1_CHECKPOINT_STATE
        print(f"\n Loading checkpoint...")
        print(f" Path: {ckpt_to_load}")
        training_client.load_state_with_optimizer(ckpt_to_load).result()
        print(f" [OK] Checkpoint loaded (weights + optimizer state)")

    # Training loop
    n_batches = math.ceil(len(train_scenes) / args.batch_size)
    start_batch = args.start_step
    global_step = args.start_step
    epoch_loss = []
    task_losses = defaultdict(list)
    total_tokens = 0
    total_response_tokens = 0
    step_times = []
    adam_params = types.AdamParams(
        learning_rate=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
    )

    print(f"\n Training: {n_batches} steps | batch={args.batch_size} "
          f"| lr={args.lr} | rank={args.lora_rank}")
    print(f" Save every {args.save_every} steps | Val every {args.val_every} steps\n")

    train_start = time.time()

    for batch_idx in range(start_batch, n_batches):
        t0 = time.time()
        start = batch_idx * args.batch_size
        batch = train_scenes[start: start + args.batch_size]
        batch_tasks = [s["task"] for s in batch]

        datums = [build_training_datum(tokenizer, s) for s in batch]
        valid_mask = [d is not None for d in datums]
        datums = [d for d in datums if d is not None]

        if not datums:
            print(f" [WARN] step {batch_idx+1} - all datums failed, skipping")
            continue

        batch_n_tokens = sum(d.model_input.length for d in datums)
        batch_resp_tokens = sum(
            int(sum(d.loss_fn_inputs["weights"].data)) for d in datums)
        total_tokens += batch_n_tokens
        total_response_tokens += batch_resp_tokens

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
        step_times.append(elapsed)

        for i, (scene, is_valid) in enumerate(zip(batch, valid_mask)):
            if is_valid:
                task_losses[scene["task"]].append(step_loss)

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        avg_step_time = sum(step_times) / len(step_times)
        remaining = n_batches - batch_idx - 1
        eta_s = remaining * avg_step_time
        eta_str = f"{int(eta_s//60)}m{int(eta_s%60):02d}s"

        step_metrics = {
            "train/loss": step_loss,
            "train/loss_avg": avg_loss,
            "train/lr": args.lr,
            "train/batch_size": len(datums),
            "train/step_time_s": elapsed,
            "train/avg_step_time_s": avg_step_time,
            "train/samples_seen": global_step * args.batch_size,
            "train/total_tokens": total_tokens,
            "train/total_response_tokens": total_response_tokens,
            "train/batch_tokens": batch_n_tokens,
            "train/batch_response_tokens": batch_resp_tokens,
        }
        for task_name, losses in task_losses.items():
            step_metrics[f"train/loss_{task_name}"] = (
                sum(losses[-20:]) / len(losses[-20:]))
        metrics_logger.log(step_metrics, step=global_step)

        task_str = ",".join(sorted(set(batch_tasks)))
        print(f" step {global_step:4d}/{n_batches} "
              f"loss={step_loss:.4f} avg={avg_loss:.4f} "
              f"tok={batch_n_tokens} "
              f"({elapsed:.1f}s) ETA={eta_str} "
              f"[{len(datums)}/{len(batch)} ok] {task_str}")

        # Validation
        if global_step % args.val_every == 0 and val_scenes:
            print(f" [VAL] {len(val_scenes)} samples...")
            val_loss = compute_val_loss(training_client, tokenizer, val_scenes)
            print(f" [VAL] step={global_step} val_loss={val_loss:.4f}")
            metrics_logger.log({"val/loss": val_loss}, step=global_step)

        # Checkpoint
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

    # Final checkpoint
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

    # Summary
    final_avg = sum(epoch_loss) / len(epoch_loss) if epoch_loss else float("nan")
    wall_time_s = time.time() - train_start
    per_task_avg = {
        t: sum(losses) / len(losses) for t, losses in task_losses.items() if losses}
    (out_dir / "summary.json").write_text(json.dumps({
        "run_name": run_name,
        "total_steps": global_step,
        "total_samples": len(train_scenes),
        "final_train_loss": final_avg,
        "per_task_avg_loss": per_task_avg,
        "total_tokens_trained": total_tokens,
        "total_response_tokens": total_response_tokens,
        "wall_time_s": round(wall_time_s, 1),
        "avg_step_time_s": round(wall_time_s / max(global_step, 1), 2),
        "finished_at": datetime.now().isoformat(),
    }, indent=2))
    metrics_logger.finish()

    latest = ckpt_tracker.latest()
    print(f"\n Done! steps={global_step} avg_loss={final_avg:.4f}")
    print(f" Wall time : {int(wall_time_s//60)}m{int(wall_time_s%60):02d}s")
    print(f" Total tokens: {total_tokens:,} (response: {total_response_tokens:,})")
    print(f" Per-task loss:")
    for t, l in sorted(per_task_avg.items(), key=lambda x: -x[1]):
        print(f" {t:25s} {l:.4f}")
    print(f" Results : {out_dir}")
    if latest:
        print(f" Checkpoint : {latest['tinker_path']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PhysSim-VLM SFT R2 on Tinker - Resume from SFT R1")

    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap total train samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="PhysSim-VLM")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Resume from a specific tinker:// state checkpoint (overrides SFT R1 default)")
    parser.add_argument("--start-step", type=int, default=0,
                        help="Skip to this batch index when resuming")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names to filter (e.g. fluid_direction,fluid_level)")
    parser.add_argument("--max-per-task", type=int, default=None,
                        help="Cap scenes per task for balanced training")
    parser.add_argument("--mcq-frac", type=float, default=0.0,
                        help="Fraction of categorical scenes to additionally "
                             "emit as MCQ-wrapped variants (0..1). Adds new "
                             "scenes on top of the originals.")
    parser.add_argument("--from-baseline", action="store_true",
                        help="Train from the base model (no checkpoint resume)")
    parser.add_argument("--include-hf", action="store_true",
                        help="Include HF R1 dataset (ttc/trajectory/stability) "
                             "alongside local R2 data")
    parser.add_argument("--task-cap", type=str, default=None,
                        help="Per-task cap overrides, e.g. "
                             "'motion_comparison=1000,counting=100'")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
