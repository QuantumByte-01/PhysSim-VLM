#!/usr/bin/env python3
"""
PhysSim-VLM: LoRA SFT Training (Epoch 1)
===============================================
Fine-tunes Qwen3-VL-30B-A3B-Instruct on the generated physics dataset
using LoRA (rank=64) via PEFT + HuggingFace Trainer.

Hardware: AMD MI300X (205 GB HBM3) · ROCm 7.0.2 · PyTorch 2.9 dev

Usage:
  python scripts/train_lora_sft.py # full run
  python scripts/train_lora_sft.py --smoke_test # 50 steps sanity check
  python scripts/train_lora_sft.py --resume # resume from checkpoint

Output: /app/PhysSim-VLM/checkpoints/lora_sft_epoch1/
"""

import os, sys, json, random, logging, argparse, time, base64, io
from pathlib import Path
from typing import Optional

# ── Must set before any torch/transformers import ────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "osmesa")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import torch
import numpy as np

# ROCm: torch._grouped_mm exists but crashes on AMD GPUs.
# Remove it so transformers falls back to the custom grouped_mm_fallback kernel.
if torch.cuda.is_available() and hasattr(torch.version, "hip"):
    if hasattr(torch, "_grouped_mm"):
        del torch._grouped_mm
    if hasattr(torch.nn.functional, "grouped_mm"):
        del torch.nn.functional.grouped_mm
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    DataCollatorWithPadding,
    ProgressCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# Liger Kernel: fused Triton kernels for RMSNorm, RoPE, SwiGLU, CrossEntropy
# Disabled for Qwen3-VL-MoE: Liger's dtype handling conflicts with the MoE
# router scatter op (RuntimeError: scatter(): Expected self.dtype == src.dtype)
LIGER_AVAILABLE = False
try:
    from liger_kernel.transformers import _apply_liger_kernel_to_instance
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "generated"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "checkpoints" / "lora_sft_epoch1"))
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

TASKS = ["ttc", "stability", "trajectory"]

# LoRA hyperparameters
LORA_RANK = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.05
LORA_TARGET_MODS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training hyperparameters
# batch_size=1 required: Qwen3-VL dynamic resolution means each sample has
# different pixel_values shape (1 frame vs 8 frames → different patch counts).
# Effective batch = 16 via gradient accumulation (maximizing MI300x 191GB VRAM).
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 16 # effective batch = 16 (max GPU utilization)
MAX_SEQ_LEN = 4096 # physics prompts are short; 4096 is sufficient
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
NUM_EPOCHS = 1
SAVE_STEPS = 200
LOGGING_STEPS = 5
DTYPE = torch.bfloat16


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

def format_answer(task: str, gt: dict) -> str:
    if task == "ttc":
        return f"{gt.get('time_to_collision', 0.0):.2f}"
    elif task == "stability":
        return "stable" if gt.get("is_stable", False) else "unstable"
    elif task == "trajectory":
        lp = gt.get("landing_position", {})
        return f"x={lp.get('x', 0.0):.2f}, y={lp.get('y', 0.0):.2f}"
    return ""


def format_reasoning(task: str, gt: dict) -> str:
    if task == "ttc":
        ttc = gt.get("time_to_collision", 0.0)
        vinfo = gt.get("video_info", {})
        dur = vinfo.get("duration_s", 0.0)
        rem = vinfo.get("time_remaining_s", 0.0)
        nf = vinfo.get("n_frames", 8)
        return (
            f"The video covers {dur:.2f}s of motion. "
            f"Across {nf} frames the objects converge steadily. "
            f"~{rem:.2f}s remain after the clip, "
            f"giving total TTC = {ttc:.2f}s from video start."
        )
    elif task == "stability":
        stable = gt.get("is_stable", False)
        disp = gt.get("max_displacement_m", 0.0)
        if stable:
            return (
                f"The stack is balanced; base supports layers above. "
                f"Max displacement = {disp:.4f}m - no collapse. Stable."
            )
        else:
            tc = gt.get("collapse_time_s", 0.0)
            return (
                f"Top-heavy or misaligned. "
                f"Displacement reaches {disp:.4f}m; collapses at {tc:.2f}s. Unstable."
            )
    elif task == "trajectory":
        lp = gt.get("landing_position", {})
        x, y = lp.get("x", 0.0), lp.get("y", 0.0)
        ht = gt.get("max_height_m", 0.0)
        ft = gt.get("flight_time_s", 0.0)
        return (
            f"Parabolic arc observed. Peak height ≈ {ht:.2f}m. "
            f"Flight time ≈ {ft:.2f}s. Lands at x={x:.2f}m, y={y:.2f}m."
        )
    return "Based on visual analysis."


CACHE_DIR = ROOT / "data" / "tokenized"


class PhysicsSceneDataset(Dataset):
    """
    Loads pre-tokenized tensors from CACHE_DIR if available (fast path),
    otherwise falls back to on-the-fly tokenization (slow path).

    Pre-tokenize with: python scripts/pretokenize_dataset.py
    Expected speedup: 5-10× vs on-the-fly.
    """

    def __init__(
        self,
        data_dir: Path,
        processor: AutoProcessor,
        split: str = "train",
        max_samples: Optional[int] = None,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.cache_dir = CACHE_DIR / split
        self.use_cache = self.cache_dir.exists() and any(self.cache_dir.glob("*.pt"))
        self.records = []

        split_file = data_dir.parent / f"{split}.json"
        if split_file.exists():
            with open(split_file) as f:
                index = {item["scene_id"]: item for item in json.load(f)}
        else:
            log.warning(f"No {split}.json found - using all scenes as train")
            index = {}

        for task in TASKS:
            task_dir = data_dir / task
            if not task_dir.exists():
                continue
            for scene_dir in sorted(task_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                if index and scene_dir.name not in index:
                    continue
                gt_path = scene_dir / "ground_truth.json"
                prompt_p = scene_dir / "prompt.txt"
                if not gt_path.exists() or not prompt_p.exists():
                    continue
                self.records.append({
                    "task": task,
                    "scene_dir": scene_dir,
                    "scene_id": scene_dir.name,
                })

        if max_samples:
            random.shuffle(self.records)
            self.records = self.records[:max_samples]

        mode = "pre-tokenized cache" if self.use_cache else "on-the-fly (slow - run pretokenize_dataset.py)"
        log.info(f"Dataset [{split}]: {len(self.records):,} samples [{mode}]")

    def __len__(self):
        return len(self.records)

    def _load_frames(self, scene_dir: Path, task: str) -> list[Image.Image]:
        if task == "stability":
            for candidate in [
                scene_dir / "scene.png",
                scene_dir / "frame_000.png",
                scene_dir / "frames" / "frame_000.png",
                scene_dir / "thumbnail.png",
            ]:
                if candidate.exists():
                    return [Image.open(candidate).convert("RGB")]
            raise FileNotFoundError(f"No image found in {scene_dir}")
        else:
            frames_dir = scene_dir / "frames"
            frame_files = sorted(frames_dir.glob("frame_*.png"))
            return [Image.open(f).convert("RGB") for f in frame_files]

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        task = rec["task"]
        sdir = rec["scene_dir"]
        scene_id = rec["scene_id"]

        # ── Fast path: load pre-tokenized tensors from cache ──────────────
        if self.use_cache:
            cache_path = self.cache_dir / f"{scene_id}.pt"
            if cache_path.exists():
                return torch.load(cache_path, weights_only=True)

        # ── Slow path: on-the-fly tokenization ────────────────────────────
        with open(sdir / "ground_truth.json") as f:
            gt = json.load(f)
        with open(sdir / "prompt.txt") as f:
            prompt_text = f.read().strip()

        frames = self._load_frames(sdir, task)

        answer = format_answer(task, gt)
        reasoning = format_reasoning(task, gt)
        assistant = f"<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>"

        user_content = []
        for _ in frames:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant}]},
        ]

        full_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_messages = [{"role": "user", "content": user_content}]
        prompt_text_only = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize full sequence
        inputs = self.processor(
            text=full_text,
            images=frames if len(frames) > 0 else None,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            max_pixels=401408, # 512*28*28 - cap visual tokens for speed
        )

        # Tokenize prompt-only to find the cutoff
        prompt_inputs = self.processor(
            text=prompt_text_only,
            images=frames if len(frames) > 0 else None,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            max_pixels=401408,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Labels: -100 for prompt tokens, actual ids for assistant tokens
        labels = input_ids.clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Include pixel_values if present (image tokens)
        # Shape: (num_patches, C, pH, pW) - do NOT squeeze, collator cats along dim=0
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]
        if "mm_token_type_ids" in inputs:
            result["mm_token_type_ids"] = inputs["mm_token_type_ids"].squeeze(0)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# HuggingFace Dataset Loader (loads from Swastikr/PhysSim-VLM-Dataset)
# ══════════════════════════════════════════════════════════════════════════════

class HFPhysicsDataset(Dataset):
    """
    Loads physics scenes directly from the HuggingFace dataset
    (Swastikr/PhysSim-VLM-Dataset) - frames stored as base64 strings.
    """

    def __init__(
        self,
        processor: AutoProcessor,
        split: str = "train",
        max_samples: Optional[int] = None,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        from datasets import load_dataset as hf_load_dataset
        self.processor = processor
        self.max_seq_len = max_seq_len

        log.info(f"Loading HF dataset Swastikr/PhysSim-VLM-Dataset [{split}]...")
        hf_ds = hf_load_dataset(
            "Swastikr/PhysSim-VLM-Dataset",
            token=HF_TOKEN,
            split=split,
        )
        if max_samples:
            hf_ds = hf_ds.select(range(min(max_samples, len(hf_ds))))
        self.records = hf_ds
        log.info(f"HF Dataset [{split}]: {len(self.records):,} samples")

    def __len__(self):
        return len(self.records)

    def _decode_frames(self, frames_b64: list) -> list:
        frames = []
        for b64_str in frames_b64:
            img_bytes = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            frames.append(img)
        return frames

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        task = rec["task"]
        prompt_text = rec["prompt"]
        assistant_text = rec["assistant_text"]
        frames = self._decode_frames(rec["frames_b64"])

        user_content = []
        for _ in frames:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]

        full_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_messages = [{"role": "user", "content": user_content}]
        prompt_text_only = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=full_text,
            images=frames if frames else None,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            max_pixels=401408, # 512*28*28 - cap visual tokens for speed
        )

        prompt_inputs = self.processor(
            text=prompt_text_only,
            images=frames if frames else None,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            max_pixels=401408,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]
        if "mm_token_type_ids" in inputs:
            result["mm_token_type_ids"] = inputs["mm_token_type_ids"].squeeze(0)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# Data Collator
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsDataCollator:
    """Pads variable-length sequences in a batch."""

    def __init__(self, pad_token_id: int):
        self.pad_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch.nn.functional as F

        max_len = max(f["input_ids"].shape[0] for f in features)
        batch = {}

        for key in ["input_ids", "attention_mask", "labels"]:
            tensors = []
            for f in features:
                t = f[key]
                pad = max_len - t.shape[0]
                if key == "input_ids":
                    t = F.pad(t, (0, pad), value=self.pad_id)
                elif key == "attention_mask":
                    t = F.pad(t, (0, pad), value=0)
                elif key == "labels":
                    t = F.pad(t, (0, pad), value=-100)
                tensors.append(t)
            batch[key] = torch.stack(tensors)

        # Qwen3-VL: with batch_size=1 there's only one sample per step - 
        # pass pixel_values, image_grid_thw, and mm_token_type_ids through as-is.
        if "pixel_values" in features[0]:
            batch["pixel_values"] = features[0]["pixel_values"]
        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = features[0]["image_grid_thw"]
        if "mm_token_type_ids" in features[0]:
            # pad mm_token_type_ids the same way as input_ids
            max_len = batch["input_ids"].shape[1]
            tensors = []
            for f in features:
                t = f["mm_token_type_ids"]
                pad = max_len - t.shape[0]
                t = F.pad(t, (0, pad), value=0)
                tensors.append(t)
            batch["mm_token_type_ids"] = torch.stack(tensors)

        return batch


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_processor(base_model: str):
    log.info(f"Loading processor: {base_model}")
    processor = AutoProcessor.from_pretrained(
        base_model,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    log.info(f"Loading model: {base_model} (dtype={DTYPE})")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        token=HF_TOKEN,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2", # ROCm flash-attn
    )

    log.info(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

    # Apply Liger Kernel fused ops (RMSNorm, RoPE, SwiGLU, CrossEntropy)
    if LIGER_AVAILABLE:
        try:
            _apply_liger_kernel_to_instance(model=model)
            log.info("✓ Liger Kernel applied - fused RMSNorm/RoPE/SwiGLU/CrossEntropy active")
        except Exception as e:
            log.warning(f"Liger Kernel apply failed (continuing without): {e}")
    else:
        log.warning("Liger Kernel not available - install liger-kernel for ~30% speedup")

    return model, processor


def apply_lora(model):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODS,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Progress bar callback
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsProgressCallback(TrainerCallback):
    """
    Rich tqdm progress bar showing:
      ▸ overall step / total steps
      ▸ epoch progress
      ▸ loss, lr, grad_norm
      ▸ steps/sec + ETA
    Replaces the default Trainer ProgressCallback.
    """

    def __init__(self):
        self._pbar: Optional[tqdm] = None
        self._epoch_pbar: Optional[tqdm] = None
        self._start_time: float = 0.0
        self._last_step: int = 0

    # ── overall bar ──────────────────────────────────────────────────────────
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kw):
        total = state.max_steps
        self._start_time = time.time()
        self._last_step = state.global_step
        self._pbar = tqdm(
            total = total,
            initial = state.global_step,
            desc = "Training",
            unit = "step",
            dynamic_ncols = True,
            colour = "cyan",
            bar_format= (
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} steps "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        )

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kw):
        if self._pbar:
            self._pbar.set_postfix_str("done ✓")
            self._pbar.close()
            self._pbar = None

    # ── per-epoch bar ─────────────────────────────────────────────────────────
    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kw):
        epoch = int(state.epoch or 0) + 1
        steps_per_epoch = max(1, state.max_steps // max(1, args.num_train_epochs))
        self._epoch_pbar = tqdm(
            total = steps_per_epoch,
            desc = f"Epoch {epoch}/{args.num_train_epochs}",
            unit = "step",
            leave = False,
            dynamic_ncols = True,
            colour = "green",
        )

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kw):
        if self._epoch_pbar:
            self._epoch_pbar.close()
            self._epoch_pbar = None

    # ── per-step update ───────────────────────────────────────────────────────
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kw):
        delta = state.global_step - self._last_step
        if delta <= 0:
            return
        self._last_step = state.global_step

        if self._pbar:
            self._pbar.update(delta)

        if self._epoch_pbar:
            self._epoch_pbar.update(delta)

    # ── log metrics ──────────────────────────────────────────────────────────
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kw):
        if not logs or not self._pbar:
            return
        postfix = {}
        if "loss" in logs:
            postfix["loss"] = f"{logs['loss']:.4f}"
        if "learning_rate" in logs:
            postfix["lr"] = f"{logs['learning_rate']:.2e}"
        if "grad_norm" in logs:
            postfix["gnorm"] = f"{logs['grad_norm']:.2f}"
        elapsed = time.time() - self._start_time
        steps_done = state.global_step
        sps = steps_done / elapsed if elapsed > 0 else 0.0
        postfix["s/s"] = f"{sps:.2f}"
        remaining = (state.max_steps - steps_done) / sps if sps > 0 else 0
        h, m = divmod(int(remaining), 3600)
        m, s = divmod(m, 60)
        postfix["ETA"] = f"{h:02d}:{m:02d}:{s:02d}"
        self._pbar.set_postfix(postfix)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model, processor = load_model_and_processor(BASE_MODEL)
    model = apply_lora(model)
    model.enable_input_require_grads()
    # gradient_checkpointing disabled - 205 GB VRAM is sufficient without it

    # Datasets - use HF dataset if local data/generated is not available
    max_train = 50 if args.smoke_test else None
    use_hf = not (DATA_DIR / "ttc").exists()
    if use_hf:
        log.info("Local data not found - loading from HuggingFace dataset (Swastikr/PhysSim-VLM-Dataset)")
        train_ds = HFPhysicsDataset(processor, split="train", max_samples=max_train)
        val_ds = HFPhysicsDataset(processor, split="val", max_samples=20 if args.smoke_test else None)
    else:
        train_ds = PhysicsSceneDataset(DATA_DIR, processor, split="train", max_samples=max_train)
        val_ds = PhysicsSceneDataset(DATA_DIR, processor, split="val", max_samples=20 if args.smoke_test else None)

    # Post-epoch eval callback (vLLM inference on val set)
    import importlib.util, sys as _sys
    _spec = importlib.util.spec_from_file_location(
        "eval_physics_val", ROOT / "scripts" / "eval_physics_val.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    PhysicsEvalCallback = _mod.PhysicsEvalCallback
    load_eval_split = _mod.load_split
    eval_records = load_eval_split("val")
    eval_callback = PhysicsEvalCallback(
        processor = processor,
        val_records = eval_records,
        results_dir = ROOT / "results",
        max_eval_samples = 50 if args.smoke_test else 300,
        epoch_offset = args.epoch_num - 1,
    )

    collator = PhysicsDataCollator(pad_token_id=processor.tokenizer.pad_token_id or 0)

    # Training args
    max_steps = 50 if args.smoke_test else -1
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        fp16=False,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_num_workers=16, # maximize CPU → GPU feed
        dataloader_prefetch_factor=2, # prefetch batches ahead
        report_to=["tensorboard", "wandb"],
        logging_dir=str(OUTPUT_DIR / "logs"),
        ddp_find_unused_parameters=False,
        gradient_checkpointing=False, # 205 GB VRAM - disabled for faster steps (no activation recompute)
        resume_from_checkpoint=args.resume or None,
        seed=42,
        # ROCm / MI300X optimizations
        dataloader_pin_memory=False, # ROCm pinned memory can be unstable
        optim="adamw_torch_fused", # fused AdamW - faster on ROCm
        tf32=False, # not available on ROCm
        torch_compile=False, # disabled: ROCm inductor has 5-10min warmup overhead
    )

    progress_callback = PhysicsProgressCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[eval_callback, progress_callback],
    )
    # Replace default ProgressCallback with ours to avoid duplicate bars
    trainer.remove_callback(ProgressCallback)

    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume or None)

    log.info("Saving final LoRA weights...")
    model.save_pretrained(OUTPUT_DIR / "final")
    processor.save_pretrained(OUTPUT_DIR / "final")

    # Save training summary
    summary = {
        "base_model": BASE_MODEL,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "epochs": NUM_EPOCHS,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "output_dir": str(OUTPUT_DIR),
    }
    with open(OUTPUT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Done. Weights saved to {OUTPUT_DIR}/final/")
    return trainer


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PhysSim-VLM LoRA SFT")
    parser.add_argument("--smoke_test", action="store_true",
                        help="50-step sanity check (no full training)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint dir to resume from")
    parser.add_argument("--epoch_num", type=int, default=1,
                        help="Epoch number label (affects output dir + results dir)")
    args = parser.parse_args()

    # Remap output dir for epoch 2+
    global OUTPUT_DIR
    if args.epoch_num > 1:
        OUTPUT_DIR = ROOT / "checkpoints" / f"lora_sft_epoch{args.epoch_num}"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("PhysSim-VLM · LoRA SFT · Epoch 1")
    log.info(f"Base model : {BASE_MODEL}")
    log.info(f"Output : {OUTPUT_DIR}")
    log.info(f"GPU : {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    log.info(f"LoRA rank : {LORA_RANK} alpha={LORA_ALPHA}")
    log.info(f"Smoke test : {args.smoke_test}")
    log.info("=" * 60)

    train(args)


if __name__ == "__main__":
    main()
