#!/usr/bin/env python3
"""
PhysSim-VLM: GRPO Training on Tinker
============================================
Group Relative Policy Optimization for physics reasoning, using MuJoCo
ground truth as continuous reward signals.

All hyperparameters are loaded from .env.grpo (see that file for docs).
API keys loaded from .env.

Training loop per step:
  1. Build G environments per prompt (PhysicsGroupBuilder)
  2. Sample G completions via Tinker rollout API
  3. Score completions with physics_reward + format_reward
  4. Compute group-relative advantages (optionally AP-GRPO modified)
  5. Forward-backward with importance_sampling loss
  6. Optimizer step

Usage:
  # Full run (reads all config from .env.grpo)
  python scripts/train_grpo_tinker.py

  # Override run name
  python scripts/train_grpo_tinker.py --run-name grpo-smoke-test

  # Smoke test: 100 steps only
  # (set GRPO_MAX_STEPS=100 in .env.grpo, or pass --max-steps 100)
  python scripts/train_grpo_tinker.py --max-steps 100

Output:
  results/grpo_tinker/<run-name>/
    metrics.jsonl per-step metrics (reward, KL, loss, etc.)
    checkpoints.jsonl tinker:// checkpoint paths
    config.json full config snapshot
"""

import os
import sys
import json
import math
import time
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Force unbuffered stdout ─────────────────────────────────────────────────
sys.stdout.reconfigure(line_buffering=True)

# ── Load environment files ──────────────────────────────────────────────────
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env") # API keys
load_dotenv(ROOT / ".env.grpo") # GRPO hyperparameters

# Propagate tokens
if os.environ.get("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
if not os.environ.get("TINKER_API_KEY"):
    raise RuntimeError("TINKER_API_KEY not set. Check .env")

import tinker
from tinker import types
from tinker_cookbook import tokenizer_utils
from datasets import load_dataset

from rewards import RewardConfig, compute_reward, ap_grpo_advantage, parse_answer
from physics_env import (
    PhysicsRLDataset, PhysicsGroupBuilder, PhysicsEnv,
    build_generation_prompt, make_sampling_params, decode_frames_b64,
)

# Optional WandB
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── Directories ─────────────────────────────────────────────────────────────
RESULTS_DIR = ROOT / "results" / "grpo_tinker"


# ── Config from .env.grpo ───────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

def _env_float(key: str, default: float) -> float:
    v = _env(key)
    return float(v) if v else default

def _env_int(key: str, default: int) -> int:
    v = _env(key)
    return int(v) if v else default

def _env_bool(key: str, default: bool) -> bool:
    v = _env(key).lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    return default


def load_grpo_config() -> dict:
    """Load all GRPO hyperparameters from environment into a dict."""
    return {
        # Model
        "base_model": _env("GRPO_BASE_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
        "lora_rank": _env_int("GRPO_LORA_RANK", 64),
        "lora_alpha": _env_int("GRPO_LORA_ALPHA", 32),

        # SFT init
        "sft_checkpoint": _env("GRPO_SFT_CHECKPOINT", ""),

        # Sampling
        "group_size": _env_int("GRPO_GROUP_SIZE", 8),
        "groups_per_batch": _env_int("GRPO_GROUPS_PER_BATCH", 4),
        "temperature": _env_float("GRPO_TEMPERATURE", 0.7),
        "max_tokens": _env_int("GRPO_MAX_TOKENS", 512),

        # Optimizer
        "learning_rate": _env_float("GRPO_LEARNING_RATE", 2e-6),
        "beta1": _env_float("GRPO_BETA1", 0.9),
        "beta2": _env_float("GRPO_BETA2", 0.95),
        "eps": _env_float("GRPO_EPS", 1e-12),
        "weight_decay": _env_float("GRPO_WEIGHT_DECAY", 0.0),
        "grad_clip": _env_float("GRPO_GRAD_CLIP", 1.0),

        # Loss
        "loss_fn": _env("GRPO_LOSS_FN", "importance_sampling"),
        "num_substeps": _env_int("GRPO_NUM_SUBSTEPS", 1),

        # KL
        "kl_penalty_coef": _env_float("GRPO_KL_PENALTY_COEF", 0.0),
        "kl_discount_factor": _env_float("GRPO_KL_DISCOUNT_FACTOR", 0.0),

        # Rewards
        "reward_decay_ttc": _env_float("GRPO_REWARD_DECAY_TTC", 3.0),
        "reward_decay_trajectory": _env_float("GRPO_REWARD_DECAY_TRAJECTORY", 1.0),
        "reward_weight_physics": _env_float("GRPO_REWARD_WEIGHT_PHYSICS", 0.8),
        "reward_weight_format": _env_float("GRPO_REWARD_WEIGHT_FORMAT", 0.2),
        "min_completion_tokens": _env_int("GRPO_MIN_COMPLETION_TOKENS", 20),

        # AP-GRPO
        "ap_enabled": _env_bool("GRPO_AP_ENABLED", True),
        "ap_alpha": _env_float("GRPO_AP_ALPHA", 1.0),

        # SNRA
        "snra_enabled": _env_bool("GRPO_SNRA_ENABLED", False),
        "snra_k_start": _env_float("GRPO_SNRA_K_START", 1.0),
        "snra_k_end": _env_float("GRPO_SNRA_K_END", 100.0),
        "snra_tau": _env_float("GRPO_SNRA_TAU", 0.5),
        "snra_steepness": _env_float("GRPO_SNRA_STEEPNESS", 10.0),

        # Groups
        "remove_constant_groups": _env_bool("GRPO_REMOVE_CONSTANT_GROUPS", True),
        "constant_group_threshold": _env_float("GRPO_CONSTANT_GROUP_THRESHOLD", 0.01),

        # Dr. GRPO (arXiv 2503.20783): remove std from advantage denominator
        "dr_grpo_enabled": _env_bool("GRPO_DR_GRPO_ENABLED", True),

        # DAPO (arXiv 2503.14476): asymmetric clipping + token-level loss
        "clip_epsilon_low": _env_float("GRPO_CLIP_EPSILON_LOW", 0.2),
        "clip_epsilon_high": _env_float("GRPO_CLIP_EPSILON_HIGH", 0.28),
        "token_level_loss": _env_bool("GRPO_TOKEN_LEVEL_LOSS", True),

        # VCRL curriculum (arXiv 2509.19803)
        "vcrl_enabled": _env_bool("GRPO_VCRL_ENABLED", True),
        "vcrl_warmup": _env_int("GRPO_VCRL_WARMUP", 50),
        "vcrl_alpha": _env_float("GRPO_VCRL_ALPHA", 2.0),
        "vcrl_ema_decay": _env_float("GRPO_VCRL_EMA_DECAY", 0.9),

        # SSR replay buffer (VL-Rethinker, arXiv 2504.08837)
        "ssr_enabled": _env_bool("GRPO_SSR_ENABLED", True),
        "ssr_buffer_size": _env_int("GRPO_SSR_BUFFER_SIZE", 500),
        "ssr_inject_count": _env_int("GRPO_SSR_INJECT_COUNT", 2),

        # Overlong penalty (DAPO)
        "overlong_penalty_start": _env_float("GRPO_OVERLONG_PENALTY_START", 0.8),
        "overlong_penalty_max": _env_float("GRPO_OVERLONG_PENALTY_MAX", 0.5),

        # Unlikeliness bonus (arXiv 2506.02355)
        "unlikeliness_bonus": _env_float("GRPO_UNLIKELINESS_BONUS", 0.05),
        "unlikeliness_threshold": _env_float("GRPO_UNLIKELINESS_THRESHOLD", 0.7),

        # Checkpointing
        "save_every": _env_int("GRPO_SAVE_EVERY", 50),
        "eval_every": _env_int("GRPO_EVAL_EVERY", 50),

        # Logging
        "wandb_project": _env("GRPO_WANDB_PROJECT", "PhysSim-VLM"),
        "run_name": _env("GRPO_RUN_NAME", ""),
        "no_wandb": _env_bool("GRPO_NO_WANDB", False),

        # Data
        "hf_dataset": _env("GRPO_HF_DATASET", "Swastikr/PhysSim-VLM-Dataset"),
        "max_samples": _env_int("GRPO_MAX_SAMPLES", 0) or None,
        "seed": _env_int("GRPO_SEED", 42),

        # Steps
        "max_steps": _env_int("GRPO_MAX_STEPS", 0) or None,

        # Per-task sample cap (0 = no cap)
        "max_samples_per_task": _env_int("GRPO_MAX_SAMPLES_PER_TASK", 0),

        # Sampling client refresh
        "refresh_every": _env_int("GRPO_REFRESH_EVERY", 4),

        # Task-adaptive temperature
        "categorical_temp": _env_float("GRPO_CATEGORICAL_TEMP", 1.0),
        "temp_schedule": _env_bool("GRPO_TEMP_SCHEDULE", True),

        # Task-balanced sampling
        "min_categorical_ratio": _env_float("GRPO_MIN_CATEGORICAL_RATIO", 0.4),

        # Soft constant-group handling
        "soft_constant_groups": _env_bool("GRPO_SOFT_CONSTANT_GROUPS", True),
        "soft_constant_advantage": _env_float("GRPO_SOFT_CONSTANT_ADVANTAGE", 0.1),

        # LR schedule
        "lr_warmup_steps": _env_int("GRPO_LR_WARMUP_STEPS", 50),
        "lr_min_ratio": _env_float("GRPO_LR_MIN_RATIO", 0.1),

        # Early stopping
        "early_stop_patience": _env_int("GRPO_EARLY_STOP_PATIENCE", 300),

        # MCQ wrap fraction (0.0 = no MCQ contamination, preserves physics alignment)
        "mcq_wrap_frac": _env_float("GRPO_MCQ_WRAP_FRAC", 0.0),

        # Static image injection fraction (anti image-only collapse)
        "static_image_frac": _env_float("GRPO_STATIC_IMAGE_FRAC", 0.40),
    }


def build_reward_config(cfg: dict) -> RewardConfig:
    """Build RewardConfig from the loaded config dict."""
    return RewardConfig(
        decay_ttc=cfg["reward_decay_ttc"],
        decay_trajectory=cfg["reward_decay_trajectory"],
        weight_physics=cfg["reward_weight_physics"],
        weight_format=cfg["reward_weight_format"],
        min_completion_tokens=cfg["min_completion_tokens"],
        max_tokens=cfg["max_tokens"],
        ap_enabled=cfg["ap_enabled"],
        ap_alpha=cfg["ap_alpha"],
        snra_enabled=cfg["snra_enabled"],
        snra_k_start=cfg["snra_k_start"],
        snra_k_end=cfg["snra_k_end"],
        snra_tau=cfg["snra_tau"],
        snra_steepness=cfg["snra_steepness"],
        overlong_penalty_start=cfg["overlong_penalty_start"],
        overlong_penalty_max=cfg["overlong_penalty_max"],
        unlikeliness_bonus=cfg["unlikeliness_bonus"],
        unlikeliness_threshold=cfg["unlikeliness_threshold"],
    )


# ── Data loading ────────────────────────────────────────────────────────────

def load_scenes(hf_dataset: str, split: str = "train",
                max_samples: int | None = None) -> list[dict]:
    """Load scenes from HuggingFace dataset (MuJoCo: ttc/trajectory/stability)."""
    hf_token = os.environ.get("HF_TOKEN", "")
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

    print(f" Loading {hf_dataset} ({split})...")
    ds = load_dataset(hf_dataset, split=split, token=hf_token, streaming=False)
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    print(f" Loaded {len(ds)} scenes")

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


def load_local_sft_r2_scenes() -> list[dict]:
    """
    Load SFT R2 scenes from local data/sft_r2/ directory.
    Frames are read from disk and base64-encoded for compatibility with PhysicsEnv.
    Includes: fluid_direction, fluid_viscosity, fluid_level, motion_comparison, object_comparison.
    """
    import base64
    sft_r2_dir = ROOT / "data" / "sft_r2"
    if not sft_r2_dir.exists():
        print(f" [WARN] SFT R2 local dir not found: {sft_r2_dir} - skipping")
        return []

    scenes = []
    errors = 0
    for task_dir in sorted(sft_r2_dir.glob("*")):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for scene_dir in sorted(task_dir.glob("*")):
            if not scene_dir.is_dir():
                continue
            try:
                prompt_path = scene_dir / "prompt.txt"
                asst_path = scene_dir / "assistant_text.txt"
                if not prompt_path.exists() or not asst_path.exists():
                    continue

                # Load frames
                frames_dir = scene_dir / "frames"
                if frames_dir.exists():
                    frame_paths = sorted(frames_dir.glob("frame_*.png"))
                else:
                    scene_png = scene_dir / "scene.png"
                    frame_paths = [scene_png] if scene_png.exists() else []

                if not frame_paths:
                    errors += 1
                    continue

                frames_b64 = []
                for fp in frame_paths:
                    frames_b64.append(base64.b64encode(fp.read_bytes()).decode())

                scenes.append({
                    "scene_id": scene_dir.name,
                    "task": task,
                    "prompt_text": prompt_path.read_text().strip(),
                    "assistant_text": asst_path.read_text().strip(),
                    "frames_b64": frames_b64,
                })
            except Exception:
                errors += 1

    print(f" Loaded {len(scenes)} SFT R2 local scenes (errors: {errors})")
    return scenes


def inject_static_image_scenes(scenes: list[dict], target_frac: float = 0.29,
                               seed: int = 42) -> list[dict]:
    """
    Ensure target_frac of scenes are single-frame (static image).
    Prevents image-only regression during GRPO by mixing in static scenes.

    Only categorical tasks (object/scene reasoning) are eligible for truncation.
    Continuous-physics tasks (TTC, trajectory) are NEVER truncated - they
    require motion to be solvable; truncation would zero their reward and
    leak gradient noise into the training loop.
    """
    import numpy as np
    from physics_env import CATEGORICAL_TASKS, CONTINUOUS_TASKS
    rng = np.random.default_rng(seed)

    single_frame = [s for s in scenes if len(s["frames_b64"]) == 1]
    multi_frame_categorical = [s for s in scenes
                               if len(s["frames_b64"]) > 1
                               and s.get("task") in CATEGORICAL_TASKS]
    multi_frame_continuous = [s for s in scenes
                              if len(s["frames_b64"]) > 1
                              and s.get("task") not in CATEGORICAL_TASKS]

    current_static = len(single_frame)
    total = len(scenes)
    target_static = int(total * target_frac)
    need_more = max(0, target_static - current_static)

    if need_more == 0:
        print(f" Static image scenes: {current_static}/{total} "
              f"({current_static/total*100:.0f}%) -- already >= {target_frac*100:.0f}%")
        return scenes

    eligible = multi_frame_categorical
    if not eligible:
        print(f" WARN: target_frac={target_frac} requested but no categorical "
              f"multi-frame scenes available; keeping {current_static}/{total} "
              f"({current_static/total*100:.0f}%) static.")
        return scenes

    take = min(need_more, len(eligible))
    truncate_indices = rng.choice(len(eligible), size=take, replace=False)
    truncated = 0
    for idx in truncate_indices:
        eligible[idx] = dict(eligible[idx])
        eligible[idx]["frames_b64"] = [eligible[idx]["frames_b64"][0]]
        truncated += 1

    result = single_frame + eligible + multi_frame_continuous
    rng.shuffle(result)
    final_static = sum(1 for s in result if len(s["frames_b64"]) == 1)
    print(f" Static image scenes: {final_static}/{len(result)} "
          f"({final_static/len(result)*100:.0f}%) -- truncated {truncated} "
          f"categorical multi-frame scenes (continuous-physics scenes preserved)")
    return result


# ── Logging ─────────────────────────────────────────────────────────────────

class MetricsLogger:
    def __init__(self, out_dir: Path, use_wandb: bool,
                 wandb_project: str, run_name: str, config: dict):
        self.jsonl_path = out_dir / "metrics.jsonl"
        self.use_wandb = use_wandb and _WANDB_AVAILABLE

        if self.use_wandb:
            wandb.init(project=wandb_project, name=run_name,
                       config=config, resume="allow")
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

    def record(self, step: int, sampler_path: str, state_path: str = "",
               tag: str = ""):
        record = {"step": step, "sampler_path": sampler_path,
                  "state_path": state_path, "tag": tag,
                  "ts": datetime.now().isoformat()}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f" [CKPT] step={step} -> {sampler_path}")


# ── VCRL: Variance-based Curriculum Sampler (arXiv 2509.19803) ─────────────

class VCRLSampler:
    """
    Variance-based Curriculum Reinforcement Learning sampler.

    Tracks per-scene reward variance via EMA and samples proportional to
    variance^alpha. Medium-difficulty scenes (some rollouts correct, some not)
    get the highest sampling weight. Easy/hard scenes are downweighted.

    Falls back to uniform sampling during warmup or before variance estimates
    are available.
    """

    def __init__(self, scenes: list[dict], alpha: float = 2.0,
                 ema_decay: float = 0.9, warmup: int = 50):
        self.scenes = scenes
        self.alpha = alpha
        self.ema_decay = ema_decay
        self.warmup = warmup

        # Per-scene running variance estimate (EMA of reward std)
        self.scene_variance = np.zeros(len(scenes))
        self.scene_seen = np.zeros(len(scenes), dtype=bool)

    def update(self, scene_indices: list[int], reward_stds: list[float]):
        """Update variance estimates for scenes used in last step."""
        for idx, std in zip(scene_indices, reward_stds):
            if not self.scene_seen[idx]:
                self.scene_variance[idx] = std
                self.scene_seen[idx] = True
            else:
                self.scene_variance[idx] = (
                    self.ema_decay * self.scene_variance[idx]
                    + (1 - self.ema_decay) * std
                )

    def sample_batch(self, batch_size: int, step: int,
                     rng: np.random.Generator,
                     min_categorical: int = 0,
                     categorical_indices: list[int] | None = None
                     ) -> list[int]:
        """
        Sample scene indices weighted by variance^alpha.

        With task-balanced sampling: ensures at least `min_categorical` scenes
        come from categorical tasks (fluid, comparison, etc.).
        """
        if step < self.warmup or not self.scene_seen.any():
            # During warmup: balanced uniform sampling
            if min_categorical > 0 and categorical_indices:
                cont_indices = [i for i in range(len(self.scenes))
                                if i not in set(categorical_indices)]
                n_cat = min(min_categorical, len(categorical_indices))
                n_cont = batch_size - n_cat
                cats = rng.choice(categorical_indices, size=n_cat,
                                  replace=False).tolist()
                conts = rng.choice(cont_indices,
                                   size=min(n_cont, len(cont_indices)),
                                   replace=False).tolist()
                result = cats + conts
                rng.shuffle(result)
                return result
            return rng.choice(len(self.scenes), size=batch_size,
                              replace=False).tolist()

        # Compute sampling weights: variance^alpha (unseen scenes get median)
        weights = self.scene_variance.copy()
        median_var = np.median(weights[self.scene_seen])
        weights[~self.scene_seen] = median_var
        weights = np.maximum(weights, 1e-8) ** self.alpha

        # Task-balanced VCRL: sample categorical and continuous separately
        if min_categorical > 0 and categorical_indices:
            cat_set = set(categorical_indices)
            cont_indices = [i for i in range(len(self.scenes)) if i not in cat_set]

            # Sample categorical
            n_cat = min(min_categorical, len(categorical_indices))
            cat_w = weights[categorical_indices]
            cat_p = cat_w / cat_w.sum()
            cats = rng.choice(categorical_indices, size=n_cat,
                              replace=False, p=cat_p).tolist()

            # Sample continuous
            n_cont = batch_size - n_cat
            cont_w = weights[cont_indices]
            cont_p = cont_w / cont_w.sum()
            conts = rng.choice(cont_indices,
                               size=min(n_cont, len(cont_indices)),
                               replace=False, p=cont_p).tolist()

            result = cats + conts
            rng.shuffle(result)
            return result

        # Normalize to probability distribution
        probs = weights / weights.sum()
        return rng.choice(len(self.scenes), size=min(batch_size, len(self.scenes)),
                          replace=False, p=probs).tolist()


# ── SSR: Selective Sample Replay (VL-Rethinker, arXiv 2504.08837) ──────────

class SSRBuffer:
    """
    Selective Sample Replay buffer for vanishing advantages.

    Stores high-|advantage| (prompt, response_tokens, advantage, reward) tuples.
    When the current batch has low signal (constant-reward groups skipped),
    inject replay samples to maintain effective batch size.
    """

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.buffer: list[dict] = [] # sorted by |advantage| descending

    def add(self, entries: list[dict]):
        """Add entries: each has 'prompt', 'tokens', 'advantage', 'reward'."""
        self.buffer.extend(entries)
        # Keep top-K by |advantage|
        self.buffer.sort(key=lambda x: abs(x["advantage"]), reverse=True)
        self.buffer = self.buffer[:self.max_size]

    def sample(self, count: int,
               rng: np.random.Generator) -> list[dict]:
        """Sample `count` entries from the buffer."""
        if not self.buffer or count <= 0:
            return []
        indices = rng.choice(len(self.buffer),
                             size=min(count, len(self.buffer)),
                             replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


# ── LR Schedule ────────────────────────────────────────────────────────────

def _get_scheduled_lr(base_lr: float, step: int, total_steps: int,
                      warmup_steps: int = 50,
                      min_lr_ratio: float = 0.1) -> float:
    """
    Linear warmup for `warmup_steps`, then cosine decay to base_lr * min_lr_ratio.
    """
    if step <= warmup_steps:
        # Linear warmup: 0 -> base_lr
        return base_lr * (step / max(warmup_steps, 1))

    # Cosine decay after warmup
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine_factor


# ── GRPO Training Step ──────────────────────────────────────────────────────

def grpo_step(training_client, sampling_client, tokenizer,
              group_builders: list[PhysicsGroupBuilder],
              reward_cfg: RewardConfig, cfg: dict,
              global_step: int, total_steps: int) -> dict:
    """
    Execute one GRPO training step:
      1. Generate G rollouts per prompt using provided sampling_client
      2. Score each with reward function
      3. Compute group-relative advantages
      4. Build training datums with advantage weights
      5. Forward-backward + optim step

    Returns dict of step metrics.
    """
    all_rewards = []
    all_advantages = []
    all_metrics = defaultdict(list)
    training_datums = []
    pending_responses = [] # staged for token-level loss normalization
    group_reward_stds = [] # per-group std for VCRL variance tracking
    ssr_entries = [] # high-advantage samples for SSR buffer
    skipped_groups = 0

    # ── 1. Fire all rollout requests in parallel ─────────────────────
    # Each sample() returns a future - fire all first, then collect.
    # This overlaps network + GPU time across groups for ~2-3x speedup.
    envs = []
    futures = []
    for builder in group_builders:
        env = builder.make_env()
        envs.append(env)
        try:
            prompt, sampling_params = env.initial_observation()
            future = sampling_client.sample(
                prompt=prompt,
                num_samples=builder.group_size,
                sampling_params=sampling_params,
            )
            futures.append((future, prompt, builder))
        except Exception as e:
            print(f" [WARN] Prompt build failed ({builder.scene['scene_id']}): {e}")
            futures.append((None, None, builder))

    # ── 1b. Collect results ────────────────────────────────────────────
    for (future, prompt, builder), env in zip(futures, envs):
        group_rewards = []
        group_responses = []

        try:
            if future is None:
                raise RuntimeError("prompt build failed")
            sample_result = future.result()

            for seq in sample_result.sequences:
                response_tokens = list(seq.tokens)
                result = env.step(response_tokens)
                group_rewards.append(result["reward"])
                group_responses.append({
                    "tokens": response_tokens,
                    "prompt": prompt,
                    "metrics": result["metrics"],
                })

        except Exception as e:
            print(f" [WARN] Rollout failed ({builder.scene['scene_id']}): {e}")
            # Pad with zero rewards so group is skipped cleanly
            group_rewards = [0.0] * builder.group_size
            group_responses = [None] * builder.group_size

        # ── 2. Handle constant reward groups ───────────────────────────
        r_arr = np.array(group_rewards)
        group_std = float(r_arr.std())
        group_reward_stds.append(group_std) # always track for VCRL

        threshold = cfg.get("constant_group_threshold", 0.01)
        is_constant = group_std <= threshold

        if is_constant and cfg["remove_constant_groups"]:
            # Soft handling: instead of skipping, assign small directional
            # advantage based on whether the group was all-correct or all-wrong.
            # This reinforces correct behavior and penalizes incorrect behavior
            # even when all rollouts agree.
            soft_constant = cfg.get("soft_constant_groups", True)
            mean_r = float(r_arr.mean())
            if soft_constant and any(r is not None for r in group_responses):
                soft_adv = cfg.get("soft_constant_advantage", 0.1)
                if mean_r >= 0.7:
                    # All correct -> small positive reinforcement
                    advantages = [soft_adv] * len(group_rewards)
                elif mean_r <= 0.2:
                    # All wrong -> small negative signal
                    advantages = [-soft_adv] * len(group_rewards)
                else:
                    # Middle ground -> skip (truly uninformative)
                    all_metrics["skipped_constant_groups"].append(1)
                    skipped_groups += 1
                    continue
                all_rewards.extend(group_rewards)
                all_advantages.extend(advantages)
                for resp in group_responses:
                    if resp and resp["metrics"]:
                        for k, v in resp["metrics"].items():
                            if isinstance(v, (int, float)):
                                all_metrics[k].append(v)
                for resp, adv, r in zip(group_responses, advantages, group_rewards):
                    if resp is not None:
                        pending_responses.append((resp, adv))
                all_metrics["soft_constant_groups"].append(1)
                continue
            else:
                all_metrics["skipped_constant_groups"].append(1)
                skipped_groups += 1
                continue

        # ── 3. Compute advantages (Dr. GRPO: no std normalization) ──────
        mean_r = float(r_arr.mean())
        if cfg.get("dr_grpo_enabled", True):
            # Dr. GRPO: A = r - mean (stable, unbiased)
            advantages = [r - mean_r for r in group_rewards]
        else:
            # Standard GRPO: A = (r - mean) / std
            std_r = float(max(r_arr.std(), 1e-6))
            advantages = [(r - mean_r) / std_r for r in group_rewards]

        # AP-GRPO modification
        advantages = ap_grpo_advantage(advantages, group_rewards, reward_cfg)

        all_rewards.extend(group_rewards)
        all_advantages.extend(advantages)

        # Collect per-group metrics
        for resp in group_responses:
            if resp and resp["metrics"]:
                for k, v in resp["metrics"].items():
                    if isinstance(v, (int, float)):
                        all_metrics[k].append(v)

        # Stage for datum building (token-level normalization applied below)
        for resp, adv, r in zip(group_responses, advantages, group_rewards):
            if resp is not None:
                pending_responses.append((resp, adv))
                # Collect high-advantage entries for SSR buffer
                if abs(adv) > 0.01:
                    ssr_entries.append({
                        "prompt": resp["prompt"],
                        "tokens": resp["tokens"],
                        "advantage": adv,
                        "reward": r,
                    })

    # ── 3.5. SSR replay injection when groups are skipped ────────────────
    # If many groups were skipped (constant rewards), inject replay samples
    # to maintain effective batch size and prevent training stall.
    ssr_injected = 0
    if (skipped_groups > 0 and cfg.get("ssr_enabled", False)
            and "_ssr_buffer" in cfg):
        ssr_buf = cfg["_ssr_buffer"]
        inject_count = min(cfg.get("ssr_inject_count", 2), len(ssr_buf))
        if inject_count > 0:
            replay_rng = np.random.default_rng(global_step)
            replays = ssr_buf.sample(inject_count, replay_rng)
            for replay in replays:
                pending_responses.append((
                    {"tokens": replay["tokens"], "prompt": replay["prompt"],
                     "metrics": {}},
                    replay["advantage"],
                ))
                ssr_injected += 1

    # ── 4. Build training datums (DAPO token-level loss if enabled) ──────
    # Token-level: weight per token = advantage / total_response_tokens
    # → all tokens across the batch weighted equally regardless of seq length
    token_level = cfg.get("token_level_loss", True)
    total_resp_tokens = max(
        sum(len(resp["tokens"]) for resp, _ in pending_responses), 1)

    for resp, adv in pending_responses:
        effective_adv = adv / total_resp_tokens if token_level else adv
        try:
            datum = _build_grpo_datum(
                tokenizer=tokenizer,
                prompt=resp["prompt"],
                response_tokens=resp["tokens"],
                advantage=effective_adv,
            )
            if datum is not None:
                training_datums.append(datum)
        except Exception as e:
            print(f" [WARN] Datum build failed: {e}")

    if not training_datums:
        return {"error": "no_datums", "reward_mean": 0.0}

    # ── 5. Forward-backward + optimizer step ────────────────────────────
    # LR schedule: linear warmup -> cosine decay
    scheduled_lr = _get_scheduled_lr(
        cfg["learning_rate"], global_step, total_steps,
        warmup_steps=cfg.get("lr_warmup_steps", 50),
        min_lr_ratio=cfg.get("lr_min_ratio", 0.1),
    )
    adam_params = types.AdamParams(
        learning_rate=scheduled_lr,
        beta1=cfg["beta1"],
        beta2=cfg["beta2"],
        eps=cfg["eps"],
        weight_decay=cfg["weight_decay"],
        grad_clip_norm=cfg["grad_clip"],
    )

    try:
        fwd_bwd = training_client.forward_backward(
            data=training_datums, loss_fn=cfg["loss_fn"])
        optim = training_client.optim_step(adam_params=adam_params)
        fwd_out = fwd_bwd.result()
        optim.result()
    except Exception as e:
        return {"error": str(e), "reward_mean": 0.0}

    # ── Aggregate metrics ───────────────────────────────────────────────
    step_metrics = {
        "grpo/reward_mean": np.mean(all_rewards) if all_rewards else 0.0,
        "grpo/reward_std": np.std(all_rewards) if all_rewards else 0.0,
        "grpo/reward_min": min(all_rewards) if all_rewards else 0.0,
        "grpo/reward_max": max(all_rewards) if all_rewards else 0.0,
        "grpo/advantage_std": np.std(all_advantages) if all_advantages else 0.0,
        "grpo/n_datums": len(training_datums),
        "grpo/n_groups": len(group_builders),
        "grpo/n_rollouts": len(all_rewards),
        "grpo/skipped_groups": skipped_groups,
        "grpo/soft_constant_groups": len(all_metrics.get("soft_constant_groups", [])),
        "grpo/ssr_injected": ssr_injected,
        "grpo/learning_rate": scheduled_lr,
    }

    # Per-metric averages
    for k, vals in all_metrics.items():
        if vals and isinstance(vals[0], (int, float)):
            step_metrics[f"grpo/{k}_mean"] = np.mean(vals)

    # Total tokens generated this step (for cost tracking)
    n_tokens_list = all_metrics.get("n_tokens", [])
    step_metrics["grpo/tokens_step"] = int(sum(n_tokens_list))

    # Internal fields consumed by training loop (not logged to wandb)
    step_metrics["_group_reward_stds"] = group_reward_stds
    step_metrics["_ssr_entries"] = ssr_entries

    return step_metrics


def _build_grpo_datum(tokenizer, prompt: types.ModelInput,
                      response_tokens: list[int],
                      advantage: float) -> "tinker.Datum | None":
    """
    Build a Tinker Datum for GRPO importance sampling loss.

    The datum contains:
      - model_input: full sequence (prompt + response)[:-1]
      - target_tokens: full sequence[1:]
      - weights: advantage value for response positions, 0 for prompt positions
    """
    # Get prompt length
    prompt_len = prompt.length

    # Build full sequence chunks: prompt chunks + response text
    chunks = list(prompt.chunks)
    chunks.append(types.EncodedTextChunk(tokens=response_tokens))

    full_input = types.ModelInput(chunks=chunks)
    full_len = full_input.length

    if full_len < 2:
        return None

    # Build target tokens and weights
    # We need: model_input = full[:-1], targets = full[1:], weights = advantage for response
    # For the flat token array, we approximate by setting weights based on position

    # Response positions start at prompt_len
    weights = []
    for i in range(1, full_len):
        if i >= prompt_len:
            # Response position: weight = advantage (can be negative for low-reward completions)
            weights.append(advantage)
        else:
            weights.append(0.0)

    # Build token ID array for targets (shifted by 1)
    # We need to extract all token IDs from the ModelInput
    flat_tokens = []
    for chunk in chunks:
        if isinstance(chunk, types.EncodedTextChunk):
            flat_tokens.extend(chunk.tokens)
        elif isinstance(chunk, types.ImageChunk):
            flat_tokens.extend([0] * chunk.expected_tokens)

    target_tokens = flat_tokens[1:] # shifted by 1

    # Remove last token from model input
    last_chunk = chunks[-1]
    if isinstance(last_chunk, types.EncodedTextChunk) and len(last_chunk.tokens) >= 2:
        chunks[-1] = types.EncodedTextChunk(tokens=list(last_chunk.tokens)[:-1])
    else:
        return None

    model_input = types.ModelInput(chunks=chunks)

    if model_input.length != len(target_tokens):
        return None

    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=[int(x) for x in target_tokens],
                dtype="int64",
                shape=[len(target_tokens)],
            ),
            "weights": tinker.TensorData(
                data=weights,
                dtype="float32",
                shape=[len(weights)],
            ),
        },
    )


# ── Main ────────────────────────────────────────────────────────────────────

def train(args):
    cfg = load_grpo_config()

    # CLI overrides
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.max_steps:
        cfg["max_steps"] = args.max_steps

    run_name = cfg["run_name"] or f"grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    reward_cfg = build_reward_config(cfg)

    print(f"\nPhysSim-VLM GRPO on Tinker")
    print(f" Run : {run_name}")
    print(f" Model : {cfg['base_model']}")
    print(f" Group size : G={cfg['group_size']}")
    print(f" Groups/batch: {cfg['groups_per_batch']}")
    print(f" Samples/step: {cfg['group_size'] * cfg['groups_per_batch']}")
    print(f" LR : {cfg['learning_rate']}")
    print(f" Temperature: {cfg['temperature']}")
    print(f" Loss fn : {cfg['loss_fn']}")
    print(f" KL penalty : {cfg['kl_penalty_coef']}")
    print(f" AP-GRPO : {cfg['ap_enabled']} (alpha={cfg['ap_alpha']})")
    print(f" SNRA : {cfg['snra_enabled']} (k: {cfg['snra_k_start']}->{cfg['snra_k_end']})")
    print(f" Dr. GRPO : {cfg['dr_grpo_enabled']} (no std normalization)")
    print(f" Token-lvl : {cfg['token_level_loss']} (DAPO token-level loss)")
    print(f" Clip eps : [{cfg['clip_epsilon_low']}, {cfg['clip_epsilon_high']}] (asymmetric)")
    print(f" VCRL : {cfg['vcrl_enabled']} (alpha={cfg['vcrl_alpha']}, warmup={cfg['vcrl_warmup']})")
    print(f" SSR : {cfg['ssr_enabled']} (buffer={cfg['ssr_buffer_size']})")
    print(f" Overlong : start={cfg['overlong_penalty_start']}, max={cfg['overlong_penalty_max']}")
    print(f" Cat. temp : {cfg['categorical_temp']} (schedule={cfg['temp_schedule']})")
    print(f" Balanced : min_cat_ratio={cfg['min_categorical_ratio']}")
    print(f" Soft const : {cfg['soft_constant_groups']} (adv={cfg['soft_constant_advantage']})")
    print(f" SFT ckpt : {cfg['sft_checkpoint'] or '(none - base model)'}")
    print(f" Out dir : {out_dir}")

    # ── Save config snapshot ────────────────────────────────────────────
    cfg["started_at"] = datetime.now().isoformat()
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    # ── Load tokenizer ──────────────────────────────────────────────────
    print(f"\n Loading tokenizer...")
    tokenizer = tokenizer_utils.get_tokenizer(cfg["base_model"])
    print(f" Tokenizer ready (vocab={tokenizer.vocab_size})")

    # ── Load dataset ────────────────────────────────────────────────────
    train_scenes = load_scenes(
        cfg["hf_dataset"], split="train", max_samples=cfg["max_samples"])

    # Also load SFT R2 local scenes (fluid + motion/object comparison) for replay
    sft_r2_scenes = load_local_sft_r2_scenes()
    if sft_r2_scenes:
        train_scenes = train_scenes + sft_r2_scenes
        print(f" Combined: {len(train_scenes)} total scenes (MuJoCo + SFT R2)")

    # ── Balance dataset: cap each task to max_samples_per_task ────────
    max_per_task = cfg.get("max_samples_per_task", 0)
    if max_per_task and max_per_task > 0:
        from collections import defaultdict as _dd
        by_task = _dd(list)
        for s in train_scenes:
            by_task[s["task"]].append(s)
        balanced = []
        for task, scenes_list in sorted(by_task.items()):
            cap = min(max_per_task, len(scenes_list))
            balanced.extend(scenes_list[:cap])
            print(f" {task}: {cap} (of {len(scenes_list)})")
        train_scenes = balanced
        print(f" Balanced to {max_per_task}/task -> {len(train_scenes)} total")

    # ── Inject static image scenes (prevent image-only regression) ───
    train_scenes = inject_static_image_scenes(
        train_scenes, target_frac=cfg["static_image_frac"], seed=cfg["seed"])

    random.Random(cfg["seed"]).shuffle(train_scenes)

    task_counts = defaultdict(int)
    for s in train_scenes:
        task_counts[s["task"]] += 1
    print(f" Dataset : {len(train_scenes)} train scenes")
    print(f" Tasks : {dict(task_counts)}")

    # ── MCQ wrapping: configurable, default 0.0 (no MCQ contamination) ──
    from physics_env import mcq_wrap_scene, CATEGORICAL_TASKS, CONTINUOUS_TASKS
    import numpy as np
    mcq_frac = cfg["mcq_wrap_frac"]
    if mcq_frac > 0.0:
        mcq_rng = np.random.default_rng(cfg["seed"] + 777)
        mcq_wrapped = 0
        for i, s in enumerate(train_scenes):
            if s["task"] in CATEGORICAL_TASKS and mcq_rng.random() < mcq_frac:
                wrapped = mcq_wrap_scene(s, rng=mcq_rng)
                if wrapped.get("mcq_wrapped"):
                    train_scenes[i] = wrapped
                    mcq_wrapped += 1
        print(f" MCQ wrapped: {mcq_wrapped}/{len(train_scenes)} scenes (frac={mcq_frac})")
    else:
        print(f" MCQ wrapping DISABLED (GRPO_MCQ_WRAP_FRAC=0.0) - "
              f"physics-pure free-text RL preserved.")

    # ── Build task-type index for balanced sampling ─────────────────────
    categorical_scene_indices = [
        i for i, s in enumerate(train_scenes) if s["task"] in CATEGORICAL_TASKS
    ]
    continuous_scene_indices = [
        i for i, s in enumerate(train_scenes) if s["task"] in CONTINUOUS_TASKS
    ]
    print(f" Categorical: {len(categorical_scene_indices)} scenes, "
          f"Continuous: {len(continuous_scene_indices)} scenes")

    # ── Build RL dataset ────────────────────────────────────────────────
    rl_dataset = PhysicsRLDataset(
        scenes=train_scenes,
        group_size=cfg["group_size"],
        groups_per_batch=cfg["groups_per_batch"],
        tokenizer=tokenizer,
        reward_cfg=reward_cfg,
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
    )

    total_steps = cfg["max_steps"] or len(rl_dataset)
    print(f" Total steps: {total_steps} "
          f"({rl_dataset.samples_per_step} samples/step)")

    # ── Metrics & checkpoints ───────────────────────────────────────────
    metrics_logger = MetricsLogger(
        out_dir, use_wandb=not cfg["no_wandb"],
        wandb_project=cfg["wandb_project"], run_name=run_name, config=cfg)
    ckpt_tracker = CheckpointTracker(out_dir)

    # ── Tinker clients ──────────────────────────────────────────────────
    print(f"\n Connecting to Tinker...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=cfg["base_model"],
        rank=cfg["lora_rank"],
    )
    print(f" Training client ready (LoRA rank={cfg['lora_rank']})")

    # Load checkpoint - either resume from a GRPO mid-run or start from SFT
    resume_ckpt = args.resume_ckpt if args.resume_ckpt else cfg["sft_checkpoint"]
    if resume_ckpt:
        print(f" Loading checkpoint: {resume_ckpt}")
        training_client.load_state_with_optimizer(resume_ckpt).result()
        print(f" Checkpoint loaded")

    # ── VCRL curriculum sampler ─────────────────────────────────────────
    vcrl = None
    if cfg["vcrl_enabled"]:
        vcrl = VCRLSampler(
            scenes=train_scenes,
            alpha=cfg["vcrl_alpha"],
            ema_decay=cfg["vcrl_ema_decay"],
            warmup=cfg["vcrl_warmup"],
        )
        print(f" VCRL sampler initialized ({len(train_scenes)} scenes)")

    # ── SSR replay buffer ───────────────────────────────────────────────
    ssr = None
    if cfg["ssr_enabled"]:
        ssr = SSRBuffer(max_size=cfg["ssr_buffer_size"])
        print(f" SSR buffer initialized (max={cfg['ssr_buffer_size']})")

    rng = np.random.default_rng(cfg["seed"])

    # ── Load PhysBench val scenes for inline eval ────────────────────────
    val_scenes_pb = None
    val_eval_every = cfg.get("eval_every", 200)
    try:
        from datasets import load_dataset as _ld
        pb_ds = _ld("Swastikr/PhysSim-VLM-Dataset", split="train",
                     token=os.environ.get("HF_TOKEN", ""), streaming=False)
        # Use first 50 scenes as quick val proxy
        val_scenes_pb = [dict(pb_ds[i]) for i in range(min(50, len(pb_ds)))]
        for v in val_scenes_pb:
            v["prompt_text"] = v["prompt"]
        print(f" Quick-val set: {len(val_scenes_pb)} samples (inline every {val_eval_every} steps)")
    except Exception as e:
        print(f" [WARN] Could not load val set for inline eval: {e}")

    # ── Training loop ───────────────────────────────────────────────────
    start_step = args.start_step
    print(f"\n{'='*60}")
    print(f" Starting GRPO training - {total_steps} steps (from step {start_step})")
    print(f" Sampling client refresh every {cfg.get('refresh_every', 4)} steps")
    print(f"{'='*60}\n")

    reward_history = []
    total_tokens_generated = 0
    sampling_client = None
    refresh_every = cfg.get("refresh_every", 4)
    best_avg_reward = -1.0
    best_avg_step = 0
    early_stop_patience = cfg.get("early_stop_patience", 300)

    for step_idx in range(start_step, total_steps):
        t0 = time.time()
        global_step = step_idx + 1

        # ── Refresh sampling client every N steps (saves ~15s/step) ─────
        if sampling_client is None or (global_step - 1) % refresh_every == 0:
            try:
                sampling_client = training_client.save_weights_and_get_sampling_client()
            except Exception as e:
                print(f" [WARN] Sampling client refresh failed: {e}")
                continue

        # ── Select scenes via VCRL curriculum or uniform ────────────
        # With task-balanced sampling: ensure min_categorical_ratio of batch
        # is categorical tasks (fluid, comparison, etc.)
        min_cat_ratio = cfg.get("min_categorical_ratio", 0.4)
        gpb = cfg["groups_per_batch"]
        min_cat_count = max(1, int(gpb * min_cat_ratio))

        if vcrl is not None:
            scene_indices = vcrl.sample_batch(
                gpb, step=global_step, rng=rng,
                min_categorical=min_cat_count,
                categorical_indices=categorical_scene_indices)
            batch_scenes = [train_scenes[i] for i in scene_indices]
        else:
            # Fall back to balanced random sampling
            n_cat = min(min_cat_count, len(categorical_scene_indices))
            n_cont = gpb - n_cat
            cat_pick = rng.choice(categorical_scene_indices, size=n_cat,
                                  replace=False).tolist()
            cont_pick = rng.choice(continuous_scene_indices,
                                   size=min(n_cont, len(continuous_scene_indices)),
                                   replace=False).tolist()
            scene_indices = cat_pick + cont_pick
            rng.shuffle(scene_indices)
            batch_scenes = [train_scenes[i] for i in scene_indices]

        # Build group builders from selected scenes
        cat_temp = cfg.get("categorical_temp", 1.0)
        do_temp_sched = cfg.get("temp_schedule", True)
        builders = [
            PhysicsGroupBuilder(
                scene=scene,
                group_size=cfg["group_size"],
                tokenizer=tokenizer,
                reward_cfg=reward_cfg,
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                training_step=global_step,
                total_steps=total_steps,
                categorical_temp=cat_temp,
                temp_schedule=do_temp_sched,
            )
            for scene in batch_scenes
        ]

        # Pass SSR buffer into cfg so grpo_step can inject replays
        step_cfg = dict(cfg)
        if ssr is not None:
            step_cfg["_ssr_buffer"] = ssr

        # Execute GRPO step
        step_metrics = grpo_step(
            training_client=training_client,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            group_builders=builders,
            reward_cfg=reward_cfg,
            cfg=step_cfg,
            global_step=global_step,
            total_steps=total_steps,
        )

        elapsed = time.time() - t0

        # ── Update VCRL variance estimates ──────────────────────────
        if vcrl is not None and scene_indices is not None:
            # Extract per-group reward std from metrics
            reward_stds = step_metrics.pop("_group_reward_stds", [])
            if reward_stds and len(reward_stds) == len(scene_indices):
                vcrl.update(scene_indices, reward_stds)
        else:
            step_metrics.pop("_group_reward_stds", None)

        # ── Feed SSR buffer with high-advantage samples ─────────────
        if ssr is not None:
            ssr_new = step_metrics.pop("_ssr_entries", [])
            if ssr_new:
                ssr.add(ssr_new)
        else:
            step_metrics.pop("_ssr_entries", None)

        # Track reward history + early stopping
        reward_mean = step_metrics.get("grpo/reward_mean", 0.0)
        reward_history.append(reward_mean)
        avg_reward = sum(reward_history[-50:]) / len(reward_history[-50:])

        # Early stopping: track best 50-step avg reward
        if avg_reward > best_avg_reward + 0.005: # min improvement threshold
            best_avg_reward = avg_reward
            best_avg_step = global_step
        elif (global_step - best_avg_step >= early_stop_patience
              and global_step > 100):
            print(f"\n [EARLY STOP] No improvement for {early_stop_patience} steps "
                  f"(best avg={best_avg_reward:.4f} at step {best_avg_step})")
            print(f" Saving final checkpoint before stopping...")
            try:
                sampler_f = training_client.save_weights_for_sampler("early_stop")
                state_f = training_client.save_state("early_stop")
                ckpt_tracker.record(global_step, sampler_f.result().path,
                                    state_path=state_f.result().path,
                                    tag="early_stop")
            except Exception as e:
                print(f" [WARN] Early stop checkpoint: {e}")
            break

        # Cumulative token tracking
        total_tokens_generated += step_metrics.get("grpo/tokens_step", 0)
        step_metrics["grpo/tokens_total"] = total_tokens_generated

        step_metrics["grpo/step_time_s"] = elapsed
        step_metrics["grpo/reward_avg_50"] = avg_reward
        if vcrl is not None:
            step_metrics["grpo/vcrl_seen_frac"] = float(vcrl.scene_seen.mean())
        if ssr is not None:
            step_metrics["grpo/ssr_buffer_size"] = len(ssr)

        metrics_logger.log(step_metrics, step=global_step)

        # Print progress
        error = step_metrics.get("error", "")
        if error:
            print(f" step {global_step:4d}/{total_steps} ERROR: {error}")
        else:
            adv_std = step_metrics.get("grpo/advantage_std", 0.0)
            n_datums = step_metrics.get("grpo/n_datums", 0)
            vcrl_tag = f" vcrl={vcrl.scene_seen.mean():.0%}" if vcrl else ""
            ssr_tag = f" ssr={len(ssr)}" if ssr else ""
            print(f" step {global_step:4d}/{total_steps} "
                  f"r={reward_mean:.3f} avg={avg_reward:.3f} "
                  f"adv_std={adv_std:.4f} "
                  f"datums={n_datums} ({elapsed:.1f}s)"
                  f"{vcrl_tag}{ssr_tag}")

        # ── Checkpoint ──────────────────────────────────────────────────
        if global_step % cfg["save_every"] == 0:
            ckpt_name = f"step_{global_step}"
            try:
                sampler_f = training_client.save_weights_for_sampler(ckpt_name)
                state_f = training_client.save_state(ckpt_name)
                sampler_res = sampler_f.result()
                state_res = state_f.result()
                ckpt_tracker.record(global_step, sampler_res.path,
                                    state_path=state_res.path, tag="periodic")
            except Exception as e:
                print(f" [WARN] Checkpoint step={global_step}: {e}")

        # ── Inline quick-val eval ──────────────────────────────────────
        if global_step % val_eval_every == 0 and val_scenes_pb:
            print(f" [EVAL] step={global_step} - running quick val on {len(val_scenes_pb)} samples...")
            eval_correct = 0
            eval_total = 0
            try:
                eval_sc = training_client.save_weights_and_get_sampling_client()
                eval_params = make_sampling_params(tokenizer, cfg["max_tokens"], 0.1)
                for vs in val_scenes_pb:
                    try:
                        prompt = build_generation_prompt(tokenizer, vs)
                        sr = eval_sc.sample(prompt=prompt, num_samples=1,
                                            sampling_params=eval_params).result()
                        resp_text = tokenizer.decode(list(sr.sequences[0].tokens),
                                                     skip_special_tokens=False)
                        pred = parse_answer(resp_text, vs["task"])
                        gt = parse_answer(vs["assistant_text"], vs["task"])
                        if pred and gt:
                            if vs["task"] == "stability":
                                if pred.get("value") == gt.get("value"):
                                    eval_correct += 1
                            elif vs["task"] in ("ttc",):
                                if abs(pred.get("value", 999) - gt.get("value", 0)) < 0.5:
                                    eval_correct += 1
                            elif vs["task"] == "trajectory":
                                dx = pred.get("x", 999) - gt.get("x", 0)
                                dy = pred.get("y", 999) - gt.get("y", 0)
                                if (dx**2 + dy**2)**0.5 < 0.5:
                                    eval_correct += 1
                        eval_total += 1
                    except Exception:
                        eval_total += 1
                val_acc = eval_correct / max(eval_total, 1)
                print(f" [EVAL] step={global_step} val_acc={val_acc:.1%} ({eval_correct}/{eval_total})")
                metrics_logger.log({"grpo/val_acc": val_acc, "grpo/val_correct": eval_correct,
                                    "grpo/val_total": eval_total}, step=global_step)
            except Exception as e:
                print(f" [EVAL] Failed: {e}")

    # ── Final checkpoint ────────────────────────────────────────────────
    print(f"\n Saving final checkpoint...")
    try:
        sampler_f = training_client.save_weights_for_sampler("final")
        state_f = training_client.save_state("final")
        sampler_res = sampler_f.result()
        state_res = state_f.result()
        ckpt_tracker.record(total_steps, sampler_res.path,
                            state_path=state_res.path, tag="final")
    except Exception as e:
        print(f" [WARN] Final checkpoint: {e}")

    # ── Summary ─────────────────────────────────────────────────────────
    final_avg = sum(reward_history[-50:]) / len(reward_history[-50:]) if reward_history else 0
    summary = {
        "run_name": run_name,
        "total_steps": total_steps,
        "final_reward_avg": final_avg,
        "total_tokens_generated": total_tokens_generated,
        "avg_tokens_per_step": total_tokens_generated / max(total_steps, 1),
        "config": cfg,
        "finished_at": datetime.now().isoformat(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    metrics_logger.finish()

    print(f"\n{'='*60}")
    print(f" GRPO training complete!")
    print(f" Steps : {total_steps}")
    print(f" Final reward : {final_avg:.4f} (avg last 50)")
    print(f" Results : {out_dir}")
    print(f"{'='*60}")

    latest = ckpt_tracker.path
    if latest.exists():
        print(f"\n Evaluate best checkpoint:")
        print(f" python scripts/eval_physbench_tinker.py "
              f"--model-path \"<checkpoint>\" --compare")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PhysSim-VLM GRPO on Tinker - config from .env.grpo")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override GRPO_RUN_NAME from .env.grpo")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override GRPO_MAX_STEPS from .env.grpo")
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Resume from a specific tinker:// state checkpoint (overrides SFT checkpoint)")
    parser.add_argument("--start-step", type=int, default=0,
                        help="Skip to this step when resuming (use with --resume-ckpt)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
