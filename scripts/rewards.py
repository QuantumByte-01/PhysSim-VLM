"""
PhysSim-VLM: Reward Functions for GRPO Training
=====================================================
Continuous physics rewards from MuJoCo ground truth + format compliance.

Reward design:
  - Physics reward: exp(-decay * |error|) - smooth, gradient-friendly
  - Format reward: binary check for <reasoning>...<answer>... tags
  - Combined: w_physics * physics + w_format * format

Supports:
  - AP-GRPO: absolute-penalized advantage (scales by reward magnitude)
  - SNRA: dynamic sharpness scheduling (smooth -> strict over training)
"""

import math
import re
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Reward function configuration, loaded from .env.grpo."""
    decay_ttc: float = 2.0
    decay_trajectory: float = 1.0
    weight_physics: float = 0.8
    weight_format: float = 0.2
    min_completion_tokens: int = 20
    max_tokens: int = 1024

    # AP-GRPO
    ap_enabled: bool = True
    ap_alpha: float = 1.0

    # SNRA
    snra_enabled: bool = True
    snra_k_start: float = 1.0
    snra_k_end: float = 15.0
    snra_tau: float = 0.6
    snra_steepness: float = 8.0

    # Overlong penalty (DAPO)
    overlong_penalty_start: float = 0.8
    overlong_penalty_max: float = 0.5

    # Unlikeliness bonus
    unlikeliness_bonus: float = 0.05
    unlikeliness_threshold: float = 0.7


# ── Answer Parsing ──────────────────────────────────────────────────────────

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_COORD_RE = re.compile(r"\(?\s*([-\d.]+)\s*[,;]\s*([-\d.]+)\s*\)?")


def parse_answer(response: str, task: str) -> dict | None:
    """
    Extract prediction from model response.

    Returns:
      {"value": float} for TTC/stability
      {"x": float, "y": float} for trajectory
      None if parsing fails
    """
    m = _ANSWER_RE.search(response)
    if not m:
        return None

    answer_text = m.group(1).strip().lower()

    if task == "stability":
        if "stable" in answer_text and "unstable" not in answer_text:
            return {"value": 1.0}
        elif "unstable" in answer_text:
            return {"value": 0.0}
        # Try numeric
        try:
            v = float(answer_text)
            return {"value": 1.0 if v >= 0.5 else 0.0}
        except ValueError:
            return None

    elif task == "ttc":
        # Extract numeric value (seconds)
        nums = re.findall(r"[-+]?\d*\.?\d+", answer_text)
        if nums:
            try:
                return {"value": float(nums[0])}
            except ValueError:
                return None
        return None

    elif task == "trajectory":
        # Extract (x, y) coordinates
        cm = _COORD_RE.search(answer_text)
        if cm:
            try:
                return {"x": float(cm.group(1)), "y": float(cm.group(2))}
            except ValueError:
                return None
        # Fallback: try to find two numbers
        nums = re.findall(r"[-+]?\d*\.?\d+", answer_text)
        if len(nums) >= 2:
            try:
                return {"x": float(nums[0]), "y": float(nums[1])}
            except ValueError:
                return None
        return None

    elif task in ("motion_comparison", "object_comparison", "manipulation",
                  "counting", "viewpoint", "light_direction",
                  "fluid_direction", "fluid_viscosity", "fluid_level"):
        # Multiple-choice or short-answer tasks: extract the answer text
        # Normalize: strip whitespace, lowercase, take first word/letter
        cleaned = answer_text.strip()
        if cleaned:
            return {"value": cleaned}
        return None

    return None


# ── Physics Reward ──────────────────────────────────────────────────────────

def physics_reward(prediction: dict | None, ground_truth: dict,
                   task: str, cfg: RewardConfig,
                   step: int = 0, total_steps: int = 1) -> float:
    """
    Compute continuous physics reward from prediction vs ground truth.

    Returns float in [0, 1]. Higher = more accurate prediction.
    Returns 0.0 if prediction is None (unparseable).
    """
    if prediction is None:
        return 0.0

    if task == "ttc":
        error = abs(prediction["value"] - ground_truth["value"])
        decay = cfg.decay_ttc
        if cfg.snra_enabled:
            return _snra_reward(error, step, total_steps, cfg)
        return math.exp(-decay * error)

    elif task == "stability":
        correct = (prediction["value"] == ground_truth["value"])
        return 1.0 if correct else 0.15 # partial credit prevents constant groups

    elif task == "trajectory":
        dx = prediction["x"] - ground_truth["x"]
        dy = prediction["y"] - ground_truth["y"]
        error = math.sqrt(dx**2 + dy**2)
        decay = cfg.decay_trajectory
        if cfg.snra_enabled:
            return _snra_reward(error, step, total_steps, cfg)
        return math.exp(-decay * error)

    elif task in ("motion_comparison", "object_comparison", "manipulation",
                  "counting", "viewpoint", "light_direction",
                  "fluid_direction", "fluid_viscosity", "fluid_level"):
        # Exact-match tasks: normalize and compare
        pred_val = prediction["value"].strip().lower()
        gt_val = ground_truth["value"].strip().lower()

        # MCQ letter matching: if ground truth is a single letter (a/b/c/d),
        # compare letter-to-letter (PhysBench format alignment)
        if len(gt_val) == 1 and gt_val in "abcd":
            pred_letter = ""
            for ch in pred_val:
                if ch in "abcd":
                    pred_letter = ch
                    break
            if pred_letter:
                return 1.0 if pred_letter == gt_val else 0.0
            return 0.0

        if pred_val == gt_val:
            return 1.0
        # Partial credit: check if ground truth is contained in prediction
        if gt_val in pred_val:
            return 0.5
        # Semantic partial credit: correct outcome type but wrong qualifier/direction
        return _semantic_partial_credit(pred_val, gt_val, task)

    return 0.0


def _snra_reward(error: float, step: int, total_steps: int,
                 cfg: RewardConfig) -> float:
    """
    Smooth Numerical Reward Activation with dynamic sharpness.
    Early: smooth (tolerant). Late: sharp (strict).
    """
    t_frac = step / max(total_steps, 1)
    k = cfg.snra_k_start + (cfg.snra_k_end - cfg.snra_k_start) * _sigmoid(
        cfg.snra_steepness * (t_frac - cfg.snra_tau))
    return 2.0 / (1.0 + math.exp(k * error**2))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# ── Format Reward ───────────────────────────────────────────────────────────

def format_reward(response: str) -> float:
    """Check if response uses <reasoning>...</reasoning><answer>...</answer> format."""
    has_reasoning = "<reasoning>" in response and "</reasoning>" in response
    has_answer = "<answer>" in response and "</answer>" in response
    return 1.0 if (has_reasoning and has_answer) else 0.0


# ── Combined Reward ─────────────────────────────────────────────────────────

def compute_reward(response: str, ground_truth: dict, task: str,
                   cfg: RewardConfig, n_tokens: int = 0,
                   step: int = 0, total_steps: int = 1) -> dict:
    """
    Compute combined reward for a single completion.

    Returns dict with:
      total: combined weighted reward
      physics: physics accuracy reward
      format: format compliance reward
      parsed: whether answer was parseable
      length_penalty: whether length penalty was applied
    """
    prediction = parse_answer(response, task)
    phys = physics_reward(prediction, ground_truth, task, cfg, step, total_steps)
    fmt = format_reward(response)

    total = cfg.weight_physics * phys + cfg.weight_format * fmt

    # Length penalty: penalize suspiciously short responses
    length_penalty = False
    if n_tokens < cfg.min_completion_tokens and n_tokens > 0:
        total *= 0.1
        length_penalty = True

    # Overlong penalty (DAPO): soft penalty for responses nearing MAX_TOKENS
    overlong_penalty = False
    threshold_tokens = int(cfg.overlong_penalty_start * cfg.max_tokens)
    if n_tokens > threshold_tokens and cfg.max_tokens > 0:
        frac = (n_tokens - threshold_tokens) / max(cfg.max_tokens - threshold_tokens, 1)
        penalty = cfg.overlong_penalty_max * min(frac, 1.0)
        total = max(total - penalty, 0.0)
        overlong_penalty = True

    # Unlikeliness bonus: reward rare correct predictions (anti-sharpening)
    unlikeliness_applied = False
    if (phys >= cfg.unlikeliness_threshold
            and n_tokens > cfg.min_completion_tokens
            and cfg.unlikeliness_bonus > 0):
        total += cfg.unlikeliness_bonus
        unlikeliness_applied = True

    return {
        "total": total,
        "physics": phys,
        "format": fmt,
        "parsed": prediction is not None,
        "length_penalty": length_penalty,
        "overlong_penalty": overlong_penalty,
        "unlikeliness_applied": unlikeliness_applied,
    }


# ── AP-GRPO Advantage Modification ─────────────────────────────────────────

def _semantic_partial_credit(pred: str, gt: str, task: str) -> float:
    """
    Partial credit for responses that get the right physical outcome type
    but wrong qualifier or direction.

    Examples:
      gt="object a" pred="object b" → 0.0 (wrong object)
      gt="left" pred="slightly left" → 0.4 (right direction, wrong magnitude)
      gt="unstable" pred="marginally stable" → 0.2 (wrong outcome, adjacent category)
      gt="both move at similar speed" pred="object a" → 0.0
    """
    # Direction tasks: same axis, wrong qualifier
    _DIRECTION_TERMS = {
        "left": ("left",), "right": ("right",),
        "up": ("up",), "down": ("down",),
        "above": ("above",), "below": ("below",),
    }
    for key, synonyms in _DIRECTION_TERMS.items():
        if key in gt:
            if any(s in pred for s in synonyms):
                return 0.4 # right direction, possibly wrong qualifier
            return 0.0

    # Comparison tasks: wrong object but reasonable
    if task in ("motion_comparison", "object_comparison"):
        gt_has_similar = "similar" in gt or "both" in gt
        pred_has_similar = "similar" in pred or "both" in pred
        if gt_has_similar and pred_has_similar:
            return 0.7 # both recognised "similar" even if phrased differently
        # Named wrong object (object a vs object b) - no credit
        return 0.0

    # Stability: adjacent categories get small credit
    if task == "stability":
        _STABILITY_ORDER = ["stable", "marginally stable", "unstable"]
        try:
            gi = next(i for i, s in enumerate(_STABILITY_ORDER) if s in gt)
            pi = next(i for i, s in enumerate(_STABILITY_ORDER) if s in pred)
            if abs(gi - pi) == 1:
                return 0.2 # adjacent stability category
        except StopIteration:
            pass
        return 0.0

    # Fluid tasks: correct fluid mentioned but wrong property
    if task in ("fluid_direction", "fluid_viscosity", "fluid_level"):
        # If at least the correct fluid is identified
        for term in ("fluid a", "fluid b", "both"):
            if term in gt and term in pred:
                return 0.3
        return 0.0

    return 0.0


def gdpo_advantages(physics_rewards: list[float], format_rewards: list[float],
                    cfg: "RewardConfig") -> list[float]:
    """
    GDPO: Decoupled per-component advantage normalization.

    Rather than normalizing (w_p*phys + w_f*fmt) together, we normalize
    physics and format advantages independently within the group, then combine.
    This prevents constant format rewards from diluting physics variance signal.

    Reference: NVIDIA GDPO (arXiv 2601.05242)
    """
    def _normalize(vals: list[float]) -> list[float]:
        if len(vals) < 2:
            return [0.0] * len(vals)
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        std = max(math.sqrt(var), 1e-8)
        return [(v - mu) / std for v in vals]

    phys_adv = _normalize(physics_rewards)
    fmt_adv = _normalize(format_rewards)

    return [
        cfg.weight_physics * pa + cfg.weight_format * fa
        for pa, fa in zip(phys_adv, fmt_adv)
    ]


def ap_grpo_advantage(advantages: list[float], rewards: list[float],
                      cfg: RewardConfig) -> list[float]:
    """
    Absolute-Penalized GRPO: scale advantage by reward magnitude.
    Prevents "least bad" completions from getting high advantage.

    A_i' = A_i * (R_i ** alpha)

    When R_i is near 1.0 (accurate): behaves like normal GRPO.
    When R_i is small (inaccurate): advantage is suppressed.
    """
    if not cfg.ap_enabled:
        return advantages

    return [
        adv * (max(r, 0.0) ** cfg.ap_alpha)
        for adv, r in zip(advantages, rewards)
    ]
