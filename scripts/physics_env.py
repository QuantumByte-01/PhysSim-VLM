"""
PhysSim-VLM: Tinker RL Environment for GRPO
==================================================
Implements Tinker's Env / EnvGroupBuilder / RLDataset abstractions
for physics-based GRPO training.

Architecture:
  PhysicsEnv - Single physics problem. Builds prompt, scores completion.
  PhysicsGroupBuilder - Creates G identical envs for one prompt (the GRPO "group").
  PhysicsRLDataset - Yields batches of GroupBuilders from the HF dataset.
"""

import base64
import math
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Callable

from PIL import Image

import tinker
from tinker import types

from rewards import RewardConfig, compute_reward


# ── Constants ───────────────────────────────────────────────────────────────

MAX_IMG_BYTES = 1_900_000
MAX_IMG_SIDE = 1024

# Tasks where the answer is categorical (binary/multiple-choice)
CATEGORICAL_TASKS = frozenset({
    "stability", "motion_comparison", "object_comparison", "manipulation",
    "counting", "viewpoint", "light_direction",
    "fluid_direction", "fluid_viscosity", "fluid_level",
})
# Tasks with continuous numerical answers
CONTINUOUS_TASKS = frozenset({"ttc", "trajectory"})

_VISION_START_STR = "<|vision_start|>"
_VISION_END_STR = "<|vision_end|>"
_IM_START_STR = "<|im_start|>"
_IM_END_STR = "<|im_end|>"


# ── Image utilities (shared with SFT script) ───────────────────────────────

def _smart_resize(height: int, width: int, factor: int = 28,
                  min_pixels: int = 3136, max_pixels: int = 235200
                  ) -> tuple[int, int]:
    """Qwen3-VL smart_resize for token count computation."""
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


def compute_image_tokens(height: int, width: int) -> int:
    """Number of visual tokens for one image in Qwen3-VL on Tinker."""
    patch_size, merge_size = 14, 2
    factor = patch_size * merge_size
    rh, rw = _smart_resize(height, width, factor)
    return (rh // patch_size // merge_size) * (rw // patch_size // merge_size)


def compress_image_bytes(raw_bytes: bytes) -> bytes:
    """Re-encode raw image bytes as JPEG under size limit."""
    img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMG_SIDE:
        scale = MAX_IMG_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    for q in [85, 75, 60, 45]:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q)
        data = buf.getvalue()
        if len(data) <= MAX_IMG_BYTES:
            return data
    raise ValueError("Cannot compress image below size limit")


def decode_frames_b64(frames_b64: list[str]) -> list[bytes]:
    """Base64 strings -> JPEG bytes list."""
    return [compress_image_bytes(base64.b64decode(b)) for b in frames_b64]


# ── MCQ Wrapping ─────────────────────────────────────────────────────────

# Possible answers per categorical task (for generating MCQ distractors)
_TASK_CHOICES = {
    "stability": ["stable", "unstable"],
    "fluid_direction": ["left", "right", "down"],
    "fluid_level": ["low (below 20%)", "medium (20-50%)",
                    "high (50-80%)", "very high (above 80%)"],
    "motion_comparison": ["A is faster", "B is faster", "same speed"],
    "object_comparison": ["A is heavier", "B is heavier", "same weight",
                          "A is larger", "B is larger", "same size"],
    "manipulation": ["falls", "stays", "slides", "tips over"],
    "counting": [str(i) for i in range(1, 11)],
    "viewpoint": ["front", "back", "left", "right", "top", "above"],
}

_LETTERS = ["A", "B", "C", "D"]


def mcq_wrap_scene(scene: dict, rng=None) -> dict:
    """
    Wrap a categorical scene's prompt in MCQ format (A/B/C/D choices).

    Modifies the prompt to include lettered choices and updates the
    assistant_text to output the correct letter. Returns a new scene dict.
    """
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()

    task = scene["task"]
    if task not in _TASK_CHOICES and task not in CATEGORICAL_TASKS:
        return scene # not a categorical task, return unchanged

    # Parse the correct answer from assistant_text
    from rewards import parse_answer
    parsed = parse_answer(scene["assistant_text"], task)
    if parsed is None or "value" not in parsed:
        return scene # can't parse, skip wrapping

    correct_answer = str(parsed["value"]).strip().lower()

    # Get possible choices for this task
    all_choices = _TASK_CHOICES.get(task, [])
    if not all_choices:
        return scene

    # Find the correct choice (case-insensitive match)
    correct_idx = None
    for i, c in enumerate(all_choices):
        if c.lower() == correct_answer or correct_answer in c.lower():
            correct_idx = i
            break
    if correct_idx is None:
        return scene # answer not in known choices, skip

    correct_choice = all_choices[correct_idx]

    # Build MCQ: correct answer + 2-3 distractors (total 3-4 choices)
    distractors = [c for i, c in enumerate(all_choices) if i != correct_idx]
    n_distractors = min(len(distractors), 3)
    selected_distractors = list(rng.choice(
        distractors, size=n_distractors, replace=False))

    choices = [correct_choice] + selected_distractors
    rng.shuffle(choices)

    # Find which letter maps to the correct answer
    correct_letter = _LETTERS[choices.index(correct_choice)]

    # Build MCQ prompt suffix
    mcq_lines = []
    for i, choice in enumerate(choices):
        mcq_lines.append(f" {_LETTERS[i]}. {choice}")
    mcq_text = "\n".join(mcq_lines)

    # Modify prompt: append MCQ choices
    new_prompt = (
        scene["prompt_text"].rstrip()
        + f"\n\nChoose the correct answer:\n{mcq_text}\n\n"
        + "<reasoning>Analyze the physics and select the best answer</reasoning>\n"
        + "<answer>A/B/C/D</answer>"
    )

    # Modify assistant_text to output the letter
    new_assistant = (
        f"<reasoning>Based on the physical analysis, the answer is "
        f"{correct_choice}.</reasoning>\n"
        f"<answer>{correct_letter}</answer>"
    )

    # Return modified copy
    wrapped = dict(scene)
    wrapped["prompt_text"] = new_prompt
    wrapped["assistant_text"] = new_assistant
    wrapped["mcq_wrapped"] = True
    wrapped["mcq_correct_letter"] = correct_letter.lower()
    return wrapped


# ── Task-adaptive temperature ─────────────────────────────────────────────

def get_task_temperature(task: str, base_temp: float,
                         categorical_temp: float = 1.0,
                         step: int = 0, total_steps: int = 1,
                         temp_schedule: bool = True) -> float:
    """
    Return temperature adapted to task type and training progress.

    Categorical tasks use higher temperature (more diverse outputs prevent
    constant-reward groups that waste GRPO signal).

    Optional schedule: start warm (explore) -> cool down (exploit).
    """
    if task in CATEGORICAL_TASKS:
        t = categorical_temp
    else:
        t = base_temp

    if temp_schedule and total_steps > 1:
        # Linear decay: start at t, end at t * 0.6
        progress = step / max(total_steps, 1)
        t = t * (1.0 - 0.4 * progress)

    return max(t, 0.3) # floor to prevent degenerate sampling


# ── Prompt building ─────────────────────────────────────────────────────────

def build_generation_prompt(tokenizer, scene: dict) -> types.ModelInput:
    """
    Build a Tinker ModelInput for generation (prompt only, no response).
    The model will generate the response during rollout.
    """
    img_bytes_list = decode_frames_b64(scene["frames_b64"])
    if not img_bytes_list:
        raise ValueError(f"No frames for {scene['scene_id']}")

    chunks = []

    # User turn prefix
    prefix_ids = tokenizer.encode(f"{_IM_START_STR}user\n", add_special_tokens=False)
    chunks.append(types.EncodedTextChunk(tokens=prefix_ids))

    # Images with vision tokens
    vs_ids = tokenizer.encode(_VISION_START_STR, add_special_tokens=False)
    ve_ids = tokenizer.encode(_VISION_END_STR, add_special_tokens=False)

    for img_bytes in img_bytes_list:
        img = Image.open(BytesIO(img_bytes))
        w, h = img.size
        n_tok = compute_image_tokens(h, w)
        chunks.append(types.EncodedTextChunk(tokens=vs_ids))
        chunks.append(types.ImageChunk(
            data=img_bytes, format="jpeg", expected_tokens=n_tok))
        chunks.append(types.EncodedTextChunk(tokens=ve_ids))

    # Prompt text + transition to assistant
    suffix_str = f"{scene['prompt_text']}{_IM_END_STR}\n{_IM_START_STR}assistant\n"
    suffix_ids = tokenizer.encode(suffix_str, add_special_tokens=False)
    chunks.append(types.EncodedTextChunk(tokens=suffix_ids))

    return types.ModelInput(chunks=chunks)


# ── Sampling params ─────────────────────────────────────────────────────────

def make_sampling_params(tokenizer, max_tokens: int,
                         temperature: float) -> types.SamplingParams:
    """Build SamplingParams: stop on <|im_end|> or max_tokens."""
    eos_ids = tokenizer.encode(_IM_END_STR, add_special_tokens=False)
    return types.SamplingParams(
        max_tokens=max_tokens,
        stop=eos_ids,
        temperature=temperature,
    )


# ── PhysicsEnv ──────────────────────────────────────────────────────────────

class PhysicsEnv:
    """
    Single physics problem environment for GRPO rollouts.

    Lifecycle:
      1. initial_observation() -> returns prompt ModelInput + stop condition
      2. Model generates completion (Tinker handles this)
      3. step(action) -> parse completion, compute reward, return StepResult
    """

    def __init__(self, tokenizer, scene: dict, reward_cfg: RewardConfig,
                 max_tokens: int, temperature: float,
                 step: int = 0, total_steps: int = 1,
                 categorical_temp: float = 1.0,
                 temp_schedule: bool = True):
        self.tokenizer = tokenizer
        self.scene = scene
        self.reward_cfg = reward_cfg
        self.max_tokens = max_tokens
        self.training_step = step
        self.total_steps = total_steps

        self.task = scene["task"]
        # Task-adaptive temperature: higher for categorical tasks
        self.temperature = get_task_temperature(
            self.task, temperature, categorical_temp,
            step, total_steps, temp_schedule)
        self.ground_truth = self._parse_ground_truth()

    def _parse_ground_truth(self) -> dict:
        """Extract ground truth from assistant_text for reward computation."""
        from rewards import parse_answer
        gt = parse_answer(self.scene["assistant_text"], self.task)
        if gt is not None:
            return gt

        # Fallback for categorical tasks: if assistant_text has no <answer> tags,
        # treat the whole text as the answer (e.g., "A", "stable", "left")
        if self.task in ("motion_comparison", "object_comparison", "manipulation",
                         "counting", "viewpoint", "light_direction", "stability",
                         "fluid_direction", "fluid_viscosity", "fluid_level"):
            cleaned = self.scene["assistant_text"].strip().lower()
            if cleaned:
                if self.task == "stability":
                    if "stable" in cleaned and "unstable" not in cleaned:
                        return {"value": 1.0}
                    elif "unstable" in cleaned:
                        return {"value": 0.0}
                return {"value": cleaned}

        raise ValueError(
            f"Cannot parse ground truth for {self.scene['scene_id']} "
            f"(task={self.task}): {self.scene['assistant_text'][:200]}")

    def initial_observation(self) -> tuple[types.ModelInput, types.SamplingParams]:
        """Build the prompt (images + question text) and sampling params."""
        prompt = build_generation_prompt(self.tokenizer, self.scene)
        sampling_params = make_sampling_params(
            self.tokenizer, self.max_tokens, self.temperature)
        return prompt, sampling_params

    def step(self, action_tokens: list[int]) -> dict:
        """
        Score the model's completion.

        Args:
            action_tokens: token IDs generated by the model

        Returns:
            dict with reward, metrics, episode_done flag
        """
        response_text = self.tokenizer.decode(action_tokens, skip_special_tokens=False)
        n_tokens = len(action_tokens)

        reward_info = compute_reward(
            response=response_text,
            ground_truth=self.ground_truth,
            task=self.task,
            cfg=self.reward_cfg,
            n_tokens=n_tokens,
            step=self.training_step,
            total_steps=self.total_steps,
        )

        return {
            "reward": reward_info["total"],
            "episode_done": True,
            "metrics": {
                "physics_reward": reward_info["physics"],
                "format_reward": reward_info["format"],
                "total_reward": reward_info["total"],
                "parsed": reward_info["parsed"],
                "length_penalty": reward_info["length_penalty"],
                "n_tokens": n_tokens,
                "task": self.task,
            },
        }


# ── PhysicsGroupBuilder ────────────────────────────────────────────────────

@dataclass(frozen=True)
class PhysicsGroupBuilder:
    """Creates G environments for the same physics problem (the GRPO "group")."""
    scene: dict
    group_size: int
    tokenizer: object
    reward_cfg: RewardConfig
    max_tokens: int
    temperature: float
    training_step: int = 0
    total_steps: int = 1
    categorical_temp: float = 1.0
    temp_schedule: bool = True

    def make_env(self) -> PhysicsEnv:
        """Create a single environment for this prompt (sampling G done externally)."""
        return PhysicsEnv(
            tokenizer=self.tokenizer,
            scene=self.scene,
            reward_cfg=self.reward_cfg,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            step=self.training_step,
            total_steps=self.total_steps,
            categorical_temp=self.categorical_temp,
            temp_schedule=self.temp_schedule,
        )

    def make_envs(self) -> list[PhysicsEnv]:
        """Create G identical environments (kept for compatibility)."""
        return [self.make_env() for _ in range(self.group_size)]

    def logging_tags(self) -> list[str]:
        task = self.scene["task"]
        # Group tasks into PhysBench categories for aggregate metrics
        category_map = {
            "ttc": "dynamics", "trajectory": "dynamics",
            "stability": "property",
            "motion_comparison": "relationships",
            "object_comparison": "relationships",
            "manipulation": "dynamics",
            "counting": "scene",
            "viewpoint": "scene",
            "light_direction": "property",
        }
        category = category_map.get(task, "other")
        return [task, category, "physics"]


# ── PhysicsRLDataset ────────────────────────────────────────────────────────

class PhysicsRLDataset:
    """
    Dataset that produces batches of PhysicsGroupBuilders.

    Each batch contains `groups_per_batch` prompts. Each prompt spawns
    `group_size` rollouts. Total samples per step = groups_per_batch * group_size.
    """

    def __init__(self, scenes: list[dict], group_size: int,
                 groups_per_batch: int, tokenizer, reward_cfg: RewardConfig,
                 max_tokens: int, temperature: float):
        self.scenes = scenes
        self.group_size = group_size
        self.groups_per_batch = groups_per_batch
        self.tokenizer = tokenizer
        self.reward_cfg = reward_cfg
        self.max_tokens = max_tokens
        self.temperature = temperature

    def get_batch(self, index: int, training_step: int = 0,
                  total_steps: int = 1) -> list[PhysicsGroupBuilder]:
        """Get a batch of GroupBuilders for the given index."""
        start = (index * self.groups_per_batch) % len(self.scenes)
        batch_scenes = []
        for i in range(self.groups_per_batch):
            idx = (start + i) % len(self.scenes)
            batch_scenes.append(self.scenes[idx])

        return [
            PhysicsGroupBuilder(
                scene=scene,
                group_size=self.group_size,
                tokenizer=self.tokenizer,
                reward_cfg=self.reward_cfg,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                training_step=training_step,
                total_steps=total_steps,
            )
            for scene in batch_scenes
        ]

    def __len__(self) -> int:
        return len(self.scenes) // self.groups_per_batch

    @property
    def samples_per_step(self) -> int:
        return self.group_size * self.groups_per_batch
