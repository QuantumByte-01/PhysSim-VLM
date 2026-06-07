---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen3-VL-30B-A3B-Instruct
tags:
- physics
- visual-reasoning
- lora
- peft
- grpo
- mujoco
- phiflow
- physics-reasoning
datasets:
- Swastikr/PhysSim-VLM-Dataset
- Swastikr/PhysSim-VLM-SFT-R2-Data
metrics:
- accuracy
pipeline_tag: image-text-to-text
---

# PhysSim-VLM: Qwen3-VL-30B LoRA for Physical Reasoning

PhysSim-VLM is a LoRA fine-tune of
[Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
trained only on synthetic simulator scenes, with answers read directly from
simulator state (no human annotation). It is the model from our AI4Physics @ ICML
2026 paper, *Synthetic Physics as Supervision*.

The pipeline has three stages:

1. **SFT R1** - rank-16 LoRA on 12,023 MuJoCo rigid-body scenes.
2. **SFT R2-redo** - resumes from R1 on a corrected PhiFlow fluid corpus plus
   categorical comparison families (2,574 scenes).
3. **GRPO R3** - rank-64 LoRA with simulator-verifiable rewards.

## Model details

| | |
|---|---|
| Base model | Qwen/Qwen3-VL-30B-A3B-Instruct (MoE, 30B total / ~3B active) |
| Method | LoRA SFT then GRPO (rank 16 for SFT, rank 64 for GRPO) |
| Trainable params | ~0.4-1.7% of backbone |
| Training platform | Thinking Machines Tinker (hosted LoRA service, no local GPU) |
| Total training cost | under $30 in hosted-API credits |

## Results (PhysBench Test, n = 9,786)

Temperature 0, 512-token cap, official PhysBench last-letter extractor.

| Stage | Overall | Dynamics | Property | Relationships | Scene |
|-------|:-------:|:--------:|:--------:|:-------------:|:-----:|
| Baseline (zero-shot) | 40.7 | 37.3 | 56.6 | 41.9 | 26.0 |
| SFT R1 | 47.5 | 44.4 | 56.9 | 43.2 | 44.8 |
| SFT R2-redo | 47.6 | 43.9 | 57.9 | 42.0 | 46.1 |
| GRPO R3 | 47.7 | 44.3 | 56.1 | 44.7 | 45.5 |

Synthetic-only SFT improves overall accuracy by **+6.9pp** and the Scene domain by
**+20.1pp**. GRPO preserves the aggregate gain (**+7.0pp** over baseline) and shifts
errors toward the un-trained `general:relationships` domain rather than raising
overall accuracy (a near-even item-level swap, McNemar p ≈ 0.84).

## Training data

Synthetic scenes generated with MuJoCo (rigid-body) and PhiFlow (continuum fluid).
Each scene yields an 8-frame rollout, a cover image, and the simulator state used as
ground truth.

| Family | Simulator | Target |
|--------|-----------|--------|
| Time-to-collision (TTC) | MuJoCo | Contact time (seconds) |
| Trajectory | MuJoCo | Landing position (x, y) |
| Stability | MuJoCo | Stable / topple |
| Mass/size, viscosity, level, direction, count/viewpoint | MuJoCo + PhiFlow | Categorical |

Datasets:
[PhysSim-VLM-Dataset](https://huggingface.co/datasets/Swastikr/PhysSim-VLM-Dataset),
[PhysSim-VLM-SFT-R2-Data](https://huggingface.co/datasets/Swastikr/PhysSim-VLM-SFT-R2-Data).

## Usage

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    torch_dtype="bfloat16",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "Swastikr/PhysSim-VLM-Qwen3VL-30B-LoRA")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
```

## Limitations

- Trained on simple geometric simulator scenes; the train-test visual gap to
  real-world PhysBench imagery is large by construction.
- The text-only `general` mode and the Relationships domain are not directly
  targeted by training and gain the least.
- A shuffled-physics control that would fully separate format-commitment from new
  physical reasoning was not run; see the paper's limitations section.

## Citation

```bibtex
@inproceedings{physsimvlm2026,
  title     = {Synthetic Physics as Supervision: Learning Real-World Physical
               Reasoning in Vision-Language Models},
  author    = {Swastik R and Natesha B V},
  booktitle = {AI4Physics Workshop at ICML},
  year      = {2026}
}
```
