# PhysSim-VLM

**Synthetic Physics as Supervision: Learning Real-World Physical Reasoning in Vision-Language Models**

Code, data-generation, and evaluation for our AI4Physics @ ICML 2026 workshop paper.
We fine-tune [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
on synthetic [MuJoCo](https://mujoco.org/) and [PhiFlow](https://github.com/tum-pbs/PhiFlow)
scenes whose answers are read directly from simulator state (no human annotation),
and measure transfer to the real-world [PhysBench](https://github.com/USC-GVL/PhysBench)
benchmark.

## Key results

On PhysBench Test (`n = 9,786`), synthetic-only supervised fine-tuning lifts
Qwen3-VL-30B accuracy from **40.7% to 47.6% (+6.9pp)**, with a **+20.1pp** gain on
the Scene domain. A follow-up GRPO stage with simulator-verifiable rewards preserves
the aggregate gain (**+7.0pp** over baseline). The full training pipeline costs under
**$30** in hosted-API credits and uses no local GPU.

| Stage | Overall | Dynamics | Property | Relationships | Scene |
|-------|:-------:|:--------:|:--------:|:-------------:|:-----:|
| Baseline (Qwen3-VL-30B) | 40.7 | 37.3 | 56.6 | 41.9 | 26.0 |
| SFT R1 | 47.5 | 44.4 | 56.9 | 43.2 | 44.8 |
| SFT R2-redo | 47.6 | 43.9 | 57.9 | 42.0 | 46.1 |
| GRPO R3 | 47.7 | 44.3 | 56.1 | 44.7 | 45.5 |

Full per-subtask numbers are in [`results/canonical_numbers.json`](results/canonical_numbers.json).

## Repository layout

| Path | Contents |
|------|----------|
| `scripts/` | Data generation (MuJoCo, PhiFlow), training (SFT, GRPO), evaluation |
| `simulation/` | Simulator-state verifier used to build ground-truth answers |
| `evaluation/` | PhysBench evaluation harness |
| `analysis/` | Scripts and tables that produce the figures and numbers in the paper |
| `results/` | Per-checkpoint accuracy summaries (JSON/CSV) |
| `docs/` | Model card |
| `paper/` | LaTeX source, bibliography, and figures |

See [STRUCTURE.md](STRUCTURE.md) for a file-level map.

## Data and models

| Artifact | Link |
|----------|------|
| Full synthetic corpus | https://huggingface.co/datasets/Swastikr/PhysSim-VLM-Dataset |
| Corrected SFT R2 split | https://huggingface.co/datasets/Swastikr/PhysSim-VLM-SFT-R2-Data |

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
# Qwen3-VL requires a recent transformers build:
pip install "git+https://github.com/huggingface/transformers.git"
```

Training and evaluation run on the hosted [Tinker](https://thinkingmachines.ai/)
LoRA service. Set credentials as environment variables (never commit them):

```bash
export TINKER_API_KEY=...
export HF_TOKEN=...
```

## Reproducing the pipeline

```bash
# 1. Generate synthetic scenes
python scripts/generate_training_data.py --tasks ttc,trajectory,stability
python scripts/generate_fluid_phiflow.py --tasks all

# 2. Supervised fine-tuning (rank-16 LoRA)
python scripts/train_sft_tinker.py
python scripts/train_sft_r2_tinker.py

# 3. GRPO with simulator-verifiable rewards (rank-64 LoRA)
python scripts/train_grpo_tinker.py

# 4. Evaluate on PhysBench Test
python scripts/eval_physbench_tinker.py --split test
```

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

## License

MIT - see [LICENSE](LICENSE).
