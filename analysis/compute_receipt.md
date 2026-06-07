# Compute Receipt - PhysSim-VLM

All training and inference performed on **Tinker (Thinking Machines-managed)** with Qwen3-VL-30B-A3B-Instruct (LoRA, rank 16, base + adapter on managed GPU). No local GPU. No human annotation.

## Per-Stage Cost & Resources

| Stage | Steps | Samples | Wall time | Avg step time | Cost (USD) | Notes |
|---|---|---|---|---|---|---|
| Data generation (MuJoCo + PhiFlow) | - | 16,617 scenes | ~30 h CPU | - | $0 | Local CPU; synthetic, deterministic |
| **SFT R1** (full epoch from base) | 751 | 12,023 | ~3.7 h | ~17.7 s | ~$8 | TTC + trajectory + stability |
| **SFT R2-redo** (resume from R1) | 644 | 2,574 | 3.71 h | 20.72 s | ~$12 | + fluid (corrected) + comparisons + manipulation |
| **GRPO Run 2** (failed, kept for ablation) | 475 | - | ~5.3 h | - | ~$11 | Regressed -2.9pp; diagnosed and fixed |
| **GRPO R3** (in progress) | 300 (cap) | 1,950 | ~3-4 h est. | ~30-40 s | ~$7 est. | KL=0.04, SNRA off, 40% static, MCQ off |
| **PhysBench eval × 4** (baseline + R1 + R2-redo + GRPO R2) | - | 9,786 each | ~1.5 h each | - | ~$12 each | Sampling-only |
| **Total committed** | - | - | - | - | **~$76** | Of $65 Tinker credits + ~$11 self-funded |

## Reproducibility

| Resource | Location |
|---|---|
| Code | https://github.com/QuantumByte-01/PhysSim-VLM |
| Dataset (R1 SFT) | https://huggingface.co/datasets/Swastikr/PhysSim-VLM-Dataset |
| Local R2 dataset | `data/sft_r2/` (8 categorical task folders) |
| SFT R1 checkpoint | `tinker://<run-id>:train:0/weights/final` |
| SFT R2-redo checkpoint | `tinker://<run-id>:train:0/weights/final` |
| Eval predictions | `results/physbench_*/predictions.json` (full responses, all 9,786 samples) |
| Hyperparameters | `.env.grpo`, frozen per-run in metadata |

## Key Compute Efficiency Claims

- **Zero human annotation**: 100% synthetic data via MuJoCo (rigid-body) + PhiFlow (fluid)
- **Single-GPU-class budget**: total $76 ≈ 2-3 days of a single A100 ≪ open-source VLM training norm
- **30B model fine-tuned without local GPU**: Tinker LoRA enables personal-budget research
- **Reproducible**: all checkpoints persisted; deterministic synthetic data generators (seed-controlled)
