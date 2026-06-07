# PhysSim-VLM - Evaluation Results

All evaluation results are stored here, one subfolder per checkpoint.
Each subfolder contains `predictions.json` and `results.md`.

---

## Result Index

| Checkpoint | Date | Overall Exact | TTC | Stability | Trajectory | Notes |
|------------|------|---------------|-----|-----------|------------|-------|
| [Zero-shot baseline](baselines/results.md) | 2026-02-19 | 40.7%* | 18.3%* | - | - | PhysBench full eval |
| [SFT Epoch 1](sft_epoch1/results.md) | - | TBD | TBD | TBD | TBD | Physics val set, 300 samples |

*PhysBench scores (different eval set - dynamics category ≈ TTC+motion)

---

## Eval Sets

| Set | Script | Samples | Notes |
|-----|--------|---------|-------|
| **Physics val** (ours) | `eval_physics_val.py` | 1,500 (300 sampled) | Generated val split, per-task exact match + continuous score |
| **PhysBench** | `physbench_eval.py` | 3,979 | Image-only, multiple choice A/B/C/D |

---

## Metrics

**Physics val set:**
- `exact_match` - within 10% for TTC, ±0.5m for trajectory, exact for stability
- `physics_score` - continuous: `exp(-3×relative_error)` for TTC, `exp(-0.5×dist)` for trajectory, 0/1 for stability

**PhysBench:**
- Accuracy - multiple choice correct answer rate

---

## Adding a New Result

Results are written automatically by the training callback after each epoch.
To run manually:
```bash
# Evaluate a specific checkpoint on val set (vLLM)
python scripts/eval_physics_val.py --checkpoint checkpoints/lora_sft_epoch1/final --epoch 1

# Evaluate zero-shot base model
python scripts/eval_physics_val.py --zero_shot --epoch 0
```
