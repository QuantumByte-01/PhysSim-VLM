# Ablation Table - PhysBench Test (9,786 samples)

| Stage | Overall | image-only | image&video | general (text-only) | Δ vs Baseline |
|---|---|---|---|---|---|
| Baseline | **40.7%** (3984/9786) | 58.4% | 38.8% | 27.5% | +0.0pp |
| SFT R1 | **47.5%** (4649/9786) | 58.5% | 50.2% | 27.2% | +6.8pp |
| SFT R2-redo | **47.6%** (4662/9786) | 58.7% | 51.5% | 23.8% | +6.9pp |
| GRPO R3 | **47.7%** (4670/9786) | 59.5% | 50.1% | 27.6% | +7.0pp |

## Per-Task-Type Decomposition

| Stage | Dynamics | Property | Relationships | Scene |
|---|---|---|---|---|
| Baseline | 37.3% | 56.6% | 41.9% | 26.0% |
| SFT R1 | 44.4% | 56.9% | 43.2% | 44.8% |
| SFT R2-redo | 43.9% | 57.9% | 42.0% | 46.1% |
| GRPO R3 | 44.3% | 56.1% | 44.7% | 45.5% |