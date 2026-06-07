# Calibration & Confidence Analysis - PhysBench Test

Without model logprobs, we use response-text proxies for calibration.

## 1. Letter-Prior Bias

Does the model prefer certain letters? Compare prediction distribution to ground-truth distribution.

**Ground truth distribution**:
  - A: 2500 (25.6%)
  - B: 2744 (28.1%)
  - C: 2537 (25.9%)
  - D: 2000 (20.4%)

| Checkpoint | A | B | C | D | Bias entropy* | Letter-bias score† |
|---|---|---|---|---|---|---|
| Baseline | 37.2% | 21.7% | 16.4% | 23.7% | 1.997 | 0.0626 |
| SFT R1 | 25.7% | 25.7% | 24.7% | 23.7% | 2.012 | 0.0034 |
| SFT R2-redo | 29.0% | 30.3% | 25.2% | 15.5% | 1.958 | 0.0139 |
| GRPO R3 | 21.1% | 25.7% | 22.5% | 30.7% | 1.986 | 0.0432 |

*Entropy: 2.0 = uniform across A-D (unbiased), <2.0 = letter-preference bias.
†Bias score: KL(prediction || ground-truth). 0 = perfectly calibrated to GT distribution.

## 2. Hedging Language (Confidence Proxy)

Frequency of uncertainty markers ("might", "perhaps", "unclear", etc.).

| Checkpoint | % responses with hedging | Hedge-on-correct | Hedge-on-wrong | Calibration gap* |
|---|---|---|---|---|
| Baseline | 39.77% | 35.74% | 42.54% | +6.79pp |
| SFT R1 | 1.49% | 1.23% | 1.73% | +0.51pp |
| SFT R2-redo | 0.13% | 0.26% | 0.02% | -0.24pp |
| GRPO R3 | 3.66% | 3.51% | 3.79% | +0.28pp |

*Calibration gap: positive = model hedges more on wrong answers (well-calibrated). Negative = model overconfident on wrong answers.

## 3. Response Decisiveness

Fraction of responses matching a decisive answer pattern (just letter, "Answer: X", etc.).

| Checkpoint | Decisive % | Decisive accuracy | Non-decisive accuracy |
|---|---|---|---|
| Baseline | 4.5% | 35.8% | 40.9% |
| SFT R1 | 22.2% | 37.2% | 50.4% |
| SFT R2-redo | 21.0% | 38.2% | 50.1% |
| GRPO R3 | 24.5% | 38.5% | 50.7% |

## 4. Subtask Improvement Breadth (Generalization Proxy)

Across all 39 subtasks, how many improve / regress vs baseline?

Broader improvement = stronger generalization (vs narrow overfit).

| Checkpoint | Subtasks improved | Subtasks regressed | No change | Mean Δ | Median Δ |
|---|---|---|---|---|---|
| Baseline | - | - | 39 | 0.0pp | 0.0pp |
| SFT R1 | 25 | 7 | 5 | +10.97pp | +6.90pp |
| SFT R2-redo | 27 | 5 | 5 | +11.58pp | +6.15pp |
| GRPO R3 | 26 | 5 | 6 | +8.64pp | +6.95pp |

## 5. Cross-Mode Consistency

If a model has "learned physics" (vs format), accuracy should be roughly similar across modes. 
Large gaps = mode-specific exploitation.

| Checkpoint | image-only | image&video | general | Std (across 3 modes) |
|---|---|---|---|---|
| Baseline | 58.4% | 38.8% | 27.5% | 12.75 |
| SFT R1 | 58.5% | 50.2% | 27.2% | 13.25 |
| SFT R2-redo | 58.7% | 51.5% | 23.8% | 15.06 |
| GRPO R3 | 59.5% | 50.1% | 27.6% | 13.40 |