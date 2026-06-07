# Chain-of-Thought (CoT) Analysis - PhysBench Test

Word counts use whitespace+regex tokenizer; format tags = `<reasoning>...<answer>` block; 
physics density = fraction of words in a curated physics-vocabulary set (gravity, force, viscosity, ...).

## Length Statistics (words per response)

| Checkpoint | Mean | Median | Mean (correct) | Mean (wrong) | Δ correct−wrong |
|---|---|---|---|---|---|
| Baseline | 129 | 137 | 127 | 130 | -3 |
| SFT_R1 | 13 | 4 | 12 | 14 | -3 |
| SFT_R2redo | 5 | 3 | 5 | 5 | +0 |
| GRPO_R3 | 18 | 5 | 17 | 18 | -1 |

## Format Tag Adoption (`<reasoning>...<answer>` schema)

| Checkpoint | % responses with full format tags |
|---|---|
| Baseline | 0.0% |
| SFT_R1 | 0.0% |
| SFT_R2redo | 0.0% |
| GRPO_R3 | 0.0% |

## Physics-Vocabulary Density (× 100 = words per 100 tokens)

| Checkpoint | Overall | Correct responses | Wrong responses | Δ correct−wrong |
|---|---|---|---|---|
| Baseline | 1.36 | 1.24 | 1.44 | -0.20 |
| SFT_R1 | 2.57 | 2.36 | 2.75 | -0.39 |
| SFT_R2redo | 2.80 | 2.60 | 2.98 | -0.38 |
| GRPO_R3 | 2.87 | 2.64 | 3.09 | -0.45 |

## Accuracy by Response-Length Bucket

| Bucket (words) | Baseline | SFT_R1 | SFT_R2redo | GRPO_R3 |
|---|---|---|---|---|
| 0-50 | 34.7% (n=424) | 48.1% (n=9114) | 47.6% (n=9784) | 47.9% (n=8664) |
| 50-100 | 84.0% (n=406) | 39.8% (n=176) | 0.0% (n=1) | 48.8% (n=400) |
| 100-200 | 39.0% (n=8956) | 38.7% (n=496) | 0.0% (n=1) | 45.4% (n=722) |
| 200-400 | - | - | - | - |
| 400-800 | - | - | - | - |
| 800-∞ | - | - | - | - |