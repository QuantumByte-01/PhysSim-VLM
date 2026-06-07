# Per-Subtask Accuracy Matrix (PhysBench Test, 39 subtasks)

Generated for paper. Bold = best per row; * = > Baseline.


## Mode: `image-only`

| Task | Subtask | N | Baseline | SFT R1 | SFT R2-redo | GRPO R3 | Δ vs Baseline |
|---|---|---|---|---|---|---|---|
| property | attribute | 783 | 51.6% | 49.6% | 47.1% | **51.9%** | -4.5 |
| dynamics | manipulation | 380 | **30.0%** | 27.6% | 27.6% | 27.4% | -2.4 |
| property | number | 363 | 78.8% | 77.7% | **81.5%** | 81.0% | +2.8 |
| relationships | depth | 244 | 82.8% | 88.1% | **88.9%** | 88.1% | +6.1 |
| relationships | location | 124 | 79.0% | 80.6% | **83.1%** | 79.0% | +4.0 |
| scene | viewpoint | 41 | 58.5% | 80.5% | 80.5% | **82.9%** | +22.0 |
| relationships | distance | 24 | 75.0% | **87.5%** | **87.5%** | 75.0% | +12.5 |
| property | color | 23 | 82.6% | 91.3% | **95.7%** | 65.2% | +13.0 |
| scene | light | 17 | 17.6% | 17.6% | 17.6% | **23.5%** | +0.0 |
| property | mass | 15 | 53.3% | 73.3% | **86.7%** | 66.7% | +33.3 |
| dynamics | collision | 12 | 50.0% | 50.0% | 50.0% | **58.3%** | +0.0 |
| relationships | size | 7 | **71.4%** | **71.4%** | **71.4%** | **71.4%** | +0.0 |

## Mode: `image&video`

| Task | Subtask | N | Baseline | SFT R1 | SFT R2-redo | GRPO R3 | Δ vs Baseline |
|---|---|---|---|---|---|---|---|
| scene | light | 1069 | 25.4% | 40.1% | **40.6%** | 40.5% | +15.2 |
| scene | viewpoint | 1008 | 23.7% | 44.8% | **47.3%** | 46.1% | +23.6 |
| dynamics | fluid | 642 | 41.0% | 48.6% | 48.1% | **51.7%** | +7.2 |
| dynamics | collision | 633 | 38.9% | 43.1% | 43.3% | **45.8%** | +4.4 |
| property | attribute | 614 | 48.0% | **54.4%** | 52.3% | 53.7% | +4.2 |
| dynamics | throwing | 410 | 35.4% | 39.8% | **44.4%** | 37.1% | +9.0 |
| dynamics | others | 279 | 53.0% | 83.5% | **84.6%** | 82.4% | +31.5 |
| property | color | 271 | 71.6% | 75.6% | **77.5%** | 77.1% | +5.9 |
| property | mass | 264 | 39.8% | 39.8% | 39.4% | **40.2%** | -0.4 |
| property | number | 209 | 61.2% | 48.3% | **65.1%** | 27.3% | +3.8 |
| relationships | location | 132 | 61.4% | 69.7% | 67.4% | **70.5%** | +6.1 |
| relationships | motion | 80 | 45.0% | 66.2% | **71.2%** | 67.5% | +26.2 |
| dynamics | chemistry | 67 | 40.3% | **70.1%** | **70.1%** | 64.2% | +29.9 |
| scene | temperature | 58 | 41.4% | **86.2%** | 82.8% | 84.5% | +41.4 |
| scene | air | 42 | 47.6% | **81.0%** | **81.0%** | 76.2% | +33.3 |
| relationships | size | 29 | 75.9% | 82.8% | **93.1%** | 86.2% | +17.2 |
| relationships | distance | 28 | 57.1% | 57.1% | 57.1% | **64.3%** | +0.0 |
| relationships | depth | 27 | 51.9% | **63.0%** | 55.6% | 59.3% | +3.7 |
| dynamics | manipulation | 9 | 55.6% | **77.8%** | 66.7% | **77.8%** | +11.1 |

## Mode: `general`

| Task | Subtask | N | Baseline | SFT R1 | SFT R2-redo | GRPO R3 | Δ vs Baseline |
|---|---|---|---|---|---|---|---|
| relationships | motion | 1353 | 27.0% | 25.3% | 22.9% | **27.6%** | -4.1 |
| dynamics | manipulation | 429 | **27.7%** | 26.6% | 19.8% | 21.2% | -7.9 |
| dynamics | others | 66 | 30.3% | 60.6% | 54.5% | **62.1%** | +24.2 |
| relationships | location | 24 | **50.0%** | 45.8% | 45.8% | 45.8% | -4.2 |
| dynamics | collision | 5 | 0.0% | **40.0%** | **40.0%** | 20.0% | +40.0 |
| scene | air | 5 | 40.0% | **60.0%** | **60.0%** | 40.0% | +20.0 |