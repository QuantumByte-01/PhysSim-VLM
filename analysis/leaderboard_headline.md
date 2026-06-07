# Leaderboard Headline - PhysSim-VLM vs Published Baselines

## Overall Rank

- **PhysSim-VLM (R2-redo)**: 47.20% → **Rank #6** of 29 models
- **Beats**: NVILA-15B, InternVL2-76B, Gemini-1.5-flash, InternVL2-40B, NVILA-Lite-15B, InternVL2.5-8B, NVILA-8B, InternVL2-26B...
- Notably outperforms: **InternVL2-76B (76B params)** by +0.43pp, **GPT-4V** by +5.94pp, **Gemini-1.5-flash** by +1.13pp

## Per-Domain Rankings

- **Property**: Our score 57.70 → **Rank #5** (top: InternVL2.5-78B at 60.32)
- **Relationships**: Our score 41.20 → **Rank #21** (top: InternVL2.5-38B at 67.51)
- **Scene**: Our score 45.90 → **Rank #1** (top: **PhysSim-VLM (R2-redo, ours)** at 45.90)
- **Dynamics**: Our score 43.40 → **Rank #8** (top: GPT-4o at 46.99)

## Spotlight: Scene Understanding (#1 Rank)

PhysSim-VLM (R2-redo) achieves **#1 on Scene** with synthetic data + LoRA on a 30B model:

| Rank | Model | Scene Score |
|---|---|---|
| 1 | **PhysSim-VLM (R2-redo, ours)** | 45.90 |
| 2 | **PhysSim-VLM (R1, ours)** | 44.60 |
| 3 | PhysSim-VLM (GRPO Run 2) | 41.30 |
| 4 | InternVL2.5-38B | 39.04 |
| 5 | NVILA-15B | 38.78 |
| 6 | NVILA-Lite-15B | 38.11 |
| 7 | InternVL2-76B | 38.07 |
| 8 | InternVL2-26B | 37.94 |
| 9 | InternVL2.5-78B | 37.32 |
| 10 | InternVL2.5-26B | 36.61 |

**Margin over GPT-4o**: +15.75pp  
**Margin over best leaderboard model (InternVL2.5-38B)**: +6.86pp