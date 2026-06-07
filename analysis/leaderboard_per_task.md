# PhysBench Sub-Leaderboards - Per Physics Domain

PhysBench tests four independent physics dimensions. We treat each as a separate leaderboard 
to demonstrate that PhysSim-VLM's gains are concentrated in **physics-grounded** dimensions 
(Scene, Dynamics) rather than relational reasoning (Relationships).


## Sub-Leaderboard: **Property**
*Object physical properties (mass, color, attribute, number)*

| Rank | Model | Size | **Property** |
|---|---|---|---|
| 1 | InternVL2.5-78B | 78B | **60.32** |
| 2 | NVILA-15B | 15B | **59.16** |
| 3 | InternVL2.5-26B | 26B | **59.08** |
| 4 | InternVL2.5-38B | 38B | **58.77** |
| 5 | **PhysSim-VLM (R2-redo, ours)** | 30B | **57.70** |
| 6 | InternVL2-76B | 76B | **57.65** |
| 7 | Gemini-1.5-flash | - | **57.41** |
| 8 | Gemini-1.5-pro | - | **57.26** |
| 9 | GPT-4o | - | **56.91** |
| 10 | Qwen3-VL-30B (Baseline) | 30B | **56.40** |
| 11 | **PhysSim-VLM (R1, ours)** | 30B | **56.30** |
| 12 | InternVL2.5-8B | 8B | **55.87** |
| 13 | InternVL2-40B | 40B | **55.79** |
| 14 | NVILA-8B | 8B | **55.79** |
| 15 | NVILA-Lite-15B | 15B | **55.44** |
| 16 | NVILA-Lite-8B | 8B | **53.81** |
| 17 | GPT-4o-mini | - | **53.54** |
| 18 | InternVL2-26B | 26B | **51.92** |
| 19 | PhysSim-VLM (GRPO Run 2) | 30B | **51.80** |
| 20 | InternVL2.5-4B | 4B | **51.03** |
| 21 | InternVL2.5-2B | 2B | **49.63** |
| 22 | GPT-4V | - | **49.59** |
| 23 | mPLUG-Owl3-7B | 7B | **49.25** |
| 24 | InternVL2-8B | 8B | **49.05** |
| 25 | LLaVA-il-dpo | 8B | **47.97** |
| 26 | LLaVA-interleave | 8B | **47.23** |
| 27 | InternVL2-4B | 4B | **47.12** |
| 28 | Phi-3.5V | 4B | **45.72** |
| 29 | Phi-3V | 4B | **43.67** |

## Sub-Leaderboard: **Relationships**
*Spatial/depth/motion relations between objects*

| Rank | Model | Size | **Relationships** |
|---|---|---|---|
| 1 | InternVL2.5-38B | 38B | **67.51** |
| 2 | GPT-4o | - | **64.80** |
| 3 | Gemini-1.5-pro | - | **63.61** |
| 4 | InternVL2.5-78B | 78B | **62.13** |
| 5 | InternVL2.5-26B | 26B | **58.33** |
| 6 | InternVL2-76B | 76B | **52.43** |
| 7 | Gemini-1.5-flash | - | **52.24** |
| 8 | InternVL2-40B | 40B | **50.05** |
| 9 | InternVL2.5-8B | 8B | **48.67** |
| 10 | GPT-4V | - | **45.77** |
| 11 | mPLUG-Owl3-7B | 7B | **45.62** |
| 12 | InternVL2-26B | 26B | **45.20** |
| 13 | InternVL2.5-4B | 4B | **44.77** |
| 14 | LLaVA-interleave | 8B | **44.62** |
| 15 | GPT-4o-mini | - | **44.24** |
| 16 | InternVL2-8B | 8B | **43.58** |
| 17 | LLaVA-il-dpo | 8B | **42.67** |
| 18 | **PhysSim-VLM (R1, ours)** | 30B | **42.60** |
| 19 | NVILA-15B | 15B | **42.34** |
| 20 | Qwen3-VL-30B (Baseline) | 30B | **41.90** |
| 21 | **PhysSim-VLM (R2-redo, ours)** | 30B | **41.20** |
| 22 | NVILA-8B | 8B | **40.29** |
| 23 | NVILA-Lite-15B | 15B | **40.15** |
| 24 | Phi-3.5V | 4B | **40.15** |
| 25 | InternVL2-4B | 4B | **39.96** |
| 26 | NVILA-Lite-8B | 8B | **39.25** |
| 27 | PhysSim-VLM (GRPO Run 2) | 30B | **38.50** |
| 28 | InternVL2.5-2B | 2B | **38.15** |
| 29 | Phi-3V | 4B | **37.92** |

## Sub-Leaderboard: **Scene**
*Scene-level physics: light, viewpoint, temperature, air, fluid*

| Rank | Model | Size | **Scene** |
|---|---|---|---|
| 1 | **PhysSim-VLM (R2-redo, ours)** | 30B | **45.90** |
| 2 | **PhysSim-VLM (R1, ours)** | 30B | **44.60** |
| 3 | PhysSim-VLM (GRPO Run 2) | 30B | **41.30** |
| 4 | InternVL2.5-38B | 38B | **39.04** |
| 5 | NVILA-15B | 15B | **38.78** |
| 6 | NVILA-Lite-15B | 15B | **38.11** |
| 7 | InternVL2-76B | 76B | **38.07** |
| 8 | InternVL2-26B | 26B | **37.94** |
| 9 | InternVL2.5-78B | 78B | **37.32** |
| 10 | InternVL2.5-26B | 26B | **36.61** |
| 11 | Gemini-1.5-pro | - | **36.52** |
| 12 | mPLUG-Owl3-7B | 7B | **35.90** |
| 13 | InternVL2-40B | 40B | **35.86** |
| 14 | LLaVA-interleave | 8B | **35.64** |
| 15 | Phi-3V | 4B | **34.93** |
| 16 | NVILA-Lite-8B | 8B | **34.62** |
| 17 | Gemini-1.5-flash | - | **34.32** |
| 18 | NVILA-8B | 8B | **33.95** |
| 19 | LLaVA-il-dpo | 8B | **33.73** |
| 20 | Phi-3.5V | 4B | **33.02** |
| 21 | InternVL2.5-4B | 4B | **31.34** |
| 22 | InternVL2-4B | 4B | **30.94** |
| 23 | GPT-4o-mini | - | **30.59** |
| 24 | GPT-4o | - | **30.15** |
| 25 | InternVL2.5-2B | 2B | **29.44** |
| 26 | InternVL2.5-8B | 8B | **29.35** |
| 27 | InternVL2-8B | 8B | **27.05** |
| 28 | GPT-4V | - | **26.34** |
| 29 | Qwen3-VL-30B (Baseline) | 30B | **26.00** |

## Sub-Leaderboard: **Dynamics**
*Physics-based motion: collision, throwing, manipulation, fluid*

| Rank | Model | Size | **Dynamics** |
|---|---|---|---|
| 1 | GPT-4o | - | **46.99** |
| 2 | InternVL2.5-78B | 78B | **46.11** |
| 3 | NVILA-15B | 15B | **45.72** |
| 4 | InternVL2.5-38B | 38B | **45.00** |
| 5 | NVILA-Lite-15B | 15B | **44.38** |
| 6 | **PhysSim-VLM (R1, ours)** | 30B | **43.60** |
| 7 | NVILA-8B | 8B | **43.43** |
| 8 | **PhysSim-VLM (R2-redo, ours)** | 30B | **43.40** |
| 9 | PhysSim-VLM (GRPO Run 2) | 30B | **43.10** |
| 10 | GPT-4o-mini | - | **42.90** |
| 11 | GPT-4V | - | **42.15** |
| 12 | InternVL2.5-26B | 26B | **41.79** |
| 13 | InternVL2.5-4B | 4B | **41.79** |
| 14 | Gemini-1.5-pro | - | **41.56** |
| 15 | InternVL2-40B | 40B | **41.33** |
| 16 | InternVL2.5-8B | 8B | **41.20** |
| 17 | NVILA-Lite-8B | 8B | **41.17** |
| 18 | Gemini-1.5-flash | - | **40.93** |
| 19 | mPLUG-Owl3-7B | 7B | **40.61** |
| 20 | InternVL2-76B | 76B | **40.12** |
| 21 | InternVL2-4B | 4B | **39.76** |
| 22 | InternVL2-8B | 8B | **39.47** |
| 23 | Phi-3.5V | 4B | **39.40** |
| 24 | InternVL2-26B | 26B | **39.34** |
| 25 | LLaVA-il-dpo | 8B | **38.78** |
| 26 | InternVL2.5-2B | 2B | **38.39** |
| 27 | LLaVA-interleave | 8B | **37.21** |
| 28 | Qwen3-VL-30B (Baseline) | 30B | **37.20** |
| 29 | Phi-3V | 4B | **36.92** |