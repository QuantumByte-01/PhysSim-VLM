# Failure Mode Taxonomy - PhysBench Test

Classification heuristic applied to all 9,786 predictions per checkpoint. 
Each response is bucketed into exactly one mode. Numbers are absolute counts.

| Mode | Baseline | SFT R1 | SFT R2-redo | GRPO R3 |
|---|---|---|---|---|
| empty_response | 89 (0.9%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| refusal | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| unparseable_predicted | 11 (0.1%) | 14 (0.1%) | 0 (0.0%) | 1 (0.0%) |
| off_by_one_mcq | 2685 (27.4%) | 2244 (22.9%) | 2315 (23.7%) | 2268 (23.2%) |
| wrong_reasoning_with_physics | 611 (6.2%) | 101 (1.0%) | 57 (0.6%) | 183 (1.9%) |
| terse_wrong_letter | 149 (1.5%) | 1014 (10.4%) | 1048 (10.7%) | 974 (10.0%) |
| terse_wrong | 1 (0.0%) | 873 (8.9%) | 979 (10.0%) | 766 (7.8%) |
| other_wrong | 2256 (23.1%) | 891 (9.1%) | 725 (7.4%) | 924 (9.4%) |
| correct | 3984 (40.7%) | 4649 (47.5%) | 4662 (47.6%) | 4670 (47.7%) |

## Wrong-only Distribution (excludes correct)

| Mode | Baseline | SFT R1 | SFT R2-redo | GRPO R3 |
|---|---|---|---|---|
| empty_response | 1.5% | 0.0% | 0.0% | 0.0% |
| refusal | 0.0% | 0.0% | 0.0% | 0.0% |
| unparseable_predicted | 0.2% | 0.3% | 0.0% | 0.0% |
| off_by_one_mcq | 46.3% | 43.7% | 45.2% | 44.3% |
| wrong_reasoning_with_physics | 10.5% | 2.0% | 1.1% | 3.6% |
| terse_wrong_letter | 2.6% | 19.7% | 20.5% | 19.0% |
| terse_wrong | 0.0% | 17.0% | 19.1% | 15.0% |
| other_wrong | 38.9% | 17.3% | 14.1% | 18.1% |

## Notable Patterns

- **Baseline**: 5802 wrong, off-by-one MCQ rate 46.3% of failures, refusals=0, empty=89, unparseable=11
- **SFT R1**: 5137 wrong, off-by-one MCQ rate 43.7% of failures, refusals=0, empty=0, unparseable=14
- **SFT R2-redo**: 5124 wrong, off-by-one MCQ rate 45.2% of failures, refusals=0, empty=0, unparseable=0
- **GRPO R3**: 5116 wrong, off-by-one MCQ rate 44.3% of failures, refusals=0, empty=0, unparseable=1