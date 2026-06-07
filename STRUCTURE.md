# Repository structure

```
PhysSim-VLM/
├── README.md                  Project overview and quick start
├── STRUCTURE.md               This file
├── LICENSE                    MIT
├── requirements.txt           Python dependencies
├── pyproject.toml             Package metadata and tooling config
│
├── scripts/                   Data generation, training, evaluation
│   ├── generate_training_data.py    MuJoCo rigid-body scenes (TTC, trajectory, stability)
│   ├── generate_fluid_phiflow.py    PhiFlow continuum-fluid scenes
│   ├── generate_sft_r2_data.py      Categorical comparison families
│   ├── prepare_dataset.py           Pack scenes into training format
│   ├── prepare_combined_sft.py      Merge SFT corpora
│   ├── train_sft_tinker.py          SFT round 1 (rank-16 LoRA)
│   ├── train_sft_r2_tinker.py       SFT round 2 (corrected fluid + categorical)
│   ├── train_grpo_tinker.py         GRPO with simulator-verifiable rewards
│   ├── rewards.py                   Reward functions for GRPO
│   ├── eval_physbench_tinker.py     PhysBench evaluation (hosted)
│   ├── eval_external_physics.py     SeePhys / PhysReason / ScienceQA probes
│   └── upload_to_hf.py              Push checkpoints / datasets to the Hub
│
├── simulation/
│   └── verifier.py            Reads simulator state into ground-truth answers
│
├── evaluation/
│   └── physbench_eval.py      PhysBench scoring and answer extraction
│
├── analysis/                  Paper figures, tables, and metrics
│   ├── build_paper_assets.py        Per-subtask matrices and cost receipts
│   ├── leaderboard_comparison.py    PhysBench leaderboard tables
│   ├── calibration.py               Calibration proxies
│   ├── cot_analysis.py              Response-length / physics-vocabulary stats
│   ├── traces_and_failures.py       Failure-mode taxonomy
│   └── *.md, *.csv, *.json          Generated tables and intermediate metrics
│
├── results/
│   ├── canonical_numbers.json       Final per-checkpoint accuracy (source of truth)
│   └── external_summary*.json       Off-distribution probe summaries
│
├── docs/
│   └── model_card.md          Model description, results, and usage
│
└── paper/                     LaTeX source, bibliography, figures
```

## Conventions

- All credentials are read from environment variables (`TINKER_API_KEY`, `HF_TOKEN`);
  none are committed. `.env` is git-ignored.
- `results/canonical_numbers.json` is the single source of truth for reported accuracy;
  tables in `analysis/` are regenerated from raw predictions.
- Large artifacts (raw datasets, checkpoints, logs) are not version-controlled; see
  the Hugging Face links in the README.
