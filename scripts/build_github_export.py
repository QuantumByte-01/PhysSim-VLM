"""Populate github/ with a clean, push-ready snapshot. Does not delete anything elsewhere."""
from __future__ import annotations

import shutil
from pathlib import Path


def copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    gh = root / "github"

    gh.mkdir(parents=True, exist_ok=True)

    # Root small files
    for name in (
        "pyproject.toml",
        "requirements.txt",
        ".python-version",
        "eval_kaggle.ipynb",
        "watchdog_eval.py",
        "watchdog_eval.sh",
        "run_pipeline.sh",
        "run_grpo.bat",
        "launch_grpo.ps1",
    ):
        s = root / name
        if s.is_file():
            shutil.copy2(s, gh / name)

    # Code
    for d in ("scripts", "analysis", "evaluation", "simulation"):
        copytree(root / d, gh / d)

    # Docs: markdown / txt only
    docs_src = root / "docs"
    docs_dst = gh / "docs"
    if docs_dst.exists():
        shutil.rmtree(docs_dst)
    docs_dst.mkdir(parents=True)
    if docs_src.is_dir():
        for p in sorted(docs_src.iterdir()):
            if p.is_file() and p.suffix.lower() in (".md", ".txt"):
                shutil.copy2(p, docs_dst / p.name)

    # Paper
    paper = gh / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    lt = root / "Paper Submisison" / "latex_template"
    tex_src = lt / "physsim_paper_conference.tex"
    tex = tex_src.read_text(encoding="utf-8")
    tex = tex.replace("../../results/figures/", "figures/")
    tex = tex.replace("../../results/samples/", "samples/")
    (paper / "physsim_paper.tex").write_text(tex, encoding="utf-8")
    for f in (
        "example_paper.bib",
        "icml2026.sty",
        "icml2026.bst",
        "algorithm.sty",
        "algorithmic.sty",
        "fancyhdr.sty",
    ):
        s = lt / f
        if s.is_file():
            shutil.copy2(s, paper / f)
    bbl = lt / "physsim_paper_conference.bbl"
    if bbl.is_file():
        shutil.copy2(bbl, paper / "physsim_paper.bbl")

    fig_src = root / "results" / "figures"
    samp_src = root / "results" / "samples"
    if fig_src.is_dir():
        copytree(fig_src, paper / "figures")
    if samp_src.is_dir():
        copytree(samp_src, paper / "samples")

    cfg = root / "configs"
    if cfg.is_dir() and any(cfg.iterdir()):
        copytree(cfg, gh / "configs")

    # Small metrics / metadata in results/
    res = root / "results"
    res_dst = gh / "results"
    if res_dst.exists():
        shutil.rmtree(res_dst)
    res_dst.mkdir(parents=True)
    if res.is_dir():
        for p in res.iterdir():
            if p.is_file() and p.suffix.lower() in (".json", ".md", ".txt", ".csv"):
                shutil.copy2(p, res_dst / p.name)

    print("github export ready:", gh)


if __name__ == "__main__":
    main()
