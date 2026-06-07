# Paper (PhysSim / synthetic physics supervision)

## Build

From this directory, with a LaTeX distribution (TeX Live / MiKTeX):

```bash
pdflatex physsim_paper.tex
bibtex physsim_paper
pdflatex physsim_paper.tex
pdflatex physsim_paper.tex
```

Figures live in `figures/`; training and benchmark sample images in
`samples/`. Paths in `physsim_paper.tex` are relative to this folder.

## Files

- `physsim_paper.tex` - main manuscript (exported from conference build)
- `physsim_paper.bbl` - pre-generated bibliography (optional; BibTeX regenerates)
- `example_paper.bib` - BibTeX database
- `icml2026.sty`, `icml2026.bst`, `algorithm*.sty`, `fancyhdr.sty` - ICML style bundle
