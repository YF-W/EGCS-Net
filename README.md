# EGCS-Net manuscript

LaTeX source for **EGCS-Net: An Explicit Geometry-Conditioned Segmentation Network for Retinal Vessel Segmentation**.

## Contents

- `main.tex`: manuscript source.
- `manuscript.pdf`: compiled manuscript.
- `figures/`: figures referenced by the manuscript.
- `references/references.bib`: bibliography.
- `.latexmkrc`: XeLaTeX and BibTeX build configuration.
- `cas-sc.cls`, `cas-common.sty`, and `thumbnails/`: unmodified Elsevier CAS template components.
- `template-notices/`: upstream template documentation and distribution manifest.

## Build

Install a current full TeX Live distribution with XeLaTeX, BibTeX, and latexmk. From this directory, run:

```sh
latexmk -interaction=nonstopmode -halt-on-error main.tex
```

The compiled manuscript is written to `build/main.pdf`. The `build/` directory is excluded from version control.

For Overleaf, upload this directory's contents and select **XeLaTeX** as the compiler and `main.tex` as the main document.

This repository contains the manuscript source and figures. Training software and experimental datasets are not included.

The bundled Elsevier template components retain their upstream copyright and LaTeX Project Public License notices. See `template-notices/README` and the headers of the template files.
