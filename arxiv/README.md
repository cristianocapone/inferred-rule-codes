# arXiv submission package

`arxiv-submission.zip` is the preprint upload: the Nature Communications layout of the manuscript with the line
numbers removed, its five figures, the compiled bibliography, and the supplementary information as an arXiv
ancillary file.

| in the zip | what it is |
|---|---|
| `main.tex` | `paper/main_nc.tex` with `lineno` and `\linenumbers` removed |
| `main.bbl` | the compiled bibliography; no `.bib` is shipped, so arXiv does not run bibtex |
| `readerbox.tex` | the definition of the reader box |
| `figures/fig1-5*.pdf` | the five figures, as referenced by `main.tex` |
| `anc/supplementary_information.pdf` | `pdf/supplement.pdf`; arXiv offers it for download and does not compile it |

It is self-contained: unzipped into an empty directory and run through `pdflatex` three times it gives the same
13-page PDF, with no undefined references and no dependency on `nature.cls` or on a bibliography style file.

To rebuild it after the manuscript changes: `python analysis/make_nc.py` to regenerate `paper/main_nc.tex`, then
`sh arxiv/build.sh`, which does all of the above and reports the page count and any undefined reference.
