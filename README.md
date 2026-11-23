# Non-linguistic communication of abstract rules emerges from reward

Companion repository for the manuscript by **Cristiano Capone**, Quantitative Health Models and Bioinspired
Technologies Lab, Istituto Superiore di Sanità, Rome.

This repository contains the manuscript, ready-to-read PDFs, the experimental implementation,
and a self-contained CPU workflow that regenerates Figures 1–5 and the nine generated
supplementary tables from the recorded experiments. The frozen inputs are included:
no original working directory, remote cluster, checkpoint download or GPU is needed for this workflow.

## Read the paper

- [Manuscript — Nature Communications layout](pdf/manuscript.pdf)
- [Supplementary information](pdf/supplement.pdf)
- [Alternative Nature preprint layout](pdf/manuscript-nature.pdf)
- [All five figures, PDF and PNG](paper/figures/)

The scientific text is maintained in [`paper/main_nature.tex`](paper/main_nature.tex).
[`analysis/make_nc.py`](analysis/make_nc.py) generates `paper/main_nc.tex` from it: the journal layout, used
for the Nature Communications submission and for the preprint, compact and with the figures set in the
text beside the passage that discusses them. Edit the canonical file, then rebuild. Both layouts carry the same
scientific content.

## Reproduce

Use **Python 3.11**. From a new clone:

```bash
git clone https://github.com/cristianocapone/inferred-rule-codes.git
cd inferred-rule-codes
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/reproduce.py verify
python scripts/reproduce.py all
python scripts/check_reference.py
```

`requirements-lock.txt` pins the complete tested analysis environment; `requirements.txt`
lists its three direct dependencies. `all` rebuilds statistics, tables, all figures, and the three manuscript PDFs.
The PDF stage also requires **TeX Live or MacTeX**, with `pdflatex`, `bibtex`, and the
standard packages used in `paper/*.tex` (including `natbib`, `lineno`, `setspace`,
`booktabs`, `geometry`, `hyperref`, `titlesec`, `caption`, `tcolorbox`, `tikz`, `cite`, `times`, `fullpage`, and the PSNFSS
fonts `mathptmx` and `helvet`). The `nature` class
and bibliography style are bundled with their notices.

If LaTeX is not installed, regenerate only the numerical results and figures:

```bash
python scripts/reproduce.py tables
python scripts/reproduce.py figures
```

Build PDFs from the checked-in figure and table outputs:

```bash
python scripts/reproduce.py pdf
```

Equivalent Make targets are `make all`, `make figures`, `make tables`, `make pdf`,
and `make verify`. Set `PYTHON=.venv/bin/python` if needed.
The runner works from any current directory, sets the required canvas size to 12,
and writes logs to `build/logs/`. It never trains a model.

## Repository layout

| Directory | Contents |
|---|---|
| `pdf/` | Compiled manuscript and supplementary PDFs |
| `paper/` | Canonical LaTeX, generated NC layout, bibliography, figures and table fragments |
| `analysis/` | Figure builders, numerical analyses and shared plotting/path helpers |
| `data/` | Frozen logs, evaluation exports, split audits and human-chain data actually read by the workflow |
| `results/` | Regenerated statistical JSON outputs |
| `reference/` | Original numerical outputs used to check that packaging preserved the results |
| `emlang/` | Synthetic generators, agents, training, evaluation and audit implementation |
| `scripts/` | Reproduction and reference-verification entry points |
| `docs/` | Figure-to-data map, provenance, experiment notes and validation record |

See [the figure and table map](docs/FIGURES.md) for the inputs of each panel and
[the provenance notes](docs/PROVENANCE.md) for the frozen snapshot and its limits.

## What is reproduced

The default workflow reruns the **analysis of recorded experiments**: it recalculates
statistics, redraws figures and compiles the manuscript. It does not rerun the GPU
training or certify every scientific claim. Training and evaluation code is included
separately; see [experiment notes](docs/EXPERIMENTS.md).

The package preserves the manuscript snapshot from **18 September 2026**. In that snapshot,
the main library uses the corrected audit with 100 novel triples. The B+dup triple analysis
still uses its archived 17 September audit; this fallback is explicit in the provenance
notes and retained to reproduce the supplied manuscript rather than silently change it.

The four regenerated statistical JSON files and nine generated table fragments were checked
against the original outputs. Portable fonts and deterministic point jitter can change the
appearance slightly; numerical results are unchanged. PDF metadata can vary between builds.

## Data and attribution

The experimental exports were produced for this study. The human iterated-learning dataset
is credited to Kirby, Cornish and Smith and is archived at
[Edinburgh DataShare, DOI 10.7488/ds/1586](https://doi.org/10.7488/ds/1586).
See [third-party notices](docs/THIRD_PARTY.md); third-party material retains its original terms.

## License

The code (`emlang/`, `analysis/`, `scripts/`, the build scripts and the `Makefile`) is released under the
[MIT License](LICENSE). The manuscript, figures, PDFs and the experimental data produced for this study are
released under [CC BY 4.0](LICENSE-CC-BY-4.0.md).
