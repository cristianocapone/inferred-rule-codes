# Provenance and scope

This standalone repository was extracted from the author's ARCAGI2 working repository.
The manuscript sources were the 18 September 2026 revision, including the Nature
Communications layout and the corrected primary audit; updated 19 September 2026 with the
corrected Bdup-triple audit and the completed pretrained-model pilot seeds (joint and staged
conditions, four seeds each, previously one or two), and on 21 September 2026 with second seeds for the
staged arm of libraries A17, A24 and Bdup and for A24 joint, and with three more seeds for each arm of the
primary contrasts (staged s5-s7, joint s3-s5, joint with the information term s3-s5). The same day the four
pretrained-model runs whose 16-bit adapters became non-finite were evaluated at their last snapshot before the event
(`data/llm_runs/<run>/evP1_snapshot.txt` names it), and three evaluations of an executor trained to read the true
rule in English were added from the companion study. On 22 September those executors were re-evaluated with the pilot's own
procedure, on the same programs and query episodes (`data/llm_runs/llm_oracle_s*/evP1_oracle.json`), which replaces
the companion study's own evaluation files; the audit's colour-map enumeration is exhaustive (the earlier cap of
5 000 maps changed no novel set); and the correlation analysis reads the per-program counts through the same
audit mask as every other script. Data were frozen when the bundle was
last updated; later changes in the original working directory do not propagate here.

`import_manifest.json` records source-relative paths and hashes at import. The source paths
are historical provenance, not runtime dependencies. `data/manifest.json` checks the current
frozen analysis inputs. Paths inside historical evaluation records may refer to checkpoints
on the original cluster; the figure pipeline does not open those checkpoints.

Only inputs actually read by a complete execution of the analysis and figure workflow are
included. GPU weights, optimizer states, machine-specific launchers, unrelated experimental
projects, submission correspondence, old manuscript drafts and review conversations are excluded.

## Split audit snapshot

The bundled 18 September audit (corrected: colour maps and tie-screening errors found by
external review, see `emlang/audit_group.py`) is used for all seven libraries, including
Bdup triples as of the 19 September update. An earlier version of this bundle fell back to
the 17 September audit for Bdup triples because the corrected result was not yet available;
that gap is closed, the fallback was removed from `analysis/audited.py` and
`analysis/make_supp_tables.py` (a missing audit is now an error), and the superseded 17 September
file was dropped from `data/`. The exact selected files are in `data/audits/` and the dependency
manifest; every frozen input is read by at least one script.

## Packaging changes

- Analysis inputs moved to `data/`; generated statistics moved to `results/`.
- Figure and table scripts moved to `analysis/`; manuscript sources remain in `paper/`.
- Figures are under `paper/figures/`, with the corresponding LaTeX paths updated.
- The build runner explicitly fixes `EMLANG_CANVAS=12`, hash seeding and Matplotlib cache paths.
- Figure jitter uses a stable SHA-256 seed instead of Python's process-dependent string hash.
- Plot fonts use Matplotlib's bundled DejaVu Sans for portability; one Figure 3 title is wrapped to avoid overlap.
- The journal layout (`analysis/make_nc.py`, the Nature Communications submission) is typeset with a
  Times-like serif body, sans-serif headings, single line spacing and the figures in the text; it is generated
  from `paper/main_nature.tex` and its text is identical.
- No scientific estimators, run-selection rules, training algorithms or manuscript claims were changed.

## Limits

The repository guarantees a documented route from the frozen logs and evaluation exports
to the generated figures, tables and PDFs. It does not include the trained checkpoints or
all external ARC/RE-ARC and model resources needed to retrain every experiment. Historical
analyses and the pretrained-model pilot retain the limitations described in the manuscript.

The complete GPU training was not repeated while packaging. A matching reproduction is
not an independent validation of the experimental design.
