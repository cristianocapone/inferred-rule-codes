# Figure and table reproduction map

Run `python scripts/reproduce.py figures` from the repository root. The command first
recalculates the information–competence statistics required by Figure 4, then builds all
five figures as vector PDFs and PNGs in `paper/figures/`.

| Figure | Script | Primary inputs |
|---|---|---|
| 1. Task and overview | `analysis/make_fig1.py` | Seeded `emlang/gen.py` examples; corrected A audit; `data/analysis/clean4/`; `data/analysis/decode/decoding.json` |
| 2. Emergence, perception, intervention and transmission | `analysis/make_fig2.py` | `data/runs/*/log.jsonl`; per-program results in `clean3/` and `clean4/`; `probe_*.json`; `transmit_*.json` |
| 3. Code structure and naming scale | `analysis/make_fig3.py` | Training evaluation logs; `probe_*.json` dictionaries; `rearc_*.json`; family decoding exports |
| 4. Collapse and information | `analysis/make_fig4.py` | Corrected split masks; `clean4/`; `codeinfo/` and `codeinfo3/`; `results/info_competence.json`; evaluation logs |
| 5. Human and model regimes | `analysis/make_fig_kirby.py` | `data/human/CornishData_KirbyCornishSmith2008.xlsx`; model evaluation logs, read with the Figure 4 diagnostic |

Shared conventions are in `analysis/figstyle.py`; all filesystem roots are defined in
`analysis/paths.py`. `analysis/audited.py` applies the frozen novelty masks to the stored
per-program exact-match counts.

Run `python scripts/reproduce.py tables` for the nine generated table fragments:

| Builder | Output in `paper/supp_tables/` |
|---|---|
| `analysis/make_supp_tables.py` | `s1_novelty.tex`, `s1_results.tex`, `s2_decoding.tex`, `s2_decomposition.tex`, `s6_stability.tex` |
| `analysis/info_competence.py` | `info_competence.tex` and `results/info_competence.json` |
| `analysis/coadapt_table.py` | `coadapt.tex` and `results/coadapt.json` |
| `analysis/llm_generality.py` | `llm_generality.tex` and `results/llm_generality.json` |
| `analysis/stats.py` | `stats_contrasts.tex` and `results/stats.json` |

Some additional supplementary tables are written directly in the supplied LaTeX. Those
are compiled as supplied; they are not claimed to have independent table generators here.

`data/manifest.json` records every retained input with a SHA-256 digest. The per-builder
dependency lists are in `docs/input_dependencies.json`.
