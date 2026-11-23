#!/bin/sh
# Rebuild arxiv-submission.zip from the manuscript sources. Run from the repository root: sh arxiv/build.sh
# The package is the Nature Communications layout without the review line numbers, its five figures, a compiled
# bibliography (arXiv then does not run bibtex) and the supplement as an ancillary file.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
PY=${PY:-python3}
mkdir -p "$WORK/figures" "$WORK/anc"
cp "$ROOT/paper/main_nc.tex" "$WORK/main.tex"
cp "$ROOT/paper/readerbox.tex" "$ROOT/paper/refs.bib" "$ROOT/paper/naturemag.bst" "$WORK/"
cp "$ROOT"/paper/figures/fig[1-5]_*.pdf "$WORK/figures/"
cp "$ROOT/pdf/supplement.pdf" "$WORK/anc/supplementary_information.pdf"
"$PY" "$ROOT/arxiv/strip_lineno.py" "$WORK/main.tex"
sed -i '' 's|{figs/|{figures/|g' "$WORK/main.tex" 2>/dev/null || sed -i 's|{figs/|{figures/|g' "$WORK/main.tex"
cd "$WORK"
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
bibtex main >/dev/null 2>&1                      # main.bbl is compiled while refs.bib and the style are here
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
SUMMARY="undefined references: $(grep -c 'undefined' main.log || true)"
SUMMARY="$SUMMARY, pages: $(pdfinfo main.pdf | awk '/^Pages/{print $2}')"
rm -f refs.bib naturemag.bst main.aux main.log main.out main.blg main.pdf
rm -f "$ROOT/arxiv/arxiv-submission.zip"
zip -qr "$ROOT/arxiv/arxiv-submission.zip" .
echo "wrote arxiv/arxiv-submission.zip -- $SUMMARY"
