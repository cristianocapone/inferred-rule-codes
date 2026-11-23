"""Cut panel a out of Figure 1 for the cover letter (figs/cover_panel_a.pdf).

The cover letter shows the human panel of Figure 1 -- worked examples, the rule passed by a gesture, the
new input answered correctly -- so that an editor sees the task before reading about it. Rather than draw
it twice, this crops the figure: it renders the page, finds the blank band that separates panel a from
panel b's heading (the last blank run in the top third), takes the ink above it, and sets that as the
crop box; the panel's title and subtitle are left out. Run it after make_fig1.py:  python analysis/make_cover_fig.py
"""
import os, subprocess, sys

from paths import FIGURES
import numpy as np
from PIL import Image

SRC = str(FIGURES / "fig1_paradigm.pdf")
DST = str(FIGURES / "cover_panel_a.pdf")
DPI, MARGIN = 200, 5

def band_near(rows, h, y, what):
    """the blank band of rows nearest the figure height y (0 at the foot, 1 at the head), as (first, last)"""
    target = int(round((1 - y) * h))
    near = [i for i in range(target - 15, target + 16) if rows[i] == 0]
    if not near:
        sys.exit(f"no blank row near {target} ({what}): Figure 1's layout has moved, and the cut needs a new fraction")
    lo = hi = min(near, key=lambda i: abs(i - target))
    while lo > 0 and rows[lo - 1] == 0: lo -= 1
    while hi + 1 < h and rows[hi + 1] == 0: hi += 1
    return lo, hi


def main():
    png = os.path.join(os.path.dirname(DST), ".cover_probe")
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", SRC, png], check=True)
    a = np.asarray(Image.open(png + ".png").convert("L"))
    os.remove(png + ".png")
    h, w = a.shape
    rows = (a < 200).sum(1)
    # Only the drawing goes on the letter: the panel's letter, title and subtitle stay in the figure, since the
    # letter's own text says the same thing. The head of the cut is the blank band between the subtitle and the
    # "worked examples" label (y = 0.914 in make_fig1.py); its foot the band above panel b's heading (y = 0.768).
    # Both cuts sit on the band's edge, with no margin across it, so nothing from the next line bleeds in.
    top = band_near(rows, h, 0.914, "below the subtitle")[1]
    bot = band_near(rows, h, 0.768, "above panel b")[0]
    cols = np.nonzero((a[top:bot] < 200).sum(0))[0]
    bp = 72.0 / DPI
    box = (cols[0] * bp - MARGIN, (h - bot) * bp, cols[-1] * bp + MARGIN, (h - top) * bp)
    subprocess.run(["pdfcrop", "--bbox", " ".join(f"{v:.1f}" for v in box), SRC, DST],
                   check=True, capture_output=True)
    out = subprocess.run(["pdfinfo", DST], capture_output=True, text=True).stdout
    print(f"panel a cut from rows {top} to {bot} of {h}; " + [l for l in out.splitlines() if l.startswith("Page size")][0])


if __name__ == "__main__":
    main()
