"""Shared figure conventions for the manuscript.

Style follows systems/computational neuroscience practice rather than ML engineering:
  * every figure opens with a schematic panel stating what is manipulated, beside the data
  * individual observations (seeds) are always drawn; summaries never replace them
  * one semantic colour per channel, held constant across every figure
  * open axes (no top/right spines), no gridlines, no chartjunk, muted palette
"""

from paths import ROOT, DATA, PAPER, FIGURES, TABLES, RESULTS
import matplotlib
import hashlib
import matplotlib.pyplot as plt
import numpy as np

# semantic colours - fixed across the whole paper
C = dict(
    cont="#7a7f87",       # continuous channel (no symbols)
    grad="#a8763e",       # gradient-trained discrete channel
    rew="#2f6f9f",        # reward-trained discrete channel
    human="#8a5a3b",      # human data / neuroscience panels
    hold="#b4443a",       # held-out conditions
    ink="#1b2430", muted="#6b7683", line="#c9d1d9", light="#eef1f4",
)
ARC_COLORS = ["#FFFFFF", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
              "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]


def use():
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["ink"], "axes.linewidth": 0.8,
        "xtick.color": C["ink"], "ytick.color": C["ink"], "text.color": C["ink"],
        "axes.labelcolor": C["ink"], "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "figure.dpi": 140, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_letter(ax, letter, dx=-0.18, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11, weight="bold", va="top", ha="left")


def points(ax, x, vals, color, width=0.30, mean_lw=1.8, jitter=0.055, size=15):
    """individual observations plus a mean bar - never a bar chart"""
    vals = np.asarray([v for v in vals if v is not None and v == v], float)
    if not len(vals): return
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(str(x).encode()).digest()[:4], "little") % 2**31)
    xs = x + rng.uniform(-jitter, jitter, len(vals))
    ax.scatter(xs, vals, s=size, facecolor=color, edgecolor="white", linewidth=0.5, zorder=3, alpha=0.95)
    m = vals.mean()
    ax.plot([x - width / 2, x + width / 2], [m, m], color=color, lw=mean_lw, solid_capstyle="round", zorder=4)
    if len(vals) > 1:
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        ax.plot([x, x], [m - se, m + se], color=color, lw=1.0, zorder=4)


def floor_line(ax, y, label="identity floor"):
    ax.axhline(y, color=C["muted"], lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(ax.get_xlim()[1], y, "  " + label, fontsize=6.8, color=C["muted"], va="center", ha="left")


def schematic_axes(fig, rect):
    ax = fig.add_axes(rect); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return ax
