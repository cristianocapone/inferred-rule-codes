"""Figure 1: the human capacity (Kean et al. 2026) and the game that models it.
Grids are drawn in their own inset axes so aspect ratios are always correct."""
import os, sys

from paths import ROOT, DATA, PAPER, FIGURES, TABLES, RESULTS
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Wedge, Polygon
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("EMLANG_CANVAS", "12")
from emlang.gen import SynthGen

ARC = ListedColormap(["#FFFFFF", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
                      "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"])
INK, MUTED, LINE = "#1b2430", "#6b7683", "#c9d1d9"
SPK, EXE, SYM, HOLD = "#2f6f9f", "#8a6d3b", "#3d6e9e", "#b4443a"
FW, FH = 9.0, 8.4          # tall enough that a, b and c can be drawn large, with a row of four charts under them
K = 1.25                   # font scale relative to the first version of the figure
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "pdf.fonttype": 42})


def put_grid(fig, g, cx, cy, cell_in=0.085, ring=None):
    """draw grid g centred at figure coords (cx, cy); returns (half-width, half-height) in fig coords"""
    g = np.asarray(g); H, W = g.shape
    w, h = W * cell_in / FW, H * cell_in / FH
    ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h]); ax.set_zorder(3)
    ax.imshow(g, cmap=ARC, vmin=0, vmax=9, interpolation="nearest")
    ax.set_xticks(np.arange(-.5, W, 1), minor=True); ax.set_yticks(np.arange(-.5, H, 1), minor=True)
    ax.grid(which="minor", color=LINE, lw=0.4); ax.tick_params(which="both", length=0, labelleft=False, labelbottom=False)
    for s in ax.spines.values():
        s.set_edgecolor(ring or LINE); s.set_linewidth(1.8 if ring else 0.5)
    return w / 2, h / 2


def put_list(fig, vals, cx, cy, cell_in=0.16):
    n = len(vals)
    w, h = n * cell_in / FW, cell_in / FH
    ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h]); ax.set_zorder(3)
    ax.imshow(np.zeros((1, n)), cmap=ListedColormap(["#f4f6f8"]), vmin=0, vmax=1)
    for j, v in enumerate(vals):
        ax.text(j, 0, str(v), ha="center", va="center", fontsize=7.5 * K, color=INK)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True); ax.grid(which="minor", color=MUTED, lw=0.5)
    ax.tick_params(which="both", length=0, labelleft=False, labelbottom=False)
    for s in ax.spines.values(): s.set_edgecolor(MUTED); s.set_linewidth(0.5)
    return w / 2, h / 2


def main(out=str(FIGURES / "fig1_paradigm.pdf")):
    G = SynthGen(0, size_lo=3, size_hi=3, n_train3=600)
    rng = np.random.default_rng(11)
    fig = plt.figure(figsize=(FW, FH))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off"); bg.set_zorder(1)
    CELL = 0.115                                  # grid cell, inches: the drawings fill the width they are given

    def txt(x, y, s, size=8, color=INK, **kw): bg.text(x, y, s, fontsize=size * K, color=color, **kw)
    def arr(x1, y1, x2, y2, color=MUTED, lw=1.3):
        bg.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))
    def panel(x, y, letter, title, sub=None):
        txt(x, y, letter, 13, INK, weight="bold")
        txt(x + 0.023, y, title, 10.5, INK, va="center")
        if sub: txt(x + 0.023, y - 0.026, sub, 7.6, MUTED, va="center")
    def rbox(x, y, w, h, label, color, sub):
        """a labelled box; the label and its gloss are placed to stay inside it"""
        bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
                                    facecolor=color, alpha=0.15, edgecolor=color, lw=0.9, zorder=2))
        txt(x + w / 2, y + h * 0.62, label, 10.5, color, ha="center", va="center", weight="bold")
        txt(x + w / 2, y + h * 0.27, sub, 6.6, MUTED, ha="center", va="center")
    def pair(gin, gout, xl, cy, gap=0.026, ring=None):
        """input grid, arrow, output grid, from the left edge xl; returns the right edge"""
        hi, ho = gin.shape[1] * CELL / FW / 2, gout.shape[1] * CELL / FW / 2
        put_grid(fig, gin, xl + hi, cy, cell_in=CELL)
        txt(xl + 2 * hi + gap / 2, cy, r"$\rightarrow$", 9.5, MUTED, ha="center", va="center")
        put_grid(fig, gout, xl + 2 * hi + gap + ho, cy, cell_in=CELL, ring=ring)
        return xl + 2 * hi + gap + 2 * ho

    # ============================================================ (a) the human capacity
    panel(0.030, 0.962, "a", "People abstract a rule from examples and communicate it without words",
          "Severe aphasia: inductive reasoning intact, the rule conveyed by gesture or sketch (Kean et al., 2026)")
    txt(0.060, 0.906, "worked examples", 7.4, MUTED)
    ex = [np.array([[2, 2, 0], [0, 4, 0], [0, 0, 0]]), np.array([[0, 3, 0], [0, 3, 7], [0, 0, 0]])]
    for k, gin in enumerate(ex):                       # the hidden rule of these examples is a quarter turn
        pair(gin, np.rot90(gin, -1), 0.070, 0.872 - k * 0.048, gap=0.020)
    arr(0.182, 0.848, 0.232, 0.848)

    FIG_C = "#59636f"                              # the two people are drawn as a plain silhouette

    def person(cx, cy, label, arm_up=False):
        """head and shoulders, the way a person is drawn on a sign; one arm raised where the rule is passed on"""
        th = np.linspace(np.pi, 0, 80)             # the shoulders: half an ellipse, closed along a flat base
        bust = np.column_stack([cx + 0.035 * np.cos(th), cy - 0.032 + 0.058 * np.sin(th)])
        bg.add_patch(Polygon(bust, closed=True, facecolor=FIG_C, edgecolor="none", zorder=3))
        bg.add_patch(Ellipse((cx, cy + 0.046), 0.032, 0.032 * FW / FH, facecolor=FIG_C, edgecolor="none", zorder=3))
        if arm_up:                                 # the raised arm, a rounded stroke out of the shoulder
            bg.plot([cx + 0.024, cx + 0.052], [cy + 0.008, cy + 0.046], color=FIG_C, lw=4.6,
                    zorder=2, solid_capstyle="round")
        txt(cx, cy - 0.050, label, 7.4, MUTED, ha="center", va="top")

    def rotation_gesture(cx, cy, r=0.036):
        """the turn itself: a thick ring, open at the lower right, with a solid head that points the way round"""
        band = r * 0.34
        bg.add_patch(Wedge((cx, cy), r, -18, 202, width=band, facecolor=SYM, edgecolor="none", zorder=3))
        a = np.deg2rad(-18); rm = r - band / 2     # the head sits on the band and continues it clockwise
        p = np.array([cx + rm * np.cos(a), cy + rm * np.sin(a)])
        t = np.array([np.sin(a), -np.cos(a)]); n = np.array([np.cos(a), np.sin(a)])
        bg.add_patch(Polygon([p + t * 0.025, p + n * 0.014, p - n * 0.014],
                             closed=True, facecolor=SYM, edgecolor="none", zorder=3))

    person(0.297, 0.848, "sender", arm_up=True)
    rotation_gesture(0.447, 0.858)
    txt(0.447, 0.802, "a turn of the hand,\nnot a word", 7.4, SYM, ha="center", va="top")
    arr(0.520, 0.848, 0.570, 0.848)
    person(0.632, 0.848, "executor")
    arr(0.692, 0.848, 0.742, 0.848)
    qin = np.array([[0, 0, 6], [0, 1, 6], [0, 0, 0]])
    xe = pair(qin, np.rot90(qin, -1), 0.782, 0.848, gap=0.020, ring="#2ECC40")
    txt((0.782 + xe) / 2, 0.796, "new input, correct output", 7.4, MUTED, ha="center", va="top")

    # ============================================================ (b) the game
    panel(0.030, 0.752, "b", "The same game between two networks",
          "over a private channel of eight invented symbols")
    yb, BH = 0.628, 0.105
    ep = G.make_episode(rng, "train1", n_demos=2)
    txt(0.060, 0.697, "worked examples", 7.4, MUTED)
    x = 0.070
    for gi, go in ep["demos"][:2]:
        x = pair(np.asarray(gi), np.asarray(go), x, yb, gap=0.020) + 0.020
    arr(x - 0.004, yb, x + 0.023, yb)
    xs_box = x + 0.030
    rbox(xs_box, yb - BH / 2, 0.125, BH, "sender", SPK, "sees the examples")
    arr(xs_box + 0.125, yb, xs_box + 0.152, yb)
    msg = [7, 3, 3, 1, 5, 3, 3, 3]
    xm = xs_box + 0.159
    for j, m in enumerate(msg):
        bg.add_patch(FancyBboxPatch((xm + j * 0.0235, yb - 0.024), 0.0195, 0.048,
                                    boxstyle="round,pad=0.002,rounding_size=0.005",
                                    facecolor=SYM, alpha=0.18 + 0.10 * (m % 3), edgecolor="none", zorder=2))
        txt(xm + 0.0098 + j * 0.0235, yb, str(m), 7.4, SYM, ha="center", va="center")
    xm_end = xm + 7 * 0.0235 + 0.0195
    txt((xm + xm_end) / 2, yb - 0.068, "eight invented symbols: the only route from the examples to the answer",
        7.2, SYM, ha="center", va="top")
    arr(xm_end + 0.008, yb, xm_end + 0.035, yb)
    xe_box = xm_end + 0.042
    rbox(xe_box, yb - BH / 2, 0.125, BH, "executor", EXE, "never sees them")
    qi, qo = ep["query"]
    xq = xe_box + 0.0625
    hw, hh = put_grid(fig, np.asarray(qi), xq, yb + 0.086, cell_in=CELL)
    txt(xq + hw + 0.008, yb + 0.086, "new input", 7.4, MUTED, ha="left", va="center")
    arr(xq, yb + 0.086 - hh - 0.004, xq, yb + BH / 2 + 0.004)
    arr(xe_box + 0.125 + 0.008, yb, xe_box + 0.125 + 0.035, yb)
    xo = xe_box + 0.125 + 0.058
    hw, hh = put_grid(fig, np.asarray(qo), xo, yb, cell_in=CELL, ring="#2ECC40")
    txt(xo, yb + hh + 0.013, "output", 7.4, "#2a8a3e", ha="center")

    # ============================================================ (c) the split
    panel(0.030, 0.472, "c", "What is trained, and what is tested",
          "single operations, pairs and triples are trained; the test is held-out triples that compute a new transformation")
    for i, (lab, split, col) in enumerate([("trained", "train1", MUTED), ("trained", "train2", MUTED),
                                           ("trained", "train3", MUTED), ("held out", "len3", HOLD)]):
        compact = lambda pr: not any(G.names[k] in ("upscale2", "mirrorH", "mirrorV") for k in pr)   # keep the drawing compact
        if split == "len3":          # a held-out triple that no training program computes, even through augmentation,
            import json as _json     # sharing no operation with the trained triple drawn beside it
            _aud = _json.load(open(os.path.join(DATA, "audits", "20260918", "group_audit_A_len3.json")))
            _progs = [pr for pr in _aud["novel_group"] if compact(pr) and not set(pr) & set(shown_triple)]
        else:
            pool = {"train1": G.train1, "train2": G.train2, "train3": G.train3}[split]
            _progs = [list(pr) for pr in pool if compact(pr)]
        for _ in range(60):          # an example earns its place only if the rule is visible in it
            e = G.make_episode(rng, "hold2", n_demos=1, prog=list(_progs[int(rng.integers(len(_progs)))]))
            gi, go = np.asarray(e["demos"][0][0]), np.asarray(e["demos"][0][1])
            if gi.shape == go.shape and np.count_nonzero(gi != go) >= 3 and len(set(gi.ravel()) - {0}) >= 2: break
        if split == "train3": shown_triple = list(e["prog"])
        xl = 0.070 + i * 0.226
        gi, go = e["demos"][0]
        xr = pair(np.asarray(gi), np.asarray(go), xl, 0.336, gap=0.020)
        txt((xl + xr) / 2, 0.414, lab, 7.8, col, weight="bold", ha="center")
        txt((xl + xr) / 2, 0.389, r" $\circ$ ".join(G.names[k] for k in e["prog"]), 6.9, MUTED, ha="center")
    txt(0.030, 0.258, "The sender must name what it sees; the executor must act on the name alone.", 8.8, INK, style="italic")

    # ============================================================ (d) the answer in brief: four square charts
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from audited import audited, decoding
    import json as _json, re as _re
    GRAD, REW, COLL, CONT = "#a8763e", "#2f6f9f", "#b4443a", "#8a8f98"
    panel(0.030, 0.206, "d", "In brief")
    AX_Y, AX_H, AX_W = 0.048, 0.116, 0.150
    xs = [0.090, 0.310, 0.530, 0.750]

    def frame(k, title):
        ax = fig.add_axes([xs[k], AX_Y, AX_W, AX_H]); ax.set_zorder(2)
        ax.set_title(title, fontsize=6.8 * K, loc="left", pad=3)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6.2 * K)
        return ax

    # d1: the code carries the rule, whatever the channel
    ax = frame(0, "a code emerges")
    for k, (lab, pat, col) in enumerate([("reward", r"cur_rl_sslpercept_nomdl_s\d", REW),
                                         ("gradient", r"cur_gumbel_sslpercept_nomdl_s\d", GRAD),
                                         ("vectors", r"cur_cont_s\d", CONT)]):
        v = audited(pat); ax.bar(k, np.mean(v), width=0.6, color=col, alpha=0.85)
        ax.scatter(np.full(len(v), k) + np.linspace(-0.1, 0.1, len(v)), v, s=6, color="white", edgecolor=col, lw=0.5, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(["reward", "gradient", "vectors"], fontsize=6.2 * K)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 0.24); ax.set_yticks([0, 0.1, 0.2])
    ax.set_ylabel("exact match,\nnew triples", fontsize=6.6 * K, labelpad=2)

    # d2: reward has a degenerate solution, and two ways around it
    ax = frame(1, "and reward can collapse")
    conds = [("joint", r"joint_rl_sslpercept_nomdl_s\d", COLL), ("joint +\ninfo. term", r"joint_rl_sslpercept_div\d?_s\d", REW),
             ("staged", r"cur_rl_sslpercept_nomdl_s\d", REW)]
    for k, (lab, pat, col) in enumerate(conds):
        v = audited(pat); ax.bar(k, np.mean(v), width=0.6, color=col, alpha=0.85)
        ax.scatter(np.full(len(v), k) + np.linspace(-0.1, 0.1, len(v)), v, s=6, color="white", edgecolor=col, lw=0.5, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in conds], fontsize=6.2 * K)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 0.24); ax.set_yticks([0, 0.1, 0.2]); ax.set_yticklabels([])

    # d3: competence goes with what the message says about the rule
    ax = frame(2, "competence follows\nwhat the message says")
    rows = _json.load(open(os.path.join(RESULTS, "info_competence.json")))["rows"]
    rw = [r for r in rows if r["group"] == "reward"]
    ax.scatter([100 * r["triple_nearest"] for r in rw], [100 * r["novel"] for r in rw], s=9,
               color="white", edgecolor=REW, lw=0.8, zorder=3)
    ax.set_xlabel("message identifies\nthe triple (%)", fontsize=6.6 * K, labelpad=1)
    ax.set_ylabel("exact match (%)", fontsize=6.6 * K, labelpad=2)
    ax.set_xlim(0, 40); ax.set_ylim(0, 20); ax.set_xticks([0, 20, 40]); ax.set_yticks([0, 10, 20])
    ax.text(0.96, 0.06, r"$\rho = 0.92$", fontsize=6.6 * K, color=REW, ha="right", transform=ax.transAxes)

    # d4: the two learning signals build different kinds of code
    ax = frame(3, "labels versus a graded map")
    D = decoding()
    def dm(pat, key): return float(np.mean([d[key] for n, d in D.items() if _re.fullmatch(pat, n)]))
    x = np.arange(2); w = 0.34
    for j, (key, alpha) in enumerate([("nearest", 0.9), ("lookup", 0.45)]):
        ax.bar(x + (j - 0.5) * w, [dm(r"scale240_rl_percept_s\d", key), dm(r"scale240_gumbel_percept_s\d", key)],
               width=w, color=[REW, GRAD], alpha=alpha)
    ax.set_xticks(x); ax.set_xticklabels(["reward", "gradient"], fontsize=6.2 * K)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 0.52); ax.set_yticks([0, 0.2, 0.4])
    ax.set_ylabel("rule decoded,\n239 rules", fontsize=6.6 * K, labelpad=2)
    ax.text(0.03, 0.97, "dark: region\nlight: exact", fontsize=5.8 * K, color=MUTED,
            ha="left", va="top", transform=ax.transAxes)

    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=170)
    print("saved", out)


if __name__ == "__main__":
    main()
