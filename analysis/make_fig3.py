"""Figure 3: reward and gradient grow different kinds of code. Dictionaries, consistency and
structure on the synthetic library; the naming law on 319 real ARC-1 rule families."""

from paths import ROOT, DATA, PAPER, FIGURES, TABLES, RESULTS
import json, os, re, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
from audited import decoding
fs.use(); C = fs.C
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(DATA, "runs"); AN = os.path.join(DATA, "analysis")
FAMILY = [("flipH", "geometry"), ("flipV", "geometry"), ("rot90", "geometry"), ("rot180", "geometry"), ("rot270", "geometry"), ("transpose", "geometry"),
          ("gravity_down", "gravity"), ("gravity_up", "gravity"), ("gravity_left", "gravity"), ("gravity_right", "gravity"),
          ("majority", "colour"), ("upscale2", "size"), ("crop", "size"), ("mirrorH", "size"), ("mirrorV", "size"), ("outline", "shape"),
          ("shift+1+0", "shift"), ("shift+0+1", "shift"), ("shift-1+0", "shift"), ("shift+0-1", "shift"),
          ("recolor7>6", "recolour"), ("recolor9>3", "recolour"), ("recolor9>1", "recolour"), ("recolor5>6", "recolour"), ("recolor6>9", "recolour"),
          ("swap95", "swap"), ("swap97", "swap"), ("swap48", "swap"), ("border1", "border"), ("border7", "border")]


def last(run, key="synth_hold2", field="exact", minstep=11000):
    f = os.path.join(RUNS, run, "log.jsonl")
    if not os.path.exists(f): return None
    ev = []
    for l in open(f, errors="ignore"):
        if '"kind": "eval"' not in l: continue
        try: ev.append(json.loads(l))
        except json.JSONDecodeError: continue
    ev = [e for e in ev if e["step"] >= minstep]
    return ev[-1].get(key, {}).get(field) if ev else None


def grp(pat, **kw):
    return [v for n in sorted(os.listdir(RUNS)) if re.match(pat + "$", n) for v in [last(n, **kw)] if v is not None]


def probe(name):
    f = os.path.join(AN, f"probe_{name}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def rearc(pat):
    out = {}
    for f in sorted(glob.glob(os.path.join(AN, "rearc_*.json"))):
        n = os.path.basename(f)[6:-5]
        if re.match(pat + "$", n):
            try:
                d = json.load(open(f)); out[n] = d["naming"]
            except Exception: pass
    return out


def raster(ax, dic, col, title, sel=slice(None)):
    """the modal message of every operation in `sel`; the thirty operations are drawn in two blocks of fifteen,
    so each row is twice as tall as it would be in one column and the names stay readable at text width"""
    names = [n for n, _ in FAMILY if n in dic][sel]
    M = np.array([dic[n]["modal"] for n in names]); pur = np.array([dic[n]["purity"] for n in names])
    V = int(M.max()) + 1
    base = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors
    soft = [tuple(0.62 * np.array(c) + 0.38) for c in base]          # blend towards white: quieter raster
    cmap = ListedColormap([soft[i % len(soft)] for i in range(max(V, 32))])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=max(V, 32) - 1, aspect="auto", interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(M[i, j]), fontsize=5.8, ha="center", va="center", color=C["ink"])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.2)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(range(1, M.shape[1] + 1), fontsize=6.6); ax.set_xlabel("position", fontsize=7.2, labelpad=1)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    fam = [f for n, f in FAMILY if n in dic][sel]
    for i in range(1, len(fam)):
        if fam[i] != fam[i - 1]: ax.axhline(i - 0.5, color="white", lw=1.4)
    ax.set_title(title, fontsize=8.5, color=col, loc="left")
    return names, pur


def main(out="figures/fig3_two_codes.pdf"):
    fig = plt.figure(figsize=(9.0, 7.4))
    # ------------------------------------------------ (a) dictionaries: modal message per primitive, in two blocks
    G = probe("cur_gumbel_sslpercept_nomdl_s0"); R = probe("cur_rl_sslpercept_nomdl_s0")
    n_ops = len([n for n, _ in FAMILY if n in G["dictionary"]]); half = (n_ops + 1) // 2
    for b, (sel, x0) in enumerate([(slice(0, half), 0.075), (slice(half, None), 0.575)]):
        ax1 = fig.add_axes([x0, 0.595, 0.135, 0.345]); ax2 = fig.add_axes([x0 + 0.150, 0.595, 0.135, 0.345])
        ax3 = fig.add_axes([x0 + 0.305, 0.595, 0.060, 0.345])
        if b == 0: fs.panel_letter(ax1, "a", -0.62, 1.10)
        names, pg = raster(ax1, G["dictionary"], C["grad"], "gradient code", sel)
        _, pr = raster(ax2, R["dictionary"], C["rew"], "reward code", sel); ax2.set_yticklabels([])
        y = np.arange(len(names))
        ax3.barh(y - 0.19, pg, height=0.36, color=C["grad"], alpha=0.85); ax3.barh(y + 0.19, pr, height=0.36, color=C["rew"], alpha=0.85)
        ax3.set_ylim(len(names) - 0.5, -0.5); ax3.set_yticks([]); ax3.set_xlim(0, 1); ax3.set_xticks([0, 0.5, 1]); ax3.set_xticklabels(["0", ".5", "1"], fontsize=6.6)
        ax3.set_xlabel("purity", fontsize=7.2, labelpad=1); ax3.spines["left"].set_visible(False)
        if b == 0:
            ax1.text(0.0, 1.13, "modal message for each single operation (one seed, self-supervised perception)",
                     fontsize=8.0, color=C["muted"], transform=ax1.transAxes)

    # ------------------------------------------------ (b) consistency and structure
    ax = fig.add_axes([0.075, 0.325, 0.30, 0.175]); fs.panel_letter(ax, "b", -0.19, 1.14)
    mets = [("synth_train1", "align_purity", "purity\n(singles)"), ("synth_hold2", "topsim", "topographic\nsimilarity"), ("synth_hold2", "posdis", "positional\ndisentangl."), ("synth_hold2", "reuse_name_A", "name reuse\nin pairs")]
    for k, (key, field, lab) in enumerate(mets):
        g = grp(r"cur_gumbel_sslpercept_nomdl_s\d", key=key, field=field); r = grp(r"cur_rl_sslpercept_nomdl_s\d", key=key, field=field)
        fs.points(ax, k - 0.17, g, C["grad"], width=0.26); fs.points(ax, k + 0.17, r, C["rew"], width=0.26)
        if field == "reuse_name_A":
            ch = grp(r"cur_rl_sslpercept_nomdl_s\d", key=key, field="reuse_name_chance")
            ax.plot([k - 0.4, k + 0.4], [np.mean(ch)] * 2, color=C["muted"], lw=0.9, ls=(0, (4, 3))); ax.text(k + 0.45, np.mean(ch) + 0.05, "chance", fontsize=5.8, color=C["muted"], ha="left", va="bottom")
    ax.set_xticks(range(len(mets))); ax.set_xticklabels([m[2] for m in mets], fontsize=6.6); ax.set_ylim(0, 1.0); ax.set_xlim(-0.6, len(mets) + 0.05)
    ax.set_title("consistency and structure of the two codes", fontsize=8.5, loc="left")
    ax.text(0.99, 0.97, "gradient", color=C["grad"], fontsize=7.2, weight="bold", ha="right", va="top", transform=ax.transAxes)
    ax.text(0.99, 0.88, "reward", color=C["rew"], fontsize=7.2, weight="bold", ha="right", va="top", transform=ax.transAxes)

    # ------------------------------------------------ (c-e) the naming law on real ARC-1 rules
    scale = rearc(r"scale\d+_(gumbel|rl)_percept_s\d"); full = rearc(r"rearc16_(gumbel|rl)_percept_s\d")
    def series(arm, field):
        xs, ys = [], []
        for n, nm in list(scale.items()) + list(full.items()):
            if f"_{arm}_" in n: xs.append(nm["n_train_rules"]); ys.append(nm[field])
        return np.array(xs), np.array(ys)
    panels = [("distinct_train_messages", "distinct messages /\nrules", True), ("purity_train_rules", "purity", False)]
    for k, (field, lab, ratio) in enumerate(panels):
        ax = fig.add_axes([0.475 + k * 0.170, 0.325, 0.125, 0.175]); fs.panel_letter(ax, "cd"[k], -0.30 if k == 0 else -0.12, 1.18)
        for arm, col, name in [("gumbel", C["grad"], "gradient"), ("rl", C["rew"], "reward")]:
            x, yv = series(arm, field)
            if ratio: yv = yv / x
            ax.scatter(x, yv, s=14, color=col, zorder=3, edgecolor="white", lw=0.4)
            ux = sorted(set(x)); ax.plot(ux, [np.mean(yv[x == u_]) for u_ in ux], color=col, lw=1.4, label=name)
        ax.set_xscale("log"); ax.set_xticks([30, 60, 120, 240]); ax.set_xticklabels(["30", "60", "120", "240"], fontsize=6.4); ax.minorticks_off()
        ax.set_ylim(0, 1.04); ax.set_title(lab, fontsize=7.4, loc="left", pad=3); ax.set_xlabel("rule families named", fontsize=7)
        if k == 0:
            ax.legend(fontsize=6.2, loc="center left", handlelength=1.3, borderpad=0.2)
            ax.text(0.0, 1.17, "naming 30 to 319 real ARC-1 rule families (RE-ARC)", fontsize=8.5, transform=ax.transAxes, va="bottom")
        else:
            ax.set_yticklabels([])
    # (e) held-out decoding: nearest neighbour (solid) and lookup (dashed); chance (dotted)
    DEC = decoding()
    ax = fig.add_axes([0.475 + 2 * 0.170, 0.325, 0.125, 0.175]); fs.panel_letter(ax, "e", -0.12, 1.18)
    for arm, col in [("gumbel", C["grad"]), ("rl", C["rew"])]:
        pts = [(d["n_rules"], d["nearest"], d["lookup"]) for n, d in DEC.items() if re.fullmatch(rf"scale\d+_{arm}_percept_s\d", n)]
        x = np.array([p[0] for p in pts]); ne = np.array([p[1] for p in pts]); lo = np.array([p[2] for p in pts])
        ux = sorted(set(x))
        ax.scatter(x, ne, s=14, color=col, zorder=3, edgecolor="white", lw=0.4)
        ax.plot(ux, [np.mean(ne[x == u_]) for u_ in ux], color=col, lw=1.4)
        ax.plot(ux, [np.mean(lo[x == u_]) for u_ in ux], color=col, lw=1.0, ls=(0, (3, 2)))
    ax.plot([30, 60, 120, 239], [1 / 30, 1 / 60, 1 / 120, 1 / 239], color=C["muted"], lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xticks([30, 60, 120, 240]); ax.set_xticklabels(["30", "60", "120", "240"], fontsize=6.4); ax.minorticks_off()
    ax.set_ylim(0, 1.04); ax.set_yticklabels([]); ax.set_xlabel("rule families named", fontsize=7)
    ax.set_title("rule decoded on new episodes", fontsize=7.4, loc="left", pad=3)
    ax.text(0.97, 0.97, "solid: nearest\ndashed: exact lookup\ndotted: chance", fontsize=5.4, color=C["muted"], ha="right", va="top", transform=ax.transAxes)

    # ------------------------------------------------ (f) what relieves saturation, at 239 rules
    ax = fig.add_axes([0.120, 0.055, 0.820, 0.155]); fs.panel_letter(ax, "f", -0.068, 1.16)
    DEC = decoding()
    conds = [("reward\nbaseline", r"scale240_rl_percept_s\d"), ("length\n8 to 16", r"scale240_rl_L16_s\d"), ("vocabulary\n32 to 128", r"scale240_rl_V128_s\d"),
             ("+ info. term\nweight 0.1", r"scale240_rl_div_s\d"), ("+ info. term\nweight 0.3", r"scale240_rl_div3_s\d"), ("gradient\nreference", r"scale240_gumbel_percept_s\d")]
    for k, (lab, pat) in enumerate(conds):
        vals = [d["nearest"] for n, d in DEC.items() if re.fullmatch(pat, n)]
        col = C["grad"] if "gradient" in lab else C["rew"]
        fs.points(ax, k, vals, col, width=0.4, size=18)
        ax.text(k, -0.035, f"n={len(vals)}", fontsize=5.4, color=C["muted"], ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.axhline(1 / 239, color=C["muted"], lw=0.8, ls=":")
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([c[0] for c in conds], fontsize=6.4); ax.tick_params(axis="x", pad=11)
    ax.set_ylim(0, 0.5); ax.set_ylabel("rule decoded,\n239 rules"); ax.set_xlim(-0.6, len(conds) - 0.4)
    ax.set_title("neither capacity increase lifts the ceiling; the information term at most modestly", fontsize=8.5, loc="left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PAPER, out.replace(".pdf", "." + ext)), bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
