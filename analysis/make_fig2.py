"""Figure 2: a discrete code emerges and composes; discreteness is not the cost, learning to see
through the channel is; the message carries the rule; a fresh learner can acquire the code."""

from paths import ROOT, DATA, PAPER, FIGURES, TABLES, RESULTS
import json, os, re, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
from audited import novel, audited
fs.use(); C = fs.C
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(DATA, "runs"); AN = os.path.join(DATA, "analysis")


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


def probe(pat):
    out = []
    for f in sorted(glob.glob(os.path.join(AN, "probe_*.json"))):
        n = os.path.basename(f)[6:-5]
        if re.match(pat + "$", n):
            try: out.append(json.load(open(f)))
            except Exception: pass
    return out


def transmit(pat):
    out = []
    for f in sorted(glob.glob(os.path.join(AN, "transmit_*.json"))):
        n = os.path.basename(f)[9:-5]
        if re.match(pat + "$", n):
            try: out.append(json.load(open(f))["curve"])
            except Exception: pass
    return out


def box(ax, x, y, w, h, text, fc, ec="none", fs_=7.2, color=None, weight="normal", style="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02", facecolor=fc, edgecolor=ec, lw=0.8, clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs_, color=color or C["ink"], weight=weight, style=style)


def arrow(ax, x0, y0, x1, y1, color=None, lw=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=lw, color=color or C["muted"]))


def main(out="figures/fig2_emergence.pdf"):
    fig = plt.figure(figsize=(9.0, 5.6))
    # ------------------------------------------------ (a) where perception sits
    ax = fs.schematic_axes(fig, [0.03, 0.55, 0.30, 0.42]); fs.panel_letter(ax, "a", -0.05, 1.02)
    ax.text(0.0, 0.97, "Where the sender learns to see", fontsize=8.6, weight="bold", va="top")
    for row, (lab, sub, col) in enumerate([
            ("perception learned through the channel", "gradient must cross 8 discrete symbols", C["hold"]),
            ("perception learned outside, then frozen", "self-supervised on demonstrations alone", C["rew"])]):
        y = 0.66 - row * 0.36
        ax.text(0.0, y + 0.19, lab, fontsize=7.2, color=col, weight="bold", va="center")
        ax.text(0.0, y + 0.13, sub, fontsize=6.4, color=C["muted"], va="center")
        box(ax, 0.02, y - 0.04, 0.22, 0.13, "perception", "#ffffff", ec=col, color=col)
        box(ax, 0.30, y - 0.04, 0.20, 0.13, "sender", C["light"])
        box(ax, 0.75, y - 0.04, 0.24, 0.13, "executor", C["light"])
        for j in range(4):
            ax.add_patch(FancyBboxPatch((0.545 + j * 0.045, y - 0.015), 0.036, 0.075, boxstyle="round,pad=0.002,rounding_size=0.008", facecolor=C["rew"], alpha=0.25, edgecolor="none", clip_on=False))
        arrow(ax, 0.24, y + 0.025, 0.30, y + 0.025); arrow(ax, 0.50, y + 0.025, 0.545, y + 0.025); arrow(ax, 0.725, y + 0.025, 0.75, y + 0.025)
        if row == 0:
            # the feedback runs below the row, so it crosses neither the boxes nor the symbols it has to pass through
            ax.annotate("", xy=(0.13, y - 0.055), xytext=(0.86, y - 0.055), clip_on=False,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, connectionstyle="arc3,rad=-0.13", mutation_scale=8))
            ax.text(0.50, y - 0.071, "learning signal for perception", fontsize=6.2, color=col, ha="center", va="center")
        else:
            ax.text(0.12, y - 0.10, "frozen", fontsize=6.4, color=col, ha="center", style="italic")
            ax.text(0.12, y - 0.16, "1-NN rule identity 0.93\n(chance 0.03)", fontsize=6.0, color=C["muted"], ha="center", va="top")

    # ------------------------------------------------ (b) exact match on unseen combinations
    ax = fig.add_axes([0.40, 0.60, 0.33, 0.34]); fs.panel_letter(ax, "b", -0.22, 1.10)
    conds = [("perception through\nthe channel", ""), ("frozen,\nsolver-derived", "percept_"), ("frozen,\nself-supervised", "sslpercept_")]
    cont = audited(r"cur_cont_s\d")
    ax.axhspan(np.mean(cont) - np.std(cont, ddof=1) / np.sqrt(len(cont)), np.mean(cont) + np.std(cont, ddof=1) / np.sqrt(len(cont)), color=C["cont"], alpha=0.18, lw=0)
    ax.axhline(np.mean(cont), color=C["cont"], lw=1.2, ls=(0, (5, 3)))
    ax.text(2.42, np.mean(cont) + 0.012, "continuous\n(no symbols)", fontsize=6.6, color=C["cont"], ha="right", va="bottom")
    for k, (lab, mid) in enumerate(conds):
        g = audited(rf"cur_gumbel_{mid}nomdl_s\d"); r = audited(rf"cur_rl_{mid}nomdl_s\d")
        fs.points(ax, k - 0.16, g, C["grad"], width=0.26); fs.points(ax, k + 0.16, r, C["rew"], width=0.26)
        ax.text(k - 0.16, -0.045, f"n={len(g)}", fontsize=5.6, color=C["grad"], ha="center", va="top", transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text(k + 0.16, -0.045, f"n={len(r)}", fontsize=5.8, color=C["rew"], ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in conds], fontsize=6.8); ax.tick_params(axis="x", pad=15)
    ax.set_ylim(0, 0.32); ax.set_xlim(-0.55, 2.55); ax.set_ylabel("exact match, new triples")
    ax.text(0.02, 0.97, "gradient", color=C["grad"], fontsize=7.4, weight="bold", transform=ax.transAxes, va="top")
    ax.text(0.02, 0.88, "reward", color=C["rew"], fontsize=7.4, weight="bold", transform=ax.transAxes, va="top")
    ax.set_title("discreteness is not the cost", fontsize=8.5, loc="left")

    # ------------------------------------------------ (c) generalisation profile
    ax = fig.add_axes([0.80, 0.60, 0.18, 0.34]); fs.panel_letter(ax, "c", -0.42, 1.10)
    splits = [(("log", "synth_train1"), "singles"), (("log", "synth_train2"), "trained\npairs"),
              (("clean", "all"), "held-out\npairs"), (("audited", "triples_audited"), "new\ntriples")]
    def vals(pat, src): return grp(pat, key=src[1]) if src[0] == "log" else (novel(pat, key=src[1]) if src[0] == "clean" else audited(pat, key=src[1]))
    for pat, col, lab in [(r"cur_cont_s\d", C["cont"], "continuous"), (r"cur_gumbel_sslpercept_nomdl_s\d", C["grad"], "gradient"), (r"cur_rl_sslpercept_nomdl_s\d", C["rew"], "reward")]:
        V = [vals(pat, src) for src, _ in splits]
        m = [np.mean(v) for v in V]; se = [np.std(v, ddof=1) / np.sqrt(len(v)) for v in V]
        ax.errorbar(range(len(splits)), m, yerr=se, color=col, lw=1.4, marker="o", ms=3.2, capsize=0, label=lab)
    ax.set_xticks(range(len(splits))); ax.set_xticklabels([s[1] for s in splits], fontsize=6.2)
    ax.set_ylim(0, 0.8); ax.set_ylabel("exact match"); ax.set_title("generalisation profile", fontsize=8.5, loc="left")
    ax.legend(loc="upper right", fontsize=6.4, handlelength=1.4, borderpad=0.2)

    # ------------------------------------------------ (d) the message carries the rule
    ax = fig.add_axes([0.075, 0.08, 0.22, 0.36]); fs.panel_letter(ax, "d", -0.32, 1.10)
    for k, (pat, col, lab) in enumerate([(r"cur_gumbel_sslpercept_nomdl_s\d", C["grad"], "gradient"), (r"cur_rl_sslpercept_nomdl_s\d", C["rew"], "reward")]):
        P = probe(pat)
        own = [p["lesion"]["base_acc"] for p in P]; rnd = [p["lesion"]["shuffled_msg_acc"] for p in P]
        for o, r in zip(own, rnd):
            ax.plot([k * 3 + 0, k * 3 + 1], [o, r], color=col, lw=0.7, alpha=0.5)
        fs.points(ax, k * 3 + 0, own, col, width=0.4); fs.points(ax, k * 3 + 1, rnd, col, width=0.4)
        ax.text(k * 3 + 0.5, 0.97, lab, color=col, fontsize=7.2, weight="bold", ha="center", transform=ax.get_xaxis_transform())
    ax.set_xticks([0, 1, 3, 4]); ax.set_xticklabels(["own", "other\nepisode's"] * 2, fontsize=6.4); ax.set_xlabel("message given to the executor", fontsize=7)
    ax.set_ylim(0, 1.0); ax.set_xlim(-0.7, 4.7); ax.set_ylabel("executor cell accuracy, held-out pairs")
    ax.set_title("the message carries the rule", fontsize=8.5, loc="left")

    # ------------------------------------------------ (e) a fresh learner acquires the code
    ax = fig.add_axes([0.385, 0.08, 0.27, 0.36]); fs.panel_letter(ax, "e", -0.24, 1.10)
    for pat, col, lab in [(r"cur_cont_s\d", C["cont"], "continuous"), (r"cur_gumbel_sslpercept_nomdl_s\d", C["grad"], "gradient"), (r"cur_rl_sslpercept_nomdl_s\d", C["rew"], "reward")]:
        curves = transmit(pat)
        if not curves: continue
        steps = [c["step"] for c in curves[0]]
        Y = np.array([[c["hold2_exact"] for c in cv] for cv in curves if len(cv) == len(steps)])
        for y in Y: ax.plot(steps, y, color=col, lw=0.6, alpha=0.35)
        ax.plot(steps, Y.mean(0), color=col, lw=1.8, label=f"{lab} (n={len(Y)})")
    ax.set_xlabel("training steps of a new executor, from messages alone"); ax.set_ylabel("exact match, held-out pairs")
    ax.set_title("a new learner acquires the code", fontsize=8.5, loc="left"); ax.legend(loc="lower right", fontsize=6.4, handlelength=1.4)
    ax.set_ylim(0, None)

    # ------------------------------------------------ (f) per-position lesion
    ax = fig.add_axes([0.745, 0.08, 0.235, 0.36]); fs.panel_letter(ax, "f", -0.30, 1.10)
    for pat, col, lab in [(r"cur_gumbel_sslpercept_nomdl_s\d", C["grad"], "gradient"), (r"cur_rl_sslpercept_nomdl_s\d", C["rew"], "reward")]:
        P = probe(pat)
        D = np.array([[pp["drop"] for pp in p["lesion"]["per_pos"]] for p in P])
        for d in D: ax.plot(range(1, D.shape[1] + 1), d, color=col, lw=0.6, alpha=0.35)
        ax.plot(range(1, D.shape[1] + 1), D.mean(0), color=col, lw=1.8, marker="o", ms=3, label=lab)
    ax.set_xlabel("message position lesioned"); ax.set_ylabel("drop in cell accuracy")
    ax.set_title("every position matters", fontsize=8.5, loc="left"); ax.set_xticks(range(1, 9)); ax.legend(fontsize=6.4, handlelength=1.4)
    ax.set_ylim(0, None)
    os.makedirs(str(FIGURES), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PAPER, out.replace(".pdf", "." + ext)), bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
