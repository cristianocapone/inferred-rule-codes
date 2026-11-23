"""Figure 4: the curriculum is a remedy for collapse, not a requirement - and the same two regimes
appear in human iterated learning. (a) hypothesis; (b) dissociation; (c) the diagnostic; (d) rescue by
an information term; (e) a second library where nothing collapses and nothing is needed; (f) human chains."""

from paths import ROOT, DATA, PAPER, FIGURES, TABLES, RESULTS
import json, os, re, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import figstyle as fs
from audited import novel, audited
fs.use(); C = fs.C
RUNS = os.path.join(DATA, "runs")
KIRBY = os.path.join(DATA, "human", "CornishData_KirbyCornishSmith2008.xlsx")


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


def uniq_per_rule(pat, n_rules):
    """distinct messages per distinct held-out rule: the logged fraction divides by 200 episodes,
    which cycle over the held-out pairs (212 in library A -> 200 distinct; 68 in library B)."""
    return [u * 200 / n_rules for u in grp(pat, field="msg_unique_frac")]


def main(out=str(FIGURES / "fig4_collapse.pdf")):
    fig = plt.figure(figsize=(9.0, 6.0))
    # ---------------- (a) hypothesis
    ax = fs.schematic_axes(fig, [0.03, 0.56, 0.27, 0.40]); fs.panel_letter(ax, "a", -0.05, 1.02)
    ax.text(0.0, 0.98, "A scalar reward admits a\ndegenerate solution", fontsize=8.6, weight="bold", va="top")
    ax.text(0.0, 0.82, "the sender says the same thing whatever it\nsees; the executor learns to ignore it", fontsize=6.8, color=C["muted"], va="top")
    for lab, msgs, col, y in [("informative code", [[7, 3, 1], [2, 9, 4], [5, 1, 8]], C["rew"], 0.58), ("collapsed code", [[3, 3, 3], [3, 3, 3], [3, 3, 3]], C["hold"], 0.32)]:
        ax.text(0.0, y + 0.08, lab, fontsize=7.4, color=col, weight="bold")
        for r, m in enumerate(msgs):
            ax.text(0.19, y - r * 0.055, f"task {r+1}", fontsize=6.3, color=C["muted"], ha="right", va="center")
            for j, sy in enumerate(m):
                ax.add_patch(FancyBboxPatch((0.22 + j * 0.075, y - r * 0.055 - 0.02), 0.062, 0.04, boxstyle="round,pad=0.002,rounding_size=0.01", facecolor=col, alpha=0.18 + 0.10 * (sy % 3), edgecolor="none"))
                ax.text(0.251 + j * 0.075, y - r * 0.055, str(sy), fontsize=6.3, color=col, ha="center", va="center")
        ax.text(0.50, y - 0.055, "distinct messages" if col == C["rew"] else "one message for all", fontsize=6.3, color=C["muted"], va="center")
    ax.text(0.0, 0.0, "If staged experience only keeps the sender out of this\nsolution, an explicit pressure against it should do the same job.", fontsize=6.6, style="italic", va="bottom")

    # ---------------- (b) the dissociation
    ax = fig.add_axes([0.385, 0.62, 0.30, 0.32]); fs.panel_letter(ax, "b", -0.22, 1.12)
    conds = [("staged", "cur"), ("joint", "joint"), ("reverse", "rev")]
    for k, (lab, pre) in enumerate(conds):
        fs.points(ax, k - 0.16, audited(rf"{pre}_rl_sslpercept_nomdl_s\d"), C["rew"], width=0.26)
        fs.points(ax, k + 0.16, audited(rf"{pre}_gumbel_sslpercept_nomdl_s\d"), C["grad"], width=0.26)
    ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in conds]); ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, 0.32)
    ax.set_ylabel("exact match, new triples"); ax.set_title("training order matters only for reward", fontsize=8.5, loc="left")
    ax.text(0.99, 0.97, "gradient", color=C["grad"], fontsize=7.2, weight="bold", ha="right", va="top", transform=ax.transAxes)
    ax.text(0.99, 0.87, "reward", color=C["rew"], fontsize=7.2, weight="bold", ha="right", va="top", transform=ax.transAxes)
    ax.set_xlabel("order of experience (singles, pairs, triples)", fontsize=7)

    # ---------------- (c) the diagnostic
    ax = fig.add_axes([0.745, 0.62, 0.24, 0.32]); fs.panel_letter(ax, "c", -0.30, 1.12)
    for k, (lab, pat, col) in enumerate([("staged", r"cur_rl_sslpercept_nomdl_s\d", C["rew"]), ("joint", r"joint_rl_sslpercept_nomdl_s\d", C["hold"]),
                                         ("reverse", r"rev_rl_sslpercept_nomdl_s\d", C["hold"]), ("joint +\ninfo. term", r"joint_rl_sslpercept_div\d?_s\d", C["rew"]),
                                         ("joint, changed-\ncell reward", r"joint_rl_sslpercept_chg_s\d", C["muted"])]):
        fs.points(ax, k, uniq_per_rule(pat, 200), col, width=0.4)
    ax.set_xticks(range(5)); ax.set_xticklabels(["staged", "joint", "reverse", "joint\n+ info.", "joint,\nchanged\ncells"], fontsize=6.3)
    ax.set_ylabel("distinct messages per rule"); ax.set_ylim(0, 0.5); ax.set_xlim(-0.5, 4.5)
    ax.set_title("collapse, measured", fontsize=8.5, loc="left")

    # ---------------- (d) competence follows the information the code carries about the rule
    import info_competence as icm
    rows = icm.load()
    ax = fig.add_axes([0.065, 0.10, 0.25, 0.34]); fs.panel_letter(ax, "d", -0.30, 1.12)
    def style(r):
        c = r["condition"]
        if r["group"] == "gradient": return dict(facecolor=C["grad"], edgecolor="white")
        if r["group"] == "soft": return dict(facecolor="white", edgecolor=C["muted"])
        if "changed-cell" in c or "slower" in c: return dict(facecolor=C["muted"], edgecolor="white")
        if c.startswith("joint +") or c.startswith("staged"): return dict(facecolor=C["rew"], edgecolor="white")
        return dict(facecolor=C["hold"], edgecolor="white")
    for r in rows:
        ax.scatter(100 * r["triple_nearest"], r["novel"], s=16, linewidth=0.7, zorder=3, **style(r))
    res = json.load(open(os.path.join(RESULTS, "info_competence.json")))
    ax.text(0.03, 0.97, f"reward runs: \u03c1 = {res['triple_nearest']['rho']:.2f} (n = {res['n_reward_runs']})\nwith open circles: \u03c1 = {res['triple_nearest_with_soft']['rho']:.2f} (n = {res['triple_nearest_with_soft']['n']})", transform=ax.transAxes, fontsize=6.2, va="top")
    ax.text(6.0, 0.075, "joint,\nreversed", fontsize=6.0, color=C["hold"], va="bottom", ha="right")
    ax.text(14.5, 0.012, "changed cells,\nslower exploration", fontsize=6.0, color=C["muted"], va="bottom")
    ax.text(13.0, 0.165, "staged,\njoint + info. term", fontsize=6.0, color=C["rew"], va="bottom", ha="center")
    ax.text(50.5, 0.212, "gradient", fontsize=6.0, color=C["grad"], va="bottom", ha="center")
    ax.text(39.0, 0.125, "soft policy\n(stability\npressure)", fontsize=6.0, color=C["muted"], va="top", ha="left")
    XMAX = 5 * np.ceil(max(100 * r["triple_nearest"] for r in rows) / 5 + 0.5)
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 0.32)
    ax.set_xlabel("message identifies the new triple (% of episodes)", fontsize=7); ax.set_ylabel("exact match, new triples")
    ax.set_title("competence goes with information", fontsize=8.5, loc="left")

    # ---------------- (e) five libraries: joint against staged, reward channel
    ax = fig.add_axes([0.385, 0.10, 0.30, 0.34]); fs.panel_letter(ax, "e", -0.22, 1.12)
    LIBS = [("A30", r"joint_rl_sslpercept_nomdl_s\d", r"cur_rl_sslpercept_nomdl_s\d", r"cur_cont_s\d", 200, 30),
            ("A24", r"A24_joint_rl_s\d", r"A24_cur_rl_s\d", r"A24_cont_s\d", 138, 24),
            ("A17", r"A17_joint_rl_s\d", r"A17_cur_rl_s\d", r"A17_cont_s\d", 67, 17),
            ("A17r", r"A17r_joint_rl_s\d", r"A17r_cur_rl_s\d", r"A17r_cont_s\d", 67, 17),
            ("A\u2212conf", r"Aclean_joint_rl_s\d", r"Aclean_cur_rl_s\d", r"Aclean_cont_s\d", 124, 23),
            ("B", r"libB_joint_rl_percept_s\d", r"libB_rl_percept_s\d", r"libB_cont_s\d", 68, 17),
            ("B+dup", r"Bdup_joint_rl_s\d", r"Bdup_cur_rl_s\d", r"Bdup_cont_s\d", 126, 23)]
    for k, (lab, pj, pc, pcont, npair, nops) in enumerate(LIBS):
        cont = audited(pcont)
        if cont: ax.plot([k - 0.42, k + 0.42], [np.mean(cont)] * 2, color=C["cont"], lw=1.1, ls=(0, (4, 3)))
        fs.points(ax, k - 0.17, audited(pj), C["hold"], width=0.26); fs.points(ax, k + 0.17, audited(pc), C["rew"], width=0.26)
    ax.set_xticks(range(len(LIBS))); ax.set_xticklabels([l[0] for l in LIBS], fontsize=6.8); ax.set_xlim(-0.6, len(LIBS) - 0.4); ax.set_ylim(0, 0.32)
    ax.set_ylabel("exact match, new triples"); ax.set_xlabel("library (operations: 30, 24, 17, 17, 23, 17, 23)", fontsize=7)
    ax.set_title("the effect depends on the library", fontsize=8.5, loc="left")
    ax.text(0.99, 0.97, "joint", color=C["hold"], fontsize=7.2, weight="bold", ha="right", va="top", transform=ax.transAxes)
    ax.text(0.99, 0.87, "staged", color=C["rew"], fontsize=7.2, weight="bold", ha="right", va="top", transform=ax.transAxes)
    ax.text(0.99, 0.77, "continuous", color=C["cont"], fontsize=6.6, ha="right", va="top", transform=ax.transAxes)

    # ---------------- (f) manipulating confusability directly changes nothing
    ax = fig.add_axes([0.765, 0.10, 0.22, 0.34]); fs.panel_letter(ax, "f", -0.32, 1.12)
    def names_per_op(pat):
        vals = []
        for f in sorted(glob.glob(os.path.join(DATA, "analysis", "codeinfo", "*.json"))):
            if re.fullmatch(pat, os.path.basename(f)[:-5]):
                d = json.load(open(f))["singles"]; vals.append(d["names"] / d["n_rules"])
        return vals
    pairs_ = [("A30", r"joint_rl_sslpercept_nomdl_s\d"), ("A\u2212conf", r"Aclean_joint_rl_s\d"), ("B", r"libB_joint_rl_percept_s\d"), ("B+dup", r"Bdup_joint_rl_s\d")]
    xs = [0, 1, 2.4, 3.4]
    for x, (lab, pat) in zip(xs, pairs_):
        fs.points(ax, x, names_per_op(pat), C["hold"], width=0.4)
    ax.annotate("", xy=(0.85, 0.50), xytext=(0.15, 0.50), arrowprops=dict(arrowstyle="-|>", color=C["muted"], lw=0.8, mutation_scale=8))
    ax.text(0.5, 0.53, "remove the 7 operations\nperception confuses", fontsize=6.0, color=C["muted"], ha="center", va="bottom")
    ax.annotate("", xy=(3.25, 0.40), xytext=(2.55, 0.40), arrowprops=dict(arrowstyle="-|>", color=C["muted"], lw=0.8, mutation_scale=8))
    ax.text(2.9, 0.37, "add 6 near-\nduplicates", fontsize=6.0, color=C["muted"], ha="center", va="top")
    ax.axhline(1.0, color=C["muted"], lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks(xs); ax.set_xticklabels([p[0] for p in pairs_], fontsize=6.8); ax.set_xlim(-0.6, 4.0); ax.set_ylim(0, 1.25)
    ax.set_ylabel("names per operation, joint training"); ax.set_title("confusability is not the cause", fontsize=8.5, loc="left")

    plt.savefig(out, bbox_inches="tight"); plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=170)
    print("saved", out)


if __name__ == "__main__":
    main()
