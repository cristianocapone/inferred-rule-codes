"""Human iterated learning (Kirby, Cornish & Smith 2008) and our sender, measured with the same
diagnostic: distinct signals per distinct meaning (rule). Left: every human chain, generation by
generation, without (Exp. 1) and with (Exp. 2) an expressivity pressure. Right: our reward-trained
sender under joint training (collapsed), with an information term, with staged experience, and on
library B - each point one seed. Same y axis."""

from paths import ROOT, DATA, PAPER, FIGURES, TABLES, RESULTS
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figstyle import C, use, panel_letter, points
from emlang.kirby_diag import load, chain_stats
use()
KIRBY = os.path.join(DATA, "human", "CornishData_KirbyCornishSmith2008.xlsx")
book = load(sys.argv[1] if len(sys.argv) > 1 else KIRBY)
exp = {"Exp. 1: transmission only": [], "Exp. 2: + expressivity pressure": []}
for name, rows in book.items():
    if name.startswith("I-"): exp["Exp. 1: transmission only"].append(chain_stats(rows))
    if name.startswith("III-"): exp["Exp. 2: + expressivity pressure"].append(chain_stats(rows))

# our sender: distinct messages per distinct held-out rule, one value per run, read from the training logs with
# the function that draws Fig. 4c (library A: 200 distinct held-out rules in the evaluation episodes; library B: 68)
import make_fig4 as m4
model = [("joint\ntraining", m4.uniq_per_rule(r"joint_rl_sslpercept_nomdl_s\d", 200), C["rew"]),
         ("+ information\nterm", m4.uniq_per_rule(r"joint_rl_sslpercept_div\d?_s\d", 200), C["rew"]),
         ("staged\nexperience", m4.uniq_per_rule(r"cur_rl_sslpercept_nomdl_s\d", 200), C["rew"]),
         ("library B,\njoint", m4.uniq_per_rule(r"libB_joint_rl_percept_s\d", 68), C["hold"])]
for lab, vals, _ in model: print(lab.replace("\n", " "), [round(v, 3) for v in vals])

fig = plt.figure(figsize=(7.2, 2.7))
ax = fig.add_axes([0.08, 0.20, 0.50, 0.70]); bx = fig.add_axes([0.66, 0.20, 0.32, 0.70])
cols = {"Exp. 1: transmission only": C["muted"], "Exp. 2: + expressivity pressure": C["human"]}
for lab, chains in exp.items():
    for st in chains:
        ax.plot([s["gen"] for s in st], [s["distinct_per_meaning"] for s in st], color=cols[lab], lw=0.8, alpha=0.45)
    G = range(11); m = [np.mean([st[g]["distinct_per_meaning"] for st in chains if g < len(st)]) for g in G]
    ax.plot(list(G), m, color=cols[lab], lw=2.2, label=lab, solid_capstyle="round")
ax.set_xlabel("generation of human learners"); ax.set_ylabel("distinct signals per meaning")
ax.set_ylim(0, 1.08); ax.set_xlim(-0.2, 10.2); ax.set_xticks(range(0, 11, 2))
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.6, frameon=False, borderaxespad=0.2)
ax.text(10.1, 0.015, "2–5 strings for 27 meanings", fontsize=6.8, color=C["muted"], ha="right", va="bottom")
ax.text(10.1, 0.99, "13–23 strings, compositional", fontsize=6.8, color=C["human"], ha="right", va="top")
panel_letter(ax, "a", dx=-0.12)
for i, (lab, vals, col) in enumerate(model):
    points(bx, i, vals, col, width=0.36)
bx.set_xticks(range(len(model))); bx.set_xticklabels([m[0] for m in model], fontsize=7)
bx.set_ylim(0, 1.08); bx.set_xlim(-0.6, len(model) - 0.4); bx.set_ylabel("distinct messages per rule")
bx.set_title("reward-trained sender", fontsize=8.5, color=C["rew"], loc="left")
panel_letter(bx, "b", dx=-0.30)
HERE = os.path.dirname(os.path.abspath(__file__))
for ext in ("pdf", "png"): fig.savefig(os.path.join(FIGURES, f"fig5_human.{ext}"), bbox_inches="tight")
print("saved figs/fig5_human.pdf/png")
