"""Generate the supplementary tables from the audit and evaluation outputs (no number is typed by hand).
Writes supp_tables/*.tex, each a complete tabular, \\input by supplement.tex."""

from paths import ROOT, DATA, PAPER, FIGURES, TABLES, RESULTS
import glob, json, os, re
import numpy as np
from audited import novel, decoding
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUD = os.path.join(DATA, "audits", "20260916")
OUT = str(TABLES); os.makedirs(OUT, exist_ok=True)


def ms(v, d=3):
    if not v: return "--"
    if len(v) == 1: return f"{v[0]:.{d}f}"
    return f"{np.mean(v):.{d}f} $\\pm$ {np.std(v, ddof=1) / np.sqrt(len(v)):.{d}f}"


def write(name, body): open(os.path.join(OUT, name + ".tex"), "w").write(body); print("wrote", name)


# S1a: novelty of the held-out programs by library and split, at the three levels of the group audit
# (emlang/audit_group.py: not computed by a training program directly; nor through a dihedral transform; nor through
# the full augmentation group of dihedral transforms and colour permutations)
from audited import audited
AUD2 = os.path.join(DATA, "audits", "20260918")
LIBS = [("A (30 operations)", "A"), ("A, first 24", "A24"), ("A, first 17", "A17"), ("A, random 17", "A17r"),
        ("A without 7 confusable", "Aclean"), ("B (17 operations)", "B"), ("B plus 6 near-duplicates", "Bdup")]
rows = []
for lab, tag in LIBS:
    cells = [lab]
    for sp in ("hold2", "len3"):
        f = os.path.join(AUD2, f"group_audit_{tag}_{sp}.json")
        d = json.load(open(f))
        cells += [str(d["n_heldout"]), str(len(d["novel_direct"])), str(len(d["novel_dihedral"])), str(len(d["novel_group"]))]
    rows.append(" & ".join(cells) + r" \\")
write("s1_novelty", r"""\begin{tabular}{lcccccccc}\toprule
 & \multicolumn{4}{c}{held-out pairs} & \multicolumn{4}{c}{held-out triples} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}
library & programs & not direct & not dihedral & new & programs & not direct & not dihedral & new \\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# S1b: every arm on all held-out programs and on the novel ones (pairs and triples of the group audit)
def pct(v):
    if not v: return "--"
    if len(v) == 1: return f"{100 * v[0]:.1f}"
    return f"{100 * np.mean(v):.1f} $\\pm$ {100 * np.std(v, ddof=1) / np.sqrt(len(v)):.1f}"
ARMS = [("A, staged, reward", r"cur_rl_sslpercept_nomdl_s\d"), ("A, joint, reward", r"joint_rl_sslpercept_nomdl_s\d"),
        ("A, reversed, reward", r"rev_rl_sslpercept_nomdl_s\d"), ("A, joint + information term", r"joint_rl_sslpercept_div\d?_s\d"),
        ("A, joint, changed-cell reward", r"joint_rl_sslpercept_chg_s\d"), ("A, staged, changed-cell reward", r"cur_rl_sslpercept_chg_s\d"),
        ("A, staged, gradient", r"cur_gumbel_sslpercept_nomdl_s\d"), ("A, joint, gradient", r"joint_gumbel_sslpercept_nomdl_s\d"),
        ("A, reversed, gradient", r"rev_gumbel_sslpercept_nomdl_s\d"), ("A, continuous", r"cur_cont_s\d"),
        ("A, reward, solver perception", r"cur_rl_percept_nomdl_s\d"), ("A, gradient, solver perception", r"cur_gumbel_percept_nomdl_s\d"),
        ("A, reward, end to end", r"cur_rl_nomdl_s\d"), ("A, gradient, end to end", r"cur_gumbel_nomdl_s\d"),
        ("A24, staged / joint, reward", (r"A24_cur_rl_s\d", r"A24_joint_rl_s\d")), ("A17, staged / joint, reward", (r"A17_cur_rl_s\d", r"A17_joint_rl_s\d")),
        ("A17r, staged / joint, reward", (r"A17r_cur_rl_s\d", r"A17r_joint_rl_s\d")), ("A$-$conf, staged / joint, reward", (r"Aclean_cur_rl_s\d", r"Aclean_joint_rl_s\d")),
        ("B, staged / joint, reward", (r"libB_rl_percept_s\d", r"libB_joint_rl_percept_s\d")), ("B, joint + information term", r"libB_joint_rl_div_s\d"),
        ("B+dup, staged / joint, reward", (r"Bdup_cur_rl_s\d", r"Bdup_joint_rl_s\d"))]
rows = []
for lab, pat in ARMS:
    pats = pat if isinstance(pat, tuple) else (pat,)
    col = lambda f: " / ".join(f(p) for p in pats)
    rows.append(f"{lab} & {col(lambda p: pct(novel(p, 'all')))} & {col(lambda p: pct(audited(p, 'pairs_audited')))} & "
                f"{col(lambda p: pct(novel(p, 'len3_all')))} & {col(lambda p: pct(audited(p, 'triples_audited')))} & "
                f"{' / '.join(str(len(novel(p, 'all'))) for p in pats)} \\\\")
write("s1_results", r"""\begin{tabular}{lccccc}\toprule
arm & pairs, all & pairs, novel & triples, all & triples, novel & runs \\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# S2: decoding
D = decoding()
def dec(pat, key): return [d[key] for n, d in D.items() if re.fullmatch(pat, n)]
rows = []
for nr in (30, 60, 120, 240):
    g, r = rf"scale{nr}_gumbel_percept_s\d", rf"scale{nr}_rl_percept_s\d"
    ch = dec(g, "chance") or dec(r, "chance")
    rows.append(f"{nr if nr != 240 else 239} & {ms(dec(g, 'nearest'))} & {ms(dec(g, 'lookup'))} & {ms(dec(r, 'nearest'))} & {ms(dec(r, 'lookup'))} & {ch[0]:.3f} & {max(dec(g, 'null_p95') + dec(r, 'null_p95')):.3f} \\\\")
write("s2_decoding", r"""\begin{tabular}{rcccccc}\toprule
 & \multicolumn{2}{c}{gradient} & \multicolumn{2}{c}{reward} & & \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}
families & nearest & lookup & nearest & lookup & chance & null, 95th pct. \\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
rows = []
for lab, pat in [("reward, baseline", r"scale240_rl_percept_s\d"), ("message length 16", r"scale240_rl_L16_s\d"),
                 ("vocabulary 128", r"scale240_rl_V128_s\d"), ("information term 0.1", r"scale240_rl_div_s\d"),
                 ("information term 0.3", r"scale240_rl_div3_s\d"), ("reward, 319 families, term 0.1", r"scale319_rl_div_s\d"),
                 ("gradient reference", r"scale240_gumbel_percept_s\d")]:
    rows.append(f"{lab} & {ms(dec(pat, 'nearest'))} & {ms(dec(pat, 'lookup'))} & {len(dec(pat, 'nearest'))} \\\\")
write("s2_decomposition", r"""\begin{tabular}{lccc}\toprule
239 families (unless stated) & nearest & lookup & runs \\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# S6: name stability (turnover of the modal name of each single operation between evaluations 500 steps apart)
import sys
sys.path.insert(0, ROOT)
from emlang.name_turnover import load as load_turnover, turnover as turnover_of
ARMS6 = [("staged", "stab_cur_rl_s"), ("joint + information term", "stab_joint_div_s"), ("joint, changed-cell reward", "stab_joint_chg_s"),
         ("\\quad + stability pressure 0.1", "stab_joint_chg_stab01_s"), ("\\quad + stability pressure 1.0", "stab_joint_chg_stab_s"),
         ("joint, plain reward", "stab_joint_rl_s"), ("\\quad + stability pressure 1.0", "stab_joint_stab_s")]
rows = []
for lab, pre in ARMS6:
    tos, purs, pairs = [], [], []
    for s in (0, 1):
        rd = os.path.join(DATA, "runs", f"{pre}{s}")
        if not os.path.exists(os.path.join(rd, "log.jsonl")): continue
        ev = load_turnover(rd); to = [t for st, t in turnover_of(ev) if st > 16000]
        tos.append(np.mean(to))
        evs = [json.loads(l) for l in open(os.path.join(rd, "log.jsonl"), errors="ignore") if '"kind": "eval"' in l]
        purs.append(np.mean([e["synth_train1"].get("align_purity", np.nan) for e in evs if e["step"] > 16000]))
        pairs.append(evs[-1]["synth_hold2"]["exact"])
    f = lambda v: " / ".join(f"{x:.2f}" for x in v)
    rows.append(f"{lab} & {f(tos)} & {f(purs)} & {' / '.join(f'{x:.3f}' for x in pairs)} \\\\")
write("s6_stability", r"""\begin{tabular}{lccc}\toprule
arm (library A, two seeds) & name turnover after 16\,000 steps & purity & exact, held-out pairs \\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
