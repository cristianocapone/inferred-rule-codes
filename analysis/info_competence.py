"""Does competence follow the information a code carries about the rule? (Fig. 4d, Supplementary Section S5)

For every reward-trained run on the main library, accuracy on the held-out triples that no training program computes,
even through the augmentation of training (emlang_runs/analysis/clean4, emlang/eval_clean.py --group_triples, ten
episodes per triple), against how often the sender's message identifies the triple on held-out episodes
(emlang_runs/analysis/codeinfo3, emlang/code_info.py --group_triples: nearest-neighbour decoding, ten episodes per
triple). Runs whose sender was held soft by a stability pressure are reported separately: their executor
is trained on noisy samples of the code, so the most probable message overstates what the executor saw.
Spearman correlation over runs, with a bootstrap interval over runs (10 000 resamples)."""

from paths import ROOT, DATA, PAPER, FIGURES, TABLES, RESULTS
import glob, json, os, re, sys
import numpy as np
from scipy import stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audited import novel_mask

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CI = os.path.join(DATA, "analysis", "codeinfo3")
C3 = os.path.join(DATA, "analysis", "clean4")
OUT = os.path.join(RESULTS)

# (label, pattern, group); group 'reward' enters the correlation, 'soft' is reported apart, 'gradient' is the reference.
# stab_cur_rl, stab_joint_chg and stab_joint_div are the same checkpoints as the runs listed here and are not counted twice.
CONDITIONS = [
    ("staged", r"cur_rl_sslpercept_nomdl_s\d", "reward"),
    ("joint", r"(joint_rl_sslpercept_nomdl|stab_joint_rl)_s\d", "reward"),     # stab_joint_rl: same configuration, run in the stability wave
    ("reversed", r"rev_rl_sslpercept_nomdl_s\d", "reward"),
    ("joint + information term", r"joint_rl_sslpercept_div\d?_s\d", "reward"),
    ("staged + information term", r"cur_rl_sslpercept_div_s\d", "reward"),
    ("joint, changed-cell reward", r"joint_rl_sslpercept_chg_s\d", "reward"),
    ("staged, changed-cell reward", r"cur_rl_sslpercept_chg_s\d", "reward"),
    ("joint, slower exploration", r"joint_rl_sslpercept_slow_s\d", "reward"),
    ("staged, per-symbol credit", r"cur_rlproc_sslpercept_nomdl_s\d", "reward"),
    ("joint, per-symbol credit", r"joint_rlproc_sslpercept_nomdl_s\d", "reward"),
    ("staged, solver perception", r"cur_rl_percept_nomdl_s\d", "reward"),
    ("joint, solver perception", r"joint_rl_percept_nomdl_s\d", "reward"),
    ("joint, changed-cell + stability 0.1", r"stab_joint_chg_stab01_s\d", "soft"),
    ("joint, changed-cell + stability 1.0", r"stab_joint_chg_stab_s\d", "soft"),
    ("joint + stability 1.0", r"stab_joint_stab_s\d", "soft"),
    ("gradient, staged", r"cur_gumbel_sslpercept_nomdl_s\d", "gradient"),
    ("gradient, joint", r"joint_gumbel_sslpercept_nomdl_s\d", "gradient"),
    ("gradient, reversed", r"rev_gumbel_sslpercept_nomdl_s\d", "gradient"),
]


def load():
    rows = []
    for lab, pat, grp in CONDITIONS:
        for f in sorted(glob.glob(os.path.join(CI, "*.json"))):
            run = os.path.basename(f)[:-5]
            if not re.fullmatch(pat, run): continue
            c3 = os.path.join(C3, run + ".json")
            if not os.path.exists(c3): print("[no accuracy]", run); continue
            ci, acc = json.load(open(f)), json.load(open(c3))["triples_audited"]
            keep = novel_mask(run, "triples_audited", acc["programs"])       # the corrected audit, as everywhere else
            novel = float(np.array(acc["exact_count"])[keep].sum() / (keep.sum() * acc["per_prog"]))
            rows.append(dict(run=run, condition=lab, group=grp, novel=novel,
                             triple_nearest=ci["triples_audited"]["decode_nearest"], triple_lookup=ci["triples_audited"]["decode_lookup"],
                             pair_nearest=ci["novel_pairs"]["decode_nearest"],
                             single_nearest=ci["singles"]["decode_nearest"], single_names=ci["singles"]["names"],
                             single_largest=ci["singles"]["largest_group"]))
    return rows


def spearman_ci(x, y, B=10_000, seed=0):
    x, y = np.asarray(x), np.asarray(y); rng = np.random.default_rng(seed); rs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1: rs.append(st.spearmanr(x[i], y[i])[0])
    return [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]


def main():
    rows = load(); os.makedirs(OUT, exist_ok=True)
    rew = [r for r in rows if r["group"] == "reward"]
    res = dict(n_reward_runs=len(rew))
    for key in ("triple_nearest", "single_nearest"):
        x, y = [r[key] for r in rew], [r["novel"] for r in rew]
        rho, p = st.spearmanr(x, y)
        res[key] = dict(rho=float(rho), p=float(p), ci=spearman_ci(x, y))
        allr = [r for r in rows if r["group"] in ("reward", "soft")]
        x2, y2 = [r[key] for r in allr], [r["novel"] for r in allr]
        rho2, p2 = st.spearmanr(x2, y2)
        res[key + "_with_soft"] = dict(rho=float(rho2), p=float(p2), n=len(allr), ci=spearman_ci(x2, y2))
    # within one condition: the six staged runs with the solver's perception differ in how much their code carries
    sub = [r for r in rows if r["condition"] == "staged, solver perception"]
    if len(sub) >= 4:
        rho, p = st.spearmanr([r["triple_nearest"] for r in sub], [r["novel"] for r in sub]); res["within_solver_staged"] = dict(rho=float(rho), p=float(p), n=len(sub))
    res["rows"] = rows
    json.dump(res, open(os.path.join(OUT, "info_competence.json"), "w"), indent=1)
    by = {}
    for r in rows: by.setdefault((r["group"], r["condition"]), []).append(r)
    print(f"{'condition':40s} {'n':>2s} {'novel exact':>14s} {'triple decoding':>16s} {'single decoding':>16s} {'names/30':>9s}")
    for (g, lab), rs in by.items():
        rng_ = lambda k, s=1: f"{min(r[k] for r in rs) * s:.1f}-{max(r[k] for r in rs) * s:.1f}"
        print(f"{lab:40s} {len(rs):2d} {rng_('novel', 100):>14s} {rng_('triple_nearest', 100):>16s} {rng_('single_nearest', 100):>16s} "
              f"{min(r['single_names'] for r in rs)}-{max(r['single_names'] for r in rs):>2}")
    for k in ("triple_nearest", "single_nearest"):
        w = res[k + '_with_soft']
        print(f"Spearman, reward runs (n={res['n_reward_runs']}), accuracy vs {k}: rho {res[k]['rho']:.2f} [{res[k]['ci'][0]:.2f}, {res[k]['ci'][1]:.2f}], p {res[k]['p']:.1e}; "
              f"with soft-policy runs rho {w['rho']:.2f} [{w['ci'][0]:.2f}, {w['ci'][1]:.2f}] (n={w['n']})")
    if "within_solver_staged" in res: print("within staged, solver perception:", res["within_solver_staged"])
    # supplementary table
    lines = [r"\begin{tabular}{lccccc}\toprule",
             r"condition (library A) & runs & exact, new triples (\%) & triple identified (\%) & operation identified (\%) & names (of 30) \\ \midrule"]
    grp_title = dict(reward="\\emph{reward channel}", soft="\\emph{reward channel, policy held soft by a stability pressure}", gradient="\\emph{gradient channel}")
    seen = set()
    for (g, lab), rs in by.items():
        if g not in seen:
            lines.append(r"\multicolumn{6}{l}{" + grp_title[g] + r"} \\"); seen.add(g)
        span = lambda k, sc=100: (f"{min(r[k] for r in rs) * sc:.1f}" if len(rs) == 1 else f"{min(r[k] for r in rs) * sc:.1f}--{max(r[k] for r in rs) * sc:.1f}")
        nm = f"{min(r['single_names'] for r in rs)}" if len(rs) == 1 else f"{min(r['single_names'] for r in rs)}--{max(r['single_names'] for r in rs)}"
        lines.append(f"\\quad {lab} & {len(rs)} & {span('novel')} & {span('triple_nearest')} & {span('single_nearest')} & {nm} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABLES, "info_competence.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
