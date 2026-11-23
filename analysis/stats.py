"""Statistics for the contrasts reported in the manuscript (Supplementary Section S8).

The unit of replication is an independent training run. Accuracy on a held-out set is a two-level quantity:
runs, and held-out programs within a run, where every run of a library is evaluated on the same programs.
Confidence intervals and p-values therefore come from a hierarchical bootstrap (Saravanan, Berman & Sober
2020) that resamples runs within each condition and, jointly for all conditions of a contrast, the held-out
programs; 10 000 resamples, percentile intervals, two-sided p = 2 min(P*(d <= 0), P*(d >= 0)).
Run-level tests are given alongside: Welch's t-test on run means, and an exact permutation test on run means
whose smallest attainable p-value is reported, because with two or three runs per condition it is large.

Inputs
  emlang_runs/analysis/clean3/<run>.json   per-program exact-match counts (emlang/eval_clean.py --per_program);
                                           each run's mean is checked against clean2, which the figures use
  emlang_runs/<run>/log.jsonl              distinct messages per held-out rule (collapse diagnostic)
  emlang_runs/analysis/decode/rearc_<run>_msgs.json   per-episode messages for the decoding contrasts
  external_rules/CornishData_KirbyCornishSmith2008.xlsx  human iterated-learning chains
Outputs
  emlang_runs/analysis/stats/stats.json, paper/supp_tables/stats_contrasts.tex
"""

from paths import ROOT, DATA, PAPER, FIGURES, TABLES, RESULTS
import glob, itertools, json, math, os, re, sys
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
CLEAN3 = os.path.join(DATA, "analysis", "clean3")
CLEAN2 = os.path.join(DATA, "analysis", "clean2")
DECODE = os.path.join(DATA, "analysis", "decode")
OUT = os.path.join(RESULTS)
KIRBY = os.path.join(DATA, "human", "CornishData_KirbyCornishSmith2008.xlsx")
B = 10_000

# ------------------------------------------------------------------ per-program data

def runs_matching(pat, d=CLEAN3):
    return sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(d, "*.json")) if re.fullmatch(pat, os.path.basename(f)[:-5]))


CLEAN4 = os.path.join(DATA, "analysis", "clean4")
AUDITED = ("triples_audited", "pairs_audited")


def matrix(pat, key):
    """runs x programs matrix of exact-match rates, the program list, and the run names.

    On the audited sets the stored per-program counts are filtered to the programs the corrected audit still
    calls new (paper/audited.py, audit_20260918)."""
    from audited import novel_mask
    rows, progs, names = [], None, []
    src = CLEAN4 if key in AUDITED else CLEAN3
    for r in runs_matching(pat, src):
        d = json.load(open(os.path.join(src, r + ".json")))[key]
        if d is None: continue
        keep = novel_mask(r, key, d["programs"]) if key in AUDITED else np.ones(len(d["programs"]), bool)
        if progs is None: progs = [p for p, k in zip(d["programs"], keep) if k]
        assert [p for p, k in zip(d["programs"], keep) if k] == progs, f"{r}: program list differs within {pat}"
        m = np.array(d["exact_count"], float)[keep] / d["per_prog"]
        ref = os.path.join(CLEAN2, r + ".json")
        if key not in AUDITED and os.path.exists(ref):
            e2 = json.load(open(ref))[key]["exact"]
            assert abs(m.mean() - e2) < 1e-9, f"{r} {key}: per-program mean {m.mean():.4f} differs from clean2 {e2:.4f}"
        rows.append(m); names.append(r)
    return (np.array(rows) if rows else np.zeros((0, 0))), progs, names


def hier_boot(mats, weights, rng, B=B):
    """bootstrap distribution of sum_c w_c * mean_c; programs resampled jointly, runs within condition."""
    P = mats[0].shape[1]
    assert all(M.shape[1] == P for M in mats)
    out = np.empty(B)
    for b in range(B):
        j = rng.integers(0, P, P); s = 0.0
        for M, w in zip(mats, weights):
            i = rng.integers(0, M.shape[0], M.shape[0])
            s += w * M[np.ix_(i, j)].mean()
        out[b] = s
    return out


def boot_summary(dist, observed):
    lo, hi = np.percentile(dist, [2.5, 97.5])
    p = 2 * min((dist <= 0).mean(), (dist >= 0).mean())
    return dict(diff=float(observed), ci=[float(lo), float(hi)], p_boot=float(min(1.0, max(p, 1.0 / len(dist)))))


def run_level(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    res = dict(n=[len(x), len(y)], mean=[float(x.mean()), float(y.mean())])
    if len(x) >= 2 and len(y) >= 2:
        t = st.ttest_ind(x, y, equal_var=False)
        res["p_welch"] = float(t.pvalue) if np.isfinite(t.pvalue) else None
        sp = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
        J = 1 - 3 / (4 * (len(x) + len(y)) - 9)
        res["hedges_g"] = float(J * (x.mean() - y.mean()) / sp) if sp > 0 else None
        allv = np.concatenate([x, y]); obs = abs(x.mean() - y.mean()); cnt = tot = 0
        for idx in itertools.combinations(range(len(allv)), len(x)):
            m = np.zeros(len(allv), bool); m[list(idx)] = True
            tot += 1; cnt += abs(allv[m].mean() - allv[~m].mean()) >= obs - 1e-12
        res["p_perm"] = cnt / tot; res["p_perm_min"] = 2 / tot if len(x) == len(y) else 1 / tot
    return res


# ------------------------------------------------------------------ contrast definitions
# (section, label, set key, {pattern: weight}); two-condition contrasts are {X: +1, Y: -1}
SSL = "sslpercept"
T = "triples_audited"      # the held-out triples no training program computes, even through augmentation
A = [
 ("Primary contrasts", "reward: staged $-$ joint", T, {rf"cur_rl_{SSL}_nomdl_s\d": 1, rf"joint_rl_{SSL}_nomdl_s\d": -1}),
 (None, "reward: joint + information term $-$ joint", T, {rf"joint_rl_{SSL}_div\d?_s\d": 1, rf"joint_rl_{SSL}_nomdl_s\d": -1}),
 (None, "interaction: (staged $-$ joint), reward $-$ gradient", T,
  {rf"cur_rl_{SSL}_nomdl_s\d": 1, rf"joint_rl_{SSL}_nomdl_s\d": -1, rf"cur_gumbel_{SSL}_nomdl_s\d": -1, rf"joint_gumbel_{SSL}_nomdl_s\d": 1}),
 ("Exploratory: discreteness and perception (staged experience)", "gradient $-$ continuous", T, {rf"cur_gumbel_{SSL}_nomdl_s\d": 1, r"cur_cont_s\d": -1}),
 (None, "reward $-$ continuous", T, {rf"cur_rl_{SSL}_nomdl_s\d": 1, r"cur_cont_s\d": -1}),
 (None, "reward $-$ gradient", T, {rf"cur_rl_{SSL}_nomdl_s\d": 1, rf"cur_gumbel_{SSL}_nomdl_s\d": -1}),
 (None, "reward: self-supervised $-$ end-to-end perception", T, {rf"cur_rl_{SSL}_nomdl_s\d": 1, r"cur_rl_nomdl_s\d": -1}),
 (None, "gradient: self-supervised $-$ end-to-end perception", T, {rf"cur_gumbel_{SSL}_nomdl_s\d": 1, r"cur_gumbel_nomdl_s\d": -1}),
 (None, "reward: self-supervised $-$ solver perception", T, {rf"cur_rl_{SSL}_nomdl_s\d": 1, r"cur_rl_percept_nomdl_s\d": -1}),
 ("Exploratory: order of experience", "reward: staged $-$ reversed", T, {rf"cur_rl_{SSL}_nomdl_s\d": 1, rf"rev_rl_{SSL}_nomdl_s\d": -1}),
 (None, "gradient: staged $-$ joint", T, {rf"cur_gumbel_{SSL}_nomdl_s\d": 1, rf"joint_gumbel_{SSL}_nomdl_s\d": -1}),
 (None, "gradient: staged $-$ reversed", T, {rf"cur_gumbel_{SSL}_nomdl_s\d": 1, rf"rev_gumbel_{SSL}_nomdl_s\d": -1}),
 ("Exploratory: information term and changed-cell reward (reward channel)", "joint + information term (weight 0.1 only) $-$ joint", T, {rf"joint_rl_{SSL}_div_s\d": 1, rf"joint_rl_{SSL}_nomdl_s\d": -1}),
 (None, "joint + information term $-$ staged", T, {rf"joint_rl_{SSL}_div\d?_s\d": 1, rf"cur_rl_{SSL}_nomdl_s\d": -1}),
 (None, "joint, changed-cell reward $-$ joint", T, {rf"joint_rl_{SSL}_chg_s\d": 1, rf"joint_rl_{SSL}_nomdl_s\d": -1}),
 (None, "joint + information term $-$ joint, changed-cell reward", T, {rf"joint_rl_{SSL}_div\d?_s\d": 1, rf"joint_rl_{SSL}_chg_s\d": -1}),
 (None, "staged, changed-cell reward $-$ staged", T, {rf"cur_rl_{SSL}_chg_s\d": 1, rf"cur_rl_{SSL}_nomdl_s\d": -1}),
 ("Exploratory: other libraries (reward channel)", "A, random 17: staged $-$ joint", T, {r"A17r_cur_rl_s\d": 1, r"A17r_joint_rl_s\d": -1}),
 (None, "A without confusable operations: staged $-$ joint", T, {r"Aclean_cur_rl_s\d": 1, r"Aclean_joint_rl_s\d": -1}),
 (None, "A, first 24: staged $-$ joint", T, {r"A24_cur_rl_s\d": 1, r"A24_joint_rl_s\d": -1}),
 (None, "A, first 17: staged $-$ joint", T, {r"A17_cur_rl_s\d": 1, r"A17_joint_rl_s\d": -1}),
 (None, "B: staged $-$ joint", T, {r"libB_rl_percept_s\d": 1, r"libB_joint_rl_percept_s\d": -1}),
 (None, "B: joint + information term $-$ joint", T, {r"libB_joint_rl_div_s\d": 1, r"libB_joint_rl_percept_s\d": -1}),
 (None, "B with near-duplicates: staged $-$ joint", T, {r"Bdup_cur_rl_s\d": 1, r"Bdup_joint_rl_s\d": -1}),
]


def per_program_contrasts(rng):
    rows = []
    for sec, lab, key, w in A:
        mats, ws, names = [], [], []
        for pat, wt in w.items():
            M, progs, nm = matrix(pat, key)
            if M.size == 0: break
            mats.append(M); ws.append(wt); names.append(nm)
        if len(mats) != len(w):
            print(f"[skip] {lab}: missing runs"); continue
        obs = sum(wt * M.mean() for M, wt in zip(mats, ws))
        r = dict(section=sec, label=lab, key=key, runs=names, **boot_summary(hier_boot(mats, ws, rng), obs))
        if len(w) == 2:
            r.update(run_level(mats[0].mean(1), mats[1].mean(1)))
        else:
            r.update(n=[M.shape[0] for M in mats], mean=[float(M.mean()) for M in mats])
        r["n_programs"] = mats[0].shape[1]
        rows.append(r)
    return rows


# ------------------------------------------------------------------ run-level measures

def collapse_contrasts():
    import make_fig4 as m4
    D = lambda pat: m4.uniq_per_rule(pat, 200)
    base = rf"joint_rl_{SSL}_nomdl_s\d"
    rows = []
    for i, (lab, pat) in enumerate([("staged $-$ joint", rf"cur_rl_{SSL}_nomdl_s\d"), ("reversed $-$ joint", rf"rev_rl_{SSL}_nomdl_s\d"),
                                    ("joint + information term $-$ joint", rf"joint_rl_{SSL}_div\d?_s\d"),
                                    ("joint, changed-cell reward $-$ joint", rf"joint_rl_{SSL}_chg_s\d")]):
        x, y = D(pat), D(base)
        r = dict(section="Exploratory: collapse diagnostic, distinct messages per held-out rule (run level)" if i == 0 else None, label=lab, key="diagnostic")
        r.update(run_level(x, y)); r["diff"] = r["mean"][0] - r["mean"][1]
        r["ci"] = welch_ci(x, y); rows.append(r)
    return rows


def welch_ci(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2: return None
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy)
    if se == 0: return [float(x.mean() - y.mean())] * 2
    dof = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    q = st.t.ppf(0.975, dof); d = x.mean() - y.mean()
    return [float(d - q * se), float(d + q * se)]


# decoding: the per-episode outcome of both decoders, with the random stream of emlang/decode_naming.py

def decode_outcomes(rules, msgs, seed=0):
    from collections import Counter, defaultdict
    import emlang.decode_naming as dn
    rng = np.random.default_rng(seed)
    Ah, Bh, n_rules = dn.halves(rules, msgs, rng)

    def look(fit, test):
        table = defaultdict(Counter)
        for r, m in fit: table[m][r] += 1
        return [(r, float(table[m].most_common(1)[0][0] == r) if m in table else 0.0) for r, m in test]

    def near(fit, test):
        F = np.array([m for _, m in fit]); FR = np.array([r for r, _ in fit], dtype=object); o = []
        for r, m in test:
            d = (F != np.array(m)[None]).sum(1); cand = np.flatnonzero(d == d.min())
            o.append((r, float(FR[rng.choice(cand)] == r)))
        return o

    lk = look(Ah, Bh) + look(Bh, Ah)
    nr = near(Ah, Bh) + near(Bh, Ah)
    return lk, nr, n_rules


def family_matrix(pats, which):
    """runs x families matrix of decoding accuracy (mean over a family's test episodes, both folds)."""
    out = {}
    for pat in pats:
        rows, fams = [], None
        for f in sorted(glob.glob(os.path.join(DECODE, "rearc_*_msgs.json"))):
            name = os.path.basename(f).replace("rearc_", "").replace("_msgs.json", "")
            if not re.fullmatch(pat, name): continue
            d = json.load(open(f))["train"]
            lk, nr, _ = decode_outcomes(d["rule"], d["msgs"])
            src = lk if which == "lookup" else nr
            acc = {}
            for r, v in src: acc.setdefault(r, []).append(v)
            fk = sorted(acc)
            if fams is None: fams = fk
            assert fk == fams
            rows.append([np.mean(acc[k]) for k in fams])
        out[pat] = np.array(rows)
    return out


def decoding_contrasts(rng):
    if not glob.glob(os.path.join(DECODE, "rearc_*_msgs.json")):
        print("[skip] decoding: no message dumps in", DECODE); return []
    rew, grad = r"scale240_rl_percept_s\d", r"scale240_gumbel_percept_s\d"
    spec = [("gradient $-$ reward, nearest neighbour", [(grad, "nearest", 1), (rew, "nearest", -1)]),
            ("gradient: nearest neighbour $-$ exact lookup", [(grad, "nearest", 1), (grad, "lookup", -1)]),
            ("reward: nearest neighbour $-$ exact lookup", [(rew, "nearest", 1), (rew, "lookup", -1)]),
            ("reward: length 16 $-$ baseline, nearest", [(r"scale240_rl_L16_s\d", "nearest", 1), (rew, "nearest", -1)]),
            ("reward: vocabulary 128 $-$ baseline, nearest", [(r"scale240_rl_V128_s\d", "nearest", 1), (rew, "nearest", -1)]),
            ("reward: information term 0.1 $-$ baseline, nearest", [(r"scale240_rl_div_s\d", "nearest", 1), (rew, "nearest", -1)]),
            ("reward: information term 0.3 $-$ baseline, nearest", [(r"scale240_rl_div3_s\d", "nearest", 1), (rew, "nearest", -1)])]
    cache, rows = {}, []
    for i, (lab, terms) in enumerate(spec):
        mats, ws = [], []
        for pat, which, w in terms:
            if (pat, which) not in cache: cache[(pat, which)] = family_matrix([pat], which)[pat]
            mats.append(cache[(pat, which)]); ws.append(w)
        if any(M.size == 0 for M in mats): print(f"[skip] {lab}"); continue
        paired = terms[0][0] == terms[1][0]
        obs = sum(w * M.mean() for M, w in zip(mats, ws))
        if paired:   # same runs, two decoders: resample runs once, jointly
            diffM = mats[0] - mats[1]
            dist = hier_boot([diffM], [1], rng)
            x = diffM.mean(1); rl = dict(n=[len(x), len(x)], mean=[float(mats[0].mean()), float(mats[1].mean())])
            if len(x) >= 2:
                t = st.ttest_1samp(x, 0.0); rl["p_welch"] = float(t.pvalue)
        else:
            dist = hier_boot(mats, ws, rng); rl = run_level(mats[0].mean(1), mats[1].mean(1))
        r = dict(section="Exploratory: decoding the rule family from the message, 239 families" if i == 0 else None, label=lab, key="decoding",
                 paired=paired, n_programs=mats[0].shape[1], **boot_summary(dist, obs)); r.update(rl)
        rows.append(r)
    return rows


def kirby_contrast():
    if not os.path.exists(KIRBY): print("[skip] Kirby: data not found"); return []
    from emlang.kirby_diag import load, chain_stats
    book = load(KIRBY); fin = {"I": [], "III": []}
    for name, rows in book.items():
        for k in fin:
            if name.startswith(k + "-"):
                s = chain_stats(rows); fin[k].append(s[-1]["distinct_per_meaning"])
    x, y = fin["III"], fin["I"]
    r = dict(section="Exploratory: human iterated learning (chain level)", label="final generation, distinct signals per meaning: pressure $-$ none",
             key="kirby"); r.update(run_level(x, y)); r["diff"] = r["mean"][0] - r["mean"][1]; r["ci"] = welch_ci(x, y)
    return [r]


# ------------------------------------------------------------------ table

def fmt_p(p):
    if p is None: return "--"
    if p < 0.001: return "$<$0.001"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def pct(v, scale):
    t = f"{100 * v:.1f}" if scale else f"{v:.2f}"
    return t.replace("-", "$-$") if t.startswith("-") and float(t) != 0 else t.lstrip("-")


def write_table(rows, path):
    lines = [r"\begin{tabular}{lcccccc}\toprule",
             r"contrast & runs & means & difference [95\% CI] & $p_\mathrm{boot}$ & $p_\mathrm{Welch}$ & $p_\mathrm{perm}$ (min) \\ \midrule"]
    for r in rows:
        if r.get("section"):
            lines.append(r"\multicolumn{7}{l}{\emph{" + r["section"] + r"}} \\")
        scale = r["key"] in ("novel", "len3_novel", "decoding", "triples_audited", "pairs_audited")
        n = " / ".join(str(k) for k in r["n"])
        means = " / ".join(pct(m, scale) for m in r["mean"]) if len(r["mean"]) == 2 else "--"
        ci = r.get("ci")
        diff = pct(r["diff"], scale) + (f" [{pct(ci[0], scale)}, {pct(ci[1], scale)}]" if ci else "")
        pp = f"{fmt_p(r['p_perm'])} ({fmt_p(r['p_perm_min'])})" if "p_perm" in r else "--"
        lines.append(f"\\quad {r['label']} & {n} & {means} & {diff} & {fmt_p(r.get('p_boot'))} & {fmt_p(r.get('p_welch'))} & {pp} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    rng = np.random.default_rng(0)
    os.makedirs(OUT, exist_ok=True)
    rows = per_program_contrasts(rng) + collapse_contrasts() + decoding_contrasts(rng) + kirby_contrast()
    json.dump(rows, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
    write_table(rows, os.path.join(TABLES, "stats_contrasts.tex"))
    for r in rows:
        ci = r.get("ci"); ci = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        print(f"{r['label'][:62]:62s} n={r['n']} diff={r['diff']:+.3f} {ci:18s} p_boot={r.get('p_boot', float('nan')):.4f} "
              f"p_welch={r.get('p_welch') if r.get('p_welch') is None else round(r['p_welch'], 5)} p_perm={r.get('p_perm', '--')}")


if __name__ == "__main__":
    main()
