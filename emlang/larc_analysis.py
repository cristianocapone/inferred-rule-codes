"""LARC x emergent-code analysis.

Human side (LARC, Acquaviva et al. 2022): for each ARC-1 task, how transmissible the rule is when
people describe it in English to another person who must apply it to a new input.
Machine side (ours): for each of the same rules, whether our emergent code gives it its own name or
merges it with other rules, and how well the executor applies it.

Question: are the rules our reward code collapses together the rules humans find hard to transmit?
"""
import argparse, json, os
import numpy as np


def spearman(a, b):
    a = np.array([np.nan if v is None else v for v in a], float)
    b = np.array([np.nan if v is None else v for v in b], float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 8: return float("nan"), float("nan"), len(a)
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    r = float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    n = len(a)
    t = r * np.sqrt(max(n - 2, 1) / max(1e-12, 1 - r ** 2))
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))     # normal approximation
    return r, p, n


def rank(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x), float); r[o] = np.arange(len(x))
    sx = x[o]; i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]: j += 1
        if j > i: r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--larc", default="emlang_runs/larc_task_stats.json")
    ap.add_argument("--runs", nargs="+", default=["rearc16_gumbel_percept_s0", "rearc16_rl_percept_s0", "rearc16_gumbel_s0"])
    ap.add_argument("--analysis", default="emlang_runs/analysis")
    a = ap.parse_args()
    L = json.load(open(a.larc))
    print(f"LARC: {len(L)} ARC-1 tasks\n")
    for run in a.runs:
        f = os.path.join(a.analysis, f"perrule_{run}.json")
        if not os.path.exists(f): continue
        R = json.load(open(f))
        rows = []
        for split in ("train", "hold"):
            for rule, d in R.get("per_rule", {}).get(split, {}).items():
                h = L.get(rule)
                if not h or d.get("cluster_size") is None: continue
                rows.append(dict(rule=rule, split=split, cluster=d["cluster_size"], exact=d["exact"], union=d["union"],
                                 build=h["build_success"], desc=h["desc_len_do"], conf=h["confidence"],
                                 verif=h["verif_success"], nb=h["n_builds"]))
        if not rows: continue
        cl = [r["cluster"] for r in rows]
        merged = [r for r in rows if r["cluster"] > 1]; alone = [r for r in rows if r["cluster"] == 1]
        print(f"=== {run}  ({len(rows)} rules matched to LARC; {len(alone)} uniquely named, {len(merged)} merged)")
        print(f"    machine cluster size: mean {np.mean(cl):.2f} max {max(cl)}")
        for hk, label in (("build", "human build success"), ("desc", "rule-description length (words)"),
                          ("conf", "describer confidence"), ("verif", "describer self-verification")):
            r, p, n = spearman(cl, [x[hk] for x in rows])
            if merged and alone:
                fm = [x[hk] for x in merged if x[hk] is not None]; fa = [x[hk] for x in alone if x[hk] is not None]
                mm = np.mean(fm) if fm else float("nan"); ma = np.mean(fa) if fa else float("nan")
                print(f"    {label:34s} rho(cluster) {r:+.3f} p={p:.3f} n={n} | merged {mm:.3f} vs uniquely-named {ma:.3f}")
            else:
                print(f"    {label:34s} rho(cluster) {r:+.3f} p={p:.3f} n={n}")
        r, p, n = spearman([x["exact"] for x in rows], [x["build"] for x in rows])
        print(f"    our per-rule exact vs human build success   rho {r:+.3f} p={p:.3f} n={n}")
        r, p, n = spearman([x["union"] for x in rows], [x["build"] for x in rows])
        print(f"    our per-rule cell accuracy vs human build   rho {r:+.3f} p={p:.3f} n={n}")
        print()


if __name__ == "__main__":
    main()
