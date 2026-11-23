"""Do the categories an emergent code imposes on ARC-1 rules align with how humans describe those rules?

Human side (LARC, Acquaviva et al. 2022): for each ARC-1 training task, the pooled natural-language
descriptions (TF-IDF cosine between tasks) and, for 354 tasks, the profile of 17 annotated concept
tags (cosine between tag distributions).
Machine side: the modal message our speaker emits for each rule; two rules are "same category" if
their modal messages coincide, and graded similarity is 1 - Hamming distance / L.

Test: Mantel correlation (Spearman on the upper triangle) between the machine-similarity matrix and the
human-similarity matrix, with a permutation null over rule labels. A partial version residualises both
matrices on difficulty confounds (|build success_i - build success_j| and |description length_i -
description length_j|) before correlating, because the reward code is known to merge the easy rules.
"""
import argparse, csv, glob, json, math, os, re, collections
import numpy as np

LARC = os.environ.get("LARC_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external_rules", "LARC"))
STOP = set("the a an of to in is are and or it its that this with for on as be by from at into each all so then which you should see make output input grid grids one there will can".split())


def human_text_sim(tasks):
    docs = {}
    for f in glob.glob(os.path.join(LARC, "dataset", "tasks_json", "*.json")):
        d = json.load(open(f)); t = d["name"].replace(".json", "")
        if t not in tasks: continue
        words = []
        for desc in d["descriptions"].values():
            for k in ("see_description", "grid_description", "do_description"):
                words += [w for w in re.findall(r"[a-z]+", str(desc.get(k, "")).lower()) if w not in STOP and len(w) > 2]
        docs[t] = collections.Counter(words)
    ids = [t for t in tasks if t in docs]
    vocab = sorted({w for t in ids for w in docs[t]})
    df = collections.Counter(w for t in ids for w in docs[t])
    idf = {w: math.log(len(ids) / df[w]) for w in vocab}
    X = np.zeros((len(ids), len(vocab)))
    col = {w: i for i, w in enumerate(vocab)}
    for i, t in enumerate(ids):
        for w, c in docs[t].items(): X[i, col[w]] = (1 + math.log(c)) * idf[w]
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return ids, X @ X.T


def human_tag_sim(tasks):
    rows = list(csv.DictReader(open(os.path.join(LARC, "dataset", "annotated_phrases.csv")), delimiter="\t"))
    tags = [k for k in rows[0] if k.startswith("tag_")]
    prof = collections.defaultdict(lambda: np.zeros(len(tags)))
    for r in rows:
        t = r["task_name"].replace(".json", "")
        if t in tasks: prof[t] += np.array([float(r[k] or 0) for k in tags])
    ids = [t for t in tasks if t in prof and prof[t].sum() > 0]
    X = np.stack([prof[t] / prof[t].sum() for t in ids])
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return ids, X @ X.T


def triu(M):
    iu = np.triu_indices(M.shape[0], 1); return M[iu]


def rank(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    # average ties
    xs = x[o]; i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]: j += 1
        if j > i: r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def mantel(A, B, n_perm=2000, seed=0, C=None):
    """Spearman Mantel between A and B (square, symmetric); label permutation on A. Optional partial
    on confound matrices C (list): both A and B are residualised on C entries by least squares."""
    n = A.shape[0]
    a, b = triu(A).astype(float), triu(B).astype(float)
    if C:
        Z = np.stack([triu(c).astype(float) for c in C] + [np.ones(len(a))], 1)
        a = a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
        b = b - Z @ np.linalg.lstsq(Z, b, rcond=None)[0]
    rb = rank(b); rb = (rb - rb.mean()) / rb.std()
    def stat(Am):
        ra = rank(Am); ra = (ra - ra.mean()) / (ra.std() + 1e-12)
        return float((ra * rb).mean())
    obs = stat(a)
    rng = np.random.default_rng(seed); null = []
    for _ in range(n_perm):
        p = rng.permutation(n)
        Ap = A[np.ix_(p, p)]
        ap = triu(Ap).astype(float)
        if C: ap = ap - Z @ np.linalg.lstsq(Z, ap, rcond=None)[0]
        null.append(stat(ap))
    null = np.array(null)
    return obs, float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perrule", nargs="+", required=True); ap.add_argument("--stats", default="emlang_runs/larc_task_stats.json")
    ap.add_argument("--n_perm", type=int, default=2000); ap.add_argument("--out", default="emlang_runs/analysis/larc_categories.json")
    a = ap.parse_args()
    st = json.load(open(a.stats))
    results = {}
    for pf in a.perrule:
        d = json.load(open(pf)); pr = d["per_rule"]["train"]
        name = os.path.basename(pf).replace(".json", "")
        tasks = [t for t in sorted(pr) if t in st and st[t].get("build_success") is not None and st[t].get("desc_len_words") is not None]
        msgs = {t: tuple(pr[t]["msg"]) for t in tasks}
        L = len(next(iter(msgs.values())))
        # machine similarities
        M = np.array([msgs[t] for t in tasks])
        same = (M[:, None, :] == M[None, :, :]).all(-1).astype(float)
        ham = (M[:, None, :] == M[None, :, :]).mean(-1)
        counts = collections.Counter(msgs.values()); giant = counts.most_common(1)[0]
        out = dict(n_rules=len(tasks), distinct=len(counts), giant_cluster=giant[1], L=L)
        for hname, (hids, H) in {"text": human_text_sim(set(tasks)), "tags": human_tag_sim(set(tasks))}.items():
            idx = [tasks.index(t) for t in hids]
            build = np.array([st[t]["build_success"] for t in hids]); dlen = np.array([st[t]["desc_len_words"] for t in hids])
            Cb = np.abs(build[:, None] - build[None, :]); Cl = np.abs(dlen[:, None] - dlen[None, :])
            for mname, Mm in (("same_message", same), ("hamming_similarity", ham)):
                A = Mm[np.ix_(idx, idx)]
                if A.std() < 1e-9: out[f"{hname}/{mname}"] = "no variance"; continue
                r, p = mantel(A, H, a.n_perm); rp, pp = mantel(A, H, a.n_perm, C=[Cb, Cl])
                within = float(H[A > 0.999].mean()) if mname == "same_message" and (A > 0.999).sum() > A.shape[0] else float("nan")
                between = float(H[A < 0.001].mean()) if mname == "same_message" else float("nan")
                out[f"{hname}/{mname}"] = dict(n=len(hids), rho=r, p=p, rho_partial=rp, p_partial=pp, within=within, between=between)
                print(f"{name:34s} {hname:5s} {mname:19s} n={len(hids):3d} rho={r:+.3f} p={p:.3f} | partial rho={rp:+.3f} p={pp:.3f}"
                      + (f" | within {within:.3f} vs between {between:.3f}" if mname == "same_message" else ""), flush=True)
            # the same test without the collapse residue: drop the giant cluster
            keep = [i for i, t in enumerate(hids) if msgs[t] != giant[0]]
            if len(keep) > 30:
                A = same[np.ix_([idx[i] for i in keep], [idx[i] for i in keep])]; Hk = H[np.ix_(keep, keep)]
                if A.std() > 1e-9:
                    r, p = mantel(A, Hk, a.n_perm); rp, pp = mantel(A, Hk, a.n_perm, C=[Cb[np.ix_(keep, keep)], Cl[np.ix_(keep, keep)]])
                    out[f"{hname}/same_message_no_giant"] = dict(n=len(keep), rho=r, p=p, rho_partial=rp, p_partial=pp)
                    print(f"{name:34s} {hname:5s} {'same_msg, no giant':19s} n={len(keep):3d} rho={r:+.3f} p={p:.3f} | partial rho={rp:+.3f} p={pp:.3f}", flush=True)
        results[name] = out
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
