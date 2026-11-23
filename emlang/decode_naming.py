"""Does a message identify its rule on episodes the decoder has not seen?

Replaces the plug-in normalised mutual information, which is 1 by construction whenever messages are
nearly all distinct (every message then identifies its rule *in the sample*, even if messages are random).

For each rule family, its episodes are split in two halves. A decoder is fitted on one half and scored on
the other, then the halves are swapped and the two scores averaged:
  - lookup:  a test message is assigned the majority rule of identical messages in the fit half
             (an unseen message scores zero) - does the *same name* recur for the rule?
  - nearest: a test message is assigned the rule of the nearest fit message in Hamming distance
             (ties broken at random) - does the rule sit in a consistent *region* of message space?
Chance is 1/number of rules. A permutation null shuffles rule labels in each fitted half and scores the same
two-fold average as the data (200 draws), reporting the mean and 95th percentile. Sanity check: independent random messages must decode at chance.
"""
import argparse, glob, json, os, re, sys
from collections import Counter, defaultdict
import numpy as np


def halves(rules, msgs, rng):
    by = defaultdict(list)
    for r, m in zip(rules, msgs): by[r].append(tuple(m))
    A, B = [], []
    for r, ms in by.items():
        idx = rng.permutation(len(ms)); h = len(ms) // 2
        A += [(r, ms[i]) for i in idx[:h]]; B += [(r, ms[i]) for i in idx[h:]]
    return A, B, len(by)


def score_lookup(fit, test):
    table = defaultdict(Counter)
    for r, m in fit: table[m][r] += 1
    return float(np.mean([table[m].most_common(1)[0][0] == r if m in table else False for r, m in test]))


def score_nearest(fit, test, rng):
    F = np.array([m for _, m in fit]); FR = np.array([r for r, _ in fit], dtype=object)
    ok = []
    for r, m in test:
        d = (F != np.array(m)[None]).sum(1)
        cand = np.flatnonzero(d == d.min()); ok.append(FR[rng.choice(cand)] == r)
    return float(np.mean(ok))


def decode(rules, msgs, seed=0, n_perm=200):
    rng = np.random.default_rng(seed)
    A, B, n_rules = halves(rules, msgs, rng)
    look = 0.5 * (score_lookup(A, B) + score_lookup(B, A))
    near = 0.5 * (score_nearest(A, B, rng) + score_nearest(B, A, rng))
    null = []
    for _ in range(n_perm):   # the same two-fold statistic as the data, with family labels shuffled in each fitted half
        la = [r for r, _ in A]; rng.shuffle(la); lb = [r for r, _ in B]; rng.shuffle(lb)
        null.append(0.5 * (score_nearest(list(zip(la, [m for _, m in A])), B, rng) +
                           score_nearest(list(zip(lb, [m for _, m in B])), A, rng)))
    null = np.array(null)
    return dict(n_rules=n_rules, n_episodes=len(rules), chance=1.0 / n_rules, lookup=look, nearest=near,
                null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                nearest_over_chance=near * n_rules)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True); ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    # sanity check the estimator on messages that carry no information about the rule
    rng = np.random.default_rng(17); rules = np.repeat(np.arange(239), 10).tolist()
    rnd = rng.integers(0, 32, (len(rules), 8)).tolist()
    s = decode(rules, rnd, n_perm=50)
    print(f"[sanity] independent random messages, 239 rules x 10: lookup {s['lookup']:.3f} nearest {s['nearest']:.3f} chance {s['chance']:.3f} null {s['null_mean']:.3f}")
    res = {}
    print(f"{'run':32s} {'rules':>5s} {'lookup':>7s} {'nearest':>8s} {'chance':>7s} {'null p95':>9s} {'x chance':>9s}")
    for f in sorted(a.files):
        d = json.load(open(f))["train"]
        if not d["msgs"]: continue
        name = os.path.basename(f).replace("rearc_", "").replace("_msgs.json", "")
        r = decode(d["rule"], d["msgs"], n_perm=a.n_perm); res[name] = r
        print(f"{name:32s} {r['n_rules']:5d} {r['lookup']:7.3f} {r['nearest']:8.3f} {r['chance']:7.3f} {r['null_p95']:9.3f} {r['nearest_over_chance']:9.1f}")
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
