"""Language-of-thought signatures measured on the emergent code (numpy only)."""
import numpy as np


def _rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x)); ranks[order] = np.arange(len(x))
    # average ties
    sx = x[order]; i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]: j += 1
        if j > i: ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    if len(a) < 3: return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def levenshtein(a, b):
    n, m = len(a), len(b)
    D = np.arange(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        prev = D.copy(); D[0] = i
        for j in range(1, m + 1):
            D[j] = min(prev[j] + 1, D[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
    return int(D[m])


def topsim(progs, msgs, max_pairs=4000, seed=0):
    """Spearman correlation between program edit-distance and message Hamming distance."""
    rng = np.random.default_rng(seed)
    n = len(progs)
    if n < 4: return float("nan")
    idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(idx) > max_pairs:
        sel = rng.choice(len(idx), max_pairs, replace=False); idx = [idx[k] for k in sel]
    dp = [levenshtein(progs[i], progs[j]) for i, j in idx]
    dm = [int((np.asarray(msgs[i]) != np.asarray(msgs[j])).sum()) for i, j in idx]
    return spearman(dp, dm)


def _entropy(counts):
    p = counts / counts.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _mutual_info(x, y):
    x = np.asarray(x); y = np.asarray(y)
    xs, xi = np.unique(x, return_inverse=True); ys, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((len(xs), len(ys)))
    np.add.at(joint, (xi, yi), 1)
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True); py = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


def posdis(attrs, msgs):
    """Positional disentanglement (Chaabouni et al. 2020): for each message position,
    (MI with best attribute - MI with second best) / H(position). attrs: [N,A] ints."""
    attrs = np.asarray(attrs); msgs = np.asarray(msgs)
    L = msgs.shape[1]; scores = []
    for j in range(L):
        col = msgs[:, j]
        _, cnt = np.unique(col, return_counts=True)
        H = _entropy(cnt.astype(np.float64))
        if H <= 1e-6: continue
        mis = sorted([_mutual_info(col, attrs[:, a]) for a in range(attrs.shape[1])], reverse=True)
        mis += [0.0]
        scores.append((mis[0] - mis[1]) / H)
    return float(np.mean(scores)) if scores else float("nan")


def bosdis(attrs, msgs, V):
    """Bag-of-symbols disentanglement: symbol COUNTS instead of positions."""
    attrs = np.asarray(attrs); msgs = np.asarray(msgs)
    scores = []
    for s in range(V):
        cnt = (msgs == s).sum(1)
        _, c = np.unique(cnt, return_counts=True)
        H = _entropy(c.astype(np.float64))
        if H <= 1e-6: continue
        mis = sorted([_mutual_info(cnt, attrs[:, a]) for a in range(attrs.shape[1])], reverse=True) + [0.0]
        scores.append((mis[0] - mis[1]) / H)
    return float(np.mean(scores)) if scores else float("nan")


def message_stats(msgs, V):
    msgs = np.asarray(msgs)
    N, L = msgs.shape
    uniq = len(set(map(tuple, msgs.tolist())))
    ent_pos = []
    for j in range(L):
        _, c = np.unique(msgs[:, j], return_counts=True); ent_pos.append(_entropy(c.astype(np.float64)))
    _, c = np.unique(msgs.reshape(-1), return_counts=True)
    return dict(unique_frac=uniq / N, ent_per_pos=float(np.mean(ent_pos)), ent_symbols=_entropy(c.astype(np.float64)),
                n_symbols_used=int(len(c)), max_ent=float(np.log2(V)))


def symbol_primitive_alignment(progs, msgs, n_prims):
    """For length-1 programs: is each primitive mapped to a consistent message?
    Returns mean purity: for each primitive, fraction of its episodes sharing the modal message."""
    by = {}
    for p, m in zip(progs, msgs):
        if len(p) != 1: continue
        by.setdefault(p[0], []).append(tuple(m))
    pur = []
    for k, ms in by.items():
        vals, cnt = np.unique(np.array(ms, dtype=object), return_counts=True, axis=0) if False else (None, None)
        from collections import Counter
        c = Counter(ms); pur.append(c.most_common(1)[0][1] / len(ms))
    return float(np.mean(pur)) if pur else float("nan")


def strip_eos(m):
    m = list(m)
    return m[:m.index(0)] if 0 in m else m


def dictionary(progs, msgs, use_eos=False):
    """modal message per singleton primitive"""
    from collections import Counter
    by = {}
    for p, m in zip(progs, msgs):
        if len(p) == 1:
            by.setdefault(p[0], []).append(tuple(strip_eos(m) if use_eos else m))
    return {k: Counter(v).most_common(1)[0][0] for k, v in by.items()}


def _contains(seq, sub):
    n, k = len(seq), len(sub)
    if k == 0: return False
    return any(tuple(seq[i:i + k]) == tuple(sub) for i in range(n - k + 1))


def symbol_reuse(progs, msgs, D1, use_eos=False, seed=0):
    """For length-2 programs (A,B): does the message reuse the stage-1 names of A and B?
    first : P(first symbol of D1[A] in msg)      (chance: same for a random C not in the program)
    name  : P(full D1[A] is a contiguous substring of msg)
    order : among messages containing both first symbols, P(A's before B's)"""
    rng = np.random.default_rng(seed)
    keys = list(D1.keys())
    stats = dict(first_A=[], first_B=[], first_chance=[], name_A=[], name_B=[], name_chance=[], order_AB=[])
    for p, m in zip(progs, msgs):
        if len(p) != 2 or p[0] not in D1 or p[1] not in D1: continue
        m = strip_eos(m) if use_eos else list(m)
        A, B = p
        C = keys[int(rng.integers(len(keys)))]
        while C in p and len(keys) > 2:
            C = keys[int(rng.integers(len(keys)))]
        fa, fb, fc = D1[A][:1], D1[B][:1], D1[C][:1]
        stats["first_A"].append(_contains(m, fa)); stats["first_B"].append(_contains(m, fb)); stats["first_chance"].append(_contains(m, fc))
        stats["name_A"].append(_contains(m, D1[A])); stats["name_B"].append(_contains(m, D1[B])); stats["name_chance"].append(_contains(m, D1[C]))
        if fa and fb and fa[0] in m and fb[0] in m and fa[0] != fb[0]:
            stats["order_AB"].append(m.index(fa[0]) < m.index(fb[0]))
    return {("reuse_" + k): (float(np.mean(v)) if v else float("nan")) for k, v in stats.items()}


def name_similarity_reuse(progs, msgs, D1, use_eos=False, seed=0):
    """Distance-based reuse for distributed codes: is msg(A,B) closer (Hamming over aligned
    positions, length-normalised) to A's / B's stage-1 name than to a random other name?
    Returns P(closer to A than to C), P(closer to B than to C), and mean normalised distances."""
    rng = np.random.default_rng(seed)
    keys = list(D1.keys())
    def ham(m, n):
        m = list(m); n = list(n); L = max(len(m), len(n))
        if L == 0: return 1.0
        return (sum(1 for i in range(L) if i >= len(m) or i >= len(n) or m[i] != n[i])) / L
    cA, cB, dA, dB, dC = [], [], [], [], []
    for p, m in zip(progs, msgs):
        if len(p) != 2 or p[0] not in D1 or p[1] not in D1: continue
        m = strip_eos(m) if use_eos else list(m)
        A, B = p
        C = keys[int(rng.integers(len(keys)))]
        while C in p and len(keys) > 2:
            C = keys[int(rng.integers(len(keys)))]
        a, b, c = ham(m, D1[A]), ham(m, D1[B]), ham(m, D1[C])
        dA.append(a); dB.append(b); dC.append(c); cA.append(a < c); cB.append(b < c)
    if not dA: return {}
    return dict(reuse_ham_closerA=float(np.mean(cA)), reuse_ham_closerB=float(np.mean(cB)),
                reuse_ham_dA=float(np.mean(dA)), reuse_ham_dB=float(np.mean(dB)), reuse_ham_dC=float(np.mean(dC)))
