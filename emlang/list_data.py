"""List-to-list rule induction — the domain of Kean et al. (PNAS 2026), in which severely aphasic
patients inferred the hidden rule mapping one list of numbers onto another and conveyed it
non-verbally.

A task is a program over lists of digits (1-9, background 0 unused). Primitives are drawn from four
families: arithmetic (add/subtract a constant, modular), order (reverse, rotate, sort), selection
(keep first/last k, drop extremes, deduplicate) and structural (repeat each element, append the
reversed list, interleave with a constant). Lists are rendered into the same grid canvas the
speaker/executor already consume — one row per list — so the architecture is unchanged and the only
thing that differs from the grid domain is the structure of the rules.

Compositional split mirrors the grid generator: all singles train; ordered pairs split train/held-out;
triples held out entirely.
"""
import os
import numpy as np
from .gen import CANVAS

MAXLEN = min(CANVAS, 12)


# ---------------------------------------------------------------- primitives
def _addc(c):
    return (lambda l: [((x - 1 + c) % 9) + 1 for x in l]), f"add{c}"


def _subc(c):
    return (lambda l: [((x - 1 - c) % 9) + 1 for x in l]), f"sub{c}"


def _reverse():
    return (lambda l: l[::-1]), "reverse"


def _rotate(k):
    return (lambda l: l[k:] + l[:k] if l else l), f"rot{k}"


def _sort(desc=False):
    return (lambda l: sorted(l, reverse=desc)), ("sortdesc" if desc else "sortasc")


def _keep_first(k):
    return (lambda l: l[:k]), f"first{k}"


def _keep_last(k):
    return (lambda l: l[-k:]), f"last{k}"


def _drop_max():
    def f(l):
        if not l: return l
        m = max(l); return [x for x in l if x != m] or l
    return f, "dropmax"


def _drop_min():
    def f(l):
        if not l: return l
        m = min(l); return [x for x in l if x != m] or l
    return f, "dropmin"


def _dedup():
    def f(l):
        out = []
        for x in l:
            if not out or out[-1] != x: out.append(x)
        return out
    return f, "dedup"


def _repeat_each(k):
    return (lambda l: [x for x in l for _ in range(k)][:MAXLEN]), f"rep{k}"


def _append_rev():
    return (lambda l: (l + l[::-1])[:MAXLEN]), "mirror"


def _interleave(c):
    def f(l):
        out = []
        for x in l: out += [x, c]
        return out[:MAXLEN]
    return f, f"weave{c}"


def _swap_ends():
    def f(l):
        if len(l) < 2: return l
        m = list(l); m[0], m[-1] = m[-1], m[0]; return m
    return f, "swapends"


def _maxfill():
    def f(l):
        if not l: return l
        m = max(l); return [m] * len(l)
    return f, "maxfill"


def build_library(seed=0, same_len=False):
    """same_len=True keeps only length-preserving operations, so the output length always equals the
    input length. Without it the model must predict the length as well, which makes the domain
    near-unlearnable for the grid architecture used here (see report 7.17)."""
    rng = np.random.default_rng(seed)
    lib = [_reverse(), _sort(False), _sort(True), _swap_ends(), _maxfill(), _rotate(1), _rotate(2)]
    for c in (1, 2, 3, 4):
        lib.append(_addc(c))
    for c in (1, 2):
        lib.append(_subc(c))
    if not same_len:
        lib += [_dedup(), _drop_max(), _drop_min(), _append_rev(), _keep_first(3), _keep_last(3), _repeat_each(2)]
        for c in (int(rng.integers(1, 10)), int(rng.integers(1, 10))):
            lib.append(_interleave(c))
    seen, out = set(), []
    for f, n in lib:
        if n not in seen:
            seen.add(n); out.append((f, n))
    return out


def random_list(rng, lo=3, hi=8):
    n = int(rng.integers(lo, hi + 1))
    return [int(x) for x in rng.integers(1, 10, size=n)]


def to_grid(l):
    """render a list as a 1 x len grid (the canvas pad marks the length)"""
    return np.array([l], dtype=np.int64) if l else np.array([[1]], dtype=np.int64)


class ListGen:
    def __init__(self, seed=0, holdout_frac=0.25, n_len3=300, n_train3=0, max_len=MAXLEN, same_len=False):
        self.lib = build_library(seed, same_len=same_len)
        self.n = len(self.lib)
        self.names = [x[1] for x in self.lib]
        self.max_len = max_len
        rng = np.random.default_rng(seed + 1)
        pairs = [(i, j) for i in range(self.n) for j in range(self.n) if i != j]
        pairs = [p for p in pairs if not self._degenerate(list(p), rng)]
        rng.shuffle(pairs)
        k = int(len(pairs) * holdout_frac)
        self.hold2, self.train2 = pairs[:k], pairs[k:]
        self.train1 = [(i,) for i in range(self.n)]
        reps = [(i, i) for i in range(self.n) if not self._degenerate([i, i], rng)]
        rng.shuffle(reps); kr = max(1, len(reps) // 3)
        self.rep2_hold, self.rep2_train = reps[:kr], reps[kr:]
        tri = set(); tries = 0
        while len(tri) < n_len3 + n_train3 and tries < 40000:
            tries += 1
            t = tuple(int(x) for x in rng.choice(self.n, 3, replace=True))
            if len(set(t)) < 2 or self._degenerate(list(t), rng): continue
            tri.add(t)
        tri = sorted(tri); rng.shuffle(tri)
        self.len3, self.train3 = sorted(tri[:n_len3]), sorted(tri[n_len3:])
        self.split_sizes = dict(train1=len(self.train1), train2=len(self.train2), hold2=len(self.hold2),
                                rep2_train=len(self.rep2_train), rep2_hold=len(self.rep2_hold),
                                train3=len(self.train3), len3=len(self.len3))
        self.stage_pools = {1: [(self.train1, 1.0)],
                            2: [(self.train1, 0.25), (self.train2, 0.6), (self.rep2_train, 0.15)],
                            3: [(self.train1, 0.15), (self.train2, 0.35), (self.rep2_train, 0.1), (self.train3, 0.4)]}
        self.stage_order = None

    def _apply(self, prog, l):
        for i in prog:
            l = self.lib[i][0](l)
            if not l or len(l) > self.max_len: return None
        return l

    def _degenerate(self, prog, rng, n=6):
        same = 0
        for _ in range(n):
            l = random_list(rng)
            o = self._apply(prog, l)
            if o is None: return True
            if o == l: same += 1
        return same >= n - 1

    def sample_program(self, rng, split):
        if isinstance(split, int):
            pools = self.stage_pools[self.stage_order[split - 1] if self.stage_order else split]
            w = np.array([p[1] for p in pools if len(p[0])]); pl = [p[0] for p in pools if len(p[0])]
            pool = pl[int(rng.choice(len(pl), p=w / w.sum()))]
            return list(pool[int(rng.integers(len(pool)))])
        pool = {"train1": self.train1, "train2": self.train2, "hold2": self.hold2, "len3": self.len3,
                "rep2_train": self.rep2_train, "rep2_hold": self.rep2_hold, "train3": self.train3}[split]
        return list(pool[int(rng.integers(len(pool)))])

    def make_episode(self, rng, split="train", n_demos=3, prog=None):
        prog = list(prog) if prog is not None else self.sample_program(rng, split)
        pairs = []
        for _ in range(n_demos + 1):
            for _try in range(30):
                l = random_list(rng)
                o = self._apply(prog, l)
                if o is not None and o != l: break
            else:
                l = random_list(rng, 3, 5); o = self._apply(prog, l) or l
            pairs.append((to_grid(l), to_grid(o)))
        return dict(demos=pairs[:n_demos], query=pairs[n_demos], prog=prog, inter=None,
                    source="list", split=str(split), tid="list_" + "_".join(self.names[i] for i in prog))

    def fixed_eval_set(self, split, n, seed=123, n_demos=3):
        rng = np.random.default_rng(seed)
        pool = {"train1": self.train1, "train2": self.train2, "hold2": self.hold2, "len3": self.len3,
                "rep2_train": self.rep2_train, "rep2_hold": self.rep2_hold, "train3": self.train3}[split]
        return [self.make_episode(rng, split, n_demos, prog=pool[i % len(pool)]) for i in range(n)]


if __name__ == "__main__":
    G = ListGen(0, n_train3=600)
    print("primitives", G.n, G.names)
    print("splits", G.split_sizes)
    rng = np.random.default_rng(0)
    for sp in ["train1", "train2", "hold2", "len3"]:
        e = G.make_episode(rng, sp)
        d = e["demos"][0]
        print(f"{sp:7s} {e['tid'][5:]:34s} {d[0].tolist()[0]} -> {d[1].tolist()[0]}")
