"""Compositional synthetic grid tasks with KNOWN program structure.

Primitives are parameterised Grid->Grid functions. A program is a sequence of 1-3
primitive instances applied left to right. The split is compositional:
  train  : all length-1 programs + a fixed subset of ordered pairs (A,B)
  hold2  : ordered pairs whose components both appear in training but whose
           combination never did (systematic generalisation)
  len3   : random triples (length generalisation; never trained)
The generator records the ground-truth program (list of primitive ids) and the
intermediate grids, so the emergent message can be scored for compositionality
(topsim / posdis) and the executor state for serial execution.
"""
import os
import numpy as np

CANVAS = int(os.environ.get("EMLANG_CANVAS", "30"))   # curriculum runs use a small canvas (e.g. 10)
PAD = 10


# ----------------------------------------------------------------- primitives
def _flipH():  return (lambda g: np.fliplr(g).copy()), "flipH", "geom"
def _flipV():  return (lambda g: np.flipud(g).copy()), "flipV", "geom"
def _rot90():  return (lambda g: np.rot90(g, 1).copy()), "rot90", "geom"
def _rot180(): return (lambda g: np.rot90(g, 2).copy()), "rot180", "geom"
def _rot270(): return (lambda g: np.rot90(g, 3).copy()), "rot270", "geom"
def _transp(): return (lambda g: g.T.copy()), "transpose", "geom"


def _gravity(d):
    def f(g):
        H, W = g.shape
        out = np.zeros_like(g)
        if d in ("down", "up"):
            for c in range(W):
                v = g[:, c][g[:, c] != 0]
                if len(v):
                    if d == "down": out[H - len(v):, c] = v
                    else: out[:len(v), c] = v
        else:
            for r in range(H):
                v = g[r][g[r] != 0]
                if len(v):
                    if d == "right": out[r, W - len(v):] = v
                    else: out[r, :len(v)] = v
        return out
    return f, f"gravity_{d}", "gravity"


def _recolor(a, b):
    def f(g):
        o = g.copy(); o[g == a] = b; return o
    return f, f"recolor{a}>{b}", "recolor"


def _swap(a, b):
    def f(g):
        o = g.copy(); o[g == a] = b; o[g == b] = a; return o
    return f, f"swap{a}{b}", "swap"


def _shift(dr, dc):
    def f(g):
        H, W = g.shape
        o = np.zeros_like(g)
        rs, re = max(dr, 0), H + min(dr, 0)
        cs, ce = max(dc, 0), W + min(dc, 0)
        o[rs:re, cs:ce] = g[rs - dr:re - dr, cs - dc:ce - dc]
        return o
    return f, f"shift{dr:+d}{dc:+d}", "shift"


def _border(c):
    def f(g):
        o = g.copy(); o[0, :] = c; o[-1, :] = c; o[:, 0] = c; o[:, -1] = c; return o
    return f, f"border{c}", "border"


def _majority():
    def f(g):
        v, n = np.unique(g[g != 0], return_counts=True)
        if len(v) == 0: return g.copy()
        o = g.copy(); o[g != 0] = int(v[np.argmax(n)]); return o
    return f, "majority", "color"


def _upscale2():
    def f(g):
        return np.kron(g, np.ones((2, 2), dtype=g.dtype))
    return f, "upscale2", "size"


def _crop_bbox():
    def f(g):
        nz = np.argwhere(g != 0)
        if len(nz) == 0: return g.copy()
        r0, c0 = nz.min(0); r1, c1 = nz.max(0) + 1
        return g[r0:r1, c0:c1].copy()
    return f, "crop", "size"


def _mirror_concat_h():
    def f(g):
        return np.concatenate([g, np.fliplr(g)], axis=1)
    return f, "mirrorH", "size"


def _mirror_concat_v():
    def f(g):
        return np.concatenate([g, np.flipud(g)], axis=0)
    return f, "mirrorV", "size"


def _outline():
    def f(g):
        H, W = g.shape
        o = np.zeros_like(g)
        for r in range(H):
            for c in range(W):
                if g[r, c] == 0: continue
                edge = r == 0 or c == 0 or r == H - 1 or c == W - 1 or \
                    g[r - 1, c] == 0 or g[r + 1, c] == 0 or g[r, c - 1] == 0 or g[r, c + 1] == 0
                if edge: o[r, c] = g[r, c]
        return o
    return f, "outline", "shape"


def build_library(seed=0):
    """Fixed library of primitive INSTANCES (fn, name, family)."""
    rng = np.random.default_rng(seed)
    lib = [_flipH(), _flipV(), _rot90(), _rot180(), _rot270(), _transp(),
           _gravity("down"), _gravity("up"), _gravity("left"), _gravity("right"),
           _majority(), _upscale2(), _crop_bbox(), _mirror_concat_h(), _mirror_concat_v(), _outline(),
           _shift(1, 0), _shift(0, 1), _shift(-1, 0), _shift(0, -1)]
    cols = np.arange(1, 10)
    seen = set(x[1] for x in lib)
    def add(p):
        if p[1] not in seen:
            seen.add(p[1]); lib.append(p)
    while sum(1 for x in lib if x[2] == "recolor") < 5:
        a, b = rng.choice(cols, 2, replace=False); add(_recolor(int(a), int(b)))
    while sum(1 for x in lib if x[2] == "swap") < 3:
        a, b = rng.choice(cols, 2, replace=False); add(_swap(int(a), int(b)))
    while sum(1 for x in lib if x[2] == "border") < 2:
        add(_border(int(rng.integers(1, 10))))
    return lib


# ------------------------------------------------------------- input grids
def random_grid(rng, size_lo=4, size_hi=12, n_colors=None):
    H = int(rng.integers(size_lo, size_hi + 1)); W = int(rng.integers(size_lo, size_hi + 1))
    n_colors = n_colors or int(rng.integers(2, 6))
    palette = rng.choice(np.arange(1, 10), n_colors, replace=False)
    kind = rng.integers(0, 3)
    g = np.zeros((H, W), dtype=np.int64)
    if kind == 0:  # sparse noise
        mask = rng.random((H, W)) < rng.uniform(0.2, 0.5)
        g[mask] = rng.choice(palette, int(mask.sum()))
    elif kind == 1:  # a few rectangles
        for _ in range(int(rng.integers(1, 4))):
            h = int(rng.integers(1, max(2, H // 2) + 1)); w = int(rng.integers(1, max(2, W // 2) + 1))
            r = int(rng.integers(0, H - h + 1)); c = int(rng.integers(0, W - w + 1))
            g[r:r + h, c:c + w] = rng.choice(palette)
    else:  # rectangles + noise
        for _ in range(int(rng.integers(1, 3))):
            h = int(rng.integers(1, max(2, H // 2) + 1)); w = int(rng.integers(1, max(2, W // 2) + 1))
            r = int(rng.integers(0, H - h + 1)); c = int(rng.integers(0, W - w + 1))
            g[r:r + h, c:c + w] = rng.choice(palette)
        mask = rng.random((H, W)) < 0.1
        g[mask] = rng.choice(palette, int(mask.sum()))
    return g


def fits(g):
    return g.shape[0] <= CANVAS and g.shape[1] <= CANVAS and g.shape[0] >= 1 and g.shape[1] >= 1


class SynthGen:
    """Holds the library and the compositional split; samples episodes."""

    def __init__(self, seed=0, holdout_frac=0.25, n_len3=300, size_lo=4, size_hi=12, n_train3=0, library="A", n_prims=0, prim_seed=None, drop_prims=""):
        if library == "B":
            from .gen_b import build_library_b
            self.lib = build_library_b(seed)
        elif library == "Bdup":
            from .gen_b import build_library_b_dup
            self.lib = build_library_b_dup(seed)
        else:
            self.lib = build_library(seed)
        if drop_prims:                   # remove named operations (e.g. the ones a frozen perception confuses)
            drop = set(drop_prims.split(","))
            self.lib = [x for x in self.lib if x[1] not in drop]
        if n_prims:                      # restrict the library, keeping the rest of the protocol fixed
            if prim_seed is None:        # first N (family-biased: geometry, gravity, size come first)
                self.lib = self.lib[:n_prims]
            else:                        # a seeded random subset, so composition is not confounded with count
                idx = sorted(np.random.default_rng(1000 + prim_seed).choice(len(self.lib), n_prims, replace=False).tolist())
                self.lib = [self.lib[i] for i in idx]
        self.n = len(self.lib)
        self.names = [x[1] for x in self.lib]
        self.fams = [x[2] for x in self.lib]
        self.size_lo, self.size_hi = size_lo, size_hi
        rng = np.random.default_rng(seed + 1)
        pairs = [(i, j) for i in range(self.n) for j in range(self.n) if i != j]
        pairs = [p for p in pairs if not self._degenerate(list(p), rng)]
        rng.shuffle(pairs)
        k = int(len(pairs) * holdout_frac)
        self.hold2 = pairs[:k]
        self.train2 = pairs[k:]
        self.train1 = [(i,) for i in range(self.n)]
        # repeats A∘A (Dehaene's "Repeat"): split into train / held-out
        reps = [(i, i) for i in range(self.n) if not self._degenerate([i, i], rng)]
        rng.shuffle(reps)
        kr = max(1, len(reps) // 3)
        self.rep2_hold, self.rep2_train = reps[:kr], reps[kr:]
        tri = set()
        tries = 0
        while len(tri) < n_len3 + n_train3 and tries < 40000:
            tries += 1
            t = tuple(int(x) for x in rng.choice(self.n, 3, replace=True))
            if len(set(t)) < 2: continue
            if self._degenerate(list(t), rng): continue
            tri.add(t)
        tri = sorted(tri); rng.shuffle(tri)
        self.len3 = sorted(tri[:n_len3])           # held-out triples (never trained)
        self.train3 = sorted(tri[n_len3:])         # training triples for curriculum stage 3
        self.split_sizes = dict(train1=len(self.train1), train2=len(self.train2), hold2=len(self.hold2),
                                rep2_train=len(self.rep2_train), rep2_hold=len(self.rep2_hold),
                                train3=len(self.train3), len3=len(self.len3))
        # curriculum stage pools (programs sampled uniformly over the listed pools, with weights)
        self.stage_pools = {
            1: [(self.train1, 1.0)],
            2: [(self.train1, 0.25), (self.train2, 0.6), (self.rep2_train, 0.15)],
            3: [(self.train1, 0.15), (self.train2, 0.35), (self.rep2_train, 0.1), (self.train3, 0.4)],
        }

    def _apply(self, prog, g):
        inter = []
        for i in prog:
            g = self.lib[i][0](g)
            if not fits(g): return None, None
            inter.append(g)
        return g, inter

    def _degenerate(self, prog, rng, n=6):
        """Program is identity (or invalid) on most random grids -> drop."""
        same = 0
        for _ in range(n):
            g = random_grid(rng, self.size_lo, self.size_hi)
            o, _ = self._apply(prog, g)
            if o is None: return True
            if o.shape == g.shape and np.array_equal(o, g): same += 1
        return same >= n - 1

    def sample_program(self, rng, split):
        if isinstance(split, int):  # curriculum stage
            pools = self.stage_pools[split]
            if getattr(self, "stage_order", None):          # reverse / custom order: stage k uses pool stage_order[k-1]
                pools = self.stage_pools[self.stage_order[split - 1]]
            w = np.array([p[1] for p in pools if len(p[0])]); pl = [p[0] for p in pools if len(p[0])]
            pool = pl[int(rng.choice(len(pl), p=w / w.sum()))]
            return list(pool[int(rng.integers(len(pool)))])
        if split == "train":
            pool = self.train1 if rng.random() < 0.3 else self.train2
        elif split == "hold2":
            pool = self.hold2
        elif split == "len3":
            pool = self.len3
        elif split == "train1":
            pool = self.train1
        elif split == "train2":
            pool = self.train2
        elif split == "rep2_train":
            pool = self.rep2_train
        elif split == "rep2_hold":
            pool = self.rep2_hold
        elif split == "train3":
            pool = self.train3
        else:
            raise ValueError(split)
        return list(pool[int(rng.integers(len(pool)))])

    def make_episode(self, rng, split="train", n_demos=3, prog=None):
        prog = list(prog) if prog is not None else self.sample_program(rng, split)
        pairs, inters = [], []
        for _ in range(n_demos + 1):
            for _attempt in range(30):
                g = random_grid(rng, self.size_lo, self.size_hi)
                o, inter = self._apply(prog, g)
                if o is None: continue
                if o.shape == g.shape and np.array_equal(o, g): continue
                break
            else:
                # give up on non-identity; accept whatever fits
                g = random_grid(rng, self.size_lo, max(self.size_lo, min(self.size_hi, 8)))
                o, inter = self._apply(prog, g)
                if o is None:
                    o, inter = g.copy(), [g.copy()] * len(prog)
            pairs.append((g, o)); inters.append(inter)
        return dict(demos=pairs[:n_demos], query=pairs[n_demos], prog=prog,
                    inter=inters[n_demos], source="synth", split=split,
                    tid="synth_" + "_".join(self.names[i] for i in prog))

    def fixed_eval_set(self, split, n, seed=123, n_demos=3):
        """Deterministic evaluation episodes covering the split's programs evenly."""
        rng = np.random.default_rng(seed)
        pool = {"train1": self.train1, "train2": self.train2, "hold2": self.hold2, "len3": self.len3,
                "rep2_train": self.rep2_train, "rep2_hold": self.rep2_hold, "train3": self.train3}[split]
        eps = []
        for k in range(n):
            prog = pool[k % len(pool)]
            eps.append(self.make_episode(rng, split, n_demos=n_demos, prog=prog))
        return eps


if __name__ == "__main__":
    G = SynthGen(0)
    print("library", G.n, G.names)
    print("splits", G.split_sizes)
    rng = np.random.default_rng(0)
    for s in ["train", "hold2", "len3"]:
        e = G.make_episode(rng, s)
        print(s, e["tid"], [d[0].shape for d in e["demos"]], "->", [d[1].shape for d in e["demos"]])
