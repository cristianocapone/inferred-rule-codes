"""How much partial credit does the executor earn by confusing one operation with another?
For each library: sample inputs, apply every operation, and measure the cell accuracy of using
op_i's output as the answer for op_j (union accuracy, the training reward). An operation with a
near-equivalent partner under this measure can be served by a shared, generic message at little cost
- the seed of collapse. Computed from the generator alone, before any training."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emlang.gen import SynthGen
from emlang.data import pad_grid
from emlang.model import union_accuracy

DROP = "swap95,transpose,swap48,rot270,majority,recolor6>9,swap97"
LIBS = [("A30", dict()), ("A24", dict(n_prims=24)), ("A17", dict(n_prims=17)), ("A17r", dict(n_prims=17, prim_seed=1)),
        ("Aclean", dict(drop_prims=DROP)), ("B", dict(library="B")), ("Bdup", dict(library="Bdup"))]
COLLAPSE = {"A30": "yes", "A24": "yes", "A17": "partial", "A17r": "no", "Aclean": "yes", "B": "no", "Bdup": "no"}


def matrix(G, n=300, seed=5):
    rng = np.random.default_rng(seed)
    outs = [[] for _ in range(G.n)]
    for _ in range(n):
        ep = G.make_episode(rng, "train1", n_demos=1, prog=[0]); x = ep["query"][0]
        for i, (f, _, _) in enumerate(G.lib):
            try: outs[i].append(pad_grid(f(x)))
            except Exception: outs[i].append(pad_grid(x))
    T = [torch.from_numpy(np.stack(o)) for o in outs]
    M = np.zeros((G.n, G.n))
    for i in range(G.n):
        for j in range(G.n):
            M[i, j] = union_accuracy(T[i], T[j]).mean().item()
    return M


if __name__ == "__main__":
    print(f"{'library':8s} {'ops':>3s} {'ops with a partner >= 0.8':>25s} {'>= 0.7':>7s} {'mean best partner':>17s} {'collapse':>8s}   families of near-equivalents (>= 0.8)")
    for name, kw in LIBS:
        G = SynthGen(0, size_lo=3, size_hi=6, n_train3=600, **kw)
        M = matrix(G); np.fill_diagonal(M, -1); best = M.max(1)
        groups = []
        seen = set()
        for i in range(G.n):
            if i in seen: continue
            grp = [i] + [j for j in range(G.n) if j != i and M[i, j] >= 0.8]
            if len(grp) > 1: groups.append([G.names[k] for k in grp]); seen.update(grp)
        print(f"{name:8s} {G.n:3d} {int((best >= 0.8).sum()):25d} {int((best >= 0.7).sum()):7d} {best.mean():17.2f} {COLLAPSE[name]:>8s}   " + "; ".join("/".join(g) for g in groups))
