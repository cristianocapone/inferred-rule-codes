"""Audit of the compositional split: which held-out pairs are genuinely new *functions*?

A held-out pair is only a test of composition if the transformation it computes is not already
available from training. Three ways it can fail to be:

  (a) it equals a training single       (flipV then flipH is exactly rot180, and rot180 is trained)
  (b) it equals a training pair or a trained self-composition
  (c) it occurs as a consecutive sub-sequence of a training triple (weaker: the intermediate grid is
      never shown, and the speaker names the whole program, so this is a caution, not leakage)

Equivalence is decided on a battery of *discriminating* grids rather than random ones: grids whose
cells are all distinct separate every geometric, shift and gravity operation; grids containing every
colour separate every recolouring and swap; non-square grids separate transpose from the rotations.
A random-grid battery is run as well, and the two must agree.

Writes the clean subset of held-out pairs so that the accuracy of every arm can be recomputed on it.
"""
import argparse, itertools, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("EMLANG_CANVAS", "12")
from emlang.gen import SynthGen, random_grid, fits


def discriminating_grids():
    """small grids built to separate operations that random grids might not"""
    gs = []
    for H, W in ((3, 3), (3, 4), (4, 3), (4, 4), (2, 5), (5, 2)):
        g = (np.arange(H * W) % 9 + 1).reshape(H, W)                 # every cell distinct where possible
        gs.append(g.copy())
        h = g.copy(); h[0, 0] = 0; gs.append(h)                       # a hole, so gravity and fills act
    gs.append(np.arange(1, 10).reshape(3, 3))                         # all nine colours exactly once
    gs.append(np.array([[1, 2, 3], [0, 0, 0], [4, 5, 6]]))            # empty row: gravity, fills
    gs.append(np.array([[1, 0, 0], [0, 5, 0], [0, 0, 9]]))            # sparse diagonal
    gs.append(np.array([[7, 7, 0], [0, 9, 9], [5, 0, 5]]))            # repeated colours: majority, swaps
    gs.append(np.array([[9, 5, 4], [8, 7, 6], [3, 2, 1]]))            # 3x3 distinct, another order
    gs.append(np.array([[1, 2], [3, 4]]))
    gs.append(np.array([[6, 9], [9, 6]]))                             # symmetric under the swap ops
    return gs


def signature(G, prog, grids):
    out = []
    for g in grids:
        o, _ = G._apply(prog, g)
        out.append(None if o is None else (o.shape, o.tobytes()))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_random", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--library", default="A"); ap.add_argument("--n_prims", type=int, default=0)
    ap.add_argument("--prim_seed", type=int, default=None); ap.add_argument("--drop_prims", default=""); ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="audit_20260916/split_audit.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    G = SynthGen(0, size_lo=3, size_hi=6, n_train3=600, library=a.library, n_prims=a.n_prims, prim_seed=a.prim_seed, drop_prims=a.drop_prims)
    audit_split(G, a, "hold2"); audit_split(G, a, "len3")


def audit_split(G, a, split):
    rng = np.random.default_rng(a.seed)
    disc = discriminating_grids()
    rand = [random_grid(rng, 3, 6) for _ in range(a.n_random)]
    name = lambda p: "∘".join(G.names[i] for i in p)

    # index every training program by its behaviour (singles, pairs, self-compositions; triples too for len3)
    train = list(G.train1) + list(G.train2) + list(G.rep2_train) + (list(G.train3) if split == "len3" else [])
    by_sig = defaultdict(list)
    for p in train: by_sig[signature(G, p, disc)].append(p)
    sub_pairs = {tuple(p[i:i + 2]) for p in G.train3 for i in range(2)}
    sub_of_train = sub_pairs if split == "hold2" else {tuple(p) for p in G.train3}
    held = G.hold2 if split == "hold2" else G.len3

    rows, disagree = [], 0
    for p in held:
        s = signature(G, p, disc)
        cands = by_sig.get(s, [])
        # a candidate must also agree on the random battery
        sr = signature(G, p, rand)
        equiv = [q for q in cands if signature(G, q, rand) == sr]
        if cands and not equiv: disagree += 1
        kind = None
        if equiv:
            kind = "equals_single" if any(len(q) == 1 for q in equiv) else "equals_trained_pair"
        rows.append(dict(pair=name(p), prog=list(p), equivalent_to=[name(q) for q in equiv], kind=kind,
                         in_training_triple=tuple(p) in sub_of_train,
                         families=[G.lib[i][2] for i in p]))
    n = len(rows)
    eq = [r for r in rows if r["kind"]]
    eq_single = [r for r in rows if r["kind"] == "equals_single"]
    tri = [r for r in rows if r["in_training_triple"]]
    clean = [r for r in rows if not r["kind"] and not r["in_training_triple"]]
    clean_fn = [r for r in rows if not r["kind"]]                     # functionally novel, triples allowed
    fam = defaultdict(int)
    for r in eq: fam["+".join(sorted(set(r["families"])))] += 1
    print(f"\n=== {a.tag or a.library}, split {split}: {n} programs ===")
    print(f"  equivalent to a training program: {len(eq)} ({len(eq)/n:.0%})  [of which equal to a single: {len(eq_single)}]")
    print(f"  occur inside a training program:  {len(tri)} ({len(tri)/n:.0%})")
    print(f"  FUNCTIONALLY NOVEL:               {len(clean_fn)} ({len(clean_fn)/n:.0%})")
    print(f"  novel AND not inside a triple:    {len(clean)} ({len(clean)/n:.0%})")
    print(f"  discriminating and random batteries disagreed on: {disagree} pairs")
    print("  families of the equivalent pairs: " + ", ".join(f"{k} {v}" for k, v in sorted(fam.items(), key=lambda x: -x[1])))
    print("  examples of equivalence: " + "; ".join(f"{r['pair']} = {r['equivalent_to'][0]}" for r in eq[:6]))
    out = a.out.replace(".json", f"_{a.tag or (a.library + (str(a.n_prims) if a.n_prims else ''))}_{split}.json")
    json.dump(dict(split=split, library=a.library, n_heldout=n, n_equivalent=len(eq), n_equal_single=len(eq_single), n_in_triple=len(tri),
                   n_functionally_novel=len(clean_fn), n_clean=len(clean), disagreements=disagree,
                   n_random_grids=a.n_random, n_discriminating_grids=len(disc), rows=rows,
                   clean_progs=[r["prog"] for r in clean], novel_progs=[r["prog"] for r in clean_fn]),
              open(out, "w"), indent=1)
    print("  wrote", out)


if __name__ == "__main__":
    main()
