"""Which held-out programs compute a transformation the augmented training distribution never contains?

Every training episode is augmented by one of the eight dihedral transforms T and a permutation s of the colours
1-9, applied to demonstrations and query alike (emlang/data.py: augment_episode). A training program P is therefore
trained as the function s T P T^-1 s^-1. A held-out program Q is *covered* if, for some training program P (single
operations, trained pairs, trained self-compositions and trained triples), some T and some s,

    Q(x) = s T P T^-1 s^-1 (x)   on every grid of a battery of discriminating grids and fresh random grids.

Three nested levels are reported for every held-out program:
    direct     covered with T and s the identity
    dihedral   covered with s the identity
    group      covered by some T and s (the full augmentation group)
and, for pairs, whether the pair occurs as consecutive operations of a training triple.

Colour permutations. Most operations treat all non-background colours alike; a few name colours (recolourings,
swaps, borders, painted rows...). Those parameter colours are found for every operation by testing commutation with
all colour transpositions (the minimum vertex cover of the failing transpositions). The colours a program names are
not always the colours that matter: in gravity_left then recolor9>3 then recolor9>1 the last step is idle, because
the first leaves no 9. s is therefore enumerated as the injective maps from the colours P names into the colours
either program names, identity first, and no candidate is dropped because the two programs name different colours.

Ties. Some operations break ties by colour index or by position (majority, keepMostCommon, keepRarest, rankRecolor,
markDensestRow/Col, sortRows/ColsByCount), so a conjugate differs from the original only on grids where such a step
meets a tie. Grids on which either program meets a tie are left out of the comparison, and a match needs at least
--min_compared tie-free grids. Screening uses output shape, zero mask and tie flags, which do not depend on s.
Two programs count as equal when they agree on every tested grid where neither meets a tie, which is equality up
to tie-breaking, and a match needs at least --min_compared such grids. Candidates are prefiltered only by output
shape and by which cells are empty, both unchanged by a colour permutation, and only on grids where neither
program meets a tie. Matches are verified on finite batteries, not proved.
"""
import argparse, itertools, json, os, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emlang.gen import SynthGen, random_grid, fits
from emlang.audit_split import discriminating_grids
from emlang.data import dihedral

INV = [0, 3, 2, 1, 4, 5, 6, 7]          # dihedral index of the inverse transform


def meets_tie(name, g):
    """does this operation break a tie on grid g?"""
    if name in ("majority", "keepMostCommon", "keepRarest") or name.startswith("rankRecolor"):
        v, c = np.unique(g[g != 0], return_counts=True)
        if len(v) < 2: return False
        s = np.sort(c)
        top, bottom = s[-1] == s[-2], s[0] == s[1]
        if name == "keepRarest": return bool(bottom)
        if name.startswith("rankRecolor"): return bool(top or bottom)
        return bool(top)
    if name.startswith("markDensestRow") or name.startswith("markDensestCol"):
        cnt = (g != 0).sum(1 if "Row" in name else 0)
        return bool(cnt.max() > 0 and (cnt == cnt.max()).sum() > 1)
    if name in ("sortRowsByCount", "sortColsByCount"):
        a = g if name == "sortRowsByCount" else g.T
        cnt = (a != 0).sum(1)
        for k in np.unique(cnt):
            rows = a[cnt == k]
            if len(rows) > 1 and any(not np.array_equal(rows[0], r) for r in rows[1:]): return True
    return False


class GroupAudit:
    def __init__(self, G, grids_params):
        self.G, self.names = G, G.names
        self.op_params = [self._params_of_op(i, grids_params) for i in range(len(G.lib))]

    def apply(self, prog, g):
        """output (None if a step leaves the canvas) and whether a tie-breaking step met a tie"""
        tie = False
        for i in prog:
            tie = tie or meets_tie(self.names[i], g)
            g = self.G.lib[i][0](g)
            if not fits(g): return None, tie
        return g, tie

    def _params_of_op(self, i, grids):
        f, name = self.G.lib[i][0], self.names[i]
        edges = []
        for c, d in itertools.combinations(range(1, 10), 2):
            tau = np.arange(10); tau[c], tau[d] = d, c
            for g in grids:
                if meets_tie(name, g): continue
                a, b = f(tau[g]), f(g)
                if a.shape != b.shape or not np.array_equal(a, tau[b]):
                    edges.append((c, d)); break
        for k in range(10):
            for S in itertools.combinations(range(1, 10), k):
                if all(c in S or d in S for c, d in edges): return list(S)
        return list(range(1, 10))

    def params(self, prog):
        out = []
        for i in prog:
            for c in self.op_params[i]:
                if c not in out: out.append(c)
        return sorted(out)

    def feature(self, prog, t, x):
        y, tie = self.apply(prog, dihedral(x, INV[t]))
        if tie: return "tie"
        if y is None: return None
        y = dihedral(y, t)
        return (y.shape, (y == 0).tobytes())

    def conj(self, prog, t, s, x):
        """s T P T^-1 s^-1 (x), with the tie flag of P's steps"""
        y, tie = self.apply(prog, dihedral(np.argsort(s)[x], INV[t]))
        return (None if y is None else s[dihedral(y, t)]), tie


def colour_maps(A, B, cap=None):
    """permutations of 0-9 (0 fixed) sending the colours A injectively into A + B, the identity first; every one of
    them (the cap is off by default, so the list is exhaustive: at most 9P6 = 60 480 maps for a program naming six colours).

    The colours a program names are not always the colours that matter, so the two programs need not name the same
    number of colours; nothing is rejected on that ground."""
    if not A: return [np.arange(10)]
    universe = sorted(set(A) | set(B))
    out, seen = [], set()
    for img in itertools.permutations(universe, len(A)):
        s = np.arange(10)
        for a, b in zip(A, img): s[a] = b
        src = [c for c in range(1, 10) if c not in A]; dst = [c for c in range(1, 10) if c not in img]
        for a, b in zip(src, dst): s[a] = b
        key = tuple(s.tolist())
        if key in seen: continue
        seen.add(key); out.append(s)
        if cap and len(out) >= cap: break
    out.sort(key=lambda z: not np.array_equal(z, np.arange(10)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="A"); ap.add_argument("--n_prims", type=int, default=0)
    ap.add_argument("--prim_seed", type=int, default=None); ap.add_argument("--drop_prims", default="")
    ap.add_argument("--tag", required=True); ap.add_argument("--n_random", type=int, default=2000)
    ap.add_argument("--n_screen", type=int, default=40); ap.add_argument("--min_compared", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260917); ap.add_argument("--outdir", default="audit_20260917")
    a = ap.parse_args()
    G = SynthGen(0, size_lo=3, size_hi=6, n_train3=600, library=a.library, n_prims=a.n_prims, prim_seed=a.prim_seed,
                 drop_prims=a.drop_prims)
    rng = np.random.default_rng(a.seed)
    screen = discriminating_grids() + [random_grid(rng, 3, 6) for _ in range(a.n_screen)]
    battery = discriminating_grids() + [random_grid(rng, 3, 6) for _ in range(a.n_random)]
    AU = GroupAudit(G, [random_grid(rng, 3, 6) for _ in range(150)])
    train = [tuple(p) for p in list(G.train1) + list(G.train2) + list(G.rep2_train) + list(G.train3)]
    t0 = time.time()
    screen_grids = battery[:a.n_screen]

    def screen_features(prog, t):
        """output shape and empty cells after the transform, per screening grid; None where the program meets a tie
        (those grids say nothing about a conjugate) . Both are unchanged by a colour permutation."""
        out = []
        for x in screen_grids:
            y, tie = AU.apply(prog, dihedral(x, INV[t]))
            if tie: out.append(None)
            elif y is None: out.append("out")
            else:
                z = dihedral(y, t); out.append((z.shape, (z == 0).tobytes()))
        return out

    train_feat = [[screen_features(p, t) for t in range(8)] for p in train]
    print(f"[{a.tag}] {len(G.lib)} operations, parameter colours "
          + ", ".join(f"{n}:{''.join(map(str, c))}" for n, c in zip(AU.names, AU.op_params) if c)
          + f"; screened {len(train)} training programs x 8 transforms in {time.time() - t0:.0f}s", flush=True)

    def compatible(fq, fp):
        """can these two be conjugate? compare only the grids where neither meets a tie; a tie says nothing"""
        return all(u == v for u, v in zip(fq, fp) if u is not None and v is not None)

    def equal(pairs, p, t, s, first=60):
        """do the conjugate of p and the held-out program agree on every grid where neither meets a tie?

        `pairs` are the battery grids on which the held-out program is tie-free, with its outputs, computed once
        per held-out program. A candidate is compared on the first `first` of them and only then on the rest,
        which decides almost every candidate after a few grids; the verdict is the same either way."""
        n = 0
        for lo, hi in ((0, first), (first, len(pairs))):
            for x, yq in pairs[lo:hi]:
                yp, tp = AU.conj(p, t, s, x)
                if tp: continue
                if (yq is None) != (yp is None): return False
                if yq is not None and (yq.shape != yp.shape or not np.array_equal(yq, yp)): return False
                n += 1
        return n >= a.min_compared

    RANK = ("direct", "dihedral", "group")

    def audit(q):
        fq = screen_features(q, 0)
        B = AU.params(q)
        pairs = [(x, AU.apply(q, x)[0]) for x in battery if not AU.apply(q, x)[1]]
        cands = []
        for i, p in enumerate(train):
            A = AU.params(p)
            maps = colour_maps(A, B)
            for t in range(8):
                if not compatible(fq, train_feat[i][t]): continue
                for s in maps:
                    fixed = all(s[c] == c for c in A)
                    level = "direct" if (t == 0 and fixed) else ("dihedral" if fixed else "group")
                    cands.append((RANK.index(level), p, t, s, level))
        cands.sort(key=lambda z: z[0])
        found = {}
        for rank, p, t, s, level in cands:
            if level in found or any(RANK.index(k) < rank for k in found): continue
            if equal(pairs, p, t, s):
                found[level] = dict(program=[AU.names[i] for i in p], t=t, colour_map=s.tolist())
                if level == "direct": break
        covered = {lvl: any(RANK.index(k) <= r for k in found) for r, lvl in enumerate(RANK)}
        return covered, found

    os.makedirs(a.outdir, exist_ok=True)
    sub_pairs = {tuple(p[i:i + 2]) for p in G.train3 for i in range(2)}
    for split, held in (("hold2", G.hold2), ("len3", G.len3)):
        t1 = time.time(); rows = []
        for q in held:
            covered, found = audit(tuple(q))
            rows.append(dict(prog=list(q), names=[AU.names[i] for i in q], covered=covered, witnesses=found,
                             in_training_triple=(tuple(q) in sub_pairs) if split == "hold2" else None))
        lv = lambda lvl: [r["prog"] for r in rows if not r["covered"][lvl]]
        rep = dict(tag=a.tag, split=split, library=a.library, n_prims=a.n_prims, prim_seed=a.prim_seed, drop_prims=a.drop_prims,
                   n_heldout=len(rows), n_random_grids=a.n_random, min_compared=a.min_compared,
                   parameter_colours={n: c for n, c in zip(AU.names, AU.op_params) if c},
                   novel_direct=lv("direct"), novel_dihedral=lv("dihedral"), novel_group=lv("group"), rows=rows)
        if split == "hold2":
            rep["novel_group_strict"] = [r["prog"] for r in rows if not r["covered"]["group"] and not r["in_training_triple"]]
        json.dump(rep, open(os.path.join(a.outdir, f"group_audit_{a.tag}_{split}.json"), "w"), indent=1)
        print(f"[{a.tag}] {split}: {len(rows)} held out; not covered directly {len(rep['novel_direct'])}, "
              f"nor through a dihedral transform {len(rep['novel_dihedral'])}, nor through the full augmentation group "
              f"{len(rep['novel_group'])}" + (f" (also not inside a training triple: {len(rep['novel_group_strict'])})" if split == "hold2" else "")
              + f"  [{time.time() - t1:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
