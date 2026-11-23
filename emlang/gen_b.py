"""A second, disjoint library of grid operations, for testing whether the results depend on the
particular set of primitives rather than on the channel.

Library A (gen.py) is built from geometry, gravity, translation, colour and size/shape families.
Library B shares no operation with it: it is built from counting, symmetry, connectivity, ordering
and masking - families that require different computations of the executor. The generator, the
compositional split and the episode format are identical, so the two libraries are interchangeable
everywhere else in the pipeline.
"""
import numpy as np
from .gen import CANVAS, PAD, random_grid, fits


def _keep_largest():
    def f(g):
        v, c = np.unique(g[g != 0], return_counts=True)
        if len(v) == 0: return g.copy()
        keep = int(v[np.argmax(c)])
        o = g.copy(); o[(g != 0) & (g != keep)] = 0; return o
    return f, "keepMostCommon", "counting"


def _keep_rarest():
    def f(g):
        v, c = np.unique(g[g != 0], return_counts=True)
        if len(v) == 0: return g.copy()
        keep = int(v[np.argmin(c)])
        o = g.copy(); o[(g != 0) & (g != keep)] = 0; return o
    return f, "keepRarest", "counting"


def _count_to_row(color):
    def f(g):
        n = int((g != 0).sum())
        n = max(1, min(n, g.shape[1]))
        o = np.zeros_like(g); o[0, :n] = color; return o
    return f, f"countRow{color}", "counting"


def _symmetrise(axis):
    def f(g):
        o = g.copy()
        m = np.fliplr(g) if axis == "h" else np.flipud(g)
        o[o == 0] = m[o == 0]; return o
    return f, f"symmetrise_{axis}", "symmetry"


def _keep_symmetric():
    def f(g):
        o = g.copy(); m = np.fliplr(g)
        o[g != m] = 0; return o
    return f, "keepSymmetric", "symmetry"


def _largest_row(color):
    def f(g):
        counts = (g != 0).sum(1)
        if counts.max() == 0: return g.copy()
        o = g.copy(); o[int(np.argmax(counts)), :] = color; return o
    return f, f"markDensestRow{color}", "counting"


def _largest_col(color):
    def f(g):
        counts = (g != 0).sum(0)
        if counts.max() == 0: return g.copy()
        o = g.copy(); o[:, int(np.argmax(counts))] = color; return o
    return f, f"markDensestCol{color}", "counting"


def _fill_row_if_any(color):
    def f(g):
        o = g.copy()
        for r in range(g.shape[0]):
            if (g[r] != 0).any(): o[r, (g[r] == 0)] = color
        return o
    return f, f"fillRows{color}", "masking"


def _fill_col_if_any(color):
    def f(g):
        o = g.copy()
        for c in range(g.shape[1]):
            if (g[:, c] != 0).any(): o[(g[:, c] == 0), c] = color
        return o
    return f, f"fillCols{color}", "masking"


def _rank_recolor(lo, hi):
    """most common colour -> lo, least common -> hi (an ordering operation)"""
    def f(g):
        v, c = np.unique(g[g != 0], return_counts=True)
        if len(v) < 2: return g.copy()
        o = g.copy(); o[g == int(v[np.argmax(c)])] = lo; o[g == int(v[np.argmin(c)])] = hi
        return o
    return f, f"rankRecolor{lo}{hi}", "ordering"


def _corners(color):
    def f(g):
        o = g.copy()
        for r in (0, g.shape[0] - 1):
            for c in (0, g.shape[1] - 1): o[r, c] = color
        return o
    return f, f"corners{color}", "masking"


def _diagonal(color):
    def f(g):
        o = g.copy()
        for i in range(min(g.shape)): o[i, i] = color
        return o
    return f, f"diagonal{color}", "masking"


def _isolate_touching():
    """keep only cells with a non-zero neighbour (connectivity)"""
    def f(g):
        H, W = g.shape; o = np.zeros_like(g)
        for r in range(H):
            for c in range(W):
                if g[r, c] == 0: continue
                nb = [g[r2, c2] for r2, c2 in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                      if 0 <= r2 < H and 0 <= c2 < W]
                if any(x != 0 for x in nb): o[r, c] = g[r, c]
        return o
    return f, "keepTouching", "connectivity"


def _isolate_lonely():
    def f(g):
        H, W = g.shape; o = np.zeros_like(g)
        for r in range(H):
            for c in range(W):
                if g[r, c] == 0: continue
                nb = [g[r2, c2] for r2, c2 in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                      if 0 <= r2 < H and 0 <= c2 < W]
                if all(x == 0 for x in nb): o[r, c] = g[r, c]
        return o
    return f, "keepLonely", "connectivity"


def _sort_rows_by_count():
    def f(g):
        order = np.argsort(-(g != 0).sum(1), kind="stable")
        return g[order].copy()
    return f, "sortRowsByCount", "ordering"


def _sort_cols_by_count():
    def f(g):
        order = np.argsort(-(g != 0).sum(0), kind="stable")
        return g[:, order].copy()
    return f, "sortColsByCount", "ordering"


def build_library_b_dup(seed=0):
    """library B plus six near-duplicates that differ from an existing operation only in the colour
    they paint: the pattern that makes recolourings and swaps confusable in library A. Prediction of the
    confusability account: joint reward training collapses here, where it does not on library B."""
    lib = build_library_b(seed)
    cols = {n: int(''.join(ch for ch in n if ch.isdigit()) or 0) for _, n, _ in lib}
    def other(c): return 1 + (c % 9)
    lib += [_count_to_row(other(cols["countRow2"] if "countRow2" in cols else 2)),
            _fill_row_if_any(other(cols.get("fillRows4", 4))), _fill_col_if_any(other(cols.get("fillCols6", 6))),
            _corners(other(cols.get("corners5", 5))), _diagonal(other(cols.get("diagonal1", 1))),
            _largest_row(other(cols.get("markDensestRow8", 8)))]
    seen, out = set(), []
    for f, n, fam in lib:
        if n not in seen: seen.add(n); out.append((f, n, fam))
    return out


def build_library_b(seed=0):
    rng = np.random.default_rng(seed)
    lib = [_keep_largest(), _keep_rarest(), _keep_symmetric(), _isolate_touching(), _isolate_lonely(),
           _sort_rows_by_count(), _sort_cols_by_count(), _symmetrise("h"), _symmetrise("v")]
    cols = list(rng.choice(np.arange(1, 10), 8, replace=False))
    lib += [_count_to_row(int(cols[0])), _largest_row(int(cols[1])), _largest_col(int(cols[2])),
            _fill_row_if_any(int(cols[3])), _fill_col_if_any(int(cols[4])),
            _corners(int(cols[5])), _diagonal(int(cols[6])),
            _rank_recolor(int(cols[7]), int(cols[0]))]
    seen, out = set(), []
    for f, n, fam in lib:
        if n not in seen:
            seen.add(n); out.append((f, n, fam))
    return out


if __name__ == "__main__":
    lib = build_library_b(0)
    print(f"library B: {len(lib)} primitives")
    for f, n, fam in lib: print(f"  {fam:13s} {n}")
    from .gen import build_library
    a = {n for _, n, _ in build_library(0)}
    b = {n for _, n, _ in lib}
    print("overlap with library A:", a & b)
