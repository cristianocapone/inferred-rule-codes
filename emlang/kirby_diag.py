"""Our collapse diagnostics applied to human iterated-learning chains (Kirby, Cornish & Smith 2008;
Cornish 2011). Each chain: 27 meanings (motion x shape x colour) named by a participant, transmitted
across 10 generations. Experiment I: plain transmission. Experiment II: homonyms filtered out of the
training set (an expressivity pressure). For each generation we compute exactly what we log for the
speaker: distinct signals per meaning, character entropy, mean length, and topographic similarity.
"""
import os, re, sys, zipfile, collections, math
import xml.etree.ElementTree as ET
import numpy as np

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "p": "http://schemas.openxmlformats.org/package/2006/relationships"}


def load(path):
    z = zipfile.ZipFile(path)
    ss = ["".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    out = {}
    for s in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", NS):
        target = rels[s.get("{%s}id" % NS["r"])]; target = target if target.startswith("xl/") else "xl/" + target.lstrip("/")
        rows = []
        for r in ET.fromstring(z.read(target)).iter("{%s}row" % NS["m"]):
            cells = {}
            for c in r.findall("m:c", NS):
                col = re.match(r"([A-Z]+)", c.get("r")).group(1); v = c.find("m:v", NS)
                cells[col] = None if v is None else (ss[int(v.text)] if c.get("t") == "s" else v.text)
            rows.append(cells)
        out[s.get("name")] = rows
    return out


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def spearman(x, y):
    def rk(v):
        o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
    x, y = rk(np.asarray(x, float)), rk(np.asarray(y, float))
    return float(np.corrcoef(x, y)[0, 1])


def chain_stats(rows):
    hdr = next(r for r in rows if r.get("B") == "motion")
    gens = [k for k in sorted(hdr) if k not in "ABCD" and hdr[k] is not None and hdr[k].strip().isdigit()]
    items = [r for r in rows if r.get("B") not in (None, "motion") and r.get("C") and r.get("D")]
    meanings = [(r["B"].strip(), r["C"].strip(), r["D"].strip()) for r in items]
    md = [[sum(a != b for a, b in zip(meanings[i], meanings[j])) for j in range(len(meanings)) if j > i] for i in range(len(meanings))]
    md = [x for row in md for x in row]
    res = []
    for g in gens:
        sig = [(r.get(g) or "").strip().lower() for r in items]
        if sum(1 for s in sig if s) < len(sig) * 0.8: continue
        n = len(sig); distinct = len(set(sig))
        chars = collections.Counter("".join(sig)); p = np.array(list(chars.values()), float); p /= p.sum()
        H = float(-(p * np.log2(p)).sum())
        sd = [lev(sig[i], sig[j]) for i in range(n) for j in range(n) if j > i]
        res.append(dict(gen=int(hdr[g]), distinct_per_meaning=distinct / n, char_entropy=H,
                        mean_len=float(np.mean([len(s) for s in sig])), topsim=spearman(md, sd)))
    return res


if __name__ == "__main__":
    book = load(sys.argv[1])
    readme = book.get("README", [])
    print("README:"); [print("  ", " | ".join(str(v)[:150] for v in r.values() if v)) for r in readme[:40]]
    by_exp = collections.defaultdict(list)
    for name, rows in book.items():
        m = re.match(r"^(I{1,3}V?|IV)-([A-D])", name)
        if not m: continue
        st = chain_stats(rows)
        if st: by_exp[m.group(1)].append((m.group(2), st))
    print(f"\n{'exp':4s} {'chain':5s} " + " ".join(f"g{g:<5d}" for g in range(10)))
    for exp in sorted(by_exp, key=lambda e: len(e)):
        for ch, st in by_exp[exp]:
            print(f"{exp:4s} {ch:5s} " + " ".join(f"{s['distinct_per_meaning']:.2f}  " for s in st) + "  <- distinct signals per meaning")
        print(f"{exp:4s} mean  " + " ".join(f"{np.mean([st[i]['distinct_per_meaning'] for _, st in by_exp[exp] if i < len(st)]):.2f}  " for i in range(10)) + "  <- mean")
        print(f"{exp:4s} topsm " + " ".join(f"{np.mean([st[i]['topsim'] for _, st in by_exp[exp] if i < len(st)]):.2f}  " for i in range(10)) + "  <- topsim")
        print(f"{exp:4s} Hchar " + " ".join(f"{np.mean([st[i]['char_entropy'] for _, st in by_exp[exp] if i < len(st)]):.2f}  " for i in range(10)) + "  <- char entropy")
