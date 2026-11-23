"""Numbers from the audit of the compositional split and of the naming estimator (audit_20260916/AUDIT.md).

novel(pattern, key): exact match of every run matching `pattern`, evaluated on the held-out programs whose
    transformation is not computed by any training program (key "novel" for pairs, "len3_novel" for triples),
    from emlang_runs/analysis/clean2/<run>.json (emlang/eval_clean.py).
decoding(): held-out decoding of the rule family from the message, per naming run
    (emlang/decode_naming.py), from emlang_runs/analysis/decode/decoding.json."""

from paths import ROOT, DATA, PAPER, FIGURES, TABLES, RESULTS
import glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(DATA, "analysis", "clean3")   # the per-program re-evaluation; reproduces clean2 exactly and also covers the later runs
DECODE = os.path.join(DATA, "analysis", "decode", "decoding.json")


def novel(pat, key="novel"):
    out = []
    for f in sorted(glob.glob(os.path.join(CLEAN, "*.json"))):
        if re.fullmatch(pat, os.path.basename(f)[:-5]):
            d = json.load(open(f))
            if d.get(key): out.append(d[key]["exact"])
    return out


def decoding():
    return json.load(open(DECODE))


CLEAN4 = os.path.join(DATA, "analysis", "clean4")
AUDIT = os.path.join(DATA, "audits", "20260918")     # the corrected audit; its novel sets are subsets of the earlier one
_cache = {}


def library_of(run):
    """which operation library a run was trained on, from its name"""
    for pre, tag in (("A17r_", "A17r"), ("A17_", "A17"), ("A24_", "A24"), ("Aclean_", "Aclean"),
                     ("Bdup_", "Bdup"), ("libB_", "B")):
        if run.startswith(pre): return tag
    return "A"


def novel_programs(run, key):
    """the held-out programs of this run's library that no training program computes, even through the dihedral
    and colour augmentation of training (audit_20260918, emlang/audit_group.py)"""
    tag, split = library_of(run), ("len3" if key.startswith("triples") else "hold2")
    if (tag, split) not in _cache:
        f = os.path.join(AUDIT, f"group_audit_{tag}_{split}.json")
        _cache[(tag, split)] = {tuple(p) for p in json.load(open(f))["novel_group"]}
    return _cache[(tag, split)]


def novel_mask(run, key, programs):
    import numpy as np
    keep = novel_programs(run, key)
    m = np.array([tuple(p) in keep for p in programs])
    assert m.sum(), f"{run} {key}: no audited program left"
    return m


def audited(pat, key="triples_audited"):
    """exact match of every run matching `pat` on the programs of novel_programs(): 'triples_audited'
    (ten episodes per triple) or 'pairs_audited' (twenty per pair). The stored per-program counts
    (emlang_runs/analysis/clean4/<run>.json) are filtered to the corrected audit here, so no run has to be
    evaluated again when the audit changes."""
    import numpy as np
    out = []
    for f in sorted(glob.glob(os.path.join(CLEAN4, "*.json"))):
        run = os.path.basename(f)[:-5]
        if not re.fullmatch(pat, run): continue
        d = json.load(open(f)).get(key)
        if not d: continue
        m = novel_mask(run, key, d["programs"])
        out.append(float(np.array(d["exact_count"])[m].sum() / (m.sum() * d["per_prog"])))
    return out
