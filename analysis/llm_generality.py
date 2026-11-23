"""A pilot: does the same mechanism appear when sender and executor are a pretrained model? (Supplement, Section S9)

Two copies of Qwen2.5-0.5B-Instruct with LoRA adapters play the same game over the same alphabet of 32 symbols
(emlang/llm_game.py, 5 000 steps). Conditions: joint experience, staged experience, and joint experience with the
information term. The reward here is token accuracy under teacher forcing, a proxy for the reward of the main
experiments. These runs never augment their episodes, so a held-out program is new to them when no training program
computes it: the 'direct' level of the corrected audit (audit_20260918, emlang/audit_group.py).

Everything is recomputed from the per-episode records of the evaluation (llm_runs/<run>/evP1.json), so the programs
counted as new follow the current audit and the code is described by the same measures as the main experiments:
names (the distinct most frequent messages over rules), purity, and how often the rule is decoded from the message
on held-out episodes. Accuracy is measured on the new triples; the code is described over all 300 held-out triples.
The reference is the constant-message control: the game's other-episode control rotates messages inside a batch
whose episodes are grouped by program, so it often returns a message of the same rule.

Runs are counted explicitly: started, completed (5 000 steps), non-finite (the fp16 adapters became non-finite late
in training; the game rolls back to its last snapshot, and these runs then degenerate), collapsed (exact match within
one point of the constant-message control), averaged. A run with a non-finite event is evaluated at its last saved
snapshot before the first event (llm_runs/<run>/evP1_snapshot.txt names it; a rule fixed before looking), every other
run at its final adapters. Averages use all completed, evaluated runs, and that count is printed in the table.

For reference, the script also reads the evaluations of an executor of the same model trained to read the true rule
in English (llm_runs/llm_oracle_s*/evP1_oracle.json, from the companion study, evaluated with this pilot's own procedure on
the same programs and query episodes): what the reader can execute when given an informative description.
Writes paper/supp_tables/llm_generality.tex and emlang_runs/analysis/stats/llm_generality.json.
"""

from paths import ROOT, DATA, PAPER, FIGURES, TABLES, RESULTS
import glob, json, os, re, sys
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from emlang.decode_naming import decode

RUNS = os.path.join(DATA, "llm_runs")
OUT = os.path.join(RESULTS)
AUDIT = os.path.join(DATA, "audits", "20260918", "group_audit_A_len3.json")
STEPS = 5000
NOVEL_PAIRS = set()
CONDITIONS = [("joint", r"llm_joint_s\d"), ("staged", r"llm_staged_s\d"), ("joint + information term", r"llm_joint_div_s\d")]


def last_train_eval(run):
    f = os.path.join(RUNS, run, "log.jsonl")
    if not os.path.exists(f): return None
    ev = [json.loads(l) for l in open(f, errors="ignore") if '"kind": "eval"' in l]
    return ev[-1] if ev else None


def code_stats(progs, msgs):
    """names, purity and held-out decoding of the rule from the message, as in the main experiments"""
    by = {}
    for p, m in zip(progs, msgs): by.setdefault(p, []).append(m)
    modal, pur = [], []
    for ms in by.values():
        m, k = Counter(ms).most_common(1)[0]; modal.append(m); pur.append(k / len(ms))
    order = {p: i for i, p in enumerate(sorted(by))}
    width = max(len(m) for m in msgs)
    toks = [[ord(c) for c in m] + [0] * (width - len(m)) for m in msgs]
    d = decode([order[p] for p in progs], toks, n_perm=50)
    return dict(rules=len(by), distinct_messages=len(set(msgs)), names=len(set(modal)),
                names_per_rule=len(set(modal)) / len(by), purity=float(np.mean(pur)),
                decode_lookup=d["lookup"], decode_nearest=d["nearest"], decode_chance=d["chance"],
                decode_null_p95=d["null_p95"])


def load(run, novel):
    ev = last_train_eval(run)
    if ev is None: return None
    tr = ev.get("synth_train1", {})
    row = dict(run=run, step=ev.get("step"), completed=ev.get("step") == STEPS, diverged=not tr.get("acc"))
    p = os.path.join(RUNS, run, "evP1.json")
    row["evaluated"] = os.path.exists(p)
    snap = os.path.join(RUNS, run, "evP1_snapshot.txt")
    row["snapshot"] = open(snap).read().strip() if os.path.exists(snap) else "final"
    row["nonfinite"] = row["snapshot"] != "final"
    if row["evaluated"]:
        ev1 = json.load(open(p))
        pe2 = ev1["audit_hold2_all"]["per_episode"]; k2 = [i for i, q in enumerate(pe2["prog"]) if tuple(q) in NOVEL_PAIRS]
        row.update(pairs_exact=float(np.mean([pe2["exact"][i] for i in k2])), pairs_const=float(np.mean([pe2["exact_const_msg"][i] for i in k2])))
        pe = ev1["audit_len3_all"]["per_episode"]
        prog = [tuple(q) for q in pe["prog"]]
        keep = [i for i, q in enumerate(prog) if q in novel]
        row.update(n_episodes=len(keep), n_programs=len({prog[i] for i in keep}),
                   exact=float(np.mean([pe["exact"][i] for i in keep])),
                   exact_const=float(np.mean([pe["exact_const_msg"][i] for i in keep])))
        row.update(code_stats(prog, pe["message"]))
    row["collapsed"] = bool(row["evaluated"] and row["exact"] <= row["exact_const"] + 0.01)
    return row


def ms(v, scale=100, d=1):
    v = [x for x in v if x is not None]
    if not v: return "--"
    if len(v) == 1: return f"{scale * v[0]:.{d}f}"
    return f"{scale * np.mean(v):.{d}f} $\\pm$ {scale * np.std(v, ddof=1) / np.sqrt(len(v)):.{d}f}"


def main():
    global NOVEL_PAIRS
    novel = {tuple(p) for p in json.load(open(AUDIT))["novel_direct"]}
    NOVEL_PAIRS = {tuple(p) for p in json.load(open(AUDIT.replace("_len3", "_hold2")))["novel_direct"]}
    oracle = []   # the English-oracle executors, evaluated with this pilot's procedure: same programs, same query episodes
    for f in sorted(glob.glob(os.path.join(RUNS, "llm_oracle_s*", "evP1_oracle.json"))):
        ev = json.load(open(f)); rec = dict(run=os.path.basename(os.path.dirname(f)))
        for key, tag, keep in (("audit_hold2_all", "pairs", NOVEL_PAIRS), ("audit_len3_all", "triples", novel)):
            pe = ev[key]["per_episode"]; k = [i for i, q in enumerate(pe["prog"]) if tuple(q) in keep]
            rec[f"{tag}_exact"] = float(np.mean([pe["exact"][i] for i in k])); rec[f"{tag}_n"] = len(k)
        oracle.append(rec)
    rows, by = [], {}
    for lab, pat in CONDITIONS:
        for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
            run = os.path.basename(d)
            if not re.fullmatch(pat, run): continue
            r = load(run, novel)
            if r is None: continue
            r["condition"] = lab; rows.append(r); by.setdefault(lab, []).append(r)
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(novel_programs=len(novel), novel_pairs=len(NOVEL_PAIRS), audit=os.path.basename(AUDIT), runs=rows, oracle_reader=oracle),
              open(os.path.join(OUT, "llm_generality.json"), "w"), indent=1)
    lines = [r"\begin{tabular}{lccccccccc}\toprule",
             r" & \multicolumn{4}{c}{runs} & \multicolumn{2}{c}{new triples (\%)} & \multicolumn{3}{c}{the code} \\"
             r" \cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}",
             r"condition & started & finished & non-finite & averaged & own message & constant message"
             r" & names & purity & rule decoded (\%) \\ \midrule"]
    for lab, rs in by.items():
        use = [r for r in rs if r["completed"] and r["evaluated"]]
        lines.append(f"{lab} & {len(rs)} & {sum(r['completed'] for r in rs)} & {sum(r['nonfinite'] for r in rs)} & {len(use)} & "
                     f"{ms([r.get('exact') for r in use])} & {ms([r.get('exact_const') for r in use])} & "
                     f"{ms([r.get('names') for r in use], scale=1, d=0)} & {ms([r.get('purity') for r in use], scale=1, d=2)} & "
                     f"{ms([r.get('decode_nearest') for r in use])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABLES, "llm_generality.tex"), "w").write("\n".join(lines) + "\n")
    print(f"new triples for these runs: {len(novel)} of 300")
    hdr = f"{'run':20s} {'condition':24s} {'step':>5s} {'div':>5s} {'coll':>5s} {'exact':>6s} {'const':>6s} {'names':>6s} {'purity':>7s} {'decode':>7s} {'null':>6s}"
    print(hdr)
    for r in rows:
        f = lambda k, s=100, w=6, d=1: " " * (w - 2) + "--" if r.get(k) is None else f"{s * r[k]:{w}.{d}f}"
        print(f"{r['run']:20s} {r['condition']:24s} {r['snapshot']:>9s} {str(r['diverged']):>5s} {str(r['collapsed']):>5s} "
              f"{f('exact')} {f('exact_const')} {f('names', 1, 6, 0)} {f('purity', 1, 7, 2)} {f('decode_nearest')} {f('decode_null_p95')}  pairs {f('pairs_exact')}/{f('pairs_const')}")
    if oracle: print("executor given the true rule in English, same unseen pairs / triples:",
                     [(round(100 * o["pairs_exact"], 1), round(100 * o["triples_exact"], 1)) for o in oracle])


if __name__ == "__main__":
    main()
