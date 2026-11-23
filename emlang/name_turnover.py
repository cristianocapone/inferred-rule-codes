"""Name turnover: how often does a single operation's modal message change between evaluations?

Reads the "dictionary" logged at every evaluation of synth_train1 (modal message per single operation) and
reports, per run: turnover between consecutive evaluations (fraction of the 30 operations whose modal message
changed), the step after which turnover stays below a threshold, and competence on unseen pairs.
Hypothesis under test: competent codes are differentiated AND stable; the changed-cell reward is differentiated
but unstable; a stability pressure restores competence."""
import argparse, glob, json, os, re
import numpy as np


def load(run_dir):
    ev = []
    for l in open(os.path.join(run_dir, "log.jsonl"), errors="ignore"):
        if '"kind": "eval"' not in l: continue
        try: e = json.loads(l)
        except Exception: continue
        d = e.get("synth_train1", {}).get("dictionary")
        if d: ev.append((e["step"], d, e.get("synth_hold2", {}).get("exact"), e.get("synth_train1", {}).get("msg_unique_frac")))
    return ev


def turnover(ev):
    out = []
    for (s0, d0, _, _), (s1, d1, _, _) in zip(ev, ev[1:]):
        keys = set(d0) & set(d1)
        out.append((s1, float(np.mean([d0[k] != d1[k] for k in keys])) if keys else float("nan")))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="emlang_runs/stab_*"); ap.add_argument("--thr", type=float, default=0.10)
    a = ap.parse_args()
    rows = []
    for rd in sorted(glob.glob(a.runs)):
        ev = load(rd)
        if len(ev) < 3: continue
        to = turnover(ev)
        late = [t for s, t in to if s > 16000]
        settle = next((s for i, (s, t) in enumerate(to) if all(tt < a.thr for _, tt in to[i:])), None)
        distinct = len({tuple(v) for v in ev[-1][1].values()}) / max(len(ev[-1][1]), 1)
        rows.append((os.path.basename(rd), ev[-1][0], float(np.mean(late)) if late else float("nan"), settle, distinct, ev[-1][2]))
    print(f"{'run':26s} {'step':>6s} {'turnover >16k':>13s} {'settles at':>11s} {'distinct/op':>11s} {'pairs exact':>11s}")
    for r in rows:
        print(f"{r[0]:26s} {r[1]:6d} {r[2]:13.3f} {str(r[3]):>11s} {r[4]:11.2f} {r[5] if r[5] is None else round(r[5], 3):>11}")


if __name__ == "__main__":
    main()
