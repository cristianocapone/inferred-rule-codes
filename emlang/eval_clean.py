"""Recompute a run's accuracy on the audited subsets of the held-out split.

Three nested test sets, from the audit (emlang/audit_split.py):
  all      every held-out program, as the manuscript reports it
  novel    programs whose transformation is not computed by any training program
  strict   novel, and not occurring as a consecutive sub-sequence of a training triple

The episodes are built exactly as in training and evaluation (same generator, same seed stream), with
--per_prog episodes for each program so that the three means are estimated on comparable evidence.
With --per_program the exact-match count of every program is stored as well (paper/stats.py resamples programs).
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emlang.gen import SynthGen
from emlang.data import collate
from emlang.model import union_accuracy, exact_match, msg_from_tokens
from emlang.probe import load


@torch.no_grad()
def run(sp, ex, args, G, progs, dev, per_prog=5, seed=12, bs=16, use_eos=False, codebook=None):
    rng = np.random.default_rng(seed)
    eps = [G.make_episode(rng, "hold2", n_demos=3, prog=list(p)) for p in progs for _ in range(per_prog)]
    ex_ok, acc = [], []
    for i in range(0, len(eps), bs):
        sub = eps[i:i + bs]; b = {k: v.to(dev) for k, v in collate(sub).items()}
        msg = sp(b["din"], b["dout"], b["dmask"], greedy=True)
        pred = ex(b["qin"], msg).argmax(1)
        ex_ok += exact_match(pred, b["qout"]).tolist(); acc += union_accuracy(pred, b["qout"]).tolist()
    per = np.array(ex_ok, dtype=int).reshape(len(progs), per_prog).sum(1)
    return float(np.mean(ex_ok)), float(np.mean(acc)), len(eps), per.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--audit", required=True); ap.add_argument("--audit_len3", default=None)
    ap.add_argument("--per_prog", type=int, default=5); ap.add_argument("--out", default=None); ap.add_argument("--device", default=None)
    ap.add_argument("--per_program", action="store_true", help="also store exact-match counts per program (for the statistics)")
    ap.add_argument("--group_pairs", default=None, help="audit_20260917/group_audit_<lib>_hold2.json: pairs not covered by the augmented training set")
    ap.add_argument("--group_triples", default=None, help="audit_20260917/group_audit_<lib>_len3.json: triples not covered by the augmented training set")
    ap.add_argument("--per_prog_pairs", type=int, default=20); ap.add_argument("--per_prog_triples", type=int, default=10)
    ap.add_argument("--only_audited", action="store_true", help="evaluate only the sets of the group audit")
    a = ap.parse_args(); dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sp, ex, args, step = load(a.ckpt, dev)
    G = SynthGen(0, size_lo=3, size_hi=6, n_train3=args.get("n_train3", 600),
                 library=args.get("library", "A"), n_prims=args.get("n_prims", 0), prim_seed=args.get("prim_seed"),
                 drop_prims=args.get("drop_prims", ""))
    au = json.load(open(a.audit))
    sets = dict(all=[r["prog"] for r in au["rows"]], novel=au["novel_progs"], strict=au["clean_progs"])
    if a.audit_len3:
        au3 = json.load(open(a.audit_len3))
        sets.update(len3_all=[r["prog"] for r in au3["rows"]], len3_novel=au3["novel_progs"])
    if a.only_audited: sets = {}
    per_set = {}
    if a.group_pairs:
        sets["pairs_audited"] = json.load(open(a.group_pairs))["novel_group"]; per_set["pairs_audited"] = a.per_prog_pairs
    if a.group_triples:
        sets["triples_audited"] = json.load(open(a.group_triples))["novel_group"]; per_set["triples_audited"] = a.per_prog_triples
    rep = dict(ckpt=a.ckpt, step=step, arm=args["arm"], per_prog=a.per_prog)
    line = []
    for k, progs in sets.items():
        if not progs: rep[k] = None; continue
        pp = per_set.get(k, a.per_prog if not k.startswith("len3") else max(2, a.per_prog // 2))
        e, c, n, per = run(sp, ex, args, G, progs, dev, pp)
        rep[k] = dict(exact=e, acc=c, n_programs=len(progs), n_episodes=n); line.append(f"{k} {e:.3f} ({len(progs)} progs)")
        if a.per_program: rep[k].update(per_prog=pp, programs=[list(p) for p in progs], exact_count=per)
    print(f"[clean-eval] {os.path.basename(os.path.dirname(a.ckpt))} step {step}: " + " | ".join(line), flush=True)
    if a.out: os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
