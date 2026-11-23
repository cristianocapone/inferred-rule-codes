"""How much does a speaker's message say about the rule? Names, pooling and held-out decoding per run.

Counting distinct messages confounds two things: names that differ between rules and noise within a rule. A code
that pools half its operations under one name but varies its messages from episode to episode has many distinct
messages and little information. This script separates them, on episodes built from the run's own generator:

  singles       every single operation, --eps_single episodes each
  novel pairs   the functionally novel held-out pairs of the audit, --eps_pair episodes each
For each set: distinct messages, distinct modal messages (names), the largest group of rules sharing one name,
purity (fraction of a rule's episodes that receive its modal message), and held-out decoding of the rule from the
message (emlang/decode_naming.py: lookup and nearest neighbour, with a permutation null)."""
import argparse, json, os, sys
from collections import Counter
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emlang.gen import SynthGen
from emlang.data import collate
from emlang.model import Speaker
from emlang.decode_naming import decode


@torch.no_grad()
def messages(sp, eps, dev, bs=64):
    out = []
    for i in range(0, len(eps), bs):
        b = {k: v.to(dev) for k, v in collate(eps[i:i + bs]).items()}
        out += sp(b["din"], b["dout"], b["dmask"], greedy=True)["tokens"].cpu().tolist()
    return out


def summarise(labels, msgs, n_rules, n_perm):
    by = {}
    for l, m in zip(labels, msgs): by.setdefault(l, []).append(tuple(m))
    modal, pur = [], []
    for l, ms in by.items():
        m, k = Counter(ms).most_common(1)[0]; modal.append(m); pur.append(k / len(ms))
    d = decode(labels, msgs, n_perm=n_perm)
    return dict(n_rules=n_rules, n_episodes=len(msgs), distinct_msgs=len(set(map(tuple, msgs))), names=len(set(modal)),
                names_per_rule=len(set(modal)) / n_rules, largest_group=max(Counter(modal).values()), purity=float(np.mean(pur)),
                decode_lookup=d["lookup"], decode_nearest=d["nearest"], decode_chance=d["chance"], decode_null_p95=d["null_p95"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--audit", required=True)
    ap.add_argument("--eps_single", type=int, default=20); ap.add_argument("--eps_pair", type=int, default=10)
    ap.add_argument("--n_perm", type=int, default=50); ap.add_argument("--device", default="cpu"); ap.add_argument("--out", required=True)
    ap.add_argument("--group_triples", default=None, help="audit_20260917/group_audit_<lib>_len3.json: also the triples not covered by the augmented training set")
    ap.add_argument("--eps_triple", type=int, default=10)
    a = ap.parse_args()
    s = torch.load(a.ckpt, map_location="cpu", weights_only=False); args = s["args"]
    sp = Speaker(args["d"], args["V"], args["L"], args["arm"], tau=args["tau"], cont_dim=args.get("cont_dim", 0),
                 use_eos=args.get("len_cost", 0) > 0).to(a.device)
    sp.load_state_dict(s["speaker"], strict=False); sp.eval()
    if args["arm"] == "cont":
        print("continuous channel: no symbols"); return
    G = SynthGen(0, size_lo=args.get("size_lo", 4), size_hi=args.get("size_hi", 12), n_train3=args.get("n_train3", 0),
                 library=args.get("library", "A"), n_prims=args.get("n_prims", 0), prim_seed=args.get("prim_seed"),
                 drop_prims=args.get("drop_prims", ""))
    rng = np.random.default_rng(5)
    rep = dict(ckpt=a.ckpt, arm=args["arm"], step=s.get("step"))
    lab, eps = [], []
    for k, p in enumerate(G.train1):
        for _ in range(a.eps_single):
            eps.append(G.make_episode(rng, "hold2", n_demos=3, prog=list(p))); lab.append(k)
    rep["singles"] = summarise(lab, messages(sp, eps, a.device), len(G.train1), a.n_perm)
    novel = json.load(open(a.audit))["novel_progs"]
    lab, eps = [], []
    for k, p in enumerate(novel):
        for _ in range(a.eps_pair):
            eps.append(G.make_episode(rng, "hold2", n_demos=3, prog=list(p))); lab.append(k)
    rep["novel_pairs"] = summarise(lab, messages(sp, eps, a.device), len(novel), a.n_perm)
    if a.group_triples:
        tri = json.load(open(a.group_triples))["novel_group"]
        lab, eps = [], []
        for k, p in enumerate(tri):
            for _ in range(a.eps_triple):
                eps.append(G.make_episode(rng, "hold2", n_demos=3, prog=list(p))); lab.append(k)
        rep["triples_audited"] = summarise(lab, messages(sp, eps, a.device), len(tri), a.n_perm)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(rep, open(a.out, "w"), indent=1)
    f = lambda d: f"names {d['names']}/{d['n_rules']} largest {d['largest_group']} purity {d['purity']:.2f} decode {d['decode_lookup']:.2f}/{d['decode_nearest']:.2f}"
    print(f"[code-info] {os.path.basename(os.path.dirname(a.ckpt))}: singles {f(rep['singles'])} | novel pairs {f(rep['novel_pairs'])}"
          + (f" | audited triples {f(rep['triples_audited'])}" if "triples_audited" in rep else ""), flush=True)


if __name__ == "__main__":
    main()
