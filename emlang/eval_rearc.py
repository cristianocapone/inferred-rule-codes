"""Evaluate a checkpoint on RE-ARC (real ARC-1 rule generators) with floor-corrected metrics.

Reports, for training rules and held-out rules:
  union            cell accuracy over non-PAD cells
  exact            whole-grid exact match
  exact_nontrivial exact match restricted to queries whose output differs from the input
                   (the identity floor on that subset is 0 by construction)
  identity_*       the same metrics for the trivial "copy the input" predictor
and, for discrete arms, the naming statistics over rules: message purity per rule and the
fraction of held-out rules whose message is closest to the message of a training rule.

Usage: EMLANG_CANVAS=16 REARC_DIR=... python emlang/eval_rearc.py --ckpt RUN/latest.pt --n 400
"""
import argparse, json, os, sys
from collections import Counter
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emlang.rearc_data import REARC
from emlang.data import collate, pad_grid
from emlang.model import Speaker, Executor, union_accuracy
from emlang.gen import PAD


def load(ckpt, dev):
    s = torch.load(ckpt, map_location=dev)
    a = s["args"]
    sp = Speaker(a["d"], a["V"], a["L"], a["arm"], tau=a["tau"], cont_dim=a.get("cont_dim", 0),
                 use_eos=a.get("len_cost", 0) > 0).to(dev)
    sp.min_len = a.get("min_len", 1)
    ex = Executor(a["d"], a["V"], a["L"], a["arm"], R=a["R"], use_ckpt=False,
                  step_emb=not a.get("no_step_emb", False), cont_dim=a.get("cont_dim", 0), ex_d=a.get("ex_d", 0)).to(dev)
    sp.load_state_dict(s["speaker"], strict=False); ex.load_state_dict(s["executor"], strict=False)
    sp.eval(); ex.eval()
    return sp, ex, a, s["step"]


@torch.no_grad()
def run(sp, ex, eps, dev, bs=32):
    out = dict(union=[], exact=[], nontrivial=[], structured=[], id_union=[], id_exact=[], msgs=[], rule=[])
    for i in range(0, len(eps), bs):
        sub = eps[i:i + bs]
        b = {k: v.to(dev) for k, v in collate(sub).items()}
        msg = sp(b["din"], b["dout"], b["dmask"], greedy=True)
        pred = ex(b["qin"], msg).argmax(1)
        out["union"] += union_accuracy(pred, b["qout"]).tolist()
        out["exact"] += (pred == b["qout"]).flatten(1).all(1).float().tolist()
        out["id_union"] += union_accuracy(b["qin"], b["qout"]).tolist()
        out["id_exact"] += (b["qin"] == b["qout"]).flatten(1).all(1).float().tolist()
        out["nontrivial"] += (~(b["qin"] == b["qout"]).flatten(1).all(1)).float().tolist()
        # structured = non-trivial AND the target is not a single uniform colour (a uniform fill or an
        # empty grid is solvable without representing the rule's spatial content)
        for e_ in sub:
            q = e_["query"][1]
            out["structured"].append(float((q != e_["query"][0]).any() if q.shape == e_["query"][0].shape else True) *
                                     float(len(set(q.flatten().tolist())) > 1))
        if msg["tokens"] is not None:
            out["msgs"] += msg["tokens"].cpu().numpy().tolist()
        out["rule"] += [e["tid"] for e in sub]
    return out


def summarize(o):
    nt = np.array(o["nontrivial"]).astype(bool)
    ex = np.array(o["exact"]); idex = np.array(o["id_exact"])
    d = dict(n=len(ex), union=float(np.mean(o["union"])), exact=float(ex.mean()),
             identity_union=float(np.mean(o["id_union"])), identity_exact=float(idex.mean()),
             nontrivial_frac=float(nt.mean()))
    d["exact_nontrivial"] = float(ex[nt].mean()) if nt.any() else float("nan")
    d["identity_exact_nontrivial"] = 0.0
    st = np.array(o["structured"]).astype(bool)
    d["structured_frac"] = float(st.mean())
    d["exact_structured"] = float(ex[st].mean()) if st.any() else float("nan")
    d["n_exact_structured"] = int((ex.astype(bool) & st).sum())
    d["n_structured"] = int(st.sum())
    return d


def per_rule(o):
    """per ARC-1 rule: modal message, how many rules share it, solve rate — for the LARC comparison"""
    from collections import Counter, defaultdict
    by = defaultdict(list); ex = defaultdict(list); un = defaultdict(list)
    for i, r in enumerate(o["rule"]):
        key = r.replace("rearc_", "")
        if o["msgs"]: by[key].append(tuple(o["msgs"][i]))
        ex[key].append(o["exact"][i]); un[key].append(o["union"][i])
    modal = {k: list(Counter(v).most_common(1)[0][0]) for k, v in by.items()} if by else {}
    share = Counter(tuple(m) for m in modal.values())
    out = {}
    for k in ex:
        m = modal.get(k)
        out[k] = dict(exact=float(np.mean(ex[k])), union=float(np.mean(un[k])), n=len(ex[k]),
                      msg=m, cluster_size=(share[tuple(m)] if m else None))
    return out


def naming(o_train, o_hold):
    """message purity per rule, and nearest-training-rule message distance for held-out rules"""
    if not o_train["msgs"]:
        return {}
    def by_rule(o):
        d = {}
        for r, m in zip(o["rule"], o["msgs"]):
            d.setdefault(r, []).append(tuple(m))
        return d
    tr, ho = by_rule(o_train), by_rule(o_hold)
    pur = [Counter(v).most_common(1)[0][1] / len(v) for v in tr.values() if len(v) >= 2]
    pur_h = [Counter(v).most_common(1)[0][1] / len(v) for v in ho.values() if len(v) >= 2]
    modal_tr = {r: np.array(Counter(v).most_common(1)[0][0]) for r, v in tr.items()}
    M = np.stack(list(modal_tr.values())) if modal_tr else None
    dists = []
    for r, v in ho.items():
        m = np.array(Counter(v).most_common(1)[0][0])
        if M is not None:
            dists.append(float((M != m[None]).mean(1).min()))
    def nmi(o):
        pairs = list(zip(o["rule"], (tuple(m) for m in o["msgs"])))
        if not pairs: return float("nan")
        def H(xs):
            _, c = np.unique(np.array([str(x) for x in xs]), return_counts=True); p = c / c.sum()
            return float(-(p * np.log2(p)).sum())
        hr = H([r for r, _ in pairs])
        return (hr + H([m for _, m in pairs]) - H(pairs)) / hr if hr > 0 else float("nan")
    return dict(nmi_train_rules=nmi(o_train), nmi_heldout_rules=nmi(o_hold),
                purity_train_rules=float(np.mean(pur)) if pur else float("nan"),
                purity_heldout_rules=float(np.mean(pur_h)) if pur_h else float("nan"),
                n_train_rules=len(tr), n_heldout_rules=len(ho),
                distinct_train_messages=len({tuple(v) for v in modal_tr.values()}),
                heldout_min_dist_to_train=float(np.mean(dists)) if dists else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--diff", default="0,0.15"); ap.add_argument("--device", default=None)
    ap.add_argument("--rules", type=int, default=0, help="restrict the training split to the first N rule families "
                    "- must match the run's --rearc_rules, otherwise naming statistics are computed over rules "
                    "the model never saw")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sp, ex, args, step = load(a.ckpt, dev)
    lo, hi = (float(x) for x in a.diff.split(","))
    R = REARC(seed=0, diff=(lo, hi), n_train_rules=a.rules or None)
    if a.rules:
        print(f"[scope] naming statistics over the {a.rules} training rule families this run actually saw", flush=True)
    rep = dict(ckpt=a.ckpt, step=step, arm=args["arm"], diff=[lo, hi], rules=a.rules or 319)
    outs = {}
    for split, n in (("train", a.n), ("hold", max(a.n, 4 * R.split_sizes["hold"]))):
        eps = R.fixed_eval_set(split, n, seed=11 if split == "train" else 12)
        outs[split] = run(sp, ex, eps, dev)
        rep[split] = summarize(outs[split])
        s = rep[split]
        print(f"[{split}] n={s['n']} union {s['union']:.3f} (identity {s['identity_union']:.3f}) | "
              f"exact {s['exact']:.3f} (identity {s['identity_exact']:.3f}) | "
              f"EXACT ON NON-TRIVIAL {s['exact_nontrivial']:.3f} ({s['nontrivial_frac']:.0%} of queries) | "
              f"EXACT ON STRUCTURED {s['exact_structured']:.3f} = {s['n_exact_structured']}/{s['n_structured']} "
              f"(non-trivial AND target not a uniform colour)", flush=True)
    rep["per_rule"] = {sp: per_rule(o) for sp, o in outs.items()}
    rep["naming"] = naming(outs["train"], outs["hold"])
    if rep["naming"]:
        n = rep["naming"]
        print(f"[naming] message purity: train rules {n['purity_train_rules']:.2f}, held-out rules {n['purity_heldout_rules']:.2f} | "
              f"{n['distinct_train_messages']} distinct messages for {n['n_train_rules']} training rules | "
              f"held-out rule message distance to nearest training rule {n['heldout_min_dist_to_train']:.2f}", flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
        # per-episode (rule, message) pairs, for decoding analyses that must generalise across episodes
        json.dump({sp: dict(rule=o["rule"], msgs=o["msgs"]) for sp, o in outs.items()},
                  open(a.out.replace(".json", "_msgs.json"), "w"))


if __name__ == "__main__":
    main()
