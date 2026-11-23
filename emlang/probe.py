"""Post-hoc probes of the emergent code from a checkpoint.
  1. dictionary : modal message per singleton primitive (train1) - is naming consistent?
  2. lesion     : replace symbol at position j by a random other symbol; accuracy drop per
                  position on hold2 (which positions carry meaning; how localised is it?); and replace the
                  whole message by uniformly random symbols, or by the message of another episode (shuffled)
  3. swap       : for hold2 programs (A,B), feed the query of (A,B) with the message of a
                  training pair (A,C) sharing the first primitive - does the output reflect A?
                  (compositional reuse: partial credit vs the intermediate grid after A)
Usage: python emlang/probe.py --ckpt emlang_runs/rl_s0/latest.pt [--n 200]"""
import argparse, json, os, sys
from collections import Counter
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emlang.gen import SynthGen, PAD
from emlang.data import collate, pad_grid
from emlang.model import Speaker, Executor, union_accuracy, exact_match, msg_from_tokens


USE_EOS = False
CODEBOOK = None


def load(ckpt, dev):
    s = torch.load(ckpt, map_location=dev)
    a = s["args"]
    sp = Speaker(a["d"], a["V"], a["L"], a["arm"], tau=a["tau"], cont_dim=a.get("cont_dim", 0), use_eos=a.get("len_cost", 0) > 0).to(dev); sp.min_len = a.get("min_len", 1)
    ex = Executor(a["d"], a["V"], a["L"], a["arm"], R=a["R"], use_ckpt=False, step_emb=not a.get("no_step_emb", False), cont_dim=a.get("cont_dim", 0), ex_d=a.get("ex_d", 0)).to(dev)
    sp.load_state_dict(s["speaker"], strict=False); ex.load_state_dict(s["executor"], strict=False); sp.eval(); ex.eval()
    return sp, ex, a, s["step"]


@torch.no_grad()
def messages(sp, eps, dev, bs=16):
    out = []
    for i in range(0, len(eps), bs):
        b = {k: v.to(dev) for k, v in collate(eps[i:i + bs]).items()}
        out.append(sp(b["din"], b["dout"], b["dmask"], greedy=True)["tokens"].cpu())
    return torch.cat(out)


@torch.no_grad()
def changed_cell_acc(pred, qin, tgt):
    """accuracy restricted to cells that the transformation CHANGES (target != input, incl. shape)"""
    ch = (tgt != qin)
    c = ((pred == tgt) & ch).flatten(1).sum(1).float()
    return c / ch.flatten(1).sum(1).clamp(min=1).float()


def execute(ex, eps, toks, dev, bs=16, target="query"):
    accs, exs = [], []
    for i in range(0, len(eps), bs):
        sub = eps[i:i + bs]
        b = {k: v.to(dev) for k, v in collate(sub).items()}
        msg = msg_from_tokens(toks[i:i + bs].to(dev), USE_EOS, CODEBOOK)
        pred = ex(b["qin"], msg).argmax(1)
        if target == "query":
            tgt = b["qout"]
        else:  # intermediate after first primitive
            tgt = torch.stack([torch.from_numpy(pad_grid(e["inter"][0])) for e in sub]).to(dev)
        accs += union_accuracy(pred, tgt).tolist(); exs += exact_match(pred, tgt).tolist()
    return np.array(accs), np.array(exs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=215)
    ap.add_argument("--device", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sp, ex, args, step = load(a.ckpt, dev)
    global USE_EOS, CODEBOOK
    USE_EOS = args.get("len_cost", 0) > 0
    CODEBOOK = sp.codebook if args["arm"] == "vq" else None
    if args["arm"] == "cont":
        print("continuous arm: no discrete probes"); return
    G = SynthGen(0, size_lo=args.get("size_lo", 4), size_hi=args.get("size_hi", 12), n_train3=args.get("n_train3", 0), library=args.get("library", "A"), n_prims=args.get("n_prims", 0), prim_seed=args.get("prim_seed"))  # the checkpoint's own task distribution
    V, L = args["V"], args["L"]
    rep = dict(ckpt=a.ckpt, step=step, arm=args["arm"], V=V, L=L)

    # 1. dictionary
    eps1 = G.fixed_eval_set("train1", 6 * G.n, seed=21)
    t1 = messages(sp, eps1, dev)
    by = {}
    for e, m in zip(eps1, t1.tolist()):
        by.setdefault(G.names[e["prog"][0]], []).append(tuple(m))
    dic = {}
    for k, ms in by.items():
        c = Counter(ms); modal, cnt = c.most_common(1)[0]
        dic[k] = dict(modal=list(modal), purity=cnt / len(ms), n_distinct=len(c))
    rep["dictionary"] = dic
    print(f"[dictionary] step {step} arm {args['arm']}  mean purity {np.mean([d['purity'] for d in dic.values()]):.2f}")
    for k, d in sorted(dic.items()):
        print(f"  {k:14s} {d['modal']}  purity {d['purity']:.2f}  distinct {d['n_distinct']}")

    # 2. lesion
    eps2 = G.fixed_eval_set("hold2", a.n, seed=12)
    t2 = messages(sp, eps2, dev)
    base_acc, base_ex = execute(ex, eps2, t2, dev)
    rng = np.random.default_rng(0)
    les = []
    for j in range(L):
        tt = t2.clone()
        rnd = torch.from_numpy(rng.integers(1, V, size=len(tt)))
        tt[:, j] = (tt[:, j] + rnd) % V  # guaranteed different symbol
        acc, exx = execute(ex, eps2, tt, dev)
        les.append(dict(pos=j, acc=float(acc.mean()), exact=float(exx.mean()), drop=float(base_acc.mean() - acc.mean())))
    rep["lesion"] = dict(base_acc=float(base_acc.mean()), base_exact=float(base_ex.mean()), per_pos=les)
    # changed-cell accuracy: own message vs random message vs identity (=0 by definition)
    cc_own, cc_rnd, cc_shuf = [], [], []
    rng2 = np.random.default_rng(1)
    # shuffled: every episode receives the message of another episode, a valid message of the learned code
    rng3 = np.random.default_rng(3); perm = rng3.permutation(len(t2))
    while np.any(perm == np.arange(len(t2))): perm = rng3.permutation(len(t2))
    ts = t2[torch.from_numpy(perm)]
    for i in range(0, len(eps2), 16):
        sub = eps2[i:i + 16]; b = {k: v.to(dev) for k, v in collate(sub).items()}
        pred = ex(b["qin"], msg_from_tokens(t2[i:i + 16].to(dev), USE_EOS, CODEBOOK)).argmax(1)
        cc_own += changed_cell_acc(pred, b["qin"], b["qout"]).tolist()
        rt = torch.from_numpy(rng2.integers(0, V, size=(len(sub), L))).to(dev)
        pred = ex(b["qin"], msg_from_tokens(rt, USE_EOS, CODEBOOK)).argmax(1)
        cc_rnd += changed_cell_acc(pred, b["qin"], b["qout"]).tolist()
        pred = ex(b["qin"], msg_from_tokens(ts[i:i + 16].to(dev), USE_EOS, CODEBOOK)).argmax(1)
        cc_shuf += changed_cell_acc(pred, b["qin"], b["qout"]).tolist()
    rep["changed_cell"] = dict(own=float(np.mean(cc_own)), random_msg=float(np.mean(cc_rnd)), shuffled_msg=float(np.mean(cc_shuf)))
    print(f"[changed-cell acc] own msg {np.mean(cc_own):.3f} | random msg {np.mean(cc_rnd):.3f}  (identity = 0)")
    print(f"[lesion] hold2 base acc {base_acc.mean():.3f} exact {base_ex.mean():.3f}")
    print("  drop per position: " + " ".join(f"p{d['pos']}:{d['drop']:+.3f}" for d in les))
    # all-random message = executor-only floor
    tt = torch.from_numpy(rng.integers(0, V, size=t2.shape))
    acc, exx = execute(ex, eps2, tt, dev)
    rep["lesion"]["random_msg_acc"] = float(acc.mean()); rep["lesion"]["random_msg_exact"] = float(exx.mean())
    acc, exx = execute(ex, eps2, ts, dev)
    rep["lesion"]["shuffled_msg_acc"] = float(acc.mean()); rep["lesion"]["shuffled_msg_exact"] = float(exx.mean())
    print(f"  shuffled message: acc {acc.mean():.3f} exact {exx.mean():.3f}  (a valid message of another episode)")
    print(f"  random message: acc {acc.mean():.3f} exact {exx.mean():.3f}  (executor-only floor)")

    # 3. swap: message of (A,C) applied to query of (A,B); score vs intermediate-after-A and vs full output
    tr2 = {p: None for p in G.train2}
    swaps = []
    for e in eps2:
        A_, B_ = e["prog"]
        cands = [p for p in G.train2 if p[0] == A_ and p[1] != B_]
        if not cands: continue
        C = cands[rng.integers(len(cands))]
        donor = G.make_episode(np.random.default_rng(hash((A_, C[1])) % (2**32)), "train2", prog=list(C))
        swaps.append((e, donor))
    if swaps:
        eq, ed = [s[0] for s in swaps], [s[1] for s in swaps]
        td = messages(sp, ed, dev); tq = messages(sp, eq, dev)
        acc_swap_inter, _ = execute(ex, eq, td, dev, target="inter")
        acc_own_inter, _ = execute(ex, eq, tq, dev, target="inter")
        acc_swap_out, _ = execute(ex, eq, td, dev)
        # reference: intermediate vs input / vs output (model-free)
        qi = torch.stack([torch.from_numpy(pad_grid(e["query"][0])) for e in eq])
        it = torch.stack([torch.from_numpy(pad_grid(e["inter"][0])) for e in eq])
        qo = torch.stack([torch.from_numpy(pad_grid(e["query"][1])) for e in eq])
        rep["swap"] = dict(n=len(swaps), donor_msg_vs_inter=float(acc_swap_inter.mean()),
                           own_msg_vs_inter=float(acc_own_inter.mean()), donor_msg_vs_output=float(acc_swap_out.mean()),
                           ref_input_vs_inter=float(union_accuracy(qi, it).mean()), ref_output_vs_inter=float(union_accuracy(qo, it).mean()))
        print(f"[swap] n={len(swaps)} donor(A,C) msg on (A,B) query -> vs inter(A): {acc_swap_inter.mean():.3f} "
              f"(own msg {acc_own_inter.mean():.3f}; refs input {rep['swap']['ref_input_vs_inter']:.3f}, output {rep['swap']['ref_output_vs_inter']:.3f}); vs full output {acc_swap_out.mean():.3f}")
    # 4. curriculum reuse (for runs trained with --curriculum): stage-1 dictionary from the singles
    #    set, then first-symbol / full-name / Hamming reuse on held-out pairs and repeats
    if args.get("curriculum"):
        from emlang import analyze as A
        D1 = A.dictionary([e["prog"] for e in eps1], t1.tolist(), USE_EOS)
        rep["reuse"] = {}
        for split, n in (("hold2", 215), ("train2", 200), ("rep2_hold", 40), ("rep2_train", 44)):
            try:
                eps_s = G.fixed_eval_set(split, n, seed=12)
            except Exception:
                continue
            ts = messages(sp, eps_s, dev).tolist()
            r = A.symbol_reuse([e["prog"] for e in eps_s], ts, D1, USE_EOS)
            r.update(A.name_similarity_reuse([e["prog"] for e in eps_s], ts, D1, USE_EOS))
            rep["reuse"][split] = r
            print(f"[reuse {split}] " + " ".join(f"{k[6:]}={v:.2f}" for k, v in r.items() if v == v))
        rep["D1"] = {G.names[k]: list(v) for k, v in D1.items()}
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
