"""Two quantities computable BEFORE reward training, as candidate predictors of collapse:
(1) how separable the library's operations are in the frozen perception space (1-NN rule identity
    of single-operation episodes; chance 1/n), and
(2) how much the executor can do without the message: cell accuracy of a trained continuous-channel
    system with its own message versus the message of another episode (the value of communication).
Run on the continuous checkpoint that supplies a library's frozen perception."""
import argparse, json, os
import numpy as np, torch, torch.nn.functional as F
from emlang.gen import SynthGen
from emlang.data import collate
from emlang.model import union_accuracy, exact_match
from emlang.probe import load


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default=None); ap.add_argument("--device", default=None)
    ap.add_argument("--perception_only", action="store_true", help="checkpoint holds only a speaker (e.g. the self-supervised encoder): compute separability, skip message value")
    ap.add_argument("--size_lo", type=int, default=3); ap.add_argument("--size_hi", type=int, default=6); ap.add_argument("--n_train3", type=int, default=600)
    ap.add_argument("--library", default="A"); ap.add_argument("--n_prims", type=int, default=0); ap.add_argument("--prim_seed", type=int, default=None)
    a = ap.parse_args(); dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.perception_only:
        from emlang.model import Speaker
        s = torch.load(a.ckpt, map_location=dev); sa = s["args"]
        sp = Speaker(sa.get("d", 128), sa.get("V", 32), sa.get("L", 8), "rl" if sa.get("arm") == "ssl" else sa.get("arm", "rl")).to(dev)
        sd = {k: v for k, v in s["speaker"].items() if not k.startswith("head")}   # drop the SSL projection head
        sp.load_state_dict(sd, strict=False); sp.eval(); ex = None; step = s.get("step", 0)
        args = dict(arm=sa.get("arm"), size_lo=a.size_lo, size_hi=a.size_hi, n_train3=a.n_train3, library=a.library, n_prims=a.n_prims, prim_seed=a.prim_seed)
    else:
        sp, ex, args, step = load(a.ckpt, dev)
    G = SynthGen(0, size_lo=args.get("size_lo", 4), size_hi=args.get("size_hi", 12), n_train3=args.get("n_train3", 0),
                 library=args.get("library", "A"), n_prims=args.get("n_prims", 0), prim_seed=args.get("prim_seed"))
    rep = dict(ckpt=a.ckpt, step=step, arm=args["arm"], n_ops=G.n, library=args.get("library", "A"), n_prims=args.get("n_prims", 0), prim_seed=args.get("prim_seed"))
    # (1) separability of the operations in perception space
    rng = np.random.default_rng(123)
    eps = [G.make_episode(rng, "train1", n_demos=3) for _ in range(20 * G.n)]
    Z = []
    for i in range(0, len(eps), 32):
        b = {k: v.to(dev) for k, v in collate(eps[i:i + 32]).items()}
        Z.append(F.normalize(sp.context(b["din"], b["dout"], b["dmask"]), dim=-1).cpu())
    Z = torch.cat(Z).numpy(); lab = np.array([e["prog"][0] for e in eps])
    S = Z @ Z.T; np.fill_diagonal(S, -1); nn = float(np.mean(lab[S.argmax(1)] == lab))
    # per-operation confusability: fraction of an op's episodes whose nearest neighbour is another op
    conf = {G.names[k]: float(np.mean(lab[S.argmax(1)][lab == k] != k)) for k in range(G.n)}
    # centroid separation: mean cosine between centroids of different ops (higher = more confusable)
    cents = np.stack([Z[lab == k].mean(0) for k in range(G.n)]); cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
    off = cents @ cents.T; np.fill_diagonal(off, np.nan); sep = float(np.nanmean(off))
    rep["perception"] = dict(nn_identity=nn, chance=1 / G.n, mean_centroid_cosine=sep, confusability=conf)
    print(f"[perception] {G.n} ops: 1-NN rule identity {nn:.3f} (chance {1/G.n:.3f}); mean between-op centroid cosine {sep:.3f}")
    worst = sorted(conf.items(), key=lambda kv: -kv[1])[:6]
    print("  most confusable:", ", ".join(f"{k} {v:.2f}" for k, v in worst))
    # (2) value of the message for the trained continuous system
    for split in (() if ex is None else ("train1", "hold2")):
        eps2 = G.fixed_eval_set(split, a.n, seed=12)
        own, other, exo, exs = [], [], [], []
        perm = np.random.default_rng(7).permutation(len(eps2))
        for i in range(0, len(eps2), 16):
            sub = eps2[i:i + 16]; b = {k: v.to(dev) for k, v in collate(sub).items()}
            msg = sp(b["din"], b["dout"], b["dmask"], greedy=True)
            pred = ex(b["qin"], msg).argmax(1)
            own += union_accuracy(pred, b["qout"]).tolist(); exo += exact_match(pred, b["qout"]).tolist()
            osub = [eps2[j] for j in perm[i:i + 16]]; ob = {k: v.to(dev) for k, v in collate(osub).items()}
            omsg = sp(ob["din"], ob["dout"], ob["dmask"], greedy=True)
            pred = ex(b["qin"], omsg).argmax(1)
            other += union_accuracy(pred, b["qout"]).tolist(); exs += exact_match(pred, b["qout"]).tolist()
        idn = float(np.mean(union_accuracy(torch.stack([torch.from_numpy(__import__("emlang.data", fromlist=["pad_grid"]).pad_grid(e["query"][0])) for e in eps2]), torch.stack([torch.from_numpy(__import__("emlang.data", fromlist=["pad_grid"]).pad_grid(e["query"][1])) for e in eps2])).tolist()))
        rep[f"value_{split}"] = dict(own_acc=float(np.mean(own)), other_msg_acc=float(np.mean(other)), identity_acc=idn,
                                    own_exact=float(np.mean(exo)), other_msg_exact=float(np.mean(exs)), value=float(np.mean(own) - np.mean(other)))
        print(f"[value {split}] own message acc {np.mean(own):.3f} (exact {np.mean(exo):.3f}) | other episode's message {np.mean(other):.3f} (exact {np.mean(exs):.3f}) | identity {idn:.3f} | value {np.mean(own)-np.mean(other):+.3f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
