"""Co-adaptation test: is a failing code unusable, or was its listener simply not coupled to it?

Rita et al. (NeurIPS 2022) decompose the objective of a signalling game into an information component (does
the message identify the input?) and a co-adaptation component (has the listener adapted to this speaker?).
Freeze a trained speaker, train a FRESH executor on its messages, and compare, on the same episodes:
  original   the executor trained together with the speaker (co-adapted)
  fresh      a new executor trained on the frozen speaker's real messages
  blind      a new executor trained on messages permuted across episodes (the message carries nothing)
If fresh >> blind, the messages carry usable information; if, in addition, fresh >> original, the original
failure was one of coupling, not of information. Accuracy is exact match on the held-out pairs of the first audit
(audit_20260916), on the pairs and triples not covered by the augmented training set (audit_20260917, when given), on
trained pairs and on single operations. The new executor is saved next to the report."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emlang.gen import SynthGen
from emlang.data import collate, augment_episode
from emlang.model import Speaker, Executor, union_accuracy, exact_match
from emlang.train import weighted_ce

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def evaluate(sp, ex, eps, dev, bs=32, shuffle=False, seed=0):
    ex.eval(); ok, acc = [], []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for i in range(0, len(eps), bs):
        b = {k: v.to(dev) for k, v in collate(eps[i:i + bs]).items()}
        msg = sp(b["din"], b["dout"], b["dmask"], greedy=True)
        if shuffle:
            perm = torch.randperm(len(eps[i:i + bs]), generator=g).to(dev)
            msg = {k: (v[perm] if torch.is_tensor(v) else v) for k, v in msg.items()}
        pred = ex(b["qin"], msg).argmax(1)
        ok += exact_match(pred, b["qout"]).tolist(); acc += union_accuracy(pred, b["qout"]).tolist()
    ex.train()
    return float(np.mean(ok)), float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=1000); ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--blind", action="store_true", help="train the fresh executor on messages permuted across episodes")
    ap.add_argument("--audit", default=os.path.join(ROOT, "audit_20260916", "split_audit_A_hold2.json"))
    ap.add_argument("--per_prog", type=int, default=3); ap.add_argument("--device", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--group_pairs", default=None, help="audit_20260917/group_audit_A_hold2.json (pairs not covered by the augmented training set)")
    ap.add_argument("--group_triples", default=None, help="audit_20260917/group_audit_A_len3.json (triples not covered by the augmented training set)")
    ap.add_argument("--per_prog_pairs", type=int, default=20); ap.add_argument("--per_prog_triples", type=int, default=10)
    ap.add_argument("--eval_executor", default=None,
                    help="evaluate a fresh executor saved by an earlier run (<report>_executor.pt) on the given "
                         "evaluation sets instead of training a new one; used to re-score the saved executors "
                         "when the audit changes")
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    s = torch.load(a.ckpt, map_location=dev, weights_only=False); args = s["args"]
    mk = lambda: Executor(args["d"], args["V"], args["L"], args["arm"], R=args["R"], step_emb=not args.get("no_step_emb", False), cont_dim=args.get("cont_dim", 0), ex_d=args.get("ex_d", 0)).to(dev)
    sp = Speaker(args["d"], args["V"], args["L"], args["arm"], tau=args["tau"], cont_dim=args.get("cont_dim", 0), use_eos=args.get("len_cost", 0) > 0).to(dev); sp.min_len = args.get("min_len", 1)
    sp.load_state_dict(s["speaker"], strict=False); sp.eval()
    for p in sp.parameters(): p.requires_grad_(False)
    ex_orig = mk(); ex_orig.load_state_dict(s["executor"], strict=False); ex_orig.eval()
    G = SynthGen(0, size_lo=args.get("size_lo", 4), size_hi=args.get("size_hi", 12), n_train3=args.get("n_train3", 0),
                 library=args.get("library", "A"), n_prims=args.get("n_prims", 0), prim_seed=args.get("prim_seed"), drop_prims=args.get("drop_prims", ""))
    novel = json.load(open(a.audit))["novel_progs"]
    rng_e = np.random.default_rng(12)
    ev = dict(novel_pairs=[G.make_episode(rng_e, "hold2", n_demos=3, prog=list(p)) for p in novel for _ in range(a.per_prog)],
              trained_pairs=G.fixed_eval_set("train2", 200, seed=11), singles=G.fixed_eval_set("train1", 150, seed=11))
    if a.group_pairs:
        ev["pairs_audited"] = [G.make_episode(rng_e, "hold2", n_demos=3, prog=list(p)) for p in json.load(open(a.group_pairs))["novel_group"] for _ in range(a.per_prog_pairs)]
    if a.group_triples:
        ev["triples_audited"] = [G.make_episode(rng_e, "hold2", n_demos=3, prog=list(p)) for p in json.load(open(a.group_triples))["novel_group"] for _ in range(a.per_prog_triples)]
    rep = dict(ckpt=a.ckpt, arm=args["arm"], blind=a.blind, steps=a.steps, original={k: evaluate(sp, ex_orig, e, dev)[0] for k, e in ev.items()})
    print("[original]", json.dumps(rep["original"]), flush=True)
    del ex_orig
    if a.eval_executor:
        w = torch.load(a.eval_executor, map_location=dev, weights_only=False)
        ex = mk(); ex.load_state_dict(w["executor"], strict=False); ex.eval()
        rep.update(blind=w["blind"], steps=w["steps"], from_executor=a.eval_executor,
                   curve=[dict(step=w["steps"], sec=0.0,
                               **{k: evaluate(sp, ex, e, dev, shuffle=w["blind"])[0] for k, e in ev.items()})])
        print(json.dumps(rep["curve"][-1]), flush=True)
        if a.out:
            os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(rep, open(a.out, "w"), indent=1)
        return
    torch.manual_seed(a.seed); ex = mk(); opt = torch.optim.AdamW(ex.parameters(), lr=a.lr, weight_decay=0.01)
    rng = np.random.default_rng(a.seed); curve = []; t0 = time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            rec = dict(step=step, sec=round(time.time() - t0, 1), **{k: evaluate(sp, ex, e, dev, shuffle=a.blind)[0] for k, e in ev.items()})
            curve.append(rec); print(json.dumps(rec), flush=True)
        if step == a.steps: break
        eps = [augment_episode(G.make_episode(rng, "train", n_demos=int(rng.integers(2, 5))), rng) for _ in range(a.bs)]
        b = {k: v.to(dev) for k, v in collate(eps).items()}
        with torch.no_grad():
            msg = sp(b["din"], b["dout"], b["dmask"], greedy=True)
            if a.blind:
                perm = torch.randperm(a.bs, device=dev)
                msg = {k: (v[perm] if torch.is_tensor(v) else v) for k, v in msg.items()}
        for g in opt.param_groups: g["lr"] = a.lr * min(1.0, (step + 1) / 200)
        loss = weighted_ce(ex(b["qin"], msg), b["qout"], args.get("pad_w", 0.2))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(ex.parameters(), 1.0); opt.step()
    rep["curve"] = curve
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(rep, open(a.out, "w"), indent=1)
        torch.save(dict(executor=ex.state_dict(), blind=a.blind, steps=a.steps, ckpt=a.ckpt), a.out.replace(".json", "_executor.pt"))


if __name__ == "__main__":
    main()
