"""Ease-of-transmission test (Chaabouni et al. 2020): freeze the trained speaker, train a
FRESH executor on its messages, and record the hold2 learning curve. A compositional code
should be acquired faster by a new learner than a holistic one.
Usage: python emlang/transmit.py --ckpt emlang_runs/rl_s0/latest.pt --steps 2000 --out x.json"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emlang.gen import SynthGen
from emlang.data import collate, augment_episode
from emlang.model import Speaker, Executor, union_accuracy, exact_match, msg_from_tokens
from emlang.train import weighted_ce


@torch.no_grad()
def evaluate(sp, ex, eps, dev, bs=16):
    ex.eval(); accs, exs = [], []
    for i in range(0, len(eps), bs):
        b = {k: v.to(dev) for k, v in collate(eps[i:i + bs]).items()}
        msg = sp(b["din"], b["dout"], b["dmask"], greedy=True)
        pred = ex(b["qin"], msg).argmax(1)
        accs += union_accuracy(pred, b["qout"]).tolist(); exs += exact_match(pred, b["qout"]).tolist()
    ex.train()
    return float(np.mean(accs)), float(np.mean(exs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=200); ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--device", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--shuffle_msgs", action="store_true", help="control: permute messages across episodes (destroys meaning)")
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    s = torch.load(a.ckpt, map_location=dev); args = s["args"]
    sp = Speaker(args["d"], args["V"], args["L"], args["arm"], tau=args["tau"], cont_dim=args.get("cont_dim", 0), use_eos=args.get("len_cost", 0) > 0).to(dev); sp.min_len = args.get("min_len", 1)
    sp.load_state_dict(s["speaker"], strict=False); sp.eval()
    for p in sp.parameters(): p.requires_grad_(False)
    torch.manual_seed(a.seed)
    ex = Executor(args["d"], args["V"], args["L"], args["arm"], R=args["R"], step_emb=not args.get("no_step_emb", False), cont_dim=args.get("cont_dim", 0), ex_d=args.get("ex_d", 0)).to(dev)
    opt = torch.optim.AdamW(ex.parameters(), lr=a.lr, weight_decay=0.01)
    G = SynthGen(0, size_lo=args.get("size_lo", 4), size_hi=args.get("size_hi", 12), n_train3=args.get("n_train3", 0), library=args.get("library", "A"), n_prims=args.get("n_prims", 0), prim_seed=args.get("prim_seed"))  # the checkpoint's own task distribution
    rng = np.random.default_rng(a.seed)
    ev = dict(train2=G.fixed_eval_set("train2", 200, seed=11), hold2=G.fixed_eval_set("hold2", 215, seed=12))
    curve = []
    t0 = time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            rec = dict(step=step, sec=time.time() - t0)
            for k, eps in ev.items():
                rec[k + "_acc"], rec[k + "_exact"] = evaluate(sp, ex, eps, dev)
            curve.append(rec); print(json.dumps(rec), flush=True)
        if step == a.steps: break
        eps = [augment_episode(G.make_episode(rng, "train", n_demos=int(rng.integers(2, 5))), rng) for _ in range(a.bs)]
        b = {k: v.to(dev) for k, v in collate(eps).items()}
        with torch.no_grad():
            msg = sp(b["din"], b["dout"], b["dmask"], greedy=True)
            if a.shuffle_msgs:
                perm = torch.randperm(a.bs, device=dev)
                msg = {k: (v[perm] if torch.is_tensor(v) else v) for k, v in msg.items()}
        for g in opt.param_groups: g["lr"] = a.lr * min(1.0, step / 100)
        logits = ex(b["qin"], msg)
        loss = weighted_ce(logits, b["qout"], args.get("pad_w", 0.2))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(ex.parameters(), 1.0); opt.step()
    if a.out:
        json.dump(dict(ckpt=a.ckpt, arm=args["arm"], shuffle=a.shuffle_msgs, curve=curve), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
