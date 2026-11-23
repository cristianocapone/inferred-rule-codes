"""Self-supervised perception for the speaker: SimCLR-style InfoNCE where the two views of an
episode are two disjoint halves of its demo pairs (same rule, different grids). No labels, no
solving. Saves a checkpoint whose 'speaker' state dict holds pair.* and ctx.* so that
train.py --init_perception <ckpt> --freeze_perception can use it.
Usage: EMLANG_CANVAS=12 python emlang/pretrain_perception.py --out ssl_perception_c12.pt --steps 6000 --no_arc
"""
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emlang.gen import SynthGen, CANVAS
from emlang.data import load_arc, split_arc_training, Mixer, collate, augment_episode, arc_episode
from emlang.model import Speaker


def halves(ep, rng):
    d = list(ep["demos"]); rng.shuffle(d)
    k = len(d) // 2
    a, b = dict(ep), dict(ep); a["demos"], b["demos"] = d[:k], d[k:]
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--no_arc", action="store_true"); ap.add_argument("--p_synth", type=float, default=0.5)
    ap.add_argument("--size_lo", type=int, default=3); ap.add_argument("--size_hi", type=int, default=6)
    ap.add_argument("--n_train3", type=int, default=600); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arc_root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ARC-AGI-2", "data"))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    synth = SynthGen(seed=0, size_lo=a.size_lo, size_hi=a.size_hi, n_train3=a.n_train3)
    arc_train = []
    if not a.no_arc:
        arc_all = load_arc(os.path.join(a.arc_root, "training")); arc_train, _ = split_arc_training(arc_all, 100, 0)
    mixer = Mixer(arc_train, synth, p_synth=1.0 if a.no_arc else a.p_synth, seed=a.seed, augment=True, stage=3)
    sp = Speaker(a.d, 32, 8, "cont").to(dev)   # only pair.* and ctx.* are used
    proj = nn.Sequential(nn.Linear(a.d, a.d), nn.GELU(), nn.Linear(a.d, 64)).to(dev)
    params = list(sp.pair.parameters()) + list(sp.ctx.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    print(f"[ssl] canvas {CANVAS} arc_train {len(arc_train)} dev {dev}", flush=True)
    t0 = time.time(); ema = None
    for step in range(a.steps + 1):
        for g in opt.param_groups: g["lr"] = a.lr * min(1.0, step / 200) * (0.5 * (1 + np.cos(np.pi * step / a.steps)))
        eps = []
        while len(eps) < a.bs:
            ep = mixer.batch(1)[0]
            if len(ep["demos"]) >= 2: eps.append(ep)   # need two non-empty halves
        va, vb = zip(*[halves(e, rng) for e in eps])
        ba = {k: v.to(dev) for k, v in collate(list(va)).items()}; bb = {k: v.to(dev) for k, v in collate(list(vb)).items()}
        za = F.normalize(proj(sp.context(ba["din"], ba["dout"], ba["dmask"])), dim=-1)
        zb = F.normalize(proj(sp.context(bb["din"], bb["dout"], bb["dmask"])), dim=-1)
        logits = za @ zb.t() / a.temp
        labels = torch.arange(len(eps), device=dev)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        with torch.no_grad():
            acc = (logits.argmax(1) == labels).float().mean().item()
        ema = (loss.item(), acc) if ema is None else (0.98 * ema[0] + 0.02 * loss.item(), 0.98 * ema[1] + 0.02 * acc)
        if step % 100 == 0:
            print(f"[{step}] infonce {ema[0]:.3f} match@bs {ema[1]:.3f} ({(time.time()-t0)/max(1,step):.2f}s/step)", flush=True)
        if step % 1000 == 0 and step > 0 or step == a.steps:
            torch.save(dict(speaker=sp.state_dict(), args=dict(arm="ssl", d=a.d, canvas=CANVAS, steps=step), step=step), a.out)
    # diagnostic: nearest-neighbour program identity on fresh synth episodes (does perception cluster by rule?)
    sp.eval(); rng2 = np.random.default_rng(123)
    eps = [synth.make_episode(rng2, "train1", n_demos=3) for _ in range(240)]
    with torch.no_grad():
        b = {k: v.to(dev) for k, v in collate(eps).items()}
        z = F.normalize(sp.context(b["din"], b["dout"], b["dmask"]), dim=-1).cpu().numpy()
    labels = np.array([e["prog"][0] for e in eps]); S = z @ z.T; np.fill_diagonal(S, -1)
    nn_acc = float(np.mean(labels[S.argmax(1)] == labels))
    print(f"[ssl-diag] 1-NN program-identity accuracy on singles (chance {1/synth.n:.3f}): {nn_acc:.3f}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
