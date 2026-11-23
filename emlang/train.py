"""Emergent symbolic reasoning language: Speaker/Executor training with three arms.

  rl     : discrete symbols sampled; speaker trained by REINFORCE with a group
           (GRPO-style) baseline on DENSE reward (union cell accuracy of the query);
           executor trained by cross-entropy on the same samples.
  gumbel : discrete symbols, straight-through Gumbel-softmax, fully backprop.
  cont   : no bottleneck - L continuous d-dim vectors (capacity control).

Data: p_synth compositional synthetic episodes + (1-p_synth) real ARC-AGI-2
training tasks (leave-one-out over demo pairs). Held-out: unseen primitive
COMPOSITIONS (hold2), unseen LENGTH (len3), 100 held-out ARC training tasks
(real test pairs), and ARC evaluation tasks as MONITORING ONLY.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emlang.gen import SynthGen, PAD, CANVAS
from emlang.data import (load_arc, split_arc_training, Mixer, collate, arc_test_episodes,
                         augment_episode, pad_grid, load_arc_kaggle)
from emlang.model import Speaker, Executor, union_accuracy, exact_match, shape_match
from emlang import analyze as A


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["rl", "gumbel", "cont", "vq"], default="rl")
    ap.add_argument("--V", type=int, default=32, help="vocabulary size")
    ap.add_argument("--L", type=int, default=8, help="message length")
    ap.add_argument("--G", type=int, default=4, help="samples per episode for the RL group baseline")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--R", type=int, default=2, help="executor blocks per symbol")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sp_lr_mult", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ent", type=float, default=0.01, help="entropy bonus (rl)")
    ap.add_argument("--ent_floor", type=float, default=0.0, help="rl: penalise mean policy entropy below this (nats)")
    ap.add_argument("--ent_floor_w", type=float, default=1.0)
    ap.add_argument("--adv_eps", type=float, default=0.05, help="rl: advantage = (r-mean)/(std+eps)")
    ap.add_argument("--div_w", type=float, default=0.0, help="anti-collapse: maximise mutual information between episode and symbol, H(marginal symbol distribution) - mean H(per-episode), per position")
    ap.add_argument("--ent_floor_start", type=float, default=None, help="anneal the entropy floor linearly from this value to --ent_floor over --ent_floor_anneal steps (keeps the speaker exploring while the executor learns)")
    ap.add_argument("--ent_floor_anneal", type=int, default=0)
    ap.add_argument("--stab_w", type=float, default=0.0, help="name-stability pressure: KL(EMA speaker || speaker) on the sampled messages, per position")
    ap.add_argument("--stab_decay", type=float, default=0.995, help="decay of the EMA speaker used by --stab_w")
    ap.add_argument("--reward", choices=["union", "changed"], default="union", help="rl reward: cell accuracy over the union of non-pad cells (pays for copying the input) or accuracy restricted to the cells the rule changes (pays nothing for copying)")
    ap.add_argument("--proc_reward", action="store_true", help="process reward: credit each symbol by the gain it produces in the executor's state (vs outcome-only REINFORCE)")
    ap.add_argument("--logp_sum", action="store_true", help="rl: use summed (not mean) log-prob over symbols")
    ap.add_argument("--init_from", default=None, help="warm-start speaker+executor weights from a checkpoint")
    ap.add_argument("--init_perception", default=None, help="load ONLY the speaker's pair encoder + context transformer from this checkpoint (any arm)")
    ap.add_argument("--freeze_perception", action="store_true", help="freeze speaker.pair and speaker.ctx (symbol policy learns alone)")
    ap.add_argument("--reset_every", type=int, default=0, help="re-initialise the EXECUTOR every N steps (transmission pressure / iterated learning)")
    ap.add_argument("--no_step_emb", action="store_true", help="executor: position-independent symbol semantics (no step embedding)")
    ap.add_argument("--cont_dim", type=int, default=0, help="cont arm: dims per symbol (0 = d); capacity-matched control")
    ap.add_argument("--ex_d", type=int, default=0, help="executor width (0 = d)")
    ap.add_argument("--len_cost", type=float, default=0.0, help="MDL pressure: cost per used symbol (enables EOS symbol 0)")
    ap.add_argument("--len_warmup", type=int, default=4000, help="linear ramp of the length cost from 0 over this many steps")
    ap.add_argument("--eos_explore", type=float, default=0.0, help="training-time prob per position of forbidding EOS (exploration of longer messages)")
    ap.add_argument("--curriculum", default=None, help="stage step budgets, e.g. '8000,12000,12000' (stage1 singles, stage2 pairs+repeats, stage3 triples); '0,0,N' = joint control")
    ap.add_argument("--size_lo", type=int, default=4); ap.add_argument("--size_hi", type=int, default=12)
    ap.add_argument("--n_train3", type=int, default=0)
    ap.add_argument("--no_arc", action="store_true", help="synthetic only (curriculum runs)")
    ap.add_argument("--data", choices=["synth", "rearc", "list"], default="synth", help="training/eval source: compositional synth or RE-ARC generators (real ARC-1 rules)")
    ap.add_argument("--rearc_rules", type=int, default=0, help="use only the first N training rule families (nested subsets) — for the naming-capacity scaling experiment")
    ap.add_argument("--rearc_diff", default="0,1", help="RE-ARC difficulty range lo,hi (0,0.15 = easy end of all 400 real ARC-1 rules; median grid side 8)")
    ap.add_argument("--kaggle_dir", default=None, help="dir with arc-agi_{evaluation,concept}_{challenges,solutions}.json -> ARC-1 eval + ConceptARC monitoring sets")
    ap.add_argument("--d1_step", type=int, default=0, help="step at which the stage-1 dictionary is snapshotted (default: end of stage 1)")
    ap.add_argument("--min_len", type=int, default=1, help="MDL arms: minimum message length (EOS forbidden earlier)")
    ap.add_argument("--prim_seed", type=int, default=None, help="with --n_prims: pick a seeded random subset instead of the first N")
    ap.add_argument("--n_prims", type=int, default=0, help="use only the first N primitives of the library (collapse-vs-task-count experiment)")
    ap.add_argument("--drop_prims", default="", help="comma-separated operation names to remove from the library")
    ap.add_argument("--library", choices=["A", "B", "Bdup"], default="A", help="grid primitive library: A (geometry/gravity/shift/colour/size) or B (counting/symmetry/connectivity/ordering/masking), disjoint from A")
    ap.add_argument("--list_same_len", action="store_true", help="list domain: keep only length-preserving operations")
    ap.add_argument("--singles_frac", type=float, default=0.0, help="override the fraction of single-primitive episodes in the stage-3 mixture (joint runs): tests whether isolated-part EXPERIENCE, rather than temporal order, is what reward-driven composition needs")
    ap.add_argument("--stage_order", default=None, help="e.g. '2,1,3' = reverse curriculum (pairs first, then singles, then triples)")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--tau_end", type=float, default=0.0, help="if >0: anneal the speaker temperature linearly from --tau to this over the run (gumbel: sharper samples late)")
    ap.add_argument("--pad_w", type=float, default=0.2, help="CE weight of PAD-target cells")
    ap.add_argument("--p_synth", type=float, default=0.5)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arc_root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ARC-AGI-2", "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_n", type=int, default=200, help="episodes per synthetic eval split")
    ap.add_argument("--arc_eval_every", type=int, default=5000, help="ARC evaluation set (monitoring only)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_ckpt", action="store_true", help="disable executor gradient checkpointing")
    return ap.parse_args()


def message_mi(probs):
    """H(mean_b p) - mean_b H(p), averaged over positions: high when different episodes get
    different symbols, zero when the speaker emits the same distribution regardless of the task."""
    p = probs.clamp_min(1e-9)
    h_marg = -(p.mean(0) * p.mean(0).clamp_min(1e-9).log()).sum(-1)   # [L]
    h_cond = -(p * p.log()).sum(-1).mean(0)                            # [L]
    return (h_marg - h_cond).mean()


def weighted_ce(logits, target, pad_w):
    ce = F.cross_entropy(logits, target, reduction="none")  # [B,H,W]
    w = torch.where(target == PAD, torch.full_like(ce, pad_w), torch.ones_like(ce))
    return (ce * w).sum() / w.sum()


def to_dev(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def run_episodes(speaker, executor, episodes, dev, bs=64, greedy=True, states=False):
    """Returns dict of per-episode arrays + messages."""
    speaker.eval(); executor.eval()
    out = dict(acc=[], exact=[], shape=[], msgs=[], cont=[], inter=[])
    for i in range(0, len(episodes), bs):
        eps = episodes[i:i + bs]
        b = to_dev(collate(eps), dev)
        msg = speaker(b["din"], b["dout"], b["dmask"], greedy=greedy)
        if states:
            logits, sts = executor(b["qin"], msg, return_states=True)
        else:
            logits = executor(b["qin"], msg)
        pred = logits.argmax(1)
        out["acc"] += union_accuracy(pred, b["qout"]).tolist()
        out["exact"] += exact_match(pred, b["qout"]).tolist()
        out["shape"] += shape_match(pred, b["qout"]).tolist()
        if msg["tokens"] is not None:
            out["msgs"] += msg["tokens"].cpu().numpy().tolist()
            if msg.get("length") is not None:
                out.setdefault("len", []); out["len"] += msg["length"].tolist()
        else:
            out["cont"] += msg["cont"].flatten(1).cpu().numpy().tolist()
        if states:
            # serial-execution probe: does some intermediate state equal the FIRST primitive's output?
            for j, ep in enumerate(eps):
                if ep.get("inter") and len(ep["prog"]) >= 2:
                    tgt = torch.from_numpy(pad_grid(ep["inter"][0])).to(dev)[None]
                    best = max(union_accuracy(s[j:j + 1].argmax(1), tgt).item() for s in sts[:-1])
                    out["inter"].append(best)
    speaker.train(); executor.train()
    return out


def language_metrics(episodes, res, V):
    m = {}
    progs = [ep["prog"] for ep in episodes]
    if res["msgs"]:
        msgs = res["msgs"]
        m["topsim"] = A.topsim(progs, msgs)
        m.update({f"msg_{k}": v for k, v in A.message_stats(msgs, V).items()})
        n_rules = len({tuple(p) for p in progs})          # ceiling of msg_unique_frac is n_rules / n_episodes
        m["msg_n_rules"] = n_rules
        m["msg_unique_per_rule"] = len({tuple(x) for x in msgs}) / max(n_rules, 1)
        two = [(p, mm) for p, mm in zip(progs, msgs) if len(p) == 2]
        if len(two) > 10:
            attrs = [p for p, _ in two]; mm = [x for _, x in two]
            m["posdis"] = A.posdis(attrs, mm); m["bosdis"] = A.bosdis(attrs, mm, V)
        m["align_purity"] = A.symbol_primitive_alignment(progs, msgs, None)
        if progs and all(len(p) == 1 for p in progs):                      # the dictionary, for name-turnover analyses
            from collections import Counter as _C
            by = {}
            for p, mm in zip(progs, msgs): by.setdefault(int(p[0]), []).append(tuple(mm))
            m["dictionary"] = {str(k): list(_C(v).most_common(1)[0][0]) for k, v in sorted(by.items())}
    elif res["cont"]:
        # continuous channel: topsim with euclidean message distance
        rng = np.random.default_rng(0); n = len(progs)
        idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if len(idx) > 4000:
            sel = rng.choice(len(idx), 4000, replace=False); idx = [idx[k] for k in sel]
        C = np.asarray(res["cont"])
        dp = [A.levenshtein(progs[i], progs[j]) for i, j in idx]
        dm = [float(np.linalg.norm(C[i] - C[j])) for i, j in idx]
        m["topsim"] = A.spearman(dp, dm)
    if res["inter"]:
        m["inter_probe"] = float(np.mean(res["inter"]))
    if res.get("len"):
        m["msg_len"] = float(np.mean(res["len"]))
    return m


def summarize(res):
    return dict(acc=float(np.mean(res["acc"])), exact=float(np.mean(res["exact"])),
                shape=float(np.mean(res["shape"])), n=len(res["acc"]))


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    def emit(d):
        d["time"] = time.time(); log.write(json.dumps(d) + "\n"); log.flush()

    # ---------------- data
    synth = (None if args.data == "list" else
             SynthGen(seed=0, size_lo=args.size_lo, size_hi=args.size_hi, n_train3=args.n_train3, library=args.library, n_prims=args.n_prims, prim_seed=args.prim_seed, drop_prims=args.drop_prims))  # library/split fixed across seeds
    if args.stage_order and synth is not None:
        synth.stage_order = [int(x) for x in args.stage_order.split(",")]
    if args.singles_frac > 0 and synth is not None:
        f = args.singles_frac
        synth.stage_pools[3] = [(synth.train1, f), (synth.train2, (1 - f) * 0.412),
                                (synth.rep2_train, (1 - f) * 0.118), (synth.train3, (1 - f) * 0.471)]
        print(f"[mixture] stage-3 singles fraction set to {f} (default 0.15)", flush=True)
    if args.no_arc:
        arc_train, arc_held, arc_eval = [], [], []
    else:
        arc_all = load_arc(os.path.join(args.arc_root, "training"))
        arc_train, arc_held = split_arc_training(arc_all, n_heldout=100, seed=0)
        arc_eval = load_arc(os.path.join(args.arc_root, "evaluation"))
    stages = [int(x) for x in args.curriculum.split(",")] if args.curriculum else None
    if stages:
        assert sum(stages) == args.steps, "curriculum budgets must sum to --steps"
        bounds = np.cumsum(stages)
        def stage_at(step):
            return int(np.searchsorted(bounds, step, side="right")) + 1 if step < bounds[-1] else len(stages)
        if args.d1_step == 0: args.d1_step = int(bounds[0]) if bounds[0] > 0 else int(bounds[1]) // 2
    if args.data == "list":
        from emlang.list_data import ListGen
        synth = ListGen(seed=0, n_train3=args.n_train3, same_len=args.list_same_len)   # same interface as SynthGen
        print(f"[data] LIST rule-induction domain: {synth.n} primitives, splits {synth.split_sizes}", flush=True)
    if args.data == "rearc":
        from emlang.rearc_data import REARC
        lo, hi = (float(x) for x in args.rearc_diff.split(","))
        rearc = REARC(seed=0, diff=(lo, hi), n_train_rules=args.rearc_rules or None)
        print(f"[data] RE-ARC difficulty range ({lo}, {hi})", flush=True)
        print(f"[data] RE-ARC tasks train {rearc.split_sizes['train']} held-out {rearc.split_sizes['hold']}", flush=True)
        mixer = Mixer(arc_train, rearc, p_synth=1.0 if args.no_arc else args.p_synth, seed=args.seed, augment=not args.no_aug, stage=None)
        ev = dict(
            rearc_train=rearc.fixed_eval_set("train", args.eval_n, seed=11),
            rearc_hold=rearc.fixed_eval_set("hold", max(args.eval_n, 4 * rearc.split_sizes["hold"]), seed=12),
        )
    else:
        mixer = Mixer(arc_train, synth, p_synth=1.0 if args.no_arc else args.p_synth, seed=args.seed, augment=not args.no_aug,
                      stage=(stage_at(0) if stages else None))
        ev = dict(
            synth_train1=synth.fixed_eval_set("train1", 4 * synth.n, seed=14),
            synth_train2=synth.fixed_eval_set("train2", args.eval_n, seed=11),
            synth_hold2=synth.fixed_eval_set("hold2", max(args.eval_n, len(synth.hold2)), seed=12),
            synth_len3=synth.fixed_eval_set("len3", args.eval_n, seed=13),
        )
    if stages and args.data == "synth":
        ev["synth_rep2_train"] = synth.fixed_eval_set("rep2_train", 4 * len(synth.rep2_train), seed=15)
        ev["synth_rep2_hold"] = synth.fixed_eval_set("rep2_hold", 8 * len(synth.rep2_hold), seed=16)
        ev["synth_train3"] = synth.fixed_eval_set("train3", args.eval_n, seed=17)
    if arc_held:
        ev["arc_heldout"] = [e for t in arc_held for e in arc_test_episodes(t)]
    arc_eval_eps = [e for t in arc_eval for e in arc_test_episodes(t)]
    monitor = {}
    if args.kaggle_dir:
        for tag, base in (("ARC1_EVAL_MONITORING", "arc-agi_evaluation"), ("CONCEPTARC_MONITORING", "arc-agi_concept")):
            cj, sj = os.path.join(args.kaggle_dir, base + "_challenges.json"), os.path.join(args.kaggle_dir, base + "_solutions.json")
            if os.path.exists(cj):
                tk = load_arc_kaggle(cj, sj if os.path.exists(sj) else None)
                monitor[tag] = [e for tt in tk for e in arc_test_episodes(tt)]
                print(f"[data] {tag}: {len(tk)} tasks, {len(monitor[tag])} test pairs (MONITORING ONLY)", flush=True)
    D1 = {}  # stage-1 dictionary snapshot: primitive -> modal message
    print(f"[data] canvas {CANVAS} arc train {len(arc_train)} heldout {len(arc_held)} tasks; "
          f"arc eval {len(arc_eval)} tasks ({len(arc_eval_eps)} pairs, MONITORING ONLY); synth {synth.split_sizes}; curriculum {stages}", flush=True)

    # ---------------- models
    speaker = Speaker(args.d, args.V, args.L, args.arm, tau=args.tau, cont_dim=args.cont_dim, use_eos=args.len_cost > 0).to(dev)
    speaker.min_len = args.min_len
    executor = Executor(args.d, args.V, args.L, args.arm, R=args.R, use_ckpt=not args.no_ckpt, step_emb=not args.no_step_emb, cont_dim=args.cont_dim, ex_d=args.ex_d).to(dev)
    n_sp = sum(p.numel() for p in speaker.parameters()); n_ex = sum(p.numel() for p in executor.parameters())
    print(f"[model] arm={args.arm} V={args.V} L={args.L} d={args.d} speaker {n_sp/1e6:.2f}M executor {n_ex/1e6:.2f}M dev={dev}", flush=True)
    speaker_ema = None
    if args.stab_w > 0:
        import copy
        speaker_ema = copy.deepcopy(speaker).eval()
        for p in speaker_ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW([
        {"params": speaker.parameters(), "lr": args.lr * args.sp_lr_mult},
        {"params": executor.parameters(), "lr": args.lr}], weight_decay=0.01, betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in opt.param_groups]
    step0 = 0
    ck = os.path.join(args.out, "latest.pt")
    if args.resume and os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        speaker.load_state_dict(s["speaker"], strict=False); executor.load_state_dict(s["executor"], strict=False); opt.load_state_dict(s["opt"])
        step0 = s["step"]; print(f"[resume] from step {step0}", flush=True)
        try:
            for line in open(os.path.join(args.out, "log.jsonl")):
                r = json.loads(line)
                if r.get("kind") == "eval" and "D1_snapshot" in r:
                    D1 = {synth.names.index(k): tuple(v) for k, v in r["D1_snapshot"].items()}
        except Exception:
            pass
    if args.init_perception and step0 == 0:
        s = torch.load(args.init_perception, map_location=dev)
        sub = {k: v for k, v in s["speaker"].items() if k.startswith("pair.") or k.startswith("ctx.")}
        missing = speaker.load_state_dict(sub, strict=False)
        print(f"[init_perception] {args.init_perception} (arm {s['args']['arm']}): loaded {len(sub)} tensors", flush=True)
    if args.freeze_perception:
        for mod in (speaker.pair, speaker.ctx):
            for p in mod.parameters(): p.requires_grad_(False)
        opt = torch.optim.AdamW([
            {"params": [p for p in speaker.parameters() if p.requires_grad], "lr": args.lr * args.sp_lr_mult},
            {"params": executor.parameters(), "lr": args.lr}], weight_decay=0.01, betas=(0.9, 0.95))
        base_lrs = [g["lr"] for g in opt.param_groups]
        print("[freeze_perception] speaker.pair + speaker.ctx frozen", flush=True)
    if args.init_from and step0 == 0:
        s = torch.load(args.init_from, map_location=dev)
        speaker.load_state_dict(s["speaker"], strict=False); executor.load_state_dict(s["executor"], strict=False)
        print(f"[init_from] {args.init_from} (step {s['step']}, arm {s['args']['arm']})", flush=True)
    scaler = torch.amp.GradScaler(enabled=args.amp and dev == "cuda")

    def lr_at(step):
        if stages:  # cosine with warm restart at each stage boundary
            s = stage_at(min(step, args.steps - 1)); start = int(bounds[s - 2]) if s > 1 else 0
            length = max(1, stages[s - 1]); local = step - start
            if local < args.warmup: return local / max(1, args.warmup)
            p = (local - args.warmup) / max(1, length - args.warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        if step < args.warmup: return step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    def evaluate(step, with_arc_eval):
        nonlocal D1
        t0 = time.time(); rec = dict(kind="eval", step=step)
        if stages: rec["stage"] = stage_at(min(step, args.steps - 1))
        use_eos = args.len_cost > 0
        for name, eps in ev.items():
            res = run_episodes(speaker, executor, eps, dev, states=name in ("synth_train2", "synth_hold2"))
            rec[name] = summarize(res)
            if name.startswith("synth") or name.startswith("rearc"):
                rec[name].update(language_metrics(eps, res, args.V))
            if stages and name == "synth_train1" and res["msgs"]:
                Dnow = A.dictionary([e["prog"] for e in eps], res["msgs"], use_eos)
                if step == args.d1_step or (not D1 and step >= args.d1_step):
                    D1 = Dnow; rec["D1_snapshot"] = {synth.names[k]: list(v) for k, v in D1.items()}
                    print(f"[D1] snapshot at step {step}: " + "; ".join(f"{synth.names[k]}={list(v)}" for k, v in list(D1.items())[:8]), flush=True)
                if D1:
                    rec[name]["d1_stability"] = float(np.mean([Dnow.get(k) == v for k, v in D1.items()]))
            if stages and D1 and res["msgs"] and name in ("synth_train2", "synth_hold2", "synth_rep2_train", "synth_rep2_hold"):
                rec[name].update(A.symbol_reuse([e["prog"] for e in eps], res["msgs"], D1, use_eos))
            if name == "arc_heldout" and args.arm != "cont":
                # pass@2 = greedy U one sampled message
                res2 = run_episodes(speaker, executor, eps, dev, greedy=False)
                rec[name]["exact_pass2"] = float(np.mean(np.maximum(res["exact"], res2["exact"])))
        if with_arc_eval:
            if arc_eval_eps:
                res = run_episodes(speaker, executor, arc_eval_eps, dev)
                rec["ARC_EVAL_MONITORING"] = summarize(res)
            for tag, eps_m in monitor.items():
                rec[tag] = summarize(run_episodes(speaker, executor, eps_m, dev))
        rec["eval_sec"] = time.time() - t0
        emit(rec)
        s = " | ".join(f"{k}: acc {v['acc']:.3f} ex {v['exact']:.3f}" + (f" ts {v['topsim']:.2f}" if 'topsim' in v and v['topsim'] == v['topsim'] else "")
                       + (f" reuseA {v['reuse_first_A']:.2f}/ch {v['reuse_first_chance']:.2f} name {v['reuse_name_A']:.2f}/{v['reuse_name_chance']:.2f} ord {v['reuse_order_AB']:.2f}" if 'reuse_first_A' in v and v['reuse_first_A'] == v['reuse_first_A'] else "")
                       + (f" len {v['msg_len']:.1f}" if 'msg_len' in v else "")
                       + (f" pd {v['posdis']:.2f}" if 'posdis' in v and v['posdis'] == v['posdis'] else "")
                       + (f" ip {v['inter_probe']:.2f}" if 'inter_probe' in v else "")
                       for k, v in rec.items() if isinstance(v, dict) and "acc" in v)
        print(f"[eval {step}] {s} ({rec['eval_sec']:.0f}s)", flush=True)
        torch.save(dict(speaker=speaker.state_dict(), executor=executor.state_dict(), opt=opt.state_dict(),
                        step=step, args=vars(args)), ck)

    # ---------------- train
    ema = {}
    def upd(k, v, a=0.02):
        ema[k] = v if k not in ema else (1 - a) * ema[k] + a * v
    t_start = time.time()
    for step in range(step0, args.steps + 1):
        for g, b in zip(opt.param_groups, base_lrs): g["lr"] = b * lr_at(step)
        if stages:
            st = stage_at(min(step, args.steps - 1))
            if st != mixer.stage:
                mixer.stage = st; print(f"[stage] -> {st} at step {step}", flush=True); emit(dict(kind="stage", step=step, stage=st))
        if stages and step == args.d1_step and step % args.eval_every != 0:
            evaluate(step, with_arc_eval=False)
        if args.tau_end > 0:
            speaker.tau = args.tau + (args.tau_end - args.tau) * min(1.0, step / max(1, args.steps))
        if args.reset_every and step > 0 and step % args.reset_every == 0 and step < args.steps:
            fresh = Executor(args.d, args.V, args.L, args.arm, R=args.R, use_ckpt=not args.no_ckpt, step_emb=not args.no_step_emb, cont_dim=args.cont_dim, ex_d=args.ex_d).to(dev)
            executor.load_state_dict(fresh.state_dict()); del fresh
            from collections import defaultdict
            opt.state = defaultdict(dict)  # drop Adam moments (params keep identity, so groups stay valid)
            print(f"[reset] executor re-initialised at step {step}", flush=True); emit(dict(kind="reset", step=step))
        eps = mixer.batch(args.bs)
        b = to_dev(collate(eps), dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
            if args.arm == "rl":
                G = args.G
                qin, qout = b["qin"].repeat_interleave(G, 0), b["qout"].repeat_interleave(G, 0)
                ctx = speaker.context(b["din"], b["dout"], b["dmask"]).repeat_interleave(G, 0)
                msg = speaker.decode(ctx, eos_explore=args.eos_explore)
                if args.proc_reward:
                    logits, states = executor(qin, msg, return_states=True)
                else:
                    logits = executor(qin, msg)
                with torch.no_grad():
                    pred = logits.argmax(1)
                    if args.reward == "changed":
                        ch = (qout != qin); r = ((pred == qout) & ch).flatten(1).sum(1).float() / ch.flatten(1).sum(1).clamp(min=1).float()
                    else:
                        r = union_accuracy(pred, qout)
                    acc_now = union_accuracy(pred, qout).mean().item()
                    if args.len_cost > 0:
                        r = r - args.len_cost * min(1.0, step / max(1, args.len_warmup)) * msg["length"]
                    r = r.view(args.bs, G)
                    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + args.adv_eps)
                    if args.proc_reward:
                        # per-symbol credit: G_i = r_final - r_{i-1}, where r_i is the accuracy of the
                        # executor's state after symbol i and r_{-1} is the accuracy of copying the input
                        accs = [union_accuracy(s.argmax(1), qout) for s in states]          # L entries
                        prev = [union_accuracy(qin, qout)] + accs[:-1]                       # r_{i-1}
                        Gp = torch.stack([accs[-1] - p for p in prev], 1)                    # [B*G, L]
                        Gp = Gp.view(args.bs, G, -1)
                        adv_pos = (Gp - Gp.mean(1, keepdim=True)) / (Gp.std(1, keepdim=True) + args.adv_eps)
                        adv_pos = adv_pos.reshape(-1, adv_pos.shape[-1])                     # [B*G, L]
                loss_ex = weighted_ce(logits.float(), qout, args.pad_w)
                ent_mean = msg["ent"].mean()
                if args.proc_reward:
                    pg = -(adv_pos * msg["logp_pos"]).sum(1).mean() / args.L
                else:
                    logp = msg["logp"] if args.logp_sum else msg["logp"] / args.L
                    pg = -(adv.reshape(-1) * logp).mean()
                floor = args.ent_floor
                if args.ent_floor_start is not None and args.ent_floor_anneal > 0:
                    floor = args.ent_floor_start + (args.ent_floor - args.ent_floor_start) * min(1.0, step / args.ent_floor_anneal)
                loss_sp = pg - args.ent * ent_mean + args.ent_floor_w * F.relu(floor - ent_mean)
                if args.div_w > 0 and msg.get("probs") is not None:
                    mi = message_mi(msg["probs"]); loss_sp = loss_sp - args.div_w * mi; upd("mi", mi.item())
                if speaker_ema is not None and msg.get("probs") is not None:
                    with torch.no_grad():
                        lp_ema = speaker_ema.score(ctx.float(), msg["tokens"])                       # [B*G,L,V]
                    lp_cur = torch.log(msg["probs"].float().clamp(min=1e-8))
                    kl = (lp_ema.exp() * (lp_ema - lp_cur)).sum(-1).mean()
                    loss_sp = loss_sp + args.stab_w * kl; upd("stab_kl", kl.item())
                loss = loss_ex + loss_sp
                upd("reward", r.mean().item()); upd("ent", msg["ent"].mean().item()); upd("loss_sp", loss_sp.item())
                upd("rstd", r.std(1).mean().item()); upd("rdiff", (r.max(1).values - r.min(1).values > 1e-6).float().mean().item())   # the speaker's learning signal: reward spread across the G sampled messages
            else:
                msg = speaker(b["din"], b["dout"], b["dmask"], eos_explore=args.eos_explore)
                logits = executor(b["qin"], msg)
                loss_ex = weighted_ce(logits.float(), b["qout"], args.pad_w)
                loss = loss_ex
                if args.len_cost > 0 and msg.get("exp_length") is not None:
                    loss = loss + args.len_cost * min(1.0, step / max(1, args.len_warmup)) * msg["exp_length"].mean()
                if args.div_w > 0 and msg.get("probs") is not None:
                    mi = message_mi(msg["probs"]); loss = loss - args.div_w * mi; upd("mi", mi.item())
                if msg.get("vq_loss") is not None:
                    loss = loss + msg["vq_loss"]; upd("vq", msg["vq_loss"].item())
                with torch.no_grad():
                    acc_now = union_accuracy(logits.argmax(1), b["qout"]).mean().item()
                if msg["ent"] is not None: upd("ent", msg["ent"].mean().item())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(list(speaker.parameters()) + list(executor.parameters()), 1.0)
        scaler.step(opt); scaler.update()
        if speaker_ema is not None:
            with torch.no_grad():
                for pe, pc in zip(speaker_ema.parameters(), speaker.parameters()):
                    pe.mul_(args.stab_decay).add_(pc.detach(), alpha=1 - args.stab_decay)
        upd("loss_ex", loss_ex.item()); upd("acc", acc_now); upd("gn", float(gn))
        if args.len_cost > 0 and msg.get("length") is not None: upd("len", msg["length"].mean().item())
        if step % 50 == 0:
            el = time.time() - t_start
            if dev == "cuda": ema["memGB"] = torch.cuda.max_memory_allocated() / 1e9
            rec = dict(kind="train", step=step, sec=el, **{k: float(v) for k, v in ema.items()})
            emit(rec)
            print(f"[{step}] " + " ".join(f"{k} {v:.3f}" for k, v in ema.items()) + f"  ({el/max(1,step-step0):.2f}s/step)", flush=True)
        if step % args.eval_every == 0 and step > step0:
            evaluate(step, with_arc_eval=((step % args.arc_eval_every == 0) or step == args.steps) and bool(arc_eval_eps or monitor))
    if args.steps % args.eval_every != 0:
        evaluate(args.steps, with_arc_eval=bool(arc_eval_eps or monitor))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
