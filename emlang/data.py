"""Episode sampling: real ARC tasks (leave-one-out over demo pairs, dihedral+colour
augmentation) mixed with compositional synthetic tasks. Tensors are padded to a
30x30 canvas with PAD=10 (the PAD class encodes output shape)."""
import glob, json, os
import numpy as np
import torch
from .gen import CANVAS, PAD, SynthGen


def load_arc(split_dir):
    tasks = []
    for f in sorted(glob.glob(os.path.join(split_dir, "*.json"))):
        d = json.load(open(f))
        tr = [(np.array(p["input"]), np.array(p["output"])) for p in d["train"]]
        te = [(np.array(p["input"]), np.array(p["output"])) for p in d["test"] if "output" in p]
        ok = all(g.shape[0] <= CANVAS and g.shape[1] <= CANVAS for pr in tr + te for g in pr)
        if ok and len(tr) >= 2 and len(te) >= 1:
            tasks.append(dict(tid=os.path.basename(f)[:-5], train=tr, test=te))
    return tasks


def split_arc_training(tasks, n_heldout=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(tasks))
    held = set(idx[:n_heldout].tolist())
    return [t for i, t in enumerate(tasks) if i not in held], [t for i, t in enumerate(tasks) if i in held]


def dihedral(a, t):
    if t == 0: return a
    if t == 1: return np.rot90(a, 1)
    if t == 2: return np.rot90(a, 2)
    if t == 3: return np.rot90(a, 3)
    if t == 4: return np.fliplr(a)
    if t == 5: return np.flipud(a)
    if t == 6: return a.T
    return np.fliplr(np.rot90(a, 1))


def augment_episode(ep, rng, color_perm=True, dihedral_t=True):
    t = int(rng.integers(8)) if dihedral_t else 0
    perm = np.arange(10)
    if color_perm:
        perm[1:] = rng.permutation(np.arange(1, 10))
    f = lambda g: perm[dihedral(g, t)].astype(np.int64)
    out = dict(ep)
    out["demos"] = [(f(a), f(b)) for a, b in ep["demos"]]
    out["query"] = (f(ep["query"][0]), f(ep["query"][1]))
    if "inter" in ep and ep["inter"] is not None:
        out["inter"] = [f(g) for g in ep["inter"]]
    return out


def arc_episode(task, rng, max_demos=4):
    """Leave-one-out: one train pair becomes the query, the rest (capped) are demos."""
    pairs = task["train"]
    q = int(rng.integers(len(pairs)))
    demos = [p for i, p in enumerate(pairs) if i != q]
    if len(demos) > max_demos:
        sel = rng.choice(len(demos), max_demos, replace=False)
        demos = [demos[i] for i in sorted(sel)]
    return dict(demos=demos, query=pairs[q], prog=None, inter=None, source="arc",
                split="arc_train", tid=task["tid"])


def arc_test_episodes(task, max_demos=5):
    """Real ARC question: demos = train pairs, query = each test pair."""
    demos = task["train"][:max_demos]
    return [dict(demos=demos, query=tp, prog=None, inter=None, source="arc",
                 split="arc_test", tid=task["tid"]) for tp in task["test"]]


def pad_grid(g):
    out = np.full((CANVAS, CANVAS), PAD, dtype=np.int64)
    h, w = g.shape
    out[:h, :w] = g
    return out


def collate(episodes, max_demos=5):
    B = len(episodes); K = max_demos
    din = np.full((B, K, CANVAS, CANVAS), PAD, np.int64)
    dout = np.full((B, K, CANVAS, CANVAS), PAD, np.int64)
    dmask = np.zeros((B, K), np.bool_)
    qin = np.zeros((B, CANVAS, CANVAS), np.int64); qout = np.zeros((B, CANVAS, CANVAS), np.int64)
    for b, ep in enumerate(episodes):
        for k, (a, o) in enumerate(ep["demos"][:K]):
            din[b, k] = pad_grid(a); dout[b, k] = pad_grid(o); dmask[b, k] = True
        qin[b] = pad_grid(ep["query"][0]); qout[b] = pad_grid(ep["query"][1])
    return dict(din=torch.from_numpy(din), dout=torch.from_numpy(dout), dmask=torch.from_numpy(dmask),
                qin=torch.from_numpy(qin), qout=torch.from_numpy(qout))


class Mixer:
    """Training stream: p_synth synthetic episodes, else real ARC leave-one-out."""

    def __init__(self, arc_train_tasks, synth: SynthGen, p_synth=0.5, seed=0, augment=True, stage=None):
        self.tasks = arc_train_tasks; self.synth = synth; self.p = p_synth
        self.rng = np.random.default_rng(seed); self.augment = augment
        self.stage = stage  # None = original mixed split; int = curriculum stage pool

    def batch(self, B):
        eps = []
        for _ in range(B):
            if self.rng.random() < self.p or not self.tasks:
                ep = self.synth.make_episode(self.rng, self.stage if self.stage is not None else "train",
                                             n_demos=int(self.rng.integers(2, 5)))
            else:
                ep = arc_episode(self.tasks[int(self.rng.integers(len(self.tasks)))], self.rng)
            if self.augment:
                ep = augment_episode(ep, self.rng)
            eps.append(ep)
        return eps


def load_arc_kaggle(challenges_json, solutions_json=None):
    """Kaggle-format ARC files (dict task_id -> {train:[{input,output}], test:[{input}]}) + solutions."""
    ch = json.load(open(challenges_json)); sol = json.load(open(solutions_json)) if solutions_json else {}
    tasks = []
    for tid, d in sorted(ch.items()):
        tr = [(np.array(p["input"]), np.array(p["output"])) for p in d["train"]]
        te = []
        for i, t in enumerate(d["test"]):
            out = t.get("output") if "output" in t else (sol.get(tid, [None] * (i + 1))[i] if tid in sol else None)
            if out is not None: te.append((np.array(t["input"]), np.array(out)))
        ok = all(g.shape[0] <= CANVAS and g.shape[1] <= CANVAS for pr in tr + te for g in pr)
        if ok and len(tr) >= 2 and te:
            tasks.append(dict(tid=tid, train=tr, test=te))
    return tasks
