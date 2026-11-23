"""RE-ARC (Hodel 2024) as an episode source: 400 procedural generators, one per ARC-AGI-1
training task, unlimited fresh examples. Tasks are split by key into train / held-out (unseen
rules). Programs are task keys (no compositional structure), so the language readouts here are
naming consistency (purity), transfer to unseen tasks, and real ARC-1 evaluation."""
import os, sys, random
import numpy as np
from .gen import CANVAS

REARC_DIR = os.environ.get("REARC_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external_rules", "re-arc_real"))


class REARC:
    def __init__(self, seed=0, n_heldout=80, diff=(0.0, 1.0), max_tries=20, n_train_rules=None):
        sys.path.insert(0, REARC_DIR)
        import generators  # noqa
        self.gens = {n[9:]: getattr(generators, n) for n in dir(generators) if n.startswith("generate_")}
        keys = sorted(self.gens); rng = np.random.default_rng(seed); rng.shuffle(keys)
        self.hold = sorted(keys[:n_heldout]); self.train = sorted(keys[n_heldout:])
        if n_train_rules is not None:               # naming-capacity scaling: use a nested subset of rules
            self.train = self.train[:n_train_rules]
        self.keys = keys; self.diff = diff; self.max_tries = max_tries
        self.split_sizes = dict(train=len(self.train), hold=len(self.hold))

    def _sample(self, key, rng):
        for _ in range(self.max_tries):
            try:
                random.seed(int(rng.integers(2**31)))   # RE-ARC generators use python's random
                d = self.gens[key](*self.diff)
                a, b = np.array(d["input"], dtype=np.int64), np.array(d["output"], dtype=np.int64)
                if a.ndim == 2 and b.ndim == 2 and max(a.shape + b.shape) <= CANVAS and min(a.shape + b.shape) >= 1:
                    return a, b
            except Exception:
                continue
        return None

    def make_episode(self, rng, split="train", n_demos=3, key=None):
        pool = self.train if split == "train" else self.hold
        for _ in range(50):
            k = key or pool[int(rng.integers(len(pool)))]
            pairs = [self._sample(k, rng) for _ in range(n_demos + 1)]
            if all(p is not None for p in pairs):
                return dict(demos=pairs[:n_demos], query=pairs[n_demos], prog=[self.keys.index(k)], inter=None,
                            source="rearc", split=split, tid="rearc_" + k)
            key = None
        raise RuntimeError("RE-ARC sampling failed repeatedly")

    def fixed_eval_set(self, split, n, seed=123, n_demos=3):
        rng = np.random.default_rng(seed); pool = self.train if split == "train" else self.hold
        return [self.make_episode(rng, split, n_demos, key=pool[i % len(pool)]) for i in range(n)]


if __name__ == "__main__":
    import time
    R = REARC(); print("split", R.split_sizes)
    rng = np.random.default_rng(0); t0 = time.time(); shapes = []
    for i in range(20):
        e = R.make_episode(rng, "train"); shapes.append(e["query"][0].shape)
    print(f"20 episodes in {time.time()-t0:.1f}s; example {e['tid']} shapes {shapes[:6]}")
