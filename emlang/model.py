"""Speaker (demos -> discrete message) and Executor (message x test input -> output).
The two agents share nothing but the channel: separate token embeddings.
Global mixing inside grids uses axial (row/column) attention, so flips/rotations/
gravity are expressible in a few FiLM-conditioned steps."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .gen import CANVAS, PAD

N_COLORS = 11  # 0-9 + PAD


class GridEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.col = nn.Embedding(N_COLORS, d)
        self.pos = nn.Parameter(torch.randn(1, d, CANVAS, CANVAS) * 0.02)

    def forward(self, g):  # [B,H,W] -> [B,d,H,W]
        return self.col(g).permute(0, 3, 1, 2) + self.pos


class FiLM(nn.Module):
    def __init__(self, d_cond, d):
        super().__init__()
        self.lin = nn.Linear(d_cond, 2 * d)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x, c):  # x [B,d,H,W], c [B,d_cond]
        g, b = self.lin(c).chunk(2, dim=-1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]


class AxialAttn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads; self.d = d
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)

    def _attn(self, x):  # x [N, T, d]
        N, T, _ = x.shape
        q, k, v = self.qkv(x).view(N, T, 3, self.h, self.d // self.h).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [N,h,T,dh]
        y = F.scaled_dot_product_attention(q, k, v)
        return self.o(y.transpose(1, 2).reshape(N, T, self.d))

    def forward(self, x, axis):  # x [B,d,H,W]
        B, d, H, W = x.shape
        if axis == "row":
            t = x.permute(0, 2, 3, 1).reshape(B * H, W, d)
            y = self._attn(t).view(B, H, W, d).permute(0, 3, 1, 2)
        else:
            t = x.permute(0, 3, 2, 1).reshape(B * W, H, d)
            y = self._attn(t).view(B, W, H, d).permute(0, 3, 2, 1)
        return y


class GridBlock(nn.Module):
    """FiLM-conditioned: conv3x3 -> row attn -> col attn -> FFN, each residual."""

    def __init__(self, d, d_cond, heads=4):
        super().__init__()
        self.n1 = nn.GroupNorm(8, d); self.f1 = FiLM(d_cond, d)
        self.conv = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GELU(), nn.Conv2d(d, d, 3, padding=1))
        self.n2 = nn.GroupNorm(8, d); self.f2 = FiLM(d_cond, d); self.row = AxialAttn(d, heads)
        self.n3 = nn.GroupNorm(8, d); self.f3 = FiLM(d_cond, d); self.col = AxialAttn(d, heads)
        self.n4 = nn.GroupNorm(8, d); self.f4 = FiLM(d_cond, d)
        self.ffn = nn.Sequential(nn.Conv2d(d, 4 * d, 1), nn.GELU(), nn.Conv2d(4 * d, d, 1))

    def forward(self, x, c):
        x = x + self.conv(self.f1(self.n1(x), c))
        x = x + self.row(self.f2(self.n2(x), c), "row")
        x = x + self.col(self.f3(self.n3(x), c), "col")
        x = x + self.ffn(self.f4(self.n4(x), c))
        return x


class PairEncoder(nn.Module):
    """(input grid, output grid) -> vector. Output cells attend to input via shared canvas."""

    def __init__(self, d, n_blocks=2, heads=4):
        super().__init__()
        self.ein = GridEmbed(d); self.eout = GridEmbed(d)
        self.mix = nn.Conv2d(2 * d, d, 1)
        self.blocks = nn.ModuleList([GridBlock(d, d, heads) for _ in range(n_blocks)])
        self.null = nn.Parameter(torch.zeros(1, d))
        self.down = nn.Sequential(nn.Conv2d(d, d, 3, stride=2, padding=1), nn.GELU(),
                                  nn.Conv2d(d, d, 3, stride=2, padding=1), nn.GELU())
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, gin, gout):  # [N,H,W] each
        x = self.mix(torch.cat([self.ein(gin), self.eout(gout)], 1))
        c = self.null.expand(x.shape[0], -1)
        for b in self.blocks:
            x = b(x, c)
        x = self.down(x)  # [N,d,8,8]
        t = x.flatten(2).transpose(1, 2)
        y, _ = self.pool(self.q.expand(t.shape[0], -1, -1), t, t)
        return self.norm(y[:, 0])


class Speaker(nn.Module):
    """demos -> context -> autoregressive message of L symbols from a vocab of V.
    mode: 'rl' (sample, REINFORCE), 'gumbel' (straight-through), 'cont' (continuous vectors)."""

    def __init__(self, d, V, L, mode, n_ctx_layers=2, heads=4, tau=1.0, cont_dim=0, use_eos=False):
        super().__init__()
        self.d, self.V, self.L, self.mode, self.tau = d, V, L, mode, tau
        self.cont_dim = cont_dim or d
        self.use_eos = use_eos  # symbol 0 = EOS: message ends at first EOS (later symbols masked to a null op)
        self.min_len = 1        # EOS forbidden at positions < min_len
        self.pair = PairEncoder(d, heads=heads)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.ctx = nn.TransformerEncoder(layer, n_ctx_layers)
        self.h0 = nn.Linear(d, d)
        self.cell = nn.GRUCell(d, d)
        self.bos = nn.Parameter(torch.zeros(1, d))
        self.tok_in = nn.Embedding(V, d)      # speaker-side embedding of its own symbols
        self.head = nn.Linear(d, V if mode not in ("cont", "vq") else self.cont_dim)
        self.cont_in = nn.Linear(self.cont_dim, d) if (mode in ("cont", "vq") and self.cont_dim != d) else nn.Identity()
        if mode == "vq":  # shared codebook: quantised vectors are what crosses the channel
            self.codebook = nn.Embedding(V, self.cont_dim); nn.init.normal_(self.codebook.weight, std=0.5)

    def context(self, din, dout, dmask):  # [B,K,H,W]
        B, K = dmask.shape
        v = self.pair(din.flatten(0, 1), dout.flatten(0, 1)).view(B, K, self.d)
        v = self.ctx(v, src_key_padding_mask=~dmask)
        m = dmask.float().unsqueeze(-1)
        return (v * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, din, dout, dmask, greedy=False, tau=None, eos_explore=0.0):
        return self.decode(self.context(din, dout, dmask), greedy=greedy, tau=tau, eos_explore=eos_explore)

    def score(self, c, tokens, tau=None):
        """teacher-forced per-position distributions of the rl decoder for given tokens: [B,L,V] log-probs"""
        tau = tau or self.tau
        h = torch.tanh(self.h0(c)); inp = self.bos.expand(c.shape[0], -1); out = []
        for i in range(self.L):
            h = self.cell(inp, h)
            out.append(F.log_softmax(self.head(h) / tau, -1))
            inp = self.tok_in(tokens[:, i])
        return torch.stack(out, 1)

    def decode(self, c, greedy=False, tau=None, eos_explore=0.0):
        """c: context [B,d]. Returns dict: tokens [B,L] (None for cont), cont [B,L,d] (cont arm),
        soft [B,L,V] (gumbel straight-through), logp [B] summed log-prob of sampled tokens, ent [B]."""
        tau = tau or self.tau
        h = torch.tanh(self.h0(c))
        B = c.shape[0]
        inp = self.bos.expand(B, -1)
        toks, logps, ents, softs, conts, p_eos, probs = [], [], [], [], [], [], []
        for i in range(self.L):
            h = self.cell(inp, h)
            out = self.head(h)
            if self.mode == "cont":
                conts.append(out); inp = self.cont_in(out)
                continue
            if self.mode == "vq":
                z = out
                d2 = (z.pow(2).sum(1, keepdim=True) - 2 * z @ self.codebook.weight.t() + self.codebook.weight.pow(2).sum(1)[None])
                t = d2.argmin(1); q = self.codebook(t)
                zq = z + (q - z).detach()                    # straight-through
                conts.append(zq); toks.append(t)
                vq_loss = F.mse_loss(q, z.detach()) + 0.25 * F.mse_loss(z, q.detach())
                ents.append(vq_loss.expand(z.shape[0]))      # carried in 'ent' slot: mean over positions = vq loss
                inp = self.cont_in(zq)
                continue
            logits = out / tau
            if self.use_eos and i < self.min_len:  # EOS forbidden before min_len symbols
                logits = logits.clone(); logits[:, 0] = -1e4
            elif self.use_eos and eos_explore > 0 and not greedy:
                # exploration: sometimes forbid EOS so the executor gets to see longer messages
                force = (torch.rand(logits.shape[0], device=logits.device) < eos_explore)
                logits = logits.clone(); logits[force, 0] = -1e4
            logp_all = F.log_softmax(logits, -1)
            probs.append(logp_all.exp())
            if self.use_eos:
                p_eos.append(logp_all[:, 0].exp())
            ent = -(logp_all.exp() * logp_all).sum(-1)
            if self.mode == "rl":
                if greedy:
                    t = logits.argmax(-1)
                else:
                    t = torch.distributions.Categorical(logits=logits).sample()
                logps.append(logp_all.gather(1, t[:, None])[:, 0])
                inp = self.tok_in(t)
            else:  # gumbel straight-through
                if greedy:
                    t = logits.argmax(-1)
                    y = F.one_hot(t, self.V).float()
                else:
                    y = F.gumbel_softmax(logits, tau=1.0, hard=True)
                    t = y.argmax(-1)
                softs.append(y)
                inp = y @ self.tok_in.weight
            toks.append(t); ents.append(ent)
        if self.mode == "cont":
            return dict(tokens=None, cont=torch.stack(conts, 1), soft=None, logp=None, ent=None)
        if self.mode == "vq":
            return dict(tokens=torch.stack(toks, 1), cont=torch.stack(conts, 1), soft=None, logp=None,
                        ent=None, vq_loss=torch.stack(ents, 1).mean())
        res = dict(tokens=torch.stack(toks, 1), ent=torch.stack(ents, 1).mean(1))
        res["probs"] = torch.stack(probs, 1) if probs else None   # [B,L,V], for the anti-collapse term
        res["logp_pos"] = torch.stack(logps, 1) if logps else None      # [B,L] per-symbol log-prob (process-reward credit assignment)
        res["logp"] = torch.stack(logps, 1).sum(1) if logps else None
        res["soft"] = torch.stack(softs, 1) if softs else None
        res["cont"] = None
        if self.use_eos:
            T = res["tokens"]
            ended = (T == 0).cumsum(1) > 0                      # True from the first EOS on (inclusive)
            res["alive"] = ~ended                               # symbols that count
            res["length"] = res["alive"].sum(1).float()         # used symbols (before EOS)
            P = torch.stack(p_eos, 1)                           # [B,L] P(EOS at i)
            surv = torch.cumprod(1 - P, 1)                      # P(not ended by i)
            res["exp_length"] = surv.sum(1)                     # differentiable expected length
            if res["logp"] is not None:                         # RL: only symbols up to and incl. first EOS carry log-prob
                lp = torch.stack(logps, 1)
                keep = (~ended) | ((T == 0) & (ended.cumsum(1) == 1))
                res["logp"] = (lp * keep.float()).sum(1)
        return res


class Executor(nn.Module):
    """Applies the message symbol by symbol to the test grid state: a serial program."""

    def __init__(self, d, V, L, mode, R=2, heads=4, use_ckpt=True, step_emb=True, cont_dim=0, ex_d=0):
        super().__init__()
        self.cont_dim = cont_dim or d
        d = ex_d or d  # executor width may differ from the speaker's
        self.d, self.V, self.L, self.mode, self.R, self.use_ckpt = d, V, L, mode, R, use_ckpt
        self.use_step = step_emb
        self.null = nn.Parameter(torch.zeros(1, d))  # symbol vector used after EOS (learned no-op)
        self.embed = GridEmbed(d)
        self.tok = nn.Embedding(V, d) if mode not in ("cont", "vq") else nn.Linear(self.cont_dim, d)
        self.step = nn.Embedding(L, d)
        self.blocks = nn.ModuleList([GridBlock(d, d, heads) for _ in range(R)])
        self.out_norm = nn.GroupNorm(8, d)
        self.head = nn.Conv2d(d, N_COLORS, 1)

    def symbol_vectors(self, msg):
        if self.mode in ("cont", "vq"):
            return self.tok(msg["cont"])                      # [B,L,d]
        if msg["soft"] is not None:
            return msg["soft"] @ self.tok.weight                # straight-through
        return self.tok(msg["tokens"])                          # [B,L,d]

    def _step(self, x, c):
        for b in self.blocks:
            x = b(x, c)
        return x

    def forward(self, qin, msg, return_states=False):
        x = self.embed(qin)
        sv = self.symbol_vectors(msg)
        if msg.get("alive") is not None:
            a = msg["alive"].float().unsqueeze(-1)
            sv = sv * a + self.null[None] * (1 - a)
        states = []
        for i in range(self.L):
            c = sv[:, i] + (self.step.weight[i][None] if self.use_step else 0)
            if self.use_ckpt and self.training and torch.is_grad_enabled():
                x = checkpoint(self._step, x, c, use_reentrant=False)
            else:
                x = self._step(x, c)
            if return_states:
                states.append(self.head(self.out_norm(x)))
        logits = self.head(self.out_norm(x))  # [B,11,H,W]
        return (logits, states) if return_states else logits


def msg_from_tokens(tokens, use_eos, codebook=None):
    """Rebuild the executor-side message dict from integer symbols (probes / transmission).
    For the VQ arm the executor consumes codebook vectors, so pass speaker.codebook."""
    d = dict(tokens=tokens, soft=None, cont=(codebook(tokens) if codebook is not None else None))
    if use_eos:
        d["alive"] = ~((tokens == 0).cumsum(1) > 0)
        d["length"] = d["alive"].sum(1).float()
    return d


def union_accuracy(pred, target):
    """Per-sample accuracy over cells where target or prediction is non-PAD."""
    region = (target != PAD) | (pred != PAD)
    correct = (pred == target) & region
    return correct.flatten(1).sum(1).float() / region.flatten(1).sum(1).clamp(min=1).float()


def exact_match(pred, target):
    return (pred == target).flatten(1).all(1).float()


def shape_match(pred, target):
    return ((pred == PAD) == (target == PAD)).flatten(1).all(1).float()
