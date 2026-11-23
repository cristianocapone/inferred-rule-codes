"""Paper 2: the speaker-executor game played by a pretrained language model over a symbolic channel.

Same library, splits and diagnostics as paper 1 (emlang/gen.py, emlang/analyze.py). Grids are text
(rows of digits separated by '/'). The speaker sees the worked examples and must emit exactly L
tokens from an alphabet of V non-linguistic symbols (existing single tokens, vocabulary masked at
generation); the executor sees the message and the query input and writes the output grid.
Speaker: REINFORCE with a group baseline and an entropy floor, as in paper 1. Executor: cross-entropy
on the output text. Reward: teacher-forced token accuracy of the executor on the output grid (a dense
proxy for cell accuracy); true exact match is measured by greedy generation at evaluation.
Two LoRA adapters on one frozen base model, one per role."""
import argparse, json, math, os, re, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emlang.gen import SynthGen
from emlang import analyze as A

CANDIDATES = list("αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ") + list("¤¶§†‡•‰€£¥©®°±µ¼½¾¿×÷") + \
             list("←↑→↓↔⇐⇒∀∂∃∅∇∈∉∏∑√∞∠∧∨∩∪∫≈≠≡≤≥⊂⊃⊕⊗⊥⌂■□▲△▼▽◆◇○●★☆♠♣♥♦")


def grid_txt(g):
    return "/".join("".join(str(int(v)) for v in row) for row in np.asarray(g))


def parse_grid(s):
    s = s.strip().split()[0] if s.strip() else ""
    rows = [r for r in s.split("/") if r]
    if not rows or any(not re.fullmatch(r"[0-9]+", r) for r in rows): return None
    w = len(rows[0])
    if any(len(r) != w for r in rows): return None
    return np.array([[int(c) for c in r] for r in rows])


def cell_acc(pred, tgt):
    """union accuracy: over the union of the two shapes, fraction of cells equal (paper 1's measure)"""
    if pred is None: return 0.0
    H = max(pred.shape[0], tgt.shape[0]); W = max(pred.shape[1], tgt.shape[1])
    P = np.full((H, W), -1); T = np.full((H, W), -2); P[:pred.shape[0], :pred.shape[1]] = pred; T[:tgt.shape[0], :tgt.shape[1]] = tgt
    return float((P == T).mean())


def speaker_prompt(ep, channel="symbols"):
    ex = "\n".join(f"{grid_txt(a)} -> {grid_txt(b)}" for a, b in ep["demos"])
    if channel == "english":
        return f"Worked examples of a grid transformation:\n{ex}\nDescribe the rule in one short English sentence:"
    return f"Worked examples of a grid transformation:\n{ex}\nName the rule with {L} symbols:"


GLOSS = {"flipH": "flip the grid left to right", "flipV": "flip the grid top to bottom", "rot90": "rotate the grid 90 degrees clockwise",
         "rot180": "rotate the grid 180 degrees", "rot270": "rotate the grid 90 degrees anticlockwise", "transpose": "transpose the grid",
         "majority": "fill the grid with its most frequent colour", "upscale2": "scale the grid up by a factor of two",
         "crop": "crop the grid to its non-empty part", "mirrorH": "mirror the grid to the right", "mirrorV": "mirror the grid downwards",
         "outline": "keep only the outline of each shape"}


def gloss(name):
    """a plain English description of one operation, for the oracle description channel"""
    if name in GLOSS: return GLOSS[name]
    if name.startswith("gravity_"): return f"move everything {name.split('_')[1]}"
    if name.startswith("recolor"): a, b = name[len("recolor"):].split(">"); return f"repaint every {a} as {b}"
    if name.startswith("swap"): return f"swap the colours {name[4]} and {name[5]}"
    if name.startswith("border"): return f"draw a border of colour {name[len('border'):]} around the grid"
    if name.startswith("shift"):
        dy, dx = int(name[5:7]), int(name[7:9])
        d = ("down" if dy > 0 else "up") if dy else ("right" if dx > 0 else "left")
        return f"shift the grid one cell {d}"
    return name


def oracle_desc(ep, names):
    return ", then ".join(gloss(names[i]) for i in ep["prog"])


def executor_prompt(msg_text, ep, desc=None):
    """desc: an English description of the rule, added as a second channel (channel "both")"""
    d = "" if desc is None else f"Description: {desc}\n"
    return f"Rule: {msg_text}\n{d}Input: {grid_txt(ep['query'][0])}\nOutput:"


def pick_alphabet(tok, V):
    ids = []
    for ch in CANDIDATES:
        t = tok.encode(ch, add_special_tokens=False)
        if len(t) == 1 and t[0] not in ids: ids.append(t[0])
        if len(ids) == V: break
    assert len(ids) == V, f"only {len(ids)} single-token symbols available"
    return ids


OUT_IDS = None   # token ids the executor may emit: digits 0-9, '/', eos (set in main)
CHANNEL = "symbols"; T_ENG = 40
SKIP_LEADING_WS = False   # english generation: a newline ends the message only after some non-whitespace token
DESC_MODE = "own"         # channel "both" at evaluation: own / other-episode / no description
DESC_LESION = (None, "language_units", "zero")   # ablate these units while the description is generated only
_WS_MASK = {}


def whitespace_mask(tok, dev):
    """bool [vocab]: tokens that decode to whitespace only (newlines, spaces)"""
    if dev not in _WS_MASK:
        _WS_MASK[dev] = torch.tensor([tok.decode([i]).strip() == "" for i in range(len(tok))], device=dev)
    return _WS_MASK[dev]


def output_ids(tok):
    ids = [tok.encode(str(d), add_special_tokens=False) for d in range(10)] + [tok.encode("/", add_special_tokens=False)]
    assert all(len(t) == 1 for t in ids), "digits and '/' must be single tokens"
    return [t[0] for t in ids] + [tok.eos_token_id]


def hidden_states(model, ids, am, n_keep):
    """last-layer hidden states of the last n_keep positions (no full-vocabulary logits)"""
    body = model.get_base_model().model
    h = body(input_ids=ids, attention_mask=am).last_hidden_state
    return h[:, -n_keep:, :]


def restricted_logits(model, h, out_ids):
    W = model.get_base_model().lm_head.weight[out_ids]        # [K, d]
    return (h.float() @ W.float().t())                        # [B, n, K]


def install_lesion(model, units_json, which="language_units", roles_note="", mode="zero"):
    """ablate the listed (layer, dim) units of the residual stream at each decoder layer's output, for every
    forward pass of this model (AlKhamissi et al. 2025 style). mode "zero" sets them to 0; mode "mean" sets them to
    their mean activation over the localiser stimuli (json "unit_means"), which removes the information a unit
    carries without shifting the residual stream by its magnitude. Returns the hook handles."""
    spec = json.load(open(units_json)); units = spec[which]
    assert mode in ("zero", "mean"), mode
    means = torch.tensor(spec["unit_means"]) if mode == "mean" else None
    by_layer = {}
    for l, d in units: by_layer.setdefault(l, []).append(d)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = base.model.layers; handles = []
    for l, dims in by_layer.items():
        idx = torch.tensor(sorted(dims)); val = means[l, idx] if means is not None else None
        def hook(mod, inp, out, idx=idx, val=val):
            h = out[0] if isinstance(out, tuple) else out
            h[:, :, idx.to(h.device)] = 0 if val is None else val.to(h.device, h.dtype)
            return out
        handles.append(layers[l].register_forward_hook(hook))
    print(f"[lesion] {which}: {len(units)} units {'zeroed' if mode == 'zero' else 'mean-ablated'} across {len(by_layer)} layers{roles_note}", flush=True)
    return handles


def load_roles(path, dev, r=16, grad_ckpt=True, roles_wanted=("speaker", "executor")):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dt = torch.float16 if dev.startswith("cuda") else torch.float32
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    roles = {}
    for role in roles_wanted:
        base = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt).to(dev)
        if grad_ckpt and dev.startswith("cuda"):
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m = get_peft_model(base, cfg); m.print_trainable_parameters(); roles[role] = m
    return tok, roles


@torch.no_grad()
def _pad_batch(tok, texts, dev):
    b = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return b["input_ids"].to(dev), b["attention_mask"].to(dev)


def speak(model, tok, prompts, alphabet, dev, greedy=False, G=1, temp=1.0):
    """sample G messages of L symbols per prompt (no gradient), then score the sampled sequence in one
    forward pass with gradient: log-prob of each symbol, entropy and probabilities of the restricted
    distribution at each of the L positions. Returns tokens [B*G, L], logp [B*G], entropy [B*G]."""
    ids0, am0 = _pad_batch(tok, prompts, dev)
    ids = ids0.repeat_interleave(G, 0); am = am0.repeat_interleave(G, 0)
    alpha = torch.tensor(alphabet, device=dev)
    with torch.no_grad():
        toks = []
        for t in range(L):
            lg = model(input_ids=ids, attention_mask=am, logits_to_keep=1).logits[:, -1, :].float()[:, alpha] / temp
            lg = torch.nan_to_num(lg, nan=0.0, posinf=30.0, neginf=-30.0)     # fp16 overflow guard: a NaN row would trip multinomial's device assert
            lp = F.log_softmax(lg, -1)
            k = lp.argmax(-1) if greedy else torch.multinomial(lp.exp(), 1).squeeze(-1)
            toks.append(k); ids = torch.cat([ids, alpha[k][:, None]], 1); am = torch.cat([am, torch.ones_like(k)[:, None]], 1)
        toks = torch.stack(toks, 1)
    if not torch.is_grad_enabled():
        speak.last_probs = None
        return toks, torch.zeros(len(toks), device=dev), torch.zeros(len(toks), device=dev)
    lg = model(input_ids=ids, attention_mask=am, logits_to_keep=L + 1).logits[:, :-1, :][:, :, alpha].float() / temp   # positions predicting the L symbols
    lg = torch.nan_to_num(lg, nan=0.0, posinf=30.0, neginf=-30.0)
    lp = F.log_softmax(lg, -1)                                                                        # [B*G, L, V]
    logp = lp.gather(2, toks[:, :, None]).squeeze(2).sum(1)
    ent = -(lp.exp() * lp).sum(-1).mean(1)
    speak.last_probs = lp.exp()
    return toks, logp, ent


def speak_english(model, tok, prompts, dev, G=1, T=40, greedy=False, temp=1.0):
    """free-text message of at most T tokens from the full vocabulary (the LARC game). Sampling with the
    KV cache under no_grad, then one scoring pass with gradient over the sampled tokens.
    Returns token ids [B*G, T] (padded with eos after the stop), logp [B*G], entropy [B*G], texts."""
    ids0, am0 = _pad_batch(tok, prompts, dev)
    ids = ids0.repeat_interleave(G, 0); am = am0.repeat_interleave(G, 0)
    eos = tok.eos_token_id; nl = tok.encode("\n", add_special_tokens=False)[0]
    body = model.get_base_model().model; head = model.get_base_model().lm_head
    was_training = model.training; model.eval()          # the KV cache is disabled under gradient checkpointing (train mode)
    with torch.no_grad():
        o = body(input_ids=ids, attention_mask=am, use_cache=True); past = o.past_key_values; h = o.last_hidden_state[:, -1, :]
        done = torch.zeros(len(ids), dtype=torch.bool, device=dev); toks = []
        started = torch.zeros(len(ids), dtype=torch.bool, device=dev); ws = whitespace_mask(tok, dev) if SKIP_LEADING_WS else None
        for t in range(T):
            lg = torch.nan_to_num(head(h).float() / temp, nan=0.0, posinf=30.0, neginf=-30.0)
            k = lg.argmax(-1) if greedy else torch.multinomial(F.softmax(lg, -1), 1).squeeze(-1)
            k = torch.where(done, torch.full_like(k, eos), k); toks.append(k)
            if ws is None: done |= (k == eos) | (k == nl)
            else: done |= (k == eos) | ((k == nl) & started); started |= ~ws[k]
            if bool(done.all()) and t < T - 1:
                toks += [torch.full_like(k, eos)] * (T - 1 - t); break
            nxt = k[:, None]; am = torch.cat([am, torch.ones_like(nxt)], 1)
            o = body(input_ids=nxt, attention_mask=am, past_key_values=past, use_cache=True); past = o.past_key_values; h = o.last_hidden_state[:, -1, :]
        toks = torch.stack(toks, 1)                                                  # [B*G, T]
    if was_training: model.train()
    full = torch.cat([ids, toks], 1); amf = torch.cat([am0.repeat_interleave(G, 0), torch.ones_like(toks)], 1)
    texts = [tok.decode([int(x) for x in row if int(x) not in (eos, nl)]).strip() for row in toks]
    if not torch.is_grad_enabled():
        speak_english.last_probs = None
        return toks, torch.zeros(len(toks), device=dev), torch.zeros(len(toks), device=dev), texts
    hs = body(input_ids=full, attention_mask=amf).last_hidden_state[:, -T - 1:-1, :]                    # positions predicting the T tokens
    lp = F.log_softmax(torch.nan_to_num(head(hs).float() / temp, nan=0.0, posinf=30.0, neginf=-30.0), -1)   # [B*G, T, Vfull]
    valid = torch.ones_like(toks, dtype=torch.bool)
    stop = ((toks == eos) | (toks == nl)).float().cumsum(1); valid = (stop - ((toks == eos) | (toks == nl)).float()) < 1   # up to and including the stop token
    logp = (lp.gather(2, toks[:, :, None]).squeeze(2) * valid).sum(1)
    ent = (-(lp.exp() * lp).sum(-1) * valid).sum(1) / valid.sum(1).clamp(min=1)
    speak_english.last_probs = None; speak_english.last_valid = valid.float()
    return toks, logp, ent, texts


@torch.no_grad()
def describe(model, tok, eps, dev, T=40, bs=16, lesion=None, lesion_set="language_units", lesion_mode="zero", trained=False):
    """the pretrained model's English description of each episode: the speaker's LoRA is disabled, so this is the
    frozen-speaker channel of the English arm, whatever the speaker has learned for the symbolic one.
    lesion: ablate these units during the description only (an aphasic describer for an intact symbolic speaker)."""
    handles = install_lesion(model, lesion, lesion_set, " (describer)", mode=lesion_mode) if lesion else []
    out = []
    try:
        import contextlib
        with (contextlib.nullcontext() if trained else model.disable_adapter()):
            for i in range(0, len(eps), bs):
                _, _, _, texts = speak_english(model, tok, [speaker_prompt(e, "english") for e in eps[i:i + bs]], dev, greedy=True, T=T)
                out += texts
    finally:
        for h in handles: h.remove()
    return out


@torch.no_grad()
def ref_logp(model, tok, prompts, toks, dev):
    """token log-probs of the sampled description under the pretrained model (adapters disabled): the KL anchor
    that keeps a reward-trained describer writing English instead of paper 1's degenerate code"""
    eos = tok.eos_token_id; T = toks.shape[1]
    ids0, am0 = _pad_batch(tok, prompts, dev)
    full = torch.cat([ids0, toks], 1); amf = torch.cat([am0, torch.ones_like(toks)], 1)
    with model.disable_adapter():
        body = model.get_base_model().model; head = model.get_base_model().lm_head
        hs = body(input_ids=full, attention_mask=amf).last_hidden_state[:, -T - 1:-1, :]
        lp = F.log_softmax(torch.nan_to_num(head(hs).float(), nan=0.0, posinf=30.0, neginf=-30.0), -1)
    return lp.gather(2, toks[:, :, None]).squeeze(2)                                   # [B, T]


def describer_step(roles, tok, eps, alphabet, dev, a, opt, scaler):
    """one REINFORCE step on the describer: the symbolic speaker and the executor are frozen, so the reward is
    exactly what the English description ADDS to an existing, working symbolic channel. A KL anchor to the
    pretrained model keeps the description English."""
    with torch.autocast(device_type="cuda" if dev.startswith("cuda") else "cpu", dtype=torch.float16, enabled=dev.startswith("cuda"), cache_enabled=False):
        with torch.no_grad():                                                   # the symbolic message: frozen speaker, greedy
            roles["speaker"].eval()
            toks_s, _, _ = speak(roles["speaker"], tok, [speaker_prompt(e) for e in eps], alphabet, dev, greedy=True)
            msgs = [msg_text(t, tok, alphabet) for t in toks_s]
        prompts = [speaker_prompt(e, "english") for e in eps]
        roles["describer"].train()
        toks_d, logp, ent, texts = speak_english(roles["describer"], tok, prompts, dev, G=a.G, T=a.T)
        eps_g = [e for e in eps for _ in range(a.G)]; msgs_g = [m for m in msgs for _ in range(a.G)]
        ctrl = None if a.desc_reward == "plain" else ([""] * len(texts) if a.desc_reward == "none" else [texts[(j + a.G) % len(texts)] for j in range(len(texts))])
        with torch.no_grad():
            roles["executor"].eval()
            ids, am, lb = executor_batch(tok, [executor_prompt(m, e, d) for m, e, d in zip(msgs_g, eps_g, texts)], [grid_txt(e["query"][1]) for e in eps_g], dev)
            _, r_own = executor_forward(roles["executor"], ids, am, lb)
            if ctrl is None: r = r_own
            else:
                ids, am, lb = executor_batch(tok, [executor_prompt(m, e, d) for m, e, d in zip(msgs_g, eps_g, ctrl)], [grid_txt(e["query"][1]) for e in eps_g], dev)
                _, r_ctrl = executor_forward(roles["executor"], ids, am, lb)
                r = r_own - r_ctrl                                              # the added value of this description
        rp = ref_logp(roles["describer"], tok, [p for p in prompts for _ in range(a.G)], toks_d, dev)
    v = speak_english.last_valid; nv = v.sum(1).clamp(min=1)
    kl_i = (logp.detach() - (rp * v).sum(1)) / nv                               # per-sample sampled-token KL to the pretrained model
    r = (r.float() - a.kl_w * kl_i).view(len(eps), a.G)                         # KL in the reward, RLHF style: as a loss term its
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + a.adv_eps)).reshape(-1)   # gradient pushes away from the samples
    kl = kl_i.mean()
    loss = -(adv.detach() * logp / nv).mean() + a.ent_floor_w * F.relu(a.ent_floor - ent.mean())
    if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True); return dict(reward=float(r.mean()), skipped=1.0)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in roles["describer"].parameters() if p.requires_grad], 1.0)
    scaler.step(opt); scaler.update()
    return dict(reward=float(r.mean()), r_own=float(r_own.mean()), kl=float(kl), ent=float(ent.mean()), words=float(np.mean([len(t.split()) for t in texts])))


def message_mi(p):
    """H(marginal symbol distribution) - mean H(per-episode distribution), per position, averaged:
    positive when messages depend on the episode, zero when every episode gets the same distribution"""
    marg = p.mean(0)                                                   # [L, V]
    h_marg = -(marg * (marg + 1e-9).log()).sum(-1).mean()
    h_cond = -(p * (p + 1e-9).log()).sum(-1).mean()
    return h_marg - h_cond


def executor_batch(tok, prompts, targets, dev):
    """teacher-forced batch: input = prompt + target + eos; labels = target + eos only (left-padded)"""
    full = [p + " " + t + tok.eos_token for p, t in zip(prompts, targets)]
    ids, am = _pad_batch(tok, full, dev)
    labels = ids.clone(); labels[am == 0] = -100
    for i, p in enumerate(prompts):
        n_p = len(tok.encode(p, add_special_tokens=False)); n_full = int(am[i].sum())
        labels[i, : ids.shape[1] - n_full + n_p] = -100                      # mask the prompt tokens
    return ids, am, labels


def executor_forward(model, ids, am, labels):
    """cross-entropy over the executor's 12-token output alphabet, at the target positions only"""
    n_keep = int((labels != -100).sum(1).max()) + 1
    h = hidden_states(model, ids, am, n_keep)[:, :-1, :]                 # positions predicting the last n_keep-1 tokens
    lg = torch.nan_to_num(restricted_logits(model, h, OUT_IDS), nan=0.0, posinf=30.0, neginf=-30.0)   # [B, n, K]
    lb = labels[:, -(n_keep - 1):]
    remap = torch.full((int(max(OUT_IDS)) + 1,), -100, device=lb.device, dtype=torch.long)
    remap[torch.tensor(OUT_IDS, device=lb.device)] = torch.arange(len(OUT_IDS), device=lb.device)
    lbk = torch.where(lb == -100, lb, remap[lb.clamp(min=0)])           # labels as indices into the output alphabet
    loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lbk.reshape(-1), ignore_index=-100)
    with torch.no_grad():
        ok = (lg.argmax(-1) == lbk) & (lbk != -100); n = (lbk != -100).sum(1).clamp(min=1)
        tok_acc = ok.sum(1).float() / n                                       # per-example reward proxy
    return loss, tok_acc


@torch.no_grad()
def execute_greedy(model, tok, prompts, dev, max_new=64):
    """greedy decoding restricted to the output alphabet, with the model's KV cache"""
    ids, am = _pad_batch(tok, prompts, dev)
    out_t = torch.tensor(OUT_IDS, device=dev); eos_k = len(OUT_IDS) - 1
    body = model.get_base_model().model; W = model.get_base_model().lm_head.weight[out_t].float()
    o = body(input_ids=ids, attention_mask=am, use_cache=True)
    past = o.past_key_values; h = o.last_hidden_state[:, -1, :]
    done = torch.zeros(len(prompts), dtype=torch.bool, device=dev); seqs = []
    for _ in range(max_new):
        k = (h.float() @ W.t()).argmax(-1)
        k = torch.where(done, torch.full_like(k, eos_k), k); done |= (k == eos_k); seqs.append(k)
        if bool(done.all()): break
        nxt = out_t[k][:, None]; am = torch.cat([am, torch.ones_like(nxt)], 1)
        o = body(input_ids=nxt, attention_mask=am, past_key_values=past, use_cache=True)
        past = o.past_key_values; h = o.last_hidden_state[:, -1, :]
    S = torch.stack(seqs, 1).tolist()
    texts = []
    for row in S:
        toks = [OUT_IDS[k] for k in row if k != eos_k]
        texts.append(tok.decode(toks))
    return texts


def msg_text(tokens, tok, alphabet):
    return "".join(tok.decode([alphabet[int(k)]]) for k in tokens)


@torch.no_grad()
def evaluate(roles, tok, G_syn, split, n, alphabet, dev, bs=16, eps=None, keep=False):
    """eps: evaluate these episodes instead of the split's fixed set. keep: also return per-episode outcomes and
    message texts (the same episodes under every lesion condition, for paired bootstrap comparisons)"""
    if eps is None: eps = G_syn.fixed_eval_set(split, n, seed=11 if split != "hold2" else 12)
    ex, acc, msgs, texts_all, descs_all = [], [], [], [], []
    ex_const, ex_other, acc_const, acc_other = [], [], [], []          # channel-bypass controls (a pretrained executor may solve from the input alone)
    const_msg = "".join(tok.decode([alphabet[0]]) for _ in range(L)) if CHANNEL != "english" else "the rule"
    for i in range(0, len(eps), bs):
        sub = eps[i:i + bs]
        if CHANNEL == "english":
            toks, _, _, texts = speak_english(roles["speaker"], tok, [speaker_prompt(e, "english") for e in sub], dev, greedy=True, T=T_ENG)
        else:
            toks, _, _ = speak(roles["speaker"], tok, [speaker_prompt(e) for e in sub], alphabet, dev, greedy=True)
            texts = [msg_text(t, tok, alphabet) for t in toks]
        if CHANNEL == "both":
            dsc = describe(roles.get("describer", roles["speaker"]), tok, sub, dev, T=T_ENG, lesion=DESC_LESION[0],
                           lesion_set=DESC_LESION[1], lesion_mode=DESC_LESION[2], trained="describer" in roles)
            if DESC_MODE == "none": dsc = ["" for _ in dsc]
            elif DESC_MODE == "other": dsc = [dsc[(j + 1) % len(dsc)] for j in range(len(dsc))]
            elif DESC_MODE == "oracle": dsc = [oracle_desc(e, G_syn.names) for e in sub]
            elif DESC_MODE == "scramble":                       # same words, order destroyed: lexical cue or language?
                rs = np.random.default_rng(1234 + i)
                dsc = [" ".join(rs.permutation(d.split()).tolist()) for d in dsc]
        else:
            dsc = [None] * len(sub)
        outs = execute_greedy(roles["executor"], tok, [executor_prompt(m, e, d) for m, e, d in zip(texts, sub, dsc)], dev)
        outs_c = execute_greedy(roles["executor"], tok, [executor_prompt(const_msg, e, d) for e, d in zip(sub, dsc)], dev)
        outs_o = execute_greedy(roles["executor"], tok, [executor_prompt(texts[(j + 1) % len(sub)], e, d) for j, (e, d) in enumerate(zip(sub, dsc))], dev)
        texts_all += texts; descs_all += dsc
        for t, e, o, oc, oo in zip(toks, sub, outs, outs_c, outs_o):
            tgt = np.asarray(e["query"][1])
            for o_, ex_l, acc_l in ((o, ex, acc), (oc, ex_const, acc_const), (oo, ex_other, acc_other)):
                pred = parse_grid(o_)
                ex_l.append(float(pred is not None and pred.shape == tgt.shape and (pred == tgt).all())); acc_l.append(cell_acc(pred, tgt))
            msgs.append(t.tolist())
    progs = [e["prog"] for e in eps]
    m = dict(exact=float(np.mean(ex)), acc=float(np.mean(acc)), n=len(eps),
             exact_const_msg=float(np.mean(ex_const)), acc_const_msg=float(np.mean(acc_const)),
             exact_other_msg=float(np.mean(ex_other)), acc_other_msg=float(np.mean(acc_other)),
             identity_acc=float(np.mean([cell_acc(np.asarray(e["query"][0]), np.asarray(e["query"][1])) for e in eps])))
    st = A.message_stats(msgs, V if CHANNEL != "english" else len(tok)); m.update({f"msg_{k}": v for k, v in st.items()})
    if CHANNEL == "english": m["examples"] = [tok.decode([int(x) for x in mm if int(x) != tok.eos_token_id]).strip() for mm in msgs[:5]]
    m["msg_n_rules"] = len({tuple(p) for p in progs}); m["msg_unique_per_rule"] = len({tuple(x) for x in msgs}) / max(m["msg_n_rules"], 1)
    if split == "train1":
        by = {}
        for p, mm in zip(progs, msgs): by.setdefault(p[0], []).append(tuple(mm))
        from collections import Counter
        m["align_purity"] = float(np.mean([Counter(v).most_common(1)[0][1] / len(v) for v in by.values()]))
    try: m["topsim"] = A.topsim(progs, msgs)
    except Exception: pass
    if keep:
        m["per_episode"] = dict(prog=[list(map(int, p)) for p in progs], exact=ex, acc=acc, exact_const_msg=ex_const, acc_const_msg=acc_const,
                                exact_other_msg=ex_other, acc_other_msg=acc_other, message=texts_all,
                                **({"description": descs_all} if CHANNEL == "both" else {}))
    return m


def subset_summary(per, keep_prog):
    """means of the per-episode outcomes over the episodes whose program satisfies keep_prog"""
    idx = [i for i, p in enumerate(per["prog"]) if keep_prog(tuple(p))]
    out = {k: float(np.mean([per[k][i] for i in idx])) if idx else None
           for k in ("exact", "acc", "exact_const_msg", "acc_const_msg", "exact_other_msg", "acc_other_msg")}
    out.update(n=len(idx), n_programs=len({tuple(per["prog"][i]) for i in idx}))
    return out


def main():
    global L, V
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--L", type=int, default=8); ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--sp_lr_mult", type=float, default=0.3)
    ap.add_argument("--ent_floor", type=float, default=0.3); ap.add_argument("--ent_floor_w", type=float, default=1.0); ap.add_argument("--adv_eps", type=float, default=0.02)
    ap.add_argument("--freeze_speaker", action="store_true", help="do not train the speaker: it writes greedily with the pretrained model (the natural-language control; use --G 1)")
    ap.add_argument("--lesion", default=None, help="units json from llm_localizer.py: ablate these units in both roles, for training and evaluation")
    ap.add_argument("--lesion_set", default="language_units", help="key of the unit set in the --lesion json")
    ap.add_argument("--lesion_mode", choices=["zero", "mean"], default="zero")
    ap.add_argument("--audit_only", action="store_true", help="eval_only: skip the fixed train1/hold2/len3 sets, evaluate only the audited held-out programs")
    ap.add_argument("--skip_leading_newlines", action="store_true", help="english: do not end a message on newlines before its first word (fixes 'mute' descriptions)")
    ap.add_argument("--lesion_roles", default="speaker,executor", help="roles the lesion is installed in (speaker-only / executor-only decompose where a channel breaks)")
    ap.add_argument("--eval_only", action="store_true", help="skip training: evaluate the adapters in --out (with --resume) and exit")
    ap.add_argument("--audit", default=None, help="eval_only: split audit json (emlang/audit_split.py) for the held-out pairs; evaluates every held-out program --per_prog times and reports all / functionally novel / equivalent")
    ap.add_argument("--audit_len3", default=None, help="eval_only: the same for the held-out triples")
    ap.add_argument("--per_prog", type=int, default=2); ap.add_argument("--eval_tag", default=None, help="eval_only: name of the output json (default eval_<lesion set or intact>[_<roles>])")
    ap.add_argument("--channel", choices=["symbols", "english", "both"], default="symbols", help="symbols: L tokens from a V-symbol alphabet; english: up to --T free tokens; both: the symbols plus the pretrained model's English description of the same episode")
    ap.add_argument("--desc_mode", choices=["own", "other", "none", "scramble", "oracle"], default="own", help="channel both: the episode's own description, another episode's, none, its own words scrambled, or the ground-truth rule in English (the ceiling of what a description could add)")
    ap.add_argument("--desc_lesion", default=None, help="channel both: ablate units (json) while generating the description only — an aphasic describer with an intact symbolic speaker")
    ap.add_argument("--desc_lesion_set", default="language_units"); ap.add_argument("--desc_lesion_mode", choices=["zero", "mean"], default="zero")
    ap.add_argument("--describer", action="store_true", help="the run has a trained describer adapter (evaluation of an induced-amplifier run)")
    ap.add_argument("--train_describer", action="store_true", help="channel both: train a third adapter, the English describer, by REINFORCE on what its description ADDS to a frozen symbolic speaker and a frozen executor (the induced-amplifier arm)")
    ap.add_argument("--desc_reward", choices=["other", "none", "plain"], default="other", help="describer reward: executor accuracy with its description minus the same episode with another episode's description / with none / no baseline")
    ap.add_argument("--kl_w", type=float, default=0.05, help="weight of the KL anchor to the pretrained model's English (0 = free rein, and the text degenerates)")
    ap.add_argument("--desc_lr", type=float, default=5e-5)
    ap.add_argument("--init_from", default=None, help="load lora_speaker / lora_executor from this run directory before training (the describer starts at the pretrained model)")
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--div_w", type=float, default=0.0, help="weight of the information term (paper 1's anti-collapse pressure)")
    ap.add_argument("--curriculum", default="1000,2000,2000", help="steps of singles; singles+pairs+repeats; +triples (joint: 0,0,N)")
    ap.add_argument("--library", default="A"); ap.add_argument("--n_prims", type=int, default=0); ap.add_argument("--prim_seed", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=500); ap.add_argument("--eval_n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default=None); ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--no_grad_ckpt", action="store_true", help="disable activation checkpointing (more memory, faster)")
    ap.add_argument("--save_every", type=int, default=250); ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max_prompt_tokens", type=int, default=420, help="resample episodes whose speaker prompt is longer (memory spikes from large grids)")
    ap.add_argument("--resume", action="store_true", help="load lora_speaker/lora_executor from --out and continue from the last evaluated step (fresh optimiser)")
    ap.add_argument("--resume_tag", default="latest", help="which saved adapters to load: latest, best (highest seen-singles accuracy), or stepN — fp16 runs can degenerate late while staying finite")
    a = ap.parse_args(); L, V = a.L, a.V
    dev = a.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True); logf = open(os.path.join(a.out, "log.jsonl"), "a")
    want = ("speaker", "executor") + (("describer",) if (a.train_describer or a.describer) else ())
    tok, roles = load_roles(a.model, dev, a.lora_r, grad_ckpt=not a.no_grad_ckpt, roles_wanted=want); alphabet = pick_alphabet(tok, V)
    global OUT_IDS, CHANNEL, T_ENG, SKIP_LEADING_WS, DESC_MODE, DESC_LESION
    OUT_IDS = output_ids(tok); CHANNEL = a.channel; T_ENG = a.T; SKIP_LEADING_WS = a.skip_leading_newlines
    DESC_MODE = a.desc_mode; DESC_LESION = (a.desc_lesion, a.desc_lesion_set, a.desc_lesion_mode)
    lesion_roles = [r for r in a.lesion_roles.split(",") if r]
    assert set(lesion_roles) <= set(roles), f"unknown role in --lesion_roles {a.lesion_roles}"
    if a.lesion:
        for role, m in roles.items():
            if role in lesion_roles: install_lesion(m, a.lesion, a.lesion_set, f" ({role})", mode=a.lesion_mode)
    start = 1
    if a.resume:
        evs = [json.loads(l) for l in open(os.path.join(a.out, "log.jsonl"), errors="ignore") if '"kind": "eval"' in l]
        start = (evs[-1]["step"] if evs else 0) + 1
        print(f"[resume] continuing from step {start}", flush=True)
    print("[alphabet]", "".join(tok.decode([i]) for i in alphabet), flush=True)
    G_syn = SynthGen(0, size_lo=3, size_hi=6, n_train3=600, library=a.library, n_prims=a.n_prims, prim_seed=a.prim_seed)
    stages = [int(x) for x in a.curriculum.split(",")]; assert sum(stages) == a.steps, "curriculum budgets must sum to --steps"
    opt_ex = torch.optim.AdamW([p for p in roles["executor"].parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    opt_sp = torch.optim.AdamW([p for p in roles["speaker"].parameters() if p.requires_grad], lr=a.lr * a.sp_lr_mult, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=dev.startswith("cuda"))
    ema = {}; t0 = time.time()

    def sample_episode(step):
        stage = 0 if step < stages[0] else (1 if step < stages[0] + stages[1] else 2)
        if stage == 0 or (stage >= 1 and rng.random() < 0.3): split = "train1"
        elif stage == 1 or rng.random() < 0.6: split = "train2" if rng.random() < 0.9 else "rep2_train"
        else: split = "train3"
        for _ in range(20):
            e = G_syn.make_episode(rng, split, n_demos=3)
            if len(tok.encode(speaker_prompt(e), add_special_tokens=False)) <= a.max_prompt_tokens: return e
        return e

    def params_finite():
        return all(torch.isfinite(p).all() for m in roles.values() for p in m.parameters() if p.requires_grad)

    def save_adapters(tag="latest"):
        for role, m in roles.items(): m.save_pretrained(os.path.join(a.out, f"lora_{role}" if tag == "latest" else f"lora_{role}_{tag}"))

    def load_from(run_dir, which, tag="latest"):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        for role in which:
            d = os.path.join(run_dir, f"lora_{role}" if tag == "latest" else f"lora_{role}_{tag}")
            set_peft_model_state_dict(roles[role], load_file(os.path.join(d, "adapter_model.safetensors"), device=dev))
        print(f"[init] loaded {list(which)} from {run_dir} ({tag})", flush=True)

    def load_adapters(tag="latest"):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        for role in (roles if not a.eval_only or "describer" not in roles or os.path.exists(os.path.join(a.out, f"lora_describer")) else ("speaker", "executor")):
            d = os.path.join(a.out, f"lora_{role}" if tag == "latest" else f"lora_{role}_{tag}")
            set_peft_model_state_dict(roles[role], load_file(os.path.join(d, "adapter_model.safetensors"), device=dev))

    def set_lr(step):
        f = min(1.0, (step - start + 1) / max(1, a.warmup))
        for g in opt_ex.param_groups: g["lr"] = a.lr * f
        for g in opt_sp.param_groups: g["lr"] = a.lr * a.sp_lr_mult * f

    if a.init_from: load_from(a.init_from, ("speaker", "executor"), a.resume_tag if a.resume_tag != "latest" else "latest")
    if a.resume: load_adapters(a.resume_tag); print(f"[resume] adapters loaded ({a.resume_tag})", flush=True)
    if a.eval_only:
        roles["speaker"].eval(); roles["executor"].eval()
        rep = dict(kind="eval", step=start - 1, lesion=a.lesion_set if a.lesion else None, lesion_roles=lesion_roles if a.lesion else None,
                   channel=a.channel, freeze_speaker=a.freeze_speaker, eval_n=a.eval_n, per_prog=a.per_prog,
                   lesion_mode=a.lesion_mode if a.lesion else None, lesion_json=a.lesion, skip_leading_newlines=a.skip_leading_newlines,
                   desc_mode=a.desc_mode if a.channel == "both" else None, desc_lesion=a.desc_lesion, desc_lesion_set=a.desc_lesion_set if a.desc_lesion else None)
        for split in (() if a.audit_only else ("train1", "hold2", "len3")):
            rep[f"synth_{split}"] = evaluate(roles, tok, G_syn, split, a.eval_n, alphabet, dev, keep=True)
        for key, path, split in (("audit_hold2", a.audit, "hold2"), ("audit_len3", a.audit_len3, "len3")):
            if not path: continue
            au = json.load(open(path)); novel = {tuple(p) for p in au["novel_progs"]}
            assert {tuple(r["prog"]) for r in au["rows"]} == {tuple(p) for p in getattr(G_syn, split)}, f"{path} does not match this generator's {split}"
            rng_a = np.random.default_rng(12 if split == "hold2" else 13)
            eps_a = [G_syn.make_episode(rng_a, split, n_demos=3, prog=list(r["prog"])) for r in au["rows"] for _ in range(a.per_prog)]
            m_a = evaluate(roles, tok, G_syn, split, len(eps_a), alphabet, dev, eps=eps_a, keep=True)
            rep[f"{key}_all"] = m_a
            rep[f"{key}_novel"] = subset_summary(m_a["per_episode"], lambda p: p in novel)
            rep[f"{key}_equivalent"] = subset_summary(m_a["per_episode"], lambda p: p not in novel)
        tag = a.eval_tag or (f"eval_{a.lesion_set if a.lesion else 'intact'}" + (f"_{'+'.join(lesion_roles)}" if a.lesion and set(lesion_roles) != set(roles) else ""))
        json.dump(rep, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
        if not a.audit_only:
            print(f"[{tag}] " + " | ".join(f"{s_}: ex {rep[s_]['exact']:.3f} (const {rep[s_]['exact_const_msg']:.3f}) acc {rep[s_]['acc']:.3f} (const {rep[s_]['acc_const_msg']:.3f})" for s_ in ("synth_train1", "synth_hold2", "synth_len3")), flush=True)
        for key in ("audit_hold2", "audit_len3"):
            if f"{key}_novel" in rep:
                s_ = rep[f"{key}_novel"]; q = rep[f"{key}_equivalent"]
                print(f"[{tag}] {key}: novel ex {s_['exact']:.3f} (const {s_['exact_const_msg']:.3f}, other {s_['exact_other_msg']:.3f}) acc {s_['acc']:.3f} (other {s_['acc_other_msg']:.3f}; n {s_['n']}) | equivalent ex {q['exact']:.3f} (n {q['n']})", flush=True)
        print("[done]", flush=True); return
    opt_desc = torch.optim.AdamW([p for p in roles["describer"].parameters() if p.requires_grad], lr=a.desc_lr, weight_decay=0.01) if a.train_describer else None
    skipped = 0; rollbacks = 0; last_good = start - 1; best_sel = -1.0
    for step in range(start, a.steps + 1):
        set_lr(step)
        eps = [sample_episode(step) for _ in range(a.bs)]
        if a.train_describer:
            stats = describer_step(roles, tok, eps, alphabet, dev, a, opt_desc, scaler)
            for k, v in stats.items(): ema[k] = v if k not in ema else 0.95 * ema[k] + 0.05 * v
            if step % 25 == 0 or step == 1:
                print(f"[{step}] " + " ".join(f"{k} {v:.3f}" for k, v in ema.items()) + f"  ({(time.time()-t0)/step:.2f}s/step)", flush=True)
            if step % a.save_every == 0 and params_finite(): save_adapters(); last_good = step
            if step % a.eval_every == 0 or step == a.steps:
                roles["describer"].eval(); rep = dict(kind="eval", step=step)
                for split in ("train1", "hold2"):
                    rep[f"synth_{split}"] = evaluate(roles, tok, G_syn, split, a.eval_n, alphabet, dev)
                logf.write(json.dumps(rep) + "\n"); logf.flush()
                print(f"[eval {step}] " + " | ".join(f"{s}: ex {rep[s]['exact']:.3f} (const {rep[s]['exact_const_msg']:.3f}, other {rep[s]['exact_other_msg']:.3f}) acc {rep[s]['acc']:.3f}" for s in ("synth_train1", "synth_hold2")), flush=True)
                if params_finite():
                    save_adapters(); save_adapters(f"step{step}")
                    sel = rep["synth_train1"]["acc"]
                    if sel > best_sel: best_sel = sel; save_adapters("best"); print(f"[eval {step}] new best (seen-singles acc {sel:.3f})", flush=True)
                roles["describer"].train()
            continue
        roles["speaker"].train(); roles["executor"].train()
        # cache_enabled=False: with the weight cache on, the fp16 copies cast inside the no-grad sampling loop are
        # reused by the scoring pass and carry no gradient (the speaker would never learn)
        with torch.autocast(device_type="cuda" if dev.startswith("cuda") else "cpu", dtype=torch.float16, enabled=dev.startswith("cuda"), cache_enabled=False):
            # --- speaker samples G messages per episode (with gradient on the log-probs)
            if a.channel in ("english",) and a.freeze_speaker:
                with torch.no_grad():
                    toks, logp, ent, texts = speak_english(roles["speaker"], tok, [speaker_prompt(e, "english") for e in eps], dev, G=a.G, T=a.T, greedy=True)
            elif a.channel == "english":
                toks, logp, ent, texts = speak_english(roles["speaker"], tok, [speaker_prompt(e, "english") for e in eps], dev, G=a.G, T=a.T)
            else:
                toks, logp, ent = speak(roles["speaker"], tok, [speaker_prompt(e) for e in eps], alphabet, dev, G=a.G)
                texts = [msg_text(t, tok, alphabet) for t in toks]
            # --- executor: teacher-forced loss and per-example token accuracy = reward
            eps_g = [e for e in eps for _ in range(a.G)]
            if a.channel != "both": descs = [None] * len(eps_g)
            elif a.desc_mode == "oracle": descs = [oracle_desc(e, G_syn.names) for e in eps_g]
            else: descs = [d for d in describe(roles.get("describer", roles["speaker"]), tok, eps, dev, T=a.T, trained="describer" in roles) for _ in range(a.G)]
            ids, am, labels = executor_batch(tok, [executor_prompt(m, e, d) for m, e, d in zip(texts, eps_g, descs)], [grid_txt(e["query"][1]) for e in eps_g], dev)
            loss_ex, r = executor_forward(roles["executor"], ids, am, labels)
        r = r.view(a.bs, a.G); adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + a.adv_eps)).reshape(-1)
        ent_mean = ent.mean()
        if a.freeze_speaker:
            loss_sp = torch.zeros((), device=dev)
        else:
            pg = -(adv.detach() * logp / (a.T if a.channel == "english" else L)).mean()
            loss_sp = pg + a.ent_floor_w * F.relu(a.ent_floor - ent_mean)
        if a.div_w > 0 and a.channel != "english":
            mi = message_mi(speak.last_probs); loss_sp = loss_sp - a.div_w * mi
        if not (torch.isfinite(loss_ex) and torch.isfinite(loss_sp)):          # fp16 overflow: drop the step, never the adapters
            skipped += 1; opt_ex.zero_grad(set_to_none=True); opt_sp.zero_grad(set_to_none=True)
            if skipped % 10 == 1: print(f"[{step}] non-finite loss, step skipped ({skipped} so far)", flush=True)
            if skipped > 200: raise RuntimeError("too many non-finite steps: adapters are probably corrupted")
            continue
        opt_ex.zero_grad(set_to_none=True); opt_sp.zero_grad(set_to_none=True)
        scaler.scale(loss_ex + loss_sp).backward()
        scaler.unscale_(opt_ex)
        if not a.freeze_speaker: scaler.unscale_(opt_sp)
        torch.nn.utils.clip_grad_norm_([p for m in roles.values() for p in m.parameters() if p.requires_grad], 1.0)
        scaler.step(opt_ex)
        if not a.freeze_speaker: scaler.step(opt_sp)
        scaler.update()
        if step % 25 == 0 and not params_finite():
            rollbacks += 1; print(f"[{step}] non-finite adapter parameters: rolling back to step {last_good} ({rollbacks} rollbacks)", flush=True)
            if rollbacks > 5: raise RuntimeError("repeated divergence")
            load_adapters(); opt_ex.state.clear(); opt_sp.state.clear(); start = step   # fresh optimiser + warmup after rollback
            continue
        if step % a.save_every == 0 and params_finite():
            save_adapters(); last_good = step
        for k, v in dict(reward=r.mean().item(), rstd=r.std(1).mean().item(), ent=ent_mean.item(), loss_ex=loss_ex.item(), **({"mi": mi.item()} if a.div_w > 0 else {})).items():
            ema[k] = v if k not in ema else 0.95 * ema[k] + 0.05 * v
        if step % 25 == 0 or step == 1:
            print(f"[{step}] " + " ".join(f"{k} {v:.3f}" for k, v in ema.items()) + f"  ({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            roles["speaker"].eval(); roles["executor"].eval()
            rep = dict(kind="eval", step=step)
            for split in ("train1", "hold2", "len3"):
                rep[f"synth_{split}"] = evaluate(roles, tok, G_syn, split, a.eval_n, alphabet, dev)
            logf.write(json.dumps(rep) + "\n"); logf.flush()
            print(f"[eval {step}] " + " | ".join(f"{s}: ex {rep[s]['exact']:.3f} (const {rep[s]['exact_const_msg']:.3f}, other {rep[s]['exact_other_msg']:.3f}) acc {rep[s]['acc']:.3f} (const {rep[s]['acc_const_msg']:.3f}, identity {rep[s]['identity_acc']:.3f}) uniq/rule {rep[s]['msg_unique_per_rule']:.2f} ent {rep[s]['msg_ent_symbols']:.2f}" + (f" purity {rep[s]['align_purity']:.2f}" if 'align_purity' in rep[s] else "") for s in ("synth_train1", "synth_hold2", "synth_len3")), flush=True)
            if params_finite():
                save_adapters(); save_adapters(f"step{step}"); last_good = step
                sel = rep["synth_train1"]["acc"]                     # selection on the seen singles only, never on a held-out split
                if sel > best_sel: best_sel = sel; save_adapters("best"); print(f"[eval {step}] new best (seen-singles acc {sel:.3f})", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
