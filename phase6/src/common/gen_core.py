"""
Phase 6 (abstractive summarization) machinery shared by all four architectures:
source selection, tokenized arrays, adapters, the decoder-only summarizer wrapper, decoding with length control and
redundancy control, ROUGE, and the submission writer. Each architecture's lm.py supplies the model-specific pieces.
"""
import os
import json
import math
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = -100


# ------------------------------------------------------------------------------------------- data
def load_split(path, with_summary):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    ids = [r["id"] for r in rows]
    docs = [r["document"] for r in rows]
    sums = [r["summary"] for r in rows] if with_summary else None
    return ids, docs, sums


_tok = None
_cfg = None


def _init_worker(cfg_dict):
    global _tok, _cfg
    import lm as LM
    _tok = LM.load_tokenizer()
    _cfg = cfg_dict


def select_source(ids, budget, head_frac, n_spread):
    """Keep the opening of the judgment (parties, facts) and evenly spaced pieces of the rest, ending at the last
    token - so the whole case is represented even when it is 50 times longer than the budget."""
    if len(ids) <= budget:
        return ids
    head = int(budget * head_frac)
    piece = (budget - head) // n_spread
    rest = len(ids) - head
    starts = [head + round(i * (rest - piece) / max(1, n_spread - 1)) for i in range(n_spread)]
    out = ids[:head]
    for s in starts:
        out = out + ids[s:s + piece]
    return out


def encode_example(pair):
    doc, summ = pair
    src = select_source(_tok.encode(doc[:_cfg["char_cap"]]), _cfg["src_len"], _cfg["head_frac"], _cfg["n_spread"])
    tgt = _tok.encode(summ)[:_cfg["tgt_len"]] if summ is not None else []
    return src, tgt, len(doc.split()), (len(summ.split()) if summ is not None else 0)


def prepare_split(docs, sums, cfg_dict, workers):
    pairs = list(zip(docs, sums if sums is not None else [None] * len(docs)))
    with Pool(workers, initializer=_init_worker, initargs=(cfg_dict,)) as pool:
        res = pool.map(encode_example, pairs, chunksize=16)
    S, T = cfg_dict["src_len"], cfg_dict["tgt_len"]
    src = np.full((len(res), S), cfg_dict["pad"], dtype=np.int32)
    tgt = np.full((len(res), T), cfg_dict["pad"], dtype=np.int32)
    src_len = np.zeros(len(res), dtype=np.int32)
    tgt_len = np.zeros(len(res), dtype=np.int32)
    doc_words = np.zeros(len(res), dtype=np.int32)
    sum_words = np.zeros(len(res), dtype=np.int32)
    for i, (s, t, dw, sw) in enumerate(res):
        src[i, :len(s)], src_len[i] = s, len(s)
        tgt[i, :len(t)], tgt_len[i] = t, len(t)
        doc_words[i], sum_words[i] = dw, sw
    return {"src": src, "src_len": src_len, "tgt": tgt, "tgt_len": tgt_len, "doc_words": doc_words, "sum_words": sum_words}


# ----------------------------------------------------------------------------------- adaptation
class LoRALinear(nn.Module):
    """Hu et al. 2021: W x + (alpha / r) B A x, B = 0 at the start."""

    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.A), self.B) * self.scale


class LoRMALinear(nn.Module):
    """Bihany, Patel & Modi 2025 (LoRMA+): h = (alpha/r B A + I) W0 x, computed as W0x -> A -> B."""

    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.out_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.linear(x, self.base.weight)
        h = h + F.linear(F.linear(self.drop(h), self.A), self.B) * self.scale
        return h + self.base.bias if self.base.bias is not None else h


class Adapter(nn.Module):
    """Bottleneck adapter on a sub-layer's output: h + up(gelu(down(h))), up = 0 at the start.
    Houlsby et al. 2019 put one after attention and one after the feed-forward block;
    Pfeiffer et al. 2021 keep only the feed-forward one."""

    def __init__(self, base, bottleneck):
        super().__init__()
        self.base = base
        d = base.out_features
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        h = self.base(x)
        return h + self.up(F.gelu(self.down(h)))


def wrap_linears(root, names, make, skip=("router",)):
    n = 0
    for parent in list(root.modules()):
        for name, child in list(parent.named_children()):
            if name in skip:
                continue
            if name in names and isinstance(child, nn.Linear):
                setattr(parent, name, make(child))
                n += 1
    return n


def configure_method(model, method, cfg):
    if method in ("full", "kd"):
        for p in model.parameters():
            p.requires_grad = True
        return "all weights trainable"
    for p in model.parameters():
        p.requires_grad = False
    if method == "lora":
        n = wrap_linears(model, cfg.ADAPTER_TARGETS, lambda m: LoRALinear(m, cfg.ADAPTER_RANK, cfg.LORA_ALPHA, cfg.ADAPTER_DROPOUT))
        return f"LoRA on {n} attention projections (r={cfg.ADAPTER_RANK})"
    if method == "lorma":
        n = wrap_linears(model, cfg.ADAPTER_TARGETS, lambda m: LoRMALinear(m, cfg.ADAPTER_RANK, cfg.LORMA_ALPHA, cfg.ADAPTER_DROPOUT))
        return f"LoRMA on {n} attention projections (r={cfg.ADAPTER_RANK})"
    if method == "houlsby":
        n = wrap_linears(model, cfg.HOULSBY_SITES, lambda m: Adapter(m, cfg.ADAPTER_BOTTLENECK))
        return f"{n} Houlsby adapters (after attention and feed-forward outputs, size {cfg.ADAPTER_BOTTLENECK})"
    if method == "pfeiffer":
        n = wrap_linears(model, cfg.PFEIFFER_SITES, lambda m: Adapter(m, cfg.ADAPTER_BOTTLENECK))
        return f"{n} Pfeiffer adapters (after the feed-forward output only, size {cfg.ADAPTER_BOTTLENECK})"
    if method == "prompt":
        # Lester et al. 2021: learned virtual tokens in front of the source, initialised from real token embeddings
        g = torch.Generator().manual_seed(cfg.SEED)
        w = model.tok_emb.weight
        pick = torch.randint(100, w.size(0) - 100, (cfg.PROMPT_LEN,), generator=g)
        model.prompt = nn.Parameter(w[pick].detach().clone())
        return f"{cfg.PROMPT_LEN} soft-prompt vectors in front of every source"
    raise ValueError(method)


def count_params(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().float().half() for k, v in model.state_dict().items() if k in names}


def build_optimizer(model, method, cfg):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "prompt":
            lr = cfg.PROMPT_LR
        elif method in ("full", "kd"):
            lr = cfg.FULL_LR
        else:
            lr = cfg.ADAPTER_LR
        wd = 0.0 if (p.ndim < 2 or name == "prompt") else cfg.WEIGHT_DECAY
        groups.setdefault((lr, wd), []).append(p)
    return torch.optim.AdamW([{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()],
                             betas=(0.9, 0.98), eps=1e-6)


# ------------------------------------------------------------------- decoder-only summarizer
def decoder_batch(src, src_len, tgt, tgt_len, sep_id, pad_id, eos_id):
    """Right-padded [source] <sep> [summary] (<eos> if the summary was not cut) and next-token labels that are
    only set where the next token belongs to the summary."""
    rows, labels = [], []
    for s, sl, t, tl in zip(src, src_len, tgt, tgt_len):
        summary = list(t[:tl]) + ([eos_id] if tl < len(t) else [])
        seq = list(s[:sl]) + [sep_id] + summary
        lab = [IGNORE] * sl + summary + [IGNORE]
        rows.append(seq)
        labels.append(lab[:len(seq)])
    L = max(map(len, rows))
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    lab = torch.full((len(rows), L), IGNORE, dtype=torch.long)
    for i, (r, l) in enumerate(zip(rows, labels)):
        ids[i, :len(r)] = torch.tensor(r)
        lab[i, :len(l)] = torch.tensor(l)
    return ids, lab


class DecoderLM(nn.Module):
    """Wraps a Phase-1 decoder-only model. Subclasses (lm.py) provide add_positions() and run_blocks()."""
    is_encoder_decoder = False

    def __init__(self):
        super().__init__()
        self.prompt = None

    def _embed(self, ids, valid):
        x = self.tok_emb(ids)
        if self.prompt is not None:
            P = self.prompt.size(0)
            x = torch.cat([self.prompt.to(x.dtype).unsqueeze(0).expand(x.size(0), -1, -1), x], dim=1)
            valid = torch.cat([valid.new_ones(valid.size(0), P), valid], dim=1)
        pos = (valid.long().cumsum(1) - 1).clamp(min=0)
        return x, valid, pos

    def teacher_forced(self, batch, device):
        ids, labels = batch
        ids, labels = ids.to(device), labels.to(device)
        x, valid, pos = self._embed(ids, torch.ones_like(ids, dtype=torch.bool))
        x = self.add_positions(x, pos)
        h, _ = self.run_blocks(x, pos, None, None)
        h = self.final_norm(h)[:, h.size(1) - ids.size(1):]
        keep = labels != IGNORE
        logits = F.linear(h[keep], self.tok_emb.weight)
        return logits.float(), labels[keep]

    def make_batch(self, src, src_len, tgt, tgt_len, tok):
        return decoder_batch(src, src_len, tgt, tgt_len, tok.sep_id, tok.pad_id, tok.eos_id)

    @torch.no_grad()
    def generate(self, sources, min_new, max_new, tok, cfg, device):
        """sources: list of token lists. Left-padded batch; every row stops on its own length budget."""
        B = len(sources)
        rows = [list(s) + [tok.sep_id] for s in sources]
        L = max(map(len, rows))
        ids = torch.full((B, L), tok.pad_id, dtype=torch.long)
        valid = torch.zeros((B, L), dtype=torch.bool)
        for i, r in enumerate(rows):
            ids[i, L - len(r):] = torch.tensor(r)
            valid[i, L - len(r):] = True
        ids, valid = ids.to(device), valid.to(device)
        x, valid, pos = self._embed(ids, valid)
        x = self.add_positions(x, pos)
        S = x.size(1)
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
        mask = (causal[None, None] & valid[:, None, None, :]) | torch.eye(S, dtype=torch.bool, device=device)[None, None]
        h, caches = self.run_blocks(x, pos, None, mask)
        logits = F.linear(self.final_norm(h[:, -1:]), self.tok_emb.weight)[:, -1].float()
        next_pos = pos[:, -1] + 1

        state = DecodeState(B, logits.size(-1), min_new, max_new, tok, cfg, device)
        while not state.done():
            nxt = state.step(logits)
            valid = torch.cat([valid, torch.ones((B, 1), dtype=torch.bool, device=device)], dim=1)
            x = self.add_positions(self.tok_emb(nxt[:, None]), next_pos[:, None])
            h, caches = self.run_blocks(x, next_pos[:, None], caches, valid[:, None, None, :])
            logits = F.linear(self.final_norm(h), self.tok_emb.weight)[:, -1].float()
            next_pos = next_pos + 1
        return state.outputs()


# ---------------------------------------------------------------------------------- decoding
class DecodeState:
    """Greedy decoding with a length budget per row, a repetition penalty (Keskar et al. 2019) and no repeated
    n-grams - the redundancy controls the summaries need."""

    def __init__(self, B, V, min_new, max_new, tok, cfg, device):
        self.B, self.tok, self.cfg = B, tok, cfg
        self.min_new = torch.tensor(min_new, device=device)
        self.max_new = torch.tensor(max_new, device=device)
        self.seen = torch.zeros((B, V), dtype=torch.bool, device=device)
        self.out = [[] for _ in range(B)]
        self.ngrams = [dict() for _ in range(B)]
        self.finished = torch.zeros(B, dtype=torch.bool, device=device)
        self.t = 0

    def done(self):
        return bool(self.finished.all())

    def step(self, logits):
        cfg, tok = self.cfg, self.tok
        logits = logits.clone()
        logits[:, tok.pad_id] = float("-inf")
        for name in ("unk_id", "bos_id"):                # arch 1's word vocabulary has these; decoding would drop them
            if hasattr(tok, name):
                logits[:, getattr(tok, name)] = float("-inf")
        logits[self.t < self.min_new, tok.eos_id] = float("-inf")
        if cfg.REPETITION_PENALTY != 1.0:
            pen = torch.where(logits > 0, logits / cfg.REPETITION_PENALTY, logits * cfg.REPETITION_PENALTY)
            logits = torch.where(self.seen, pen, logits)
        n = cfg.NO_REPEAT_NGRAM
        if n > 1:
            for i in range(self.B):
                if not self.finished[i] and len(self.out[i]) >= n - 1:
                    banned = self.ngrams[i].get(tuple(self.out[i][-(n - 1):]))
                    if banned:
                        logits[i, list(banned)] = float("-inf")
        nxt = logits.argmax(-1)
        nxt = torch.where(self.finished, torch.full_like(nxt, tok.pad_id), nxt)
        for i, t in enumerate(nxt.tolist()):
            if self.finished[i]:
                continue
            seq = self.out[i]
            seq.append(t)
            if n > 1 and len(seq) >= n:
                self.ngrams[i].setdefault(tuple(seq[-n:-1]), set()).add(seq[-1])
        self.seen[torch.arange(self.B), nxt] = True
        self.t += 1
        self.finished |= (nxt == tok.eos_id) | (self.t >= self.max_new)
        return nxt

    def outputs(self):
        return [[t for t in seq if t not in (self.tok.eos_id, self.tok.pad_id)] for seq in self.out]


def length_budget(doc_words, cfg, tokens_per_word, ceiling):
    """Summary length follows document length (the training median ratio), clipped to sensible bounds."""
    words = np.clip(np.asarray(doc_words) / cfg.LENGTH_RATIO, cfg.MIN_SUMMARY_WORDS, cfg.MAX_SUMMARY_WORDS)
    target = words * tokens_per_word
    max_new = np.minimum(np.ceil(target * (1 + cfg.LENGTH_SLACK)), ceiling).astype(int)
    min_new = np.minimum(np.floor(target * (1 - cfg.LENGTH_SLACK)), max_new - 1).astype(int)
    return min_new.tolist(), max_new.tolist()


def tidy(text, sentence_case):
    text = " ".join(text.split())
    if sentence_case:                                    # arch 1 generates lower-case words
        out, cap = [], True
        for w in text.split(" "):
            out.append(w[:1].upper() + w[1:] if cap and w[:1].isalpha() else w)
            cap = w.endswith((".", "?", "!"))
        text = " ".join(out)
    return text


# ------------------------------------------------------------------------------------------ ROUGE
def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    if not r or not h:
        return 0.0, 0.0
    from collections import Counter
    overlap = sum((Counter(r) & Counter(h)).values())
    p1, r1 = overlap / len(h), overlap / len(r)
    f1 = 2 * p1 * r1 / (p1 + r1) if p1 + r1 else 0.0
    lcs = _lcs(r, h)
    pl, rl = lcs / len(h), lcs / len(r)
    fl = 2 * pl * rl / (pl + rl) if pl + rl else 0.0
    return f1, fl


def write_submission(ids, summaries, out_dir, zip_name):
    import zipfile
    assert len(ids) == len(summaries) == len(set(map(str, ids)))
    payload = [{"id": i, "summary": s if s.strip() else "."} for i, s in zip(ids, summaries)]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "prediction.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    zip_path = os.path.join(out_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname="prediction.json")
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"]
    return path, zip_path
