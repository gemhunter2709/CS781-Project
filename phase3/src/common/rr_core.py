"""
Phase 3 (rhetorical roles) machinery shared by all four architectures.

Model, bottom to top:
  1. the Phase-1 backbone reads packs of consecutive sentences (up to PACK_TOKENS), so every sentence is encoded
     next to its neighbours (Cohan et al. 2019), and each sentence is mean-pooled;
  2. relative position + domain (CL / IT) embeddings are added;
  3. a BiLSTM runs over the whole document's sentence vectors;
  4. a linear-chain CRF picks the label sequence (roles come in long blocks: 94% of CL and 75% of IT sentences
     share the previous sentence's label);
  5. an auxiliary head predicts whether the role changes at each sentence (label-shift prediction, Malik et al. 2022).
"""
import json
import math
import random
import zipfile
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute", "Dissent", "PrecedentReliedUpon",
          "PrecedentNotReliedUpon", "PrecedentOverruled", "RulingByLowerCourt", "RatioOfTheDecision",
          "RulingByPresentCourt", "None"]
LABEL_ID = {l: i for i, l in enumerate(LABELS)}
N_LABELS = len(LABELS)
DOMAINS = ["CL", "IT"]
N_POS_BUCKETS = 20


# --------------------------------------------------------------------------------------------- data
def load_split(path, domain):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    docs = []
    if isinstance(raw, dict):                                   # test files: {doc_id: [sentences]}
        raw = [{"id": k, "text": v} for k, v in raw.items()]
    for d in raw:
        labels = [LABEL_ID[l] for l in d["labels"]] if "labels" in d else None
        docs.append({"id": str(d["id"]), "domain": DOMAINS.index(domain), "sents": list(d["text"]), "labels": labels})
    return docs


def tokenize_docs(docs, tok, max_sent_tokens):
    """Every sentence becomes [separator] + its first max_sent_tokens tokens."""
    for d in docs:
        d["tok"] = [[tok.sep_id] + tok.encode(s)[:max_sent_tokens] for s in d["sents"]]
    return docs


def pack(sent_tokens, pack_tokens):
    """Greedily put consecutive sentences into sequences of at most pack_tokens tokens.
    Returns the packs and, for every sentence, (pack index, first token, last token + 1)."""
    packs, where, cur = [], [], []
    for toks in sent_tokens:
        if cur and len(cur) + len(toks) > pack_tokens:
            packs.append(cur)
            cur = []
        where.append((len(packs), len(cur), len(cur) + len(toks)))
        cur = cur + toks
    if cur:
        packs.append(cur)
    return packs, where


def batch_packs(packs, where, pad_id, device, one_per_sentence=False):
    if one_per_sentence:                                        # sentence-only ablation: no neighbours in view
        packs = [packs[p][a:b] for p, a, b in where]
        where = [(i, 0, b - a) for i, (_, a, b) in enumerate(where)]
    L = max(len(p) for p in packs)
    ids = torch.full((len(packs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(packs), L), dtype=torch.bool)
    sent_of_token = torch.full((len(packs), L), -1, dtype=torch.long)
    for i, p in enumerate(packs):
        ids[i, :len(p)] = torch.tensor(p)
        mask[i, :len(p)] = True
    for s, (p, a, b) in enumerate(where):
        sent_of_token[p, a:b] = s
    return ids.to(device), mask.to(device), sent_of_token.to(device)


def position_buckets(n):
    return torch.tensor([min(N_POS_BUCKETS - 1, int(N_POS_BUCKETS * i / n)) for i in range(n)], dtype=torch.long)


# ----------------------------------------------------------------------------------- adapters
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


class IA3Out(nn.Module):
    """Liu et al. 2022 (IA3): a learned vector rescales a projection's output (keys / values). Starts at 1."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.scale = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x):
        return self.base(x) * self.scale


class IA3In(nn.Module):
    """IA3 on the feed-forward block: rescale the hidden activation just before the down projection."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.scale = nn.Parameter(torch.ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.scale)


def wrap_linears(root, names, make):
    swapped = 0
    for parent in list(root.modules()):
        for name, child in list(parent.named_children()):
            if name in names and isinstance(child, nn.Linear):
                setattr(parent, name, make(child))
                swapped += 1
    return swapped


# ------------------------------------------------------------------------------------------ CRF
class CRF(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(n, n))           # trans[prev, cur]
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))

    def nll(self, em, tags):
        """Negative log-likelihood of one tag sequence. em: (T, n) emissions."""
        em = em.float()
        T = em.size(0)
        gold = self.start[tags[0]] + em[0, tags[0]] + self.end[tags[-1]]
        if T > 1:
            gold = gold + self.trans[tags[:-1], tags[1:]].sum() + em[torch.arange(1, T), tags[1:]].sum()
        alpha = self.start + em[0]
        for t in range(1, T):
            alpha = torch.logsumexp(alpha[:, None] + self.trans, dim=0) + em[t]
        return torch.logsumexp(alpha + self.end, dim=0) - gold

    @torch.no_grad()
    def viterbi(self, em):
        em = em.float().cpu().numpy()
        trans = self.trans.detach().float().cpu().numpy()
        score = self.start.detach().float().cpu().numpy() + em[0]
        back = np.zeros(em.shape, dtype=np.int64)
        for t in range(1, len(em)):
            cand = score[:, None] + trans
            back[t] = cand.argmax(0)
            score = cand.max(0) + em[t]
        score = score + self.end.detach().float().cpu().numpy()
        path = [int(score.argmax())]
        for t in range(len(em) - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return path[::-1]


# ---------------------------------------------------------------------------------------- model
class RRModel(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        H = cfg.CONTEXT_DIM
        self.proj = nn.Sequential(nn.Linear(backbone.d_model, H), nn.LayerNorm(H))
        self.pos_emb = nn.Embedding(N_POS_BUCKETS, H)
        self.dom_emb = nn.Embedding(len(DOMAINS), H)
        self.drop = nn.Dropout(cfg.HEAD_DROPOUT)
        self.use_lstm = cfg.DOC_CONTEXT == "bilstm"
        self.lstm = nn.LSTM(H, H // 2, num_layers=2, bidirectional=True, batch_first=True, dropout=cfg.HEAD_DROPOUT)
        self.emit = nn.Linear(H, N_LABELS)
        self.shift = nn.Linear(2 * H, 1)
        self.crf = CRF(N_LABELS)
        self.prompt = None                      # (P, d_model) soft prompt, only for prompt tuning
        self.pack_tokens = cfg.PACK_TOKENS
        self.sentence_only = not cfg.PACK_CONTEXT

    def sentence_vectors(self, doc_tok, pad_id, device, pack_batch=None):
        """Encode a run of sentences (list of token lists) -> (S, d_model) mean-pooled sentence vectors."""
        packs, where = pack(doc_tok, self.pack_tokens)
        ids, mask, sot = batch_packs(packs, where, pad_id, device, self.sentence_only)
        S = len(doc_tok)
        out = torch.zeros(S, self.backbone.d_model, device=device)
        count = torch.zeros(S, 1, device=device)
        step = pack_batch or ids.size(0)
        for b in range(0, ids.size(0), step):
            h = self.backbone(ids[b:b + step], mask[b:b + step], prompt=self.prompt).float()
            s = sot[b:b + step].reshape(-1)
            keep = s >= 0
            out = out.index_add(0, s[keep], h.reshape(-1, h.size(-1))[keep])
            count = count.index_add(0, s[keep], torch.ones(int(keep.sum()), 1, device=device))
        return out / count.clamp(min=1)

    def head(self, sent_vecs, domain, n_total=None, offset=0):
        """Sentence vectors of a contiguous run -> (context states, emissions, shift logits)."""
        S = sent_vecs.size(0)
        n_total = n_total or S
        buckets = position_buckets(n_total)[offset:offset + S].to(sent_vecs.device)
        x = self.proj(sent_vecs) + self.pos_emb(buckets) + self.dom_emb.weight[domain]
        x = self.drop(x)
        if self.use_lstm:
            x, _ = self.lstm(x.unsqueeze(0))
            x = x.squeeze(0)
        ctx = self.drop(x)
        em = self.emit(ctx).float()
        shift = self.shift(torch.cat([ctx[:-1], ctx[1:]], dim=-1)).squeeze(-1).float() if S > 1 else em.new_zeros(0)
        return ctx, em, shift

    def forward(self, doc_tok, domain, pad_id, device, n_total=None, offset=0, pack_batch=None):
        vecs = self.sentence_vectors(doc_tok, pad_id, device, pack_batch)
        return self.head(vecs, domain, n_total, offset)


# ----------------------------------------------------------------------------- methods / params
def configure_method(model, method, cfg, tok_emb_weight):
    bb = model.backbone
    if method in ("full", "kd"):
        for p in model.parameters():
            p.requires_grad = True
        return "all weights trainable"

    for p in bb.parameters():
        p.requires_grad = False
    if method == "lora":
        n = wrap_linears(bb.layers, cfg.ADAPTER_TARGETS, lambda m: LoRALinear(m, cfg.ADAPTER_RANK, cfg.LORA_ALPHA, cfg.ADAPTER_DROPOUT))
        return f"LoRA on {n} projections (r={cfg.ADAPTER_RANK})"
    if method == "lorma":
        n = wrap_linears(bb.layers, cfg.ADAPTER_TARGETS, lambda m: LoRMALinear(m, cfg.ADAPTER_RANK, cfg.LORMA_ALPHA, cfg.ADAPTER_DROPOUT))
        return f"LoRMA on {n} projections (r={cfg.ADAPTER_RANK})"
    if method == "ia3":
        n_out = wrap_linears(bb.layers, cfg.IA3_OUT_TARGETS, IA3Out)
        n_in = wrap_linears(bb.layers, cfg.IA3_IN_TARGETS, IA3In)
        return f"IA3 vectors on {n_out} key/value projections and {n_in} feed-forward layers"
    if method == "prompt":
        # Lester et al. 2021: learned virtual tokens, initialised from real token embeddings
        rng = torch.Generator().manual_seed(cfg.SEED)
        pick = torch.randint(100, tok_emb_weight.size(0) - 100, (cfg.PROMPT_LEN,), generator=rng)
        model.prompt = nn.Parameter(tok_emb_weight[pick].detach().clone())
        return f"{cfg.PROMPT_LEN} soft-prompt vectors prepended to every pack"
    raise ValueError(f"unknown method {method}")


def count_params(model):
    every = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return every, train


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().float().half() for k, v in model.state_dict().items() if k in names}


def layer_depth(name, n_layers):
    if ".layers." in name:
        return int(name.split(".layers.")[1].split(".")[0]) + 1
    if "emb" in name and name.startswith("backbone."):
        return 0
    return n_layers + 1


def build_optimizer(model, method, cfg):
    n_layers = len(model.backbone.layers)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not name.startswith("backbone.") and name != "prompt":
            lr = cfg.HEAD_LR                                    # BiLSTM / CRF / embeddings of the task head
        elif name == "prompt":
            lr = cfg.PROMPT_LR
        elif method in ("full", "kd"):
            lr = cfg.FULL_LR * cfg.LLRD ** (n_layers + 1 - layer_depth(name, n_layers))
        else:
            lr = cfg.ADAPTER_LR
        wd = 0.0 if (p.ndim < 2 or name == "prompt") else cfg.WEIGHT_DECAY
        groups.setdefault((lr, wd), []).append(p)
    return torch.optim.AdamW([{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()],
                             betas=(0.9, 0.98), eps=1e-6)


def class_weights(docs, power):
    counts = np.ones(N_LABELS)
    for d in docs:
        counts += np.bincount(d["labels"], minlength=N_LABELS)
    w = (counts.sum() / counts) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32), torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)


# --------------------------------------------------------------------------------------- metric
def macro_scores(gold, pred):
    """Macro P / R / F1 over the labels that occur in gold or prediction (sklearn's default behaviour)."""
    gold, pred = np.asarray(gold), np.asarray(pred)
    per = {}
    for lab in sorted(set(gold.tolist()) | set(pred.tolist())):
        tp = int(((gold == lab) & (pred == lab)).sum())
        p = tp / max(1, int((pred == lab).sum()))
        r = tp / max(1, int((gold == lab).sum()))
        per[LABELS[lab]] = {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "gold": int((gold == lab).sum())}
    macro = {k: float(np.mean([v[k] for v in per.values()])) for k in ("p", "r", "f1")}
    return macro, per


def top_confusions(gold, pred, k=8):
    pairs = {}
    for g, p in zip(gold, pred):
        if g != p:
            pairs[(LABELS[g], LABELS[p])] = pairs.get((LABELS[g], LABELS[p]), 0) + 1
    return sorted(pairs.items(), key=lambda kv: -kv[1])[:k]


# ------------------------------------------------------------------------------------ inference
@torch.no_grad()
def document_emissions(model, docs, pad_id, device, pack_batch):
    model.eval()
    out = []
    for d in docs:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, em, _ = model(d["tok"], d["domain"], pad_id, device, pack_batch=pack_batch)
        out.append(F.log_softmax(em.float(), -1))
    return out


def decode(model, emissions, log_prior, decoder, tau):
    preds = []
    for em in emissions:
        adj = em - tau * log_prior.to(em.device)                # logit adjustment (Menon et al. 2021) for macro-F1
        preds.append(model.crf.viterbi(adj) if decoder == "crf" else adj.argmax(-1).tolist())
    return preds


def tune_decoding(model, docs, emissions, log_prior, taus):
    """Pick decoder (CRF / argmax) and prior-correction strength on the dev documents."""
    gold = [l for d in docs for l in d["labels"]]
    best = None
    grid = []
    for decoder in ("crf", "argmax"):
        for tau in taus:
            pred = [l for p in decode(model, emissions, log_prior, decoder, tau) for l in p]
            f1 = macro_scores(gold, pred)[0]["f1"]
            grid.append({"decoder": decoder, "tau": tau, "f1": f1})
            if best is None or f1 > best["f1"]:
                best = {"decoder": decoder, "tau": tau, "f1": f1}
    return best, grid


def write_submission(test_docs, preds, out_dir, zip_name):
    os.makedirs(out_dir, exist_ok=True)
    payload = {}
    for d, p in zip(test_docs, preds):
        assert len(p) == len(d["sents"]), f"{d['id']}: {len(p)} predictions for {len(d['sents'])} sentences"
        payload[d["id"]] = [LABELS[i] for i in p]
    path = os.path.join(out_dir, "prediction.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    zip_path = os.path.join(out_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname="prediction.json")
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"]
    return path, zip_path


def training_spans(docs, span, rng):
    """Cut every document into runs of `span` consecutive sentences, starting at a random offset each epoch."""
    spans = []
    for di, d in enumerate(docs):
        n = len(d["sents"])
        if n <= span:
            spans.append((di, 0, n))
            continue
        start = rng.randrange(0, span)
        cuts = [0] + list(range(start, n, span))
        cuts = sorted(set(c for c in cuts if c < n)) + [n]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a >= 2:
                spans.append((di, a, b))
    rng.shuffle(spans)
    return spans
