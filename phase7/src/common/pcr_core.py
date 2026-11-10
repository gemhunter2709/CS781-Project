"""
Phase 7 (prior case retrieval) machinery shared by all four architectures.

Ranking = dense similarity from the fine-tuned Phase-1 model, fused with BM25. The dense part is a bi-encoder: a case is
read as a few 512-token chunks spread over the whole judgment, the backbone encodes each chunk, chunks are mean-pooled
into one normalised vector, and queries and candidates share the encoder. It is trained contrastively (InfoNCE) with
in-batch negatives plus BM25 hard negatives. Every query also appears in its own candidate pool, so it is never
allowed to retrieve itself.
"""
import os
import re
import json
import math
from collections import Counter
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOP_K = 7


# ------------------------------------------------------------------------------------------- data
def load_split(query_path, cand_path):
    with open(query_path, encoding="utf-8") as f:
        queries = json.load(f)
    with open(cand_path, encoding="utf-8") as f:
        cands = json.load(f)
    as_text = lambda v: " ".join(v["text"]) if isinstance(v["text"], list) else v["text"]
    q_ids = list(queries)
    q_text = [as_text(queries[k]) for k in q_ids]
    rel = [list(queries[k].get("relevant_candidates", [])) for k in q_ids]
    c_ids = list(cands)
    c_text = [as_text(cands[k]) for k in c_ids]
    return q_ids, q_text, rel, c_ids, c_text


# ------------------------------------------------------------------------------------------ BM25
WORD = re.compile(r"[a-z0-9]+")


def words(text):
    return WORD.findall(text.lower())


class BM25:
    """Okapi BM25 over the candidate pool, with an inverted index so scoring touches only matching documents."""

    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.lens.mean())
        self.post = {}
        for i, d in enumerate(docs):
            for w, f in Counter(d).items():
                self.post.setdefault(w, []).append((i, f))
        self.idf = {w: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.post.items()}
        self.post = {w: (np.array([i for i, _ in p]), np.array([f for _, f in p], dtype=np.float32)) for w, p in self.post.items()}

    def scores(self, query_words, min_idf, max_terms):
        qc = Counter(w for w in query_words if self.idf.get(w, 0.0) > min_idf)
        terms = sorted(qc, key=lambda w: -qc[w] * self.idf[w])[:max_terms]      # the query's most telling words
        s = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lens / self.avg)
        for w in terms:
            idx, f = self.post[w]
            s[idx] += self.idf[w] * f * (self.k1 + 1) / (f + norm[idx])
        return s


def bm25_matrix(q_text, c_text, cfg):
    bm = BM25([words(t) for t in c_text])
    return np.stack([bm.scores(words(t), cfg.BM25_MIN_IDF, cfg.BM25_MAX_TERMS) for t in q_text])


# ----------------------------------------------------------------------------------------- chunks
_tok = None
_cfg = None


def _init_worker(cfg_dict):
    global _tok, _cfg
    import backbone as BB
    _tok = BB.load_tokenizer()
    _cfg = cfg_dict


def chunk_text(text):
    """Up to K chunks of W tokens spread evenly over the case (first and last included)."""
    W, K = _cfg["W"], _cfg["K"]
    ids = _tok.encode(text[:_cfg["char_cap"]]) or [_cfg["pad"]]
    n = len(ids)
    starts = list(range(0, n, W)) if n <= K * W else [round(i * (n - W) / (K - 1)) for i in range(K)]
    out = np.full((K, W), _cfg["pad"], dtype=np.int32)
    lens = np.zeros(K, dtype=np.int16)
    for k, s in enumerate(starts[:K]):
        piece = ids[s:s + W]
        out[k, :len(piece)] = piece
        lens[k] = len(piece)
    return out, lens


def prepare_texts(texts, cfg_dict, workers):
    with Pool(workers, initializer=_init_worker, initargs=(cfg_dict,)) as pool:
        res = pool.map(chunk_text, texts, chunksize=16)
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])


# ----------------------------------------------------------------------------------- adaptation
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
    """Liu et al. 2022 (IA3): learned rescaling of a projection's output (keys / values). Starts at 1."""

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


# ----------------------------------------------------------------------------------------- model
class Retriever(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(backbone.d_model, cfg.EMB_DIM)
        self.prompt = None

    def forward(self, ids, lens):
        """ids (B, K, W) -> L2-normalised case embeddings (B, EMB_DIM). Empty chunks never reach the backbone."""
        B, K, W = ids.shape
        flat_ids, flat_lens = ids.view(B * K, W), lens.view(B * K)
        real = flat_lens > 0
        L = int(flat_lens[real].max())
        mask = torch.arange(L, device=ids.device)[None, :] < flat_lens[real][:, None]
        h = self.backbone(flat_ids[real][:, :L], mask, prompt=self.prompt).float()
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        chunks = torch.zeros(B * K, pooled.size(-1), device=ids.device).index_put((real.nonzero().squeeze(1),), pooled)
        present = real.view(B, K, 1).float()
        doc = (chunks.view(B, K, -1) * present).sum(1) / present.sum(1).clamp(min=1)
        return F.normalize(self.proj(doc), dim=-1)


def configure_method(model, method, cfg):
    bb = model.backbone
    if method in ("full", "kd"):
        for p in model.parameters():
            p.requires_grad = True
        return "all weights trainable"
    for p in bb.parameters():
        p.requires_grad = False
    if method == "lorma":
        n = wrap_linears(bb.layers, cfg.ADAPTER_TARGETS, lambda m: LoRMALinear(m, cfg.ADAPTER_RANK, cfg.LORMA_ALPHA, cfg.ADAPTER_DROPOUT))
        return f"LoRMA on {n} attention projections (r={cfg.ADAPTER_RANK})"
    if method == "ia3":
        n_out = wrap_linears(bb.layers, cfg.IA3_OUT_TARGETS, IA3Out)
        n_in = wrap_linears(bb.layers, cfg.IA3_IN_TARGETS, IA3In)
        return f"IA3 vectors on {n_out} key/value projections and {n_in} feed-forward layers"
    if method == "partial":
        for layer in list(bb.layers)[-cfg.PARTIAL_TOP_LAYERS:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in bb.final_norm.parameters():
            p.requires_grad = True
        return f"top {cfg.PARTIAL_TOP_LAYERS} backbone layers + final norm trainable"
    if method == "prompt":
        # Lester et al. 2021: learned virtual tokens in front of every chunk, initialised from real token embeddings
        g = torch.Generator().manual_seed(cfg.SEED)
        w = bb.tok_emb.weight
        pick = torch.randint(100, w.size(0) - 100, (cfg.PROMPT_LEN,), generator=g)
        model.prompt = nn.Parameter(w[pick].detach().clone())
        return f"{cfg.PROMPT_LEN} soft-prompt vectors in front of every chunk"
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
        elif not name.startswith("backbone."):
            lr = cfg.HEAD_LR
        elif method in ("full", "kd"):
            lr = cfg.FULL_LR
        elif method == "partial":
            lr = cfg.PARTIAL_LR
        else:
            lr = cfg.ADAPTER_LR
        wd = 0.0 if (p.ndim < 2 or name == "prompt") else cfg.WEIGHT_DECAY
        groups.setdefault((lr, wd), []).append(p)
    return torch.optim.AdamW([{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()],
                             betas=(0.9, 0.98), eps=1e-6)


# --------------------------------------------------------------------------------------- scoring
@torch.no_grad()
def embed(model, ids, lens, device, batch):
    model.eval()
    out = []
    for b in range(0, len(ids), batch):
        x = torch.from_numpy(ids[b:b + batch].astype(np.int64)).to(device)
        l = torch.from_numpy(lens[b:b + batch].astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(x, l).float().cpu())
    return torch.cat(out).numpy()


def zscore(m):
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-6)


def rank(scores, q_ids, c_ids):
    """Top-7 candidate ids per query; a query's own id is never returned."""
    pos = {c: j for j, c in enumerate(c_ids)}
    s = scores.copy()
    out = []
    for i, q in enumerate(q_ids):
        if q in pos:
            s[i, pos[q]] = -np.inf
        top = np.argsort(-s[i])[:TOP_K]
        out.append([c_ids[j] for j in top])
    return out


def micro_f1_at_7(preds, rel):
    tp = sum(len(set(p) & set(r)) for p, r in zip(preds, rel))
    n_pred = sum(len(p) for p in preds)
    n_gold = sum(len(r) for r in rel)
    p = tp / max(1, n_pred)
    r = tp / max(1, n_gold)
    return {"f1": 2 * p * r / (p + r) if p + r else 0.0, "p": p, "r": r}


def tune_fusion(dense, bm25, q_ids, c_ids, rel, grid):
    """Weight alpha on the dense score (1 - alpha on BM25) that gives the best dev micro-F1@7."""
    dz, bz = zscore(dense), zscore(bm25)
    results = []
    for a in grid:
        results.append((a, micro_f1_at_7(rank(a * dz + (1 - a) * bz, q_ids, c_ids), rel)))
    best = max(results, key=lambda t: t[1]["f1"])
    return best[0], results


def write_submission(q_ids, preds, out_dir, zip_name, pool):
    import zipfile
    pool = set(pool)
    payload = {}
    for q, p in zip(q_ids, preds):
        assert len(p) == TOP_K == len(set(p)) and all(c in pool and c != q for c in p), q
        payload[q] = p
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "prediction.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    zip_path = os.path.join(out_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname="prediction.json")
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"]
    return path, zip_path
