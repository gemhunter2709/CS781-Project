"""
Phase 4 (court judgment prediction) machinery shared by all four architectures.

Every case is turned into a few 512-token chunks: the opening of the judgment plus its last part (ILDC, Malik et al.
2021, found the closing section - right before the removed verdict - to be the most predictive). The Phase-1 backbone
encodes each chunk, chunks are mean-pooled, tagged with where they came from, and combined by learned attention pooling
into one document vector (a hierarchical chunk -> document representation) that a linear layer classifies.
"""
import os
import json
import math
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------------------- data
def load_cases(path, with_labels):
    """Only id, text and (for train/dev) label are read. The public test file also carries a 'label' field;
    it is deliberately ignored."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    labels = [int(r["label"]) for r in rows] if with_labels else None
    return ids, texts, labels


_tok = None
_cfg = None


def _init_worker(cfg_dict):
    global _tok, _cfg
    import backbone as BB
    _tok = BB.load_tokenizer()
    _cfg = cfg_dict


def chunk_text(text):
    """-> (K, W) token ids and per-chunk lengths. Short cases are simply cut into consecutive chunks."""
    W, head, tail = _cfg["W"], _cfg["head"], _cfg["tail"]
    K = head + tail
    budget = K * W
    approx = budget * _cfg["chars_per_token"]
    if len(text) <= approx:
        ids = _tok.encode(text)
    else:
        ids = None
    if ids is not None and len(ids) <= budget:
        pieces = [ids[i:i + W] for i in range(0, len(ids), W)]
    else:
        h = _tok.encode(text[:head * W * _cfg["chars_per_token"]])[:head * W]
        t = _tok.encode(text[-tail * W * _cfg["chars_per_token"]:])[-tail * W:]
        pieces = [h[i:i + W] for i in range(0, len(h), W)][:head] + [t[i:i + W] for i in range(0, len(t), W)][-tail:]
    out = np.full((K, W), _cfg["pad"], dtype=np.int32)
    lens = np.zeros(K, dtype=np.int16)
    for k, p in enumerate(pieces[:K]):
        out[k, :len(p)] = p
        lens[k] = len(p)
    return out, lens


def prepare_split(texts, cfg_dict, workers):
    with Pool(workers, initializer=_init_worker, initargs=(cfg_dict,)) as pool:
        res = pool.map(chunk_text, texts, chunksize=64)
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


class BitFitLinear(nn.Module):
    """BitFit (Ben Zaken et al. 2022) trains only bias terms. Arch 2-4 were pre-trained without biases, so a
    zero-initialised bias is added there (the model is unchanged at the start); arch 1 keeps its own biases."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.bias = nn.Parameter(torch.zeros(base.out_features)) if base.bias is None else None

    def forward(self, x):
        out = self.base(x)
        return out + self.bias if self.bias is not None else out


class Adapter(nn.Module):
    """Houlsby et al. 2019 bottleneck adapter on a sub-layer's output: h + up(gelu(down(h))). up = 0 at the start."""

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
    """Wrap matching nn.Linear children. MoE routers are always left alone: arch 4 reads router.weight directly,
    and the pre-trained routing should not drift."""
    n = 0
    for parent in list(root.modules()):
        for name, child in list(parent.named_children()):
            if name in skip:
                continue
            if (names is None or name in names) and isinstance(child, nn.Linear):
                setattr(parent, name, make(child))
                n += 1
    return n


# ---------------------------------------------------------------------------------------- model
class CJPEModel(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        D = backbone.d_model
        K = cfg.HEAD_CHUNKS + cfg.TAIL_CHUNKS
        self.chunk_pos = nn.Embedding(K, D)                 # which part of the judgment a chunk came from
        nn.init.zeros_(self.chunk_pos.weight)
        self.att = nn.Sequential(nn.Linear(D, cfg.POOL_DIM), nn.Tanh(), nn.Linear(cfg.POOL_DIM, 1))
        self.drop = nn.Dropout(cfg.HEAD_DROPOUT)
        self.cls = nn.Linear(D, 2)

    def encode_chunks(self, ids, lens):
        """ids (B, K, W), lens (B, K) -> chunk vectors (B, K, D); empty chunks are never run through the backbone."""
        B, K, W = ids.shape
        flat_ids, flat_lens = ids.view(B * K, W), lens.view(B * K)
        real = flat_lens > 0
        vecs = torch.zeros(B * K, self.backbone.d_model, device=ids.device)
        if real.any():
            sub = flat_ids[real]
            L = int(flat_lens[real].max())
            sub = sub[:, :L]
            mask = torch.arange(L, device=ids.device)[None, :] < flat_lens[real][:, None]
            h = self.backbone(sub, mask).float()
            pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            vecs = vecs.index_put((real.nonzero().squeeze(1),), pooled)
        return vecs.view(B, K, -1), (lens > 0)

    def forward(self, ids, lens):
        chunks, present = self.encode_chunks(ids, lens)
        chunks = chunks + self.chunk_pos.weight[None]
        scores = self.att(chunks).squeeze(-1).masked_fill(~present, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        doc = (weights[..., None] * chunks).sum(1)
        return self.cls(self.drop(doc)).float(), doc, weights


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
    if method == "bitfit":
        n = wrap_linears(bb.layers, None, BitFitLinear)
        for name, p in bb.named_parameters():
            if name.endswith(".bias"):
                p.requires_grad = True
        return f"bias terms of {n} linear layers trainable"
    if method == "adapter":
        n = wrap_linears(bb.layers, cfg.ADAPTER_SITES, lambda m: Adapter(m, cfg.ADAPTER_BOTTLENECK))
        return f"{n} bottleneck adapters (size {cfg.ADAPTER_BOTTLENECK}) after attention and feed-forward outputs"
    if method == "unfreeze":
        return "gradual unfreezing: head first, then one more backbone layer at a time from the top"
    raise ValueError(method)


def set_unfrozen(model, n_top):
    """Gradual unfreezing (Howard & Ruder 2018): only the top n_top layers (+ final norm) of the backbone train."""
    layers = list(model.backbone.layers)
    for i, layer in enumerate(layers):
        for p in layer.parameters():
            p.requires_grad = i >= len(layers) - n_top
    for p in model.backbone.final_norm.parameters():
        p.requires_grad = n_top > 0


def count_params(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_state(model, method):
    if method == "unfreeze":                                # every layer may have moved by the end
        names = {n for n, _ in model.named_parameters()}
    else:
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
        if method != "unfreeze" and not p.requires_grad:
            continue
        if not name.startswith("backbone."):
            lr = cfg.HEAD_LR
        elif method in ("full", "kd", "unfreeze"):
            lr = cfg.FULL_LR * cfg.LLRD ** (n_layers + 1 - layer_depth(name, n_layers))
        elif method == "bitfit":
            lr = cfg.BITFIT_LR
        else:
            lr = cfg.ADAPTER_LR
        wd = 0.0 if p.ndim < 2 else cfg.WEIGHT_DECAY
        groups.setdefault((lr, wd), []).append(p)
    return torch.optim.AdamW([{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()],
                             betas=(0.9, 0.98), eps=1e-6)


# --------------------------------------------------------------------------------------- metric
def binary_scores(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    per = {}
    for c in (0, 1):
        tp = int(((gold == c) & (pred == c)).sum())
        p = tp / max(1, int((pred == c).sum()))
        r = tp / max(1, int((gold == c).sum()))
        per[c] = (p, r, 2 * p * r / (p + r) if p + r else 0.0)
    return {"f1": (per[0][2] + per[1][2]) / 2, "p": (per[0][0] + per[1][0]) / 2, "r": (per[0][1] + per[1][1]) / 2,
            "acc": float((gold == pred).mean())}


def best_threshold(gold, prob1):
    """Decision threshold on P(class 1) that maximises dev macro-F1 (train is 59/41, dev and test are balanced)."""
    best = (0.5, binary_scores(gold, (prob1 >= 0.5).astype(int))["f1"])
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), np.quantile(prob1, np.linspace(0.02, 0.98, 97))]))
    for t in candidates:
        f1 = binary_scores(gold, (prob1 >= t).astype(int))["f1"]
        if f1 > best[1]:
            best = (float(t), f1)
    return best


@torch.no_grad()
def predict_probs(model, ids, lens, device, batch):
    model.eval()
    out = []
    for b in range(0, len(ids), batch):
        x = torch.from_numpy(ids[b:b + batch].astype(np.int64)).to(device)
        l = torch.from_numpy(lens[b:b + batch].astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _, _ = model(x, l)
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def write_submission(test_ids, preds, out_dir, zip_name):
    import zipfile
    assert len(test_ids) == len(preds) == len(set(test_ids))
    payload = [{"id": i, "pred": int(p)} for i, p in zip(test_ids, preds)]
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
