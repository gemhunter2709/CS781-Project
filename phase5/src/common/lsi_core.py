"""
Phase 5 (legal statute identification) machinery shared by all four architectures.

A document becomes up to MAX_CHUNKS chunks of 512 tokens spread evenly from its start to its end (statutes can be
implied anywhere in the facts). The Phase-1 backbone encodes every chunk; each token is tagged with the chunk it came
from; then label-wise attention (CAML, Mullenbach et al. 2018; LWAN for legal text, Chalkidis et al. 2019) lets every
statute attend to the tokens that matter for *it* before its own score is computed.
"""
import os
import json
import math
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_LABELS = 100


# ------------------------------------------------------------------------------------------- data
def load_docs(path, with_labels):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    ids = [r["id"] for r in rows]
    texts = [r["text"] if isinstance(r["text"], str) else " ".join(r["text"]) for r in rows]
    labels = [sorted(set(int(l) for l in r["labels"])) for r in rows] if with_labels else None
    return ids, texts, labels


_tok = None
_cfg = None


def _init_worker(cfg_dict):
    global _tok, _cfg
    import backbone as BB
    _tok = BB.load_tokenizer()
    _cfg = cfg_dict


def chunk_text(text):
    """Up to K chunks of W tokens; long documents are sampled at K evenly spaced places (first and last included)."""
    W, K = _cfg["W"], _cfg["K"]
    ids = _tok.encode(text[:_cfg["char_cap"]]) or [_cfg["pad"]]      # an empty document still gets one token
    n = len(ids)
    if n <= K * W:
        starts = list(range(0, n, W)) or [0]
    else:
        starts = [round(i * (n - W) / (K - 1)) for i in range(K)]
    out = np.full((K, W), _cfg["pad"], dtype=np.int32)
    lens = np.zeros(K, dtype=np.int16)
    for k, s in enumerate(starts[:K]):
        piece = ids[s:s + W]
        out[k, :len(piece)] = piece
        lens[k] = len(piece)
    return out, lens


def prepare_split(texts, cfg_dict, workers):
    with Pool(workers, initializer=_init_worker, initargs=(cfg_dict,)) as pool:
        res = pool.map(chunk_text, texts, chunksize=64)
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])


def multi_hot(labels):
    y = np.zeros((len(labels), N_LABELS), dtype=np.float32)
    for i, ls in enumerate(labels):
        y[i, ls] = 1.0
    return y


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
    """MoE routers are always left alone: arch 4 reads router.weight directly."""
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
class LSIModel(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        D = backbone.d_model
        self.chunk_pos = nn.Embedding(cfg.MAX_CHUNKS, D)
        nn.init.zeros_(self.chunk_pos.weight)
        self.drop = nn.Dropout(cfg.HEAD_DROPOUT)
        self.proj = nn.Linear(D, cfg.LWAN_DIM)
        self.label_query = nn.Parameter(torch.randn(N_LABELS, cfg.LWAN_DIM) * 0.02)   # one attention per statute
        self.label_out = nn.Parameter(torch.randn(N_LABELS, cfg.LWAN_DIM) * 0.02)     # one scorer per statute
        self.label_bias = nn.Parameter(torch.zeros(N_LABELS))
        self.prompt = None

    def token_states(self, ids, lens):
        """ids (B, K, W) -> token states (B, K*W, D) and mask; empty chunks never reach the backbone."""
        B, K, W = ids.shape
        flat_ids, flat_lens = ids.view(B * K, W), lens.view(B * K)
        real = flat_lens > 0
        L = int(flat_lens.max())
        states = torch.zeros(B * K, L, self.backbone.d_model, device=ids.device)
        sub_ids = flat_ids[real][:, :L]
        mask = torch.arange(L, device=ids.device)[None, :] < flat_lens[real][:, None]
        h = self.backbone(sub_ids, mask, prompt=self.prompt).float()
        states = states.index_put((real.nonzero().squeeze(1),), h)
        full_mask = torch.arange(L, device=ids.device)[None, :] < flat_lens[:, None]
        states = states.view(B, K, L, -1) + self.chunk_pos.weight[None, :, None, :]
        return states.view(B, K * L, -1), full_mask.view(B, K * L)

    def forward(self, ids, lens):
        h, mask = self.token_states(ids, lens)
        z = torch.tanh(self.proj(self.drop(h)))                                  # (B, T, A)
        scores = torch.einsum("bta,la->blt", z, self.label_query)                # (B, L, T)
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        att = torch.softmax(scores, dim=-1)
        per_label = torch.einsum("blt,bta->bla", att, z)                         # (B, L, A)
        logits = (per_label * self.label_out[None]).sum(-1) + self.label_bias
        doc = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return logits.float(), doc


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
    if method == "adapter":
        n = wrap_linears(bb.layers, cfg.ADAPTER_SITES, lambda m: Adapter(m, cfg.ADAPTER_BOTTLENECK))
        return f"{n} bottleneck adapters (size {cfg.ADAPTER_BOTTLENECK}) after attention and feed-forward outputs"
    if method == "partial":
        for layer in list(bb.layers)[-cfg.PARTIAL_TOP_LAYERS:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in bb.final_norm.parameters():
            p.requires_grad = True
        return f"top {cfg.PARTIAL_TOP_LAYERS} backbone layers + final norm trainable"
    if method == "prompt":
        # Lester et al. 2021: learned virtual tokens, initialised from real token embeddings
        g = torch.Generator().manual_seed(cfg.SEED)
        w = bb.tok_emb.weight
        pick = torch.randint(100, w.size(0) - 100, (cfg.PROMPT_LEN,), generator=g)
        model.prompt = nn.Parameter(w[pick].detach().clone())
        return f"{cfg.PROMPT_LEN} soft-prompt vectors prepended to every chunk"
    raise ValueError(method)


def count_params(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        if name == "prompt":
            lr = cfg.PROMPT_LR
        elif not name.startswith("backbone."):
            lr = cfg.HEAD_LR
        elif method in ("full", "kd"):
            lr = cfg.FULL_LR * cfg.LLRD ** (n_layers + 1 - layer_depth(name, n_layers))
        elif method == "partial":
            lr = cfg.PARTIAL_LR
        else:
            lr = cfg.ADAPTER_LR
        wd = 0.0 if (p.ndim < 2 or name == "prompt") else cfg.WEIGHT_DECAY
        groups.setdefault((lr, wd), []).append(p)
    return torch.optim.AdamW([{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()],
                             betas=(0.9, 0.98), eps=1e-6)


# ----------------------------------------------------------------------------------------- loss
def asymmetric_loss(logits, y, gamma_neg, gamma_pos, clip):
    """Ridnik et al. 2021 (ASL): easy negatives are down-weighted and very-low-probability negatives ignored,
    which suits 100 statutes where each document has ~4 positives."""
    p = torch.sigmoid(logits)
    p_neg = (p - clip).clamp(min=0) if clip > 0 else p
    loss_pos = y * torch.log(p.clamp(min=1e-8)) * (1 - p) ** gamma_pos
    loss_neg = (1 - y) * torch.log((1 - p_neg).clamp(min=1e-8)) * p_neg ** gamma_neg
    return -(loss_pos + loss_neg).sum(1).mean()


def task_loss(logits, y, cfg):
    if cfg.LOSS == "asl":
        return asymmetric_loss(logits, y, cfg.ASL_GAMMA_NEG, cfg.ASL_GAMMA_POS, cfg.ASL_CLIP)
    return F.binary_cross_entropy_with_logits(logits, y, reduction="sum") / logits.size(0)


# -------------------------------------------------------------------------------------- metrics
def scores(Y, P):
    """Macro-F1 (over the 100 statutes), micro-F1 and subset accuracy for binary matrices Y (gold) and P (pred)."""
    tp = (Y * P).sum(0)
    fp = ((1 - Y) * P).sum(0)
    fn = (Y * (1 - P)).sum(0)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    present = Y.sum(0) + P.sum(0) > 0
    micro = 2 * tp.sum() / max(1.0, 2 * tp.sum() + fp.sum() + fn.sum())
    return {"macro_f1": float(f1[present].mean()), "micro_f1": float(micro),
            "subset_acc": float((Y == P).all(1).mean()), "per_label_f1": f1.tolist()}


def decide(probs, thresholds, at_least_one=True):
    P = (probs >= thresholds[None, :]).astype(np.float32)
    if at_least_one:
        empty = P.sum(1) == 0
        P[empty, probs[empty].argmax(1)] = 1.0           # every training document has at least one statute
    return P


def tune_thresholds(Y, probs, min_pos):
    """Per-statute threshold that maximises that statute's dev F1 (macro-F1 is their average).
    Statutes with fewer than min_pos dev positives fall back to the best single global threshold."""
    grid = np.linspace(0.05, 0.95, 91)
    glob = max(grid, key=lambda t: scores(Y, decide(probs, np.full(N_LABELS, t)))["macro_f1"])
    th = np.full(N_LABELS, glob)
    for l in range(N_LABELS):
        if Y[:, l].sum() < min_pos:
            continue
        y, p = Y[:, l], probs[:, l]
        best, best_f1 = glob, -1.0
        for t in grid:
            pred = p >= t
            tp = (pred & (y == 1)).sum()
            f1 = 2 * tp / max(1, pred.sum() + y.sum())
            if f1 > best_f1:
                best, best_f1 = t, f1
        th[l] = best
    return float(glob), th


@torch.no_grad()
def predict_probs(model, ids, lens, device, batch):
    model.eval()
    out = []
    for b in range(0, len(ids), batch):
        x = torch.from_numpy(ids[b:b + batch].astype(np.int64)).to(device)
        l = torch.from_numpy(lens[b:b + batch].astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = model(x, l)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def write_submission(test_ids, P, out_dir, zip_name):
    import zipfile
    assert len(test_ids) == len(P) == len(set(map(str, test_ids)))
    payload = [{"id": i, "pred": sorted(int(l) for l in np.nonzero(row)[0])} for i, row in zip(test_ids, P)]
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
