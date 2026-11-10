"""
Phase 2 (legal NER) machinery shared by all four architectures:
data loading, BIO tagging, sliding windows, LoRA / LoRMA adapters, the tagging model,
training, span decoding and the strict (exact offsets + label) metric.
"""
import os
import json
import math
import random
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]
TAGS = ["O"] + [f"{p}-{lab}" for lab in LABELS for p in ("B", "I")]
TAG_ID = {t: i for i, t in enumerate(TAGS)}
N_TAGS = len(TAGS)
IGNORE = -100


# --------------------------------------------------------------------------------------------- data
def load_docs(text_path, gold_path=None):
    with open(text_path, encoding="utf-8") as f:
        texts = json.load(f)
    gold = {}
    if gold_path:
        with open(gold_path, encoding="utf-8") as f:
            gold = json.load(f)
    docs = []
    for doc_id, text in texts.items():
        ents = [(int(e[0]), int(e[1]), e[2]) for e in gold.get(doc_id, [])]
        docs.append({"id": str(doc_id), "text": text, "ents": sorted(ents)})
    return docs


def split_dev(fold_docs, n_dev_per_fold, seed):
    """Hold out a few whole documents from every fold, so the dev set looks like the test set."""
    rng = random.Random(seed)
    train, dev = [], []
    for docs in fold_docs:
        docs = sorted(docs, key=lambda d: d["id"])
        rng.shuffle(docs)
        dev += docs[:n_dev_per_fold]
        train += docs[n_dev_per_fold:]
    return train, dev


# ------------------------------------------------------------------------------------ BIO encoding
def encode_doc(doc, tokenizer):
    """
    Tokenize with character offsets. starts/ends are the *content* span of every token (leading or
    trailing whitespace removed); pure-whitespace tokens get an empty span and never carry a label.
    """
    ids, starts, ends = tokenizer.encode_with_offsets(doc["text"])
    ids, starts, ends = np.asarray(ids, np.int64), np.asarray(starts, np.int64), np.asarray(ends, np.int64)
    real = ends > starts
    tags = np.where(real, TAG_ID["O"], IGNORE).astype(np.int64)

    for s, e, lab in doc["ents"]:
        hit = np.nonzero(real & (starts < e) & (ends > s))[0]
        if len(hit) == 0:
            continue
        tags[hit[0]] = TAG_ID[f"B-{lab}"]
        tags[hit[1:]] = TAG_ID[f"I-{lab}"]
    return {"id": doc["id"], "text": doc["text"], "ents": doc["ents"],
            "ids": ids, "starts": starts, "ends": ends, "tags": tags}


def make_windows(n, size, stride):
    if n <= size:
        return [(0, n)]
    wins = [(s, s + size) for s in range(0, n - size + 1, stride)]
    if wins[-1][1] < n:
        wins.append((n - size, n))
    return wins


def owner_of_tokens(windows, n):
    """For each token, the window in which it sits furthest from an edge (most context on both sides)."""
    best = np.full(n, -1, dtype=np.int64)
    best_margin = np.full(n, -1, dtype=np.int64)
    for k, (ws, we) in enumerate(windows):
        pos = np.arange(ws, we)
        margin = np.minimum(pos - ws, we - 1 - pos)
        better = margin > best_margin[ws:we]
        best[ws:we][better] = k
        best_margin[ws:we][better] = margin[better]
    return best


def window_items(enc_docs, size, stride):
    items = []
    for d, doc in enumerate(enc_docs):
        for ws, we in make_windows(len(doc["ids"]), size, stride):
            items.append((d, ws, we))
    return items


def collate(enc_docs, items, pad_id):
    T = max(we - ws for _, ws, we in items)
    B = len(items)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    tags = torch.full((B, T), IGNORE, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.bool)
    for r, (d, ws, we) in enumerate(items):
        doc = enc_docs[d]
        L = we - ws
        ids[r, :L] = torch.from_numpy(doc["ids"][ws:we])
        tags[r, :L] = torch.from_numpy(doc["tags"][ws:we])
        mask[r, :L] = True
    return ids, tags, mask


# ----------------------------------------------------------------------------------- adapters
class LoRALinear(nn.Module):
    """Hu et al. 2021: W x + (alpha / r) * B A x, with B = 0 at the start."""

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
    """
    Bihany, Patel & Modi 2025 (LoRMA+): h = (alpha/r * B A + I) W0 x.
    Computed as W0x first, then A, then B, which costs the same as LoRA. B = 0 makes it the identity at the start.
    """

    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        d = base.out_features
        self.A = nn.Parameter(torch.empty(r, d))
        self.B = nn.Parameter(torch.zeros(d, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.linear(x, self.base.weight)
        h = h + F.linear(F.linear(self.drop(h), self.A), self.B) * self.scale
        if self.base.bias is not None:
            h = h + self.base.bias
        return h


def inject_adapters(root, target_names, kind, r, alpha, dropout):
    wrapper = {"lora": LoRALinear, "lorma": LoRMALinear}[kind]
    swapped = 0
    for parent in list(root.modules()):
        for name, child in list(parent.named_children()):
            if name in target_names and isinstance(child, nn.Linear):
                setattr(parent, name, wrapper(child, r, alpha, dropout))
                swapped += 1
    return swapped


# --------------------------------------------------------------------------------------- model
class Tagger(nn.Module):
    def __init__(self, backbone, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(backbone.d_model, N_TAGS)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids, mask):
        h = self.backbone(ids, mask)
        return self.head(self.drop(h.float()))


def configure_method(model, method, cfg):
    """Freeze / unfreeze / add adapters according to the experiment. Returns a short description."""
    bb = model.backbone
    if method in ("full", "kd"):
        for p in model.parameters():
            p.requires_grad = True
        return "all weights trainable"

    for p in bb.parameters():
        p.requires_grad = False

    if method == "partial":
        top = list(bb.layers)[-cfg.PARTIAL_TOP_LAYERS:]
        for layer in top:
            for p in layer.parameters():
                p.requires_grad = True
        for p in bb.final_norm.parameters():
            p.requires_grad = True
        return f"top {cfg.PARTIAL_TOP_LAYERS} layers + final norm trainable"

    if method in ("lora", "lorma"):
        n = inject_adapters(bb.layers, cfg.ADAPTER_TARGETS, method, cfg.ADAPTER_RANK,
                            cfg.LORA_ALPHA if method == "lora" else cfg.LORMA_ALPHA, cfg.ADAPTER_DROPOUT)
        return f"{n} linear layers wrapped with {method.upper()} (r={cfg.ADAPTER_RANK})"

    raise ValueError(f"unknown method {method}")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def trainable_state(model):
    """Only what training changed: adapters + head for PEFT runs, everything for full / KD."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().half() if v.is_floating_point() else v for k, v in model.state_dict().items() if k in names}


# ----------------------------------------------------------------------------------- optimizer
def layer_depth(name, n_layers):
    if ".layers." in name:
        return int(name.split(".layers.")[1].split(".")[0]) + 1
    if "emb" in name and name.startswith("backbone."):
        return 0
    return n_layers + 1                      # final norm and the head


def build_optimizer(model, method, cfg):
    n_layers = len(model.backbone.layers)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head."):
            lr = cfg.HEAD_LR
        elif method in ("full", "kd"):
            # layer-wise LR decay (ULMFiT / ELECTRA): lower layers move less
            lr = cfg.FULL_LR * cfg.LLRD ** (n_layers + 1 - layer_depth(name, n_layers))
        elif method == "partial":
            lr = cfg.PARTIAL_LR
        else:
            lr = cfg.ADAPTER_LR
        wd = 0.0 if p.ndim < 2 else cfg.WEIGHT_DECAY
        groups.setdefault((lr, wd), []).append(p)
    return torch.optim.AdamW([{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()],
                             betas=(0.9, 0.98), eps=1e-6)


# ------------------------------------------------------------------------------------ decoding
def _allowed():
    allow = np.ones((N_TAGS, N_TAGS), dtype=bool)          # allow[prev, cur]
    for cur, tag in enumerate(TAGS):
        if tag.startswith("I-"):
            lab = tag[2:]
            allow[:, cur] = False
            allow[TAG_ID[f"B-{lab}"], cur] = True
            allow[TAG_ID[f"I-{lab}"], cur] = True
    start_ok = np.array([not t.startswith("I-") for t in TAGS])
    return allow, start_ok


ALLOW, START_OK = _allowed()


def viterbi(logp):
    """Best tag path that is a valid BIO sequence (no I-X without a preceding B-X / I-X)."""
    T = len(logp)
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    trans = np.where(ALLOW, 0.0, -1e9)
    score = np.where(START_OK, logp[0], -1e9)
    back = np.zeros((T, N_TAGS), dtype=np.int64)
    for t in range(1, T):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def greedy(logp):
    return logp.argmax(-1)


def tags_to_spans(tag_ids, starts, ends):
    """Turn a tag sequence over non-whitespace tokens into (start, end, label) character spans."""
    spans, cur = [], None
    for t, s, e in zip(tag_ids, starts, ends):
        tag = TAGS[t]
        if tag == "O":
            if cur:
                spans.append(tuple(cur))
            cur = None
            continue
        kind, lab = tag.split("-", 1)
        if kind == "B" or cur is None or cur[2] != lab:     # an orphan I- simply opens a new entity
            if cur:
                spans.append(tuple(cur))
            cur = [int(s), int(e), lab]
        else:
            cur[1] = int(e)
    if cur:
        spans.append(tuple(cur))
    return spans


# -------------------------------------------------------------------------------------- metric
def strict_scores(pred, gold):
    """pred / gold: {doc_id: iterable of (start, end, label)}. Exact span AND label must match."""
    tp, n_pred, n_gold = ({lab: 0 for lab in LABELS} for _ in range(3))
    for doc_id in gold:
        g, p = set(map(tuple, gold[doc_id])), set(map(tuple, pred.get(doc_id, [])))
        for s, e, lab in g:
            n_gold[lab] += 1
        for s, e, lab in p:
            n_pred[lab] += 1
            if (s, e, lab) in g:
                tp[lab] += 1
    per = {}
    for lab in LABELS:
        if n_gold[lab] == 0 and n_pred[lab] == 0:
            continue
        p = tp[lab] / n_pred[lab] if n_pred[lab] else 0.0
        r = tp[lab] / n_gold[lab] if n_gold[lab] else 0.0
        per[lab] = {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "gold": n_gold[lab]}
    macro = {k: float(np.mean([v[k] for v in per.values()])) if per else 0.0 for k in ("p", "r", "f1")}
    return macro, per


def oracle_spans(enc_doc):
    """Decode the gold tags themselves: shows the best score this tokenization allows."""
    real = enc_doc["ends"] > enc_doc["starts"]
    return tags_to_spans(enc_doc["tags"][real], enc_doc["starts"][real], enc_doc["ends"][real])


# ----------------------------------------------------------------------------------- inference
@torch.no_grad()
def doc_log_probs(model, enc_docs, cfg, device, pad_id, batch_size):
    """Per-token log-probabilities for whole documents, stitched from overlapping windows."""
    model.eval()
    out = []
    for doc in enc_docs:
        n = len(doc["ids"])
        wins = make_windows(n, cfg.WINDOW, cfg.STRIDE)
        owner = owner_of_tokens(wins, n)
        lp = np.zeros((n, N_TAGS), dtype=np.float32)
        items = [(0, ws, we) for ws, we in wins]
        for b in range(0, len(items), batch_size):
            chunk = items[b:b + batch_size]
            ids, _, mask = collate([doc], chunk, pad_id)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(ids.to(device), mask.to(device))
            logp = F.log_softmax(logits.float(), -1).cpu().numpy()
            for r, (_, ws, we) in enumerate(chunk):
                k = b + r
                own = np.nonzero(owner[ws:we] == k)[0]
                lp[ws + own] = logp[r, own]
        out.append(lp)
    return out


def decode_docs(enc_docs, log_probs, decoder):
    fn = viterbi if decoder == "viterbi" else greedy
    pred = {}
    for doc, lp in zip(enc_docs, log_probs):
        real = doc["ends"] > doc["starts"]
        tags = fn(lp[real])
        pred[doc["id"]] = tags_to_spans(tags, doc["starts"][real], doc["ends"][real])
    return pred


def evaluate(model, enc_docs, cfg, device, pad_id, batch_size):
    lps = doc_log_probs(model, enc_docs, cfg, device, pad_id, batch_size)
    gold = {d["id"]: d["ents"] for d in enc_docs}
    results = {}
    for dec in ("viterbi", "greedy"):
        macro, per = strict_scores(decode_docs(enc_docs, lps, dec), gold)
        results[dec] = {"macro": macro, "per_label": per}
    return results


def boundary_report(pred, gold):
    """Splits the errors into: exact, right label but wrong boundary, overlapping with the wrong label, missed."""
    counts = {"exact": 0, "boundary": 0, "wrong_label": 0, "missed": 0, "spurious": 0}
    for doc_id, g in gold.items():
        p = list(pred.get(doc_id, []))
        pset = set(p)
        for s, e, lab in g:
            if (s, e, lab) in pset:
                counts["exact"] += 1
                continue
            over = [q for q in p if q[0] < e and q[1] > s]
            if any(q[2] == lab for q in over):
                counts["boundary"] += 1
            elif over:
                counts["wrong_label"] += 1
            else:
                counts["missed"] += 1
        for s, e, lab in p:
            if not any(gs < e and ge > s for gs, ge, _ in g):
                counts["spurious"] += 1
    return counts


def write_submission(pred, test_docs, out_dir, zip_name):
    os.makedirs(out_dir, exist_ok=True)
    payload = {}
    for doc in test_docs:
        n = len(doc["text"])
        spans = sorted(set(pred.get(doc["id"], [])))
        assert all(0 <= s < e <= n and lab in LABELS for s, e, lab in spans), f"bad span in {doc['id']}"
        payload[doc["id"]] = [[s, e, lab] for s, e, lab in spans]
    pred_path = os.path.join(out_dir, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    zip_path = os.path.join(out_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"]
    return pred_path, zip_path
