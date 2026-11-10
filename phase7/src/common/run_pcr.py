"""
One Phase-7 experiment = one call:

    python run_pcr.py --method lorma

Contrastive training of the bi-encoder for a fixed wall-clock budget (best checkpoint by dev micro-F1@7), then the dense
scores are fused with BM25 - the mixing weight is tuned on dev - and the top 7 test candidates are written out.
"""
import os
import sys
import json
import math
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import backbone as BB
from hub_files import fetch_from_hub
from pcr_core import (Retriever, configure_method, count_params, trainable_state, build_optimizer, embed, rank,
                      micro_f1_at_7, tune_fusion, write_submission, zscore)


def log(*a):
    print(*a, flush=True)


def build_model(method, keep_layers=None):
    model = Retriever(BB.build_backbone(keep_layers=keep_layers, bidirectional=C.BIDIRECTIONAL), C)
    note = configure_method(model, method, C)
    return model, note


def load_finetuned(method, device):
    path = os.path.join(C.OUT_DIR, method, f"pcr_{method}.pt")
    if not os.path.exists(path):
        path = fetch_from_hub(C.PCR_REPO, f"{method}/pcr_{method}.pt", os.path.join(C.OUT_DIR, "_hub"))
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"No fine-tuned '{method}' model locally or on {C.PCR_REPO}; run '{method}' first.")
    blob = torch.load(path, map_location="cpu")
    model, _ = build_model(method, blob.get("keep_layers"))
    _, unexpected = model.load_state_dict({k: v.float() for k, v in blob["weights"].items()}, strict=False)
    assert not unexpected, unexpected
    return model.to(device).eval()


def push(files, message):
    if not os.environ.get("HF_TOKEN"):
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(C.PCR_REPO, private=True, exist_ok=True, token=os.environ["HF_TOKEN"])
        for local, remote in files:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=C.PCR_REPO,
                            token=os.environ["HF_TOKEN"], commit_message=message)
        log(f"pushed to {C.PCR_REPO}: {message}")
    except Exception as e:
        log(f"HF push failed ({type(e).__name__}: {e}) - the files are still in {C.OUT_DIR}")


def load(split):
    z = np.load(os.path.join(C.DATA_DIR, f"{split}.npz"))
    meta = json.load(open(os.path.join(C.DATA_DIR, f"{split}_meta.json")))
    return {k: z[k] for k in z.files}, meta


def dense_scores(model, d, device):
    q = embed(model, d["q_ids"], d["q_lens"], device, C.EVAL_BATCH)
    c = embed(model, d["c_ids"], d["c_lens"], device, C.EVAL_BATCH)
    return q @ c.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["full", "lorma", "ia3", "partial", "prompt", "kd"])
    method = ap.parse_args().method
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(C.OUT_DIR, method)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    tr, tr_meta = load("train")
    dv, dv_meta = load("dev")
    te, te_meta = load("test")
    c_index = {c: j for j, c in enumerate(tr_meta["c_ids"])}
    # (query row, positive candidate rows, hard-negative candidate rows) for every training query with a positive
    examples = []
    for i, (q, rel) in enumerate(zip(tr_meta["q_ids"], tr_meta["rel"])):
        pos = [c_index[c] for c in rel if c in c_index and c != q]
        if pos:
            examples.append((i, pos, [j for j in tr["hard_neg"][i] if j >= 0], c_index.get(q, -1)))
    if C.TRAIN_QUERY_LIMIT:
        examples = examples[:C.TRAIN_QUERY_LIMIT]
    log(f"[{method}] train queries {len(examples)} (pool {len(tr_meta['c_ids'])}) | dev {len(dv_meta['q_ids'])} "
        f"(pool {len(dv_meta['c_ids'])}) | test {len(te_meta['q_ids'])} (pool {len(te_meta['c_ids'])})")

    keep = BB.student_layers() if method == "kd" else None
    model, note = build_model(method, keep)
    model.to(device)
    total, trainable = count_params(model)
    log(f"[{method}] {note} | params {total / 1e6:.1f}M, trainable {trainable / 1e6:.2f}M | budget {C.TRAIN_MINUTES} min")

    teacher = None
    if method == "kd":
        teacher = load_finetuned(C.KD_TEACHER, device)
        log(f"[kd] teacher = fine-tuned '{C.KD_TEACHER}' ({count_params(teacher)[0] / 1e6:.1f}M) -> student keeping "
            f"layers {keep} ({total / 1e6:.1f}M); it learns the teacher's similarity distribution and embeddings")

    optimizer = build_optimizer(model, method, C)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    budget = C.TRAIN_MINUTES * 60
    step_cap = max(1, C.MAX_EPOCHS * len(examples) // (C.MICRO_QUERIES * C.GRAD_ACCUM))
    rng = np.random.RandomState(C.SEED)
    order = rng.permutation(len(examples))
    cursor, step, seen, running = 0, 0, 0, 0.0
    best, history = {"f1": -1.0, "step": 0}, []
    t_train = time.time()

    def progress():
        return max(step / step_cap, (time.time() - t_train) / budget)

    def as_tensor(rows_ids, rows_lens):
        return (torch.from_numpy(rows_ids.astype(np.int64)).to(device), torch.from_numpy(rows_lens.astype(np.int64)).to(device))

    def sample_chunks(ids, lens):
        # training reads TRAIN_CHUNKS random chunks of each case (all of them at evaluation) - cheaper and a mild augmentation
        k = C.TRAIN_CHUNKS
        if k >= ids.shape[1]:
            return ids, lens
        out_ids = np.zeros((len(ids), k, ids.shape[2]), dtype=ids.dtype)
        out_lens = np.zeros((len(ids), k), dtype=lens.dtype)
        for r in range(len(ids)):
            real = np.flatnonzero(lens[r] > 0)
            pick = np.sort(rng.choice(real, size=min(k, len(real)), replace=False))
            out_ids[r, :len(pick)], out_lens[r, :len(pick)] = ids[r, pick], lens[r, pick]
        return out_ids, out_lens

    while progress() < 1.0:
        warm = min(1.0, (step + 1) / C.WARMUP_STEPS)
        lr_scale = warm * (C.MIN_LR_RATIO + (1 - C.MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress()))))
        for g, lr in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr * lr_scale
        model.train()
        for _ in range(C.GRAD_ACCUM):
            if cursor + C.MICRO_QUERIES > len(order):
                order, cursor = rng.permutation(len(examples)), 0
            batch = [examples[k] for k in order[cursor:cursor + C.MICRO_QUERIES]]
            cursor += C.MICRO_QUERIES
            q_rows = [b[0] for b in batch]
            pos_rows = [b[1][rng.randint(len(b[1]))] for b in batch]
            neg_rows = [b[2][rng.randint(len(b[2]))] if b[2] else rng.randint(len(tr["c_ids"])) for b in batch]
            cand_rows = pos_rows + neg_rows
            # a candidate that is relevant to another query in the batch must not count as its negative
            relevant = [set(b[1]) | {b[3]} for b in batch]                 # ... and neither may its own copy in the pool
            mask = torch.tensor([[j != i and cand_rows[j] in relevant[i] for j in range(len(cand_rows))]
                                 for i in range(len(batch))], device=device)
            target = torch.arange(len(batch), device=device)
            q_in = as_tensor(*sample_chunks(tr["q_ids"][q_rows], tr["q_lens"][q_rows]))
            c_in = as_tensor(*sample_chunks(tr["c_ids"][cand_rows], tr["c_lens"][cand_rows]))
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                q_emb = model(*q_in)
                c_emb = model(*c_in)
                logits = (q_emb.float() @ c_emb.float().T / C.TEMPERATURE).masked_fill(mask, -1e4)
                loss = F.cross_entropy(logits, target)                              # InfoNCE
                if teacher is not None:
                    with torch.no_grad():
                        tq = teacher(*q_in).float()
                        tc = teacher(*c_in).float()
                        t_logits = (tq @ tc.T / C.TEMPERATURE).masked_fill(mask, -1e4)
                    T = C.KD_TEMPERATURE
                    kl = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(t_logits / T, -1), reduction="batchmean") * T * T
                    rep = 2 - F.cosine_similarity(q_emb.float(), tq).mean() - F.cosine_similarity(c_emb.float(), tc).mean()
                    loss = C.KD_ALPHA * loss + (1 - C.KD_ALPHA) * kl + C.KD_HIDDEN_WEIGHT * rep
            scaler.scale(loss / C.GRAD_ACCUM).backward()
            running += loss.item()
            seen += 1
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % C.EVAL_EVERY_STEPS == 0 or progress() >= 1.0:
            dense = dense_scores(model, dv, device)
            alpha, _ = tune_fusion(dense, dv["bm25"], dv_meta["q_ids"], dv_meta["c_ids"], dv_meta["rel"], C.FUSION_GRID)
            hybrid = micro_f1_at_7(rank(alpha * zscore(dense) + (1 - alpha) * zscore(dv["bm25"]), dv_meta["q_ids"], dv_meta["c_ids"]), dv_meta["rel"])
            dense_only = micro_f1_at_7(rank(dense, dv_meta["q_ids"], dv_meta["c_ids"]), dv_meta["rel"])
            history.append({"step": step, "train_loss": running / max(1, seen), "dev_dense_f1": dense_only["f1"],
                            "dev_hybrid_f1": hybrid["f1"], "alpha": alpha, "minutes": (time.time() - t0) / 60})
            log(f"[{method}] step {step} | loss {running / max(1, seen):.4f} | dev F1@7 dense {dense_only['f1']:.4f}, "
                f"hybrid {hybrid['f1']:.4f} (alpha {alpha:.1f}) | {100 * min(1.0, progress()):.0f}% of budget")
            running, seen = 0.0, 0
            if hybrid["f1"] > best["f1"]:
                best = {"f1": hybrid["f1"], "step": step}
                torch.save({"method": method, "keep_layers": keep, "weights": trainable_state(model), "step": step},
                           os.path.join(out_dir, f"pcr_{method}.pt"))

    # --------------------------------------- best checkpoint: fusion weight on dev, then rank the test pool
    blob = torch.load(os.path.join(out_dir, f"pcr_{method}.pt"), map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in blob["weights"].items()}, strict=False)
    t_inf = time.time()
    dense = dense_scores(model, dv, device)
    dev_seconds = time.time() - t_inf
    alpha, grid = tune_fusion(dense, dv["bm25"], dv_meta["q_ids"], dv_meta["c_ids"], dv_meta["rel"], C.FUSION_GRID)
    scores = {a: r for a, r in grid}
    dense_only = micro_f1_at_7(rank(dense, dv_meta["q_ids"], dv_meta["c_ids"]), dv_meta["rel"])
    bm25_only = micro_f1_at_7(rank(dv["bm25"], dv_meta["q_ids"], dv_meta["c_ids"]), dv_meta["rel"])

    test_dense = dense_scores(model, te, device)
    preds = rank(alpha * zscore(test_dense) + (1 - alpha) * zscore(te["bm25"]), te_meta["q_ids"], te_meta["c_ids"])
    _, zip_path = write_submission(te_meta["q_ids"], preds, out_dir, f"submission_{C.ARCH}_{method}.zip", te_meta["c_ids"])

    summary = {
        "arch": C.ARCH, "method": method, "best_step": best["step"], "alpha": alpha,
        "dev_f1": scores[alpha]["f1"], "dev_p": scores[alpha]["p"], "dev_r": scores[alpha]["r"],
        "dev_dense_only_f1": dense_only["f1"], "dev_bm25_only_f1": bm25_only["f1"],
        "fusion_grid": [{"alpha": a, "f1": r["f1"]} for a, r in grid],
        "params_total": total, "params_trainable": trainable, "dev_encode_seconds": round(dev_seconds, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0,
        "minutes": round((time.time() - t0) / 60, 1), "steps": step, "history": history,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[{method}] DONE | dev micro-F1@7 {scores[alpha]['f1']:.4f} with alpha {alpha:.1f} (dense only {dense_only['f1']:.4f}, "
        f"BM25 only {bm25_only['f1']:.4f}) -> {zip_path}")

    push([(os.path.join(out_dir, f"pcr_{method}.pt"), f"{method}/pcr_{method}.pt"),
          (os.path.join(out_dir, "results.json"), f"{method}/results.json"),
          (zip_path, f"{method}/{os.path.basename(zip_path)}")], f"{C.ARCH} {method}: dev F1@7 {scores[alpha]['f1']:.4f}")



if __name__ == "__main__":
    main()
