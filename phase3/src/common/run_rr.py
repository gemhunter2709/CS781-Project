"""
One Phase-3 experiment = one call:

    python run_rr.py --method lorma

Trains on CL + IT together (a domain embedding tells them apart), keeps the epoch with the best dev macro-F1,
tunes the decoder (CRF vs argmax) and the prior correction on dev, writes the test predictions + submission zip
and pushes everything to Hugging Face.
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import backbone as BB
from rr_core import (load_split, tokenize_docs, RRModel, configure_method, count_params, trainable_state,
                     build_optimizer, class_weights, macro_scores, top_confusions, document_emissions, decode,
                     tune_decoding, write_submission, training_spans, DOMAINS, LABELS)


def log(*a):
    print(*a, flush=True)


def build_model(method, keep_layers=None):
    model = RRModel(BB.build_backbone(keep_layers=keep_layers, bidirectional=C.BIDIRECTIONAL), C)
    note = configure_method(model, method, C, model.backbone.tok_emb.weight)
    return model, note


def load_finetuned(method, device):
    """Rebuild a model trained earlier by this notebook (the distillation teacher)."""
    path = os.path.join(C.OUT_DIR, method, f"rr_{method}.pt")
    if not os.path.exists(path):
        path = BB.fetch_from_hub(C.RR_REPO, f"{method}/rr_{method}.pt", os.path.join(C.OUT_DIR, "_hub"))
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"No fine-tuned '{method}' model locally or on {C.RR_REPO}; run '{method}' first.")
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
        api.create_repo(C.RR_REPO, private=True, exist_ok=True, token=os.environ["HF_TOKEN"])
        for local, remote in files:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=C.RR_REPO,
                            token=os.environ["HF_TOKEN"], commit_message=message)
        log(f"pushed to {C.RR_REPO}: {message}")
    except Exception as e:
        log(f"HF push failed ({type(e).__name__}: {e}) - the files are still in {C.OUT_DIR}")


def evaluate(model, docs, pad_id, device, log_prior):
    ems = document_emissions(model, docs, pad_id, device, C.EVAL_PACK_BATCH)
    gold = [l for d in docs for l in d["labels"]]
    pred = [l for p in decode(model, ems, log_prior, "crf", 0.0) for l in p]
    return macro_scores(gold, pred)[0]["f1"], ems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["full", "lora", "lorma", "ia3", "prompt", "kd"])
    method = ap.parse_args().method

    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(C.OUT_DIR, method)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ---------------------------------------------------------------- data
    tok = BB.load_tokenizer()
    train = [d for dom in DOMAINS for d in load_split(C.FILES[f"{dom}_train"], dom)]
    dev = [d for dom in DOMAINS for d in load_split(C.FILES[f"{dom}_dev"], dom)]
    test = [d for dom in DOMAINS for d in load_split(C.FILES[f"{dom}_test"], dom)]
    if C.TRAIN_ON_DEV:
        train = train + dev                                     # final run only: dev is then no longer a fair check
    if C.TRAIN_DOC_LIMIT:
        train = train[:C.TRAIN_DOC_LIMIT // 2] + train[-(C.TRAIN_DOC_LIMIT // 2):]
    for docs in (train, dev, test):
        tokenize_docs(docs, tok, C.MAX_SENT_TOKENS)
    log(f"[{method}] train docs {len(train)} ({sum(len(d['sents']) for d in train):,} sentences) | "
        f"dev docs {len(dev)} | test docs {len(test)}")

    weights, log_prior = class_weights(train, C.CLASS_WEIGHT_POWER)
    weights = weights.to(device)

    # --------------------------------------------------------------- model
    keep = BB.student_layers() if method == "kd" else None
    model, note = build_model(method, keep)
    model.to(device)
    total, trainable = count_params(model)
    log(f"[{method}] {note} | params {total / 1e6:.1f}M, trainable {trainable / 1e6:.2f}M ({100 * trainable / total:.2f}%)")

    teacher = None
    if method == "kd":
        teacher = load_finetuned(C.KD_TEACHER, device)
        log(f"[kd] teacher = fine-tuned '{C.KD_TEACHER}' ({count_params(teacher)[0] / 1e6:.1f}M) -> "
            f"student with backbone layers {keep} ({total / 1e6:.1f}M)")

    optimizer = build_optimizer(model, method, C)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = random.Random(C.SEED)
    n_spans = len(training_spans(train, C.SPAN_SENTS, random.Random(0)))
    total_steps = -(-n_spans // C.GRAD_ACCUM) * C.EPOCHS
    warmup = max(1, int(C.WARMUP_FRAC * total_steps))

    best, stale, step, history = {"f1": -1.0}, 0, 0, []
    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        spans = training_spans(train, C.SPAN_SENTS, rng)
        running, seen = 0.0, 0
        for i, (di, a, b) in enumerate(spans):
            d = train[di]
            gold = torch.tensor(d["labels"][a:b], device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                ctx, em, shift = model(d["tok"][a:b], d["domain"], tok.pad_id, device, n_total=len(d["sents"]), offset=a)
                loss = model.crf.nll(em, gold) / len(gold)
                loss = loss + C.CE_WEIGHT * F.cross_entropy(em, gold, weight=weights)
                if len(gold) > 1:
                    changed = (gold[1:] != gold[:-1]).float()
                    loss = loss + C.SHIFT_WEIGHT * F.binary_cross_entropy_with_logits(shift, changed)
                if teacher is not None:
                    with torch.no_grad():
                        t_ctx, t_em, _ = teacher(d["tok"][a:b], d["domain"], tok.pad_id, device, n_total=len(d["sents"]), offset=a)
                    T = C.KD_TEMPERATURE
                    kl = F.kl_div(F.log_softmax(em / T, -1), F.softmax(t_em.float() / T, -1), reduction="batchmean") * T * T
                    hid = F.mse_loss(ctx.float(), t_ctx.float())     # match the teacher's sentence representations too
                    loss = C.KD_ALPHA * loss + (1 - C.KD_ALPHA) * kl + C.KD_HIDDEN_WEIGHT * hid
            scaler.scale(loss / C.GRAD_ACCUM).backward()
            running += loss.item()
            seen += 1

            if seen % C.GRAD_ACCUM == 0 or i == len(spans) - 1:
                frac = (step + 1) / warmup if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
                for g, lr in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = lr * frac
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

        f1, _ = evaluate(model, dev, tok.pad_id, device, log_prior)
        history.append({"epoch": epoch, "train_loss": running / max(1, seen), "dev_macro_f1": f1})
        log(f"[{method}] epoch {epoch}/{C.EPOCHS} | loss {running / max(1, seen):.4f} | dev macro-F1 (CRF) {f1:.4f} | "
            f"{(time.time() - t0) / 60:.1f} min")
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch}
            torch.save({"method": method, "keep_layers": keep, "weights": trainable_state(model), "dev_f1": f1,
                        "epoch": epoch}, os.path.join(out_dir, f"rr_{method}.pt"))
            stale = 0
        else:
            stale += 1
            if stale >= C.PATIENCE:
                log(f"[{method}] no dev improvement for {C.PATIENCE} epochs - stopping early")
                break

    # ------------------------------------------- best epoch: tune decoding on dev, predict test
    blob = torch.load(os.path.join(out_dir, f"rr_{method}.pt"), map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in blob["weights"].items()}, strict=False)
    t_inf = time.time()
    dev_ems = document_emissions(model, dev, tok.pad_id, device, C.EVAL_PACK_BATCH)
    dev_seconds = time.time() - t_inf
    choice, grid = tune_decoding(model, dev, dev_ems, log_prior, C.TAU_GRID)
    dev_pred = decode(model, dev_ems, log_prior, choice["decoder"], choice["tau"])

    gold_all = [l for d in dev for l in d["labels"]]
    pred_all = [l for p in dev_pred for l in p]
    macro, per = macro_scores(gold_all, pred_all)
    by_domain = {}
    for k, dom in enumerate(DOMAINS):
        g = [l for d in dev if d["domain"] == k for l in d["labels"]]
        p = [l for d, pr in zip(dev, dev_pred) if d["domain"] == k for l in pr]
        by_domain[dom] = macro_scores(g, p)[0]

    test_ems = document_emissions(model, test, tok.pad_id, device, C.EVAL_PACK_BATCH)
    test_pred = decode(model, test_ems, log_prior, choice["decoder"], choice["tau"])
    _, zip_path = write_submission(test, test_pred, out_dir, f"submission_{C.ARCH}_{method}.zip")

    summary = {
        "arch": C.ARCH, "method": method, "best_epoch": best["epoch"], "decoder": choice["decoder"], "tau": choice["tau"],
        "dev_macro": macro, "dev_by_domain": by_domain, "dev_per_label": per, "decoding_grid": grid,
        "confusions": [[g, p, n] for (g, p), n in top_confusions(gold_all, pred_all)],
        "params_total": total, "params_trainable": trainable, "dev_inference_seconds": round(dev_seconds, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0,
        "train_minutes": round((time.time() - t0) / 60, 1), "history": history,
        "trained_on_dev": bool(C.TRAIN_ON_DEV),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[{method}] DONE | dev macro-F1 {macro['f1']:.4f} (CL {by_domain['CL']['f1']:.4f}, IT {by_domain['IT']['f1']:.4f}) "
        f"| decoder {choice['decoder']}, tau {choice['tau']} | epoch {best['epoch']} -> {zip_path}")

    push([(os.path.join(out_dir, f"rr_{method}.pt"), f"{method}/rr_{method}.pt"),
          (os.path.join(out_dir, "results.json"), f"{method}/results.json"),
          (zip_path, f"{method}/{os.path.basename(zip_path)}")], f"{C.ARCH} {method}: dev macro-F1 {macro['f1']:.4f}")


if __name__ == "__main__":
    main()
