"""
One Phase-2 experiment = one call:

    python run_ner.py --method lorma

Fine-tunes the Phase-1 model (or a distilled student of it) for legal NER, keeps the epoch with the best
dev strict macro-F1, writes the test predictions + submission zip and pushes everything to Hugging Face.
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
from ner_core import (load_docs, split_dev, encode_doc, window_items, collate, Tagger, configure_method,
                      count_params, trainable_state, build_optimizer, evaluate, doc_log_probs, decode_docs,
                      strict_scores, boundary_report, write_submission, N_TAGS, IGNORE)


def log(*a):
    print(*a, flush=True)


def build_tagger(method, keep_layers=None):
    model = Tagger(BB.build_backbone(keep_layers=keep_layers, bidirectional=C.BIDIRECTIONAL), C.HEAD_DROPOUT)
    note = configure_method(model, method, C)
    return model, note


def load_finetuned(method, device):
    """Rebuild a model trained earlier by this notebook (used as the KD teacher)."""
    path = os.path.join(C.OUT_DIR, method, f"ner_{method}.pt")
    if not os.path.exists(path):
        path = BB.fetch_from_hub(C.NER_REPO, f"{method}/ner_{method}.pt", os.path.join(C.OUT_DIR, "_hub"))
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"No fine-tuned '{method}' model found locally or on {C.NER_REPO}. "
                                f"Run the '{method}' experiment first - it is the distillation teacher.")
    blob = torch.load(path, map_location="cpu")
    model, _ = build_tagger(method, blob.get("keep_layers"))
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in blob["weights"].items()}, strict=False)
    assert not unexpected, unexpected
    return model.to(device).eval()


def class_weights(enc_docs, device):
    if C.CLASS_WEIGHTING != "sqrt_inv":
        return None
    counts = np.ones(N_TAGS)
    for d in enc_docs:
        t = d["tags"][d["tags"] != IGNORE]
        counts += np.bincount(t, minlength=N_TAGS)
    w = 1.0 / np.sqrt(counts / counts.sum())
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


def push(files, message):
    if not os.environ.get("HF_TOKEN"):
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(C.NER_REPO, private=True, exist_ok=True, token=os.environ["HF_TOKEN"])
        for local, remote in files:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=C.NER_REPO,
                            token=os.environ["HF_TOKEN"], commit_message=message)
        log(f"pushed to {C.NER_REPO}: {message}")
    except Exception as e:
        log(f"HF push failed ({type(e).__name__}: {e}) - the files are still in {C.OUT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["full", "partial", "lora", "lorma", "kd"])
    args = ap.parse_args()
    method = args.method

    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(C.OUT_DIR, method)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ---------------------------------------------------------------- data
    tok = BB.load_tokenizer()
    folds = [load_docs(C.FOLD_TEXT[i], C.FOLD_GOLD[i]) for i in range(2)]
    train_docs, dev_docs = split_dev(folds, C.DEV_DOCS_PER_FOLD, C.SEED)
    if C.TRAIN_DOC_LIMIT:                      # quick smoke test only
        train_docs, dev_docs = train_docs[:C.TRAIN_DOC_LIMIT], dev_docs[:max(1, C.TRAIN_DOC_LIMIT // 4)]
    test_docs = load_docs(C.TEST_TEXT)
    enc_train = [encode_doc(d, tok) for d in train_docs]
    enc_dev = [encode_doc(d, tok) for d in dev_docs]
    enc_test = [encode_doc(d, tok) for d in test_docs]
    log(f"[{method}] train docs {len(enc_train)} | dev docs {len(enc_dev)} | test docs {len(enc_test)} | "
        f"train tokens {sum(len(d['ids']) for d in enc_train):,}")

    # --------------------------------------------------------------- model
    keep = BB.student_layers() if method == "kd" else None
    model, note = build_tagger(method, keep)
    model.to(device)
    total, trainable = count_params(model)
    log(f"[{method}] {note} | params {total / 1e6:.1f}M, trainable {trainable / 1e6:.2f}M "
        f"({100 * trainable / total:.2f}%) | bidirectional attention: {C.BIDIRECTIONAL}")

    teacher = None
    if method == "kd":
        teacher = load_finetuned(C.KD_TEACHER, device)       # fp32 weights; autocast runs it in fp16 on the GPU
        t_total, _ = count_params(teacher)
        log(f"[kd] teacher = fine-tuned '{C.KD_TEACHER}' model ({t_total / 1e6:.1f}M) -> "
            f"student keeps layers {keep} ({total / 1e6:.1f}M)")

    # ------------------------------------------------------------ training
    items = window_items(enc_train, C.WINDOW, C.STRIDE)
    steps_per_epoch = math_ceil(len(items), C.MICRO_BATCH * C.GRAD_ACCUM)
    total_steps = steps_per_epoch * C.EPOCHS
    warmup = max(1, int(C.WARMUP_FRAC * total_steps))
    optimizer = build_optimizer(model, method, C)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights = class_weights(enc_train, device)
    rng = random.Random(C.SEED)

    best = {"f1": -1.0}
    stale, step = 0, 0
    history = []
    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        rng.shuffle(items)
        running, seen = 0.0, 0
        for b in range(0, len(items), C.MICRO_BATCH):
            ids, tags, mask = collate(enc_train, items[b:b + C.MICRO_BATCH], tok.pad_id)
            ids, tags, mask = ids.to(device), tags.to(device), mask.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(ids, mask).float()
                loss = F.cross_entropy(logits.view(-1, N_TAGS), tags.view(-1), weight=weights, ignore_index=IGNORE)
                if teacher is not None:
                    # Hinton et al. 2015: match the teacher's softened tag distribution on every real token
                    with torch.no_grad():
                        t_logits = teacher(ids, mask).float()
                    keep_tok = tags.view(-1) != IGNORE
                    T = C.KD_TEMPERATURE
                    kl = F.kl_div(F.log_softmax(logits.view(-1, N_TAGS)[keep_tok] / T, -1),
                                  F.softmax(t_logits.view(-1, N_TAGS)[keep_tok] / T, -1),
                                  reduction="batchmean") * T * T
                    loss = C.KD_ALPHA * loss + (1 - C.KD_ALPHA) * kl
            scaler.scale(loss / C.GRAD_ACCUM).backward()
            running += loss.item()
            seen += 1

            if seen % C.GRAD_ACCUM == 0 or b + C.MICRO_BATCH >= len(items):
                # linear warm-up, then linear decay to zero
                if step < warmup:
                    frac = (step + 1) / warmup
                else:
                    frac = max(0.0, (total_steps - step) / max(1, total_steps - warmup))
                for g, lr in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = lr * frac
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

        res = evaluate(model, enc_dev, C, device, tok.pad_id, C.EVAL_BATCH)
        dec = max(res, key=lambda k: res[k]["macro"]["f1"])
        f1 = res[dec]["macro"]["f1"]
        history.append({"epoch": epoch, "train_loss": running / max(1, seen),
                        "dev_f1_viterbi": res["viterbi"]["macro"]["f1"], "dev_f1_greedy": res["greedy"]["macro"]["f1"]})
        log(f"[{method}] epoch {epoch}/{C.EPOCHS} | loss {running / max(1, seen):.4f} | dev strict macro-F1 "
            f"viterbi {res['viterbi']['macro']['f1']:.4f}  greedy {res['greedy']['macro']['f1']:.4f} | "
            f"{(time.time() - t0) / 60:.1f} min")
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "decoder": dec, "dev": res[dec]}
            torch.save({"method": method, "keep_layers": keep, "weights": trainable_state(model),
                        "dev_f1": f1, "decoder": dec, "epoch": epoch}, os.path.join(out_dir, f"ner_{method}.pt"))
            stale = 0
        else:
            stale += 1
            if stale >= C.PATIENCE:
                log(f"[{method}] no dev improvement for {C.PATIENCE} epochs - stopping early")
                break

    # ------------------------------------------------- best model -> test
    blob = torch.load(os.path.join(out_dir, f"ner_{method}.pt"), map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in blob["weights"].items()}, strict=False)
    t_inf = time.time()
    dev_lp = doc_log_probs(model, enc_dev, C, device, tok.pad_id, C.EVAL_BATCH)
    dev_pred = decode_docs(enc_dev, dev_lp, best["decoder"])
    dev_seconds = time.time() - t_inf
    errors = boundary_report(dev_pred, {d["id"]: d["ents"] for d in enc_dev})
    test_pred = decode_docs(enc_test, doc_log_probs(model, enc_test, C, device, tok.pad_id, C.EVAL_BATCH), best["decoder"])
    pred_path, zip_path = write_submission(test_pred, test_docs, out_dir, f"submission_{C.ARCH}_{method}.zip")

    summary = {
        "arch": C.ARCH, "method": method, "best_epoch": best["epoch"], "decoder": best["decoder"],
        "dev_macro": best["dev"]["macro"], "dev_per_label": best["dev"]["per_label"], "dev_errors": errors,
        "params_total": total, "params_trainable": trainable, "dev_inference_seconds": round(dev_seconds, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0,
        "train_minutes": round((time.time() - t0) / 60, 1), "history": history,
        "test_entities": sum(len(v) for v in test_pred.values()),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[{method}] DONE | best dev strict macro-F1 {best['f1']:.4f} (epoch {best['epoch']}, {best['decoder']}) | "
        f"errors {errors} | {summary['test_entities']} test entities -> {zip_path}")

    push([(os.path.join(out_dir, f"ner_{method}.pt"), f"{method}/ner_{method}.pt"),
          (os.path.join(out_dir, "results.json"), f"{method}/results.json"),
          (zip_path, f"{method}/{os.path.basename(zip_path)}")], f"{C.ARCH} {method}: dev F1 {best['f1']:.4f}")


def math_ceil(a, b):
    return -(-a // b)


if __name__ == "__main__":
    main()
