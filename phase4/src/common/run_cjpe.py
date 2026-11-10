"""
One Phase-4 experiment = one call:

    python run_cjpe.py --method lorma

Trains for a fixed wall-clock budget (37K long cases are far more than one Kaggle session can read in full), checks a
dev subset along the way and keeps the best checkpoint, tunes the decision threshold on the whole dev set, then writes
the test predictions + submission zip and pushes everything to Hugging Face.
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
from cjpe_core import (CJPEModel, configure_method, set_unfrozen, count_params, trainable_state, build_optimizer,
                       binary_scores, best_threshold, predict_probs, write_submission)


def log(*a):
    print(*a, flush=True)


def build_model(method, keep_layers=None):
    model = CJPEModel(BB.build_backbone(keep_layers=keep_layers, bidirectional=C.BIDIRECTIONAL), C)
    note = configure_method(model, method, C)
    return model, note


def load_finetuned(method, device):
    path = os.path.join(C.OUT_DIR, method, f"cjpe_{method}.pt")
    if not os.path.exists(path):
        path = BB.fetch_from_hub(C.CJPE_REPO, f"{method}/cjpe_{method}.pt", os.path.join(C.OUT_DIR, "_hub"))
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"No fine-tuned '{method}' model locally or on {C.CJPE_REPO}; run '{method}' first.")
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
        api.create_repo(C.CJPE_REPO, private=True, exist_ok=True, token=os.environ["HF_TOKEN"])
        for local, remote in files:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=C.CJPE_REPO,
                            token=os.environ["HF_TOKEN"], commit_message=message)
        log(f"pushed to {C.CJPE_REPO}: {message}")
    except Exception as e:
        log(f"HF push failed ({type(e).__name__}: {e}) - the files are still in {C.OUT_DIR}")


def load_npz(name):
    z = np.load(os.path.join(C.DATA_DIR, f"{name}.npz"))
    return z["ids"], z["lens"], (z["labels"] if "labels" in z.files else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["full", "lorma", "bitfit", "adapter", "unfreeze", "kd"])
    method = ap.parse_args().method
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(C.OUT_DIR, method)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    tr_ids, tr_lens, tr_y = load_npz("train")
    dv_ids, dv_lens, dv_y = load_npz("dev")
    te_ids, te_lens, _ = load_npz("test")
    with open(os.path.join(C.DATA_DIR, "test_ids.json")) as f:
        test_case_ids = json.load(f)
    if C.TRAIN_DOC_LIMIT:
        tr_ids, tr_lens, tr_y = tr_ids[:C.TRAIN_DOC_LIMIT], tr_lens[:C.TRAIN_DOC_LIMIT], tr_y[:C.TRAIN_DOC_LIMIT]
    sub = np.random.RandomState(0).permutation(len(dv_y))[:C.DEV_SUBSET]          # quick check during training
    if C.DEV_FINAL_LIMIT:
        dv_ids, dv_lens, dv_y = dv_ids[:C.DEV_FINAL_LIMIT], dv_lens[:C.DEV_FINAL_LIMIT], dv_y[:C.DEV_FINAL_LIMIT]
        sub = sub[sub < len(dv_y)]
    log(f"[{method}] train {len(tr_y):,} cases (class 1: {tr_y.mean():.0%}) | dev {len(dv_y):,} | test {len(test_case_ids):,} "
        f"| {tr_ids.shape[1]} chunks x {tr_ids.shape[2]} tokens per case")

    keep = BB.student_layers() if method == "kd" else None
    model, note = build_model(method, keep)
    model.to(device)
    if method == "unfreeze":
        set_unfrozen(model, 0)
    total, trainable = count_params(model)
    log(f"[{method}] {note} | params {total / 1e6:.1f}M, trainable now {trainable / 1e6:.2f}M | budget {C.TRAIN_MINUTES} min")

    teacher = None
    if method == "kd":
        teacher = load_finetuned(C.KD_TEACHER, device)
        log(f"[kd] teacher = fine-tuned '{C.KD_TEACHER}' ({count_params(teacher)[0] / 1e6:.1f}M) -> "
            f"student with backbone layers {keep} ({total / 1e6:.1f}M)")

    counts = np.bincount(tr_y, minlength=2)
    class_w = torch.tensor(len(tr_y) / (2 * counts), dtype=torch.float32, device=device)   # balanced loss

    optimizer = build_optimizer(model, method, C)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    docs_per_step = C.MICRO_DOCS * C.GRAD_ACCUM
    step_cap = max(1, C.MAX_EPOCHS * len(tr_y) // docs_per_step)
    budget = C.TRAIN_MINUTES * 60
    n_layers = len(model.backbone.layers)

    order = np.random.RandomState(C.SEED).permutation(len(tr_y))
    cursor, step, seen, running = 0, 0, 0, 0.0
    best, history = {"f1": -1.0}, []
    t_train = time.time()

    def progress():
        return max(step / step_cap, (time.time() - t_train) / budget)

    while progress() < 1.0:
        p = progress()
        if method == "unfreeze":
            set_unfrozen(model, min(n_layers, 1 + int(p / C.UNFREEZE_BY * n_layers)))   # all layers live by UNFREEZE_BY
        warm = min(1.0, (step + 1) / C.WARMUP_STEPS)
        lr_scale = warm * (C.MIN_LR_RATIO + (1 - C.MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
        for g, lr in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr * lr_scale

        model.train()
        for _ in range(C.GRAD_ACCUM):
            if cursor + C.MICRO_DOCS > len(order):
                order = np.random.RandomState(C.SEED + step).permutation(len(tr_y))
                cursor = 0
            idx = np.sort(order[cursor:cursor + C.MICRO_DOCS])
            cursor += C.MICRO_DOCS
            x = torch.from_numpy(tr_ids[idx].astype(np.int64)).to(device)
            l = torch.from_numpy(tr_lens[idx].astype(np.int64)).to(device)
            y = torch.from_numpy(tr_y[idx].astype(np.int64)).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, doc, _ = model(x, l)
                loss = F.cross_entropy(logits, y, weight=class_w)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits, t_doc, _ = teacher(x, l)
                    T = C.KD_TEMPERATURE
                    kl = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(t_logits.float() / T, -1), reduction="batchmean") * T * T
                    rep = (1 - F.cosine_similarity(doc.float(), t_doc.float(), dim=-1)).mean()
                    loss = C.KD_ALPHA * loss + (1 - C.KD_ALPHA) * kl + C.KD_HIDDEN_WEIGHT * rep
            scaler.scale(loss / C.GRAD_ACCUM).backward()
            running += loss.item()
            seen += 1
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % C.EVAL_EVERY_STEPS == 0 or progress() >= 1.0:
            probs = predict_probs(model, dv_ids[sub], dv_lens[sub], device, C.EVAL_BATCH)
            thr, f1 = best_threshold(dv_y[sub], probs)
            history.append({"step": step, "docs_seen": step * docs_per_step, "train_loss": running / max(1, seen),
                            "dev_subset_f1": f1, "minutes": (time.time() - t0) / 60})
            log(f"[{method}] step {step} | {step * docs_per_step:,} cases seen | loss {running / max(1, seen):.4f} | "
                f"dev-subset macro-F1 {f1:.4f} (thr {thr:.2f}) | {100 * min(1.0, progress()):.0f}% of budget")
            running, seen = 0.0, 0
            if f1 > best["f1"]:
                best = {"f1": f1, "step": step}
                torch.save({"method": method, "keep_layers": keep, "weights": trainable_state(model, method),
                            "step": step}, os.path.join(out_dir, f"cjpe_{method}.pt"))

    # ------------------------------------- best checkpoint: threshold on full dev, then test
    blob = torch.load(os.path.join(out_dir, f"cjpe_{method}.pt"), map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in blob["weights"].items()}, strict=False)
    t_inf = time.time()
    dev_probs = predict_probs(model, dv_ids, dv_lens, device, C.EVAL_BATCH)
    dev_seconds = time.time() - t_inf
    thr, _ = best_threshold(dv_y, dev_probs)
    dev_pred = (dev_probs >= thr).astype(int)
    dev = binary_scores(dv_y, dev_pred)
    dev_at_half = binary_scores(dv_y, (dev_probs >= 0.5).astype(int))
    fp = int(((dev_pred == 1) & (dv_y == 0)).sum())
    fn = int(((dev_pred == 0) & (dv_y == 1)).sum())

    test_probs = predict_probs(model, te_ids, te_lens, device, C.EVAL_BATCH)
    test_pred = (test_probs >= thr).astype(int)
    _, zip_path = write_submission(test_case_ids, test_pred, out_dir, f"submission_{C.ARCH}_{method}.zip")
    np.save(os.path.join(out_dir, "dev_probs.npy"), dev_probs)

    summary = {
        "arch": C.ARCH, "method": method, "best_step": best["step"], "threshold": thr,
        "dev_macro_f1": dev["f1"], "dev_macro_p": dev["p"], "dev_macro_r": dev["r"], "dev_acc": dev["acc"],
        "dev_macro_f1_at_0.5": dev_at_half["f1"], "dev_false_pos": fp, "dev_false_neg": fn,
        "test_pred_class1_share": float(test_pred.mean()),
        "params_total": total, "params_trainable": count_params(model)[1] if method != "unfreeze" else total,
        "dev_inference_seconds": round(dev_seconds, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0,
        "minutes": round((time.time() - t0) / 60, 1), "cases_seen": step * docs_per_step, "history": history,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[{method}] DONE | dev macro-F1 {dev['f1']:.4f} (threshold {thr:.2f}; {dev_at_half['f1']:.4f} at 0.5) | "
        f"FP {fp} FN {fn} | test predicted class 1: {test_pred.mean():.0%} -> {zip_path}")

    push([(os.path.join(out_dir, f"cjpe_{method}.pt"), f"{method}/cjpe_{method}.pt"),
          (os.path.join(out_dir, "results.json"), f"{method}/results.json"),
          (zip_path, f"{method}/{os.path.basename(zip_path)}")], f"{C.ARCH} {method}: dev macro-F1 {dev['f1']:.4f}")


if __name__ == "__main__":
    main()
