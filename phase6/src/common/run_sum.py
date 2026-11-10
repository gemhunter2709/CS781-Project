"""
One Phase-6 experiment = one call:

    python run_sum.py --method lorma

Trains for a fixed wall-clock budget (the best checkpoint is chosen by teacher-forced loss on held-out documents),
generates summaries for some held-out documents to measure ROUGE-1 / ROUGE-L and length, then summarizes the test set
and pushes checkpoint, results and submission zip to Hugging Face.
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
import lm as LM
from gen_core import (configure_method, count_params, trainable_state, build_optimizer, length_budget, tidy, rouge,
                      write_submission)


def log(*a):
    print(*a, flush=True)


def build_model(method, keep_layers=None):
    model = LM.build_lm(keep_layers)
    note = configure_method(model, method, C)
    return model, note


def load_finetuned(method, device):
    path = os.path.join(C.OUT_DIR, method, f"sum_{method}.pt")
    if not os.path.exists(path):
        path = LM.fetch_from_hub(C.SUM_REPO, f"{method}/sum_{method}.pt", os.path.join(C.OUT_DIR, "_hub"))
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"No fine-tuned '{method}' model locally or on {C.SUM_REPO}; run '{method}' first.")
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
        api.create_repo(C.SUM_REPO, private=True, exist_ok=True, token=os.environ["HF_TOKEN"])
        for local, remote in files:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=C.SUM_REPO,
                            token=os.environ["HF_TOKEN"], commit_message=message)
        log(f"pushed to {C.SUM_REPO}: {message}")
    except Exception as e:
        log(f"HF push failed ({type(e).__name__}: {e}) - the files are still in {C.OUT_DIR}")


def load_npz(name):
    z = np.load(os.path.join(C.DATA_DIR, f"{name}.npz"))
    return {k: z[k] for k in z.files}


@torch.no_grad()
def dev_loss(model, d, idx, tok, device):
    model.eval()
    total, n = 0.0, 0
    for b in range(0, len(idx), C.MICRO_DOCS):
        j = idx[b:b + C.MICRO_DOCS]
        batch = model.make_batch(d["src"][j], d["src_len"][j], d["tgt"][j], d["tgt_len"][j], tok)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits, gold = model.teacher_forced(batch, device)
        total += F.cross_entropy(logits, gold, reduction="sum").item()
        n += gold.numel()
    return total / max(1, n)


def summarize(model, d, idx, tok, device, tokens_per_word):
    srcs = [d["src"][i][:d["src_len"][i]].tolist() for i in idx]
    ceiling = C.MAX_NEW_TOKENS
    min_new, max_new = length_budget(d["doc_words"][idx], C, tokens_per_word, ceiling)
    out = [None] * len(idx)
    order = np.argsort([len(s) for s in srcs])                    # similar lengths together -> little padding
    for b in range(0, len(order), C.GEN_BATCH):
        part = order[b:b + C.GEN_BATCH]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            toks = model.generate([srcs[i] for i in part], [min_new[i] for i in part], [max_new[i] for i in part],
                                  tok, C, device)
        for i, t in zip(part, toks):
            out[i] = tidy(tok.decode(t), C.SENTENCE_CASE)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["full", "lorma", "lora", "houlsby", "pfeiffer", "prompt", "kd"])
    method = ap.parse_args().method
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(C.OUT_DIR, method)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    tok = LM.load_tokenizer()
    train, test = load_npz("train"), load_npz("test")
    meta = json.load(open(os.path.join(C.DATA_DIR, "meta.json")))
    if C.LENGTH_RATIO == "auto":
        C.LENGTH_RATIO = meta["length_ratio"]          # median document/summary word ratio of the training set
    n = len(train["src_len"])
    perm = np.random.RandomState(C.SEED).permutation(n)
    dev_idx, tr_idx = np.sort(perm[:C.DEV_DOCS]), perm[C.DEV_DOCS:]
    if C.TRAIN_DOC_LIMIT:
        tr_idx = tr_idx[:C.TRAIN_DOC_LIMIT]
    loss_idx, gen_idx = dev_idx[:C.DEV_LOSS_DOCS], dev_idx[:C.DEV_GEN_DOCS]
    log(f"[{method}] train {len(tr_idx):,} docs | held-out {len(dev_idx)} | test {len(test['src_len'])} | "
        f"source {int(train['src_len'].mean())} tokens, summary {int(train['tgt_len'].mean())} tokens on average")

    keep = LM.student_layers() if method == "kd" else None
    model, note = build_model(method, keep)
    model.to(device)
    total, trainable = count_params(model)
    log(f"[{method}] {note} | params {total / 1e6:.1f}M, trainable {trainable / 1e6:.2f}M | budget {C.TRAIN_MINUTES} min")

    teacher = None
    if method == "kd":
        teacher = load_finetuned(C.KD_TEACHER, device)
        log(f"[kd] teacher = fine-tuned '{C.KD_TEACHER}' ({count_params(teacher)[0] / 1e6:.1f}M) -> student keeping "
            f"layers {keep} ({total / 1e6:.1f}M); it learns the teacher's next-token distributions (word-level KD)")

    optimizer = build_optimizer(model, method, C)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    docs_per_step = C.MICRO_DOCS * C.GRAD_ACCUM
    step_cap = max(1, C.MAX_EPOCHS * len(tr_idx) // docs_per_step)
    budget = C.TRAIN_MINUTES * 60
    order = np.random.RandomState(C.SEED).permutation(tr_idx)
    cursor, step, seen, running = 0, 0, 0, 0.0
    best, history = {"loss": float("inf"), "step": 0}, []
    t_train = time.time()

    def progress():
        return max(step / step_cap, (time.time() - t_train) / budget)

    while progress() < 1.0:
        warm = min(1.0, (step + 1) / C.WARMUP_STEPS)
        lr_scale = warm * (C.MIN_LR_RATIO + (1 - C.MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress()))))
        for g, lr in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr * lr_scale
        model.train()
        for _ in range(C.GRAD_ACCUM):
            if cursor + C.MICRO_DOCS > len(order):
                order = np.random.RandomState(C.SEED + step).permutation(tr_idx)
                cursor = 0
            j = np.sort(order[cursor:cursor + C.MICRO_DOCS])
            cursor += C.MICRO_DOCS
            batch = model.make_batch(train["src"][j], train["src_len"][j], train["tgt"][j], train["tgt_len"][j], tok)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, gold = model.teacher_forced(batch, device)
                loss = F.cross_entropy(logits, gold, label_smoothing=C.LABEL_SMOOTHING)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits, _ = teacher.teacher_forced(batch, device)
                    T = C.KD_TEMPERATURE
                    kl = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(t_logits.float() / T, -1),
                                  reduction="batchmean") * T * T
                    loss = C.KD_ALPHA * loss + (1 - C.KD_ALPHA) * kl
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
            dl = dev_loss(model, train, loss_idx, tok, device)
            history.append({"step": step, "docs_seen": step * docs_per_step, "train_loss": running / max(1, seen),
                            "dev_loss": dl, "minutes": (time.time() - t0) / 60})
            log(f"[{method}] step {step} | {step * docs_per_step:,} docs seen | loss {running / max(1, seen):.4f} | "
                f"held-out loss {dl:.4f} (ppl {math.exp(min(dl, 20)):.1f}) | {100 * min(1.0, progress()):.0f}% of budget")
            running, seen = 0.0, 0
            if dl < best["loss"]:
                best = {"loss": dl, "step": step}
                torch.save({"method": method, "keep_layers": keep, "weights": trainable_state(model), "step": step},
                           os.path.join(out_dir, f"sum_{method}.pt"))

    # --------------------------------------------- best checkpoint: summaries for held-out docs, then the test set
    blob = torch.load(os.path.join(out_dir, f"sum_{method}.pt"), map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in blob["weights"].items()}, strict=False)
    model.eval()
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with open(os.path.join(C.DATA_DIR, "train_summaries.json")) as f:
        references = json.load(f)
    t_gen = time.time()
    dev_out = summarize(model, train, gen_idx, tok, device, meta["tokens_per_word"])
    gen_seconds = time.time() - t_gen
    pairs = [rouge(references[int(i)], h) for i, h in zip(gen_idx, dev_out)]
    r1 = float(np.mean([p[0] for p in pairs]))
    rl = float(np.mean([p[1] for p in pairs]))
    words = [len(h.split()) for h in dev_out]
    ref_words = [len(references[int(i)].split()) for i in gen_idx]

    test_out = summarize(model, test, np.arange(len(test["src_len"])), tok, device, meta["tokens_per_word"])
    with open(os.path.join(C.DATA_DIR, "test_ids.json")) as f:
        test_ids = json.load(f)
    _, zip_path = write_submission(test_ids, test_out, out_dir, f"submission_{C.ARCH}_{method}.zip")
    with open(os.path.join(out_dir, "dev_samples.json"), "w") as f:
        json.dump([{"reference": references[int(i)][:1500], "generated": h[:1500]} for i, h in zip(gen_idx[:5], dev_out[:5])],
                  f, indent=1, ensure_ascii=False)

    summary = {
        "arch": C.ARCH, "method": method, "best_step": best["step"], "heldout_loss": best["loss"],
        "heldout_ppl": math.exp(min(best["loss"], 20)), "dev_rouge1_f1": r1, "dev_rougeL_f1": rl,
        "dev_words_generated": float(np.mean(words)), "dev_words_reference": float(np.mean(ref_words)),
        "test_words_generated": float(np.mean([len(s.split()) for s in test_out])),
        "params_total": total, "params_trainable": trainable,
        "seconds_per_summary": round(gen_seconds / max(1, len(gen_idx)), 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0,
        "minutes": round((time.time() - t0) / 60, 1), "docs_seen": step * docs_per_step, "history": history,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[{method}] DONE | held-out loss {best['loss']:.4f} | ROUGE-1 {r1:.4f} ROUGE-L {rl:.4f} on {len(gen_idx)} held-out docs | "
        f"{np.mean(words):.0f} words generated vs {np.mean(ref_words):.0f} in references -> {zip_path}")

    push([(os.path.join(out_dir, f"sum_{method}.pt"), f"{method}/sum_{method}.pt"),
          (os.path.join(out_dir, "results.json"), f"{method}/results.json"),
          (os.path.join(out_dir, "dev_samples.json"), f"{method}/dev_samples.json"),
          (zip_path, f"{method}/{os.path.basename(zip_path)}")], f"{C.ARCH} {method}: ROUGE-L {rl:.4f}")


if __name__ == "__main__":
    main()
