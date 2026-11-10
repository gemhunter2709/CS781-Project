"""
One-off data preparation (skipped automatically for anything already on disk):

  1. learn the byte-level BPE merges from a spread-out sample of the training text
  2. tokenize the corpus with all CPU cores into flat uint16 files (train.bin / val.bin)
  3. mirror tokenizer + bins to the Hugging Face repo so a Kaggle restart never repeats this

Training later memory-maps those files, so RAM usage during training stays tiny.
"""
import os
import sys
import json
import time
from collections import deque
from multiprocessing import Pool

import numpy as np
import ijson
from huggingface_hub import hf_hub_download, HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from bpe_tokenizer import ByteBPE

TOKEN = os.environ.get("HF_TOKEN")


def local_train_files():
    paths = []
    for name in C.TRAIN_FILES:
        paths.append(hf_hub_download(C.HF_DATASET_REPO, name, repo_type="dataset", token=TOKEN))
    return paths


def iter_texts(path):
    """Stream the 'text' field of every record without loading the whole json file."""
    with open(path, "rb") as f:
        for rec in ijson.items(f, "item"):
            text = rec.get("text", "")
            if len(text) >= C.MIN_DOC_CHARS:
                yield text


# ------------------------------------------------------------------ stage 1
def tokenizer_sample(paths):
    budget = C.TOK_SAMPLE_MB_PER_FILE * 1_000_000
    for p in paths:
        used = 0
        for i, text in enumerate(iter_texts(p)):
            if i % C.TOK_SAMPLE_STRIDE:
                continue
            yield text
            used += len(text)
            if used >= budget:
                break
        print(f"  sampled {used / 1e6:.0f} MB from {os.path.basename(p)}", flush=True)


def train_tokenizer(paths):
    print("[1/2] Training BPE tokenizer ...", flush=True)
    t0 = time.time()
    tok = ByteBPE.train(tokenizer_sample(paths), vocab_size=C.VOCAB_SIZE)
    tok.save(C.TOKENIZER_PATH)
    print(f"  done in {(time.time() - t0) / 60:.1f} min, {len(tok.merges):,} merges", flush=True)


# ------------------------------------------------------------------ stage 2
_tok = None


def _init_worker(path):
    global _tok
    _tok = ByteBPE.load(path)


def _encode_batch(batch):
    """batch: list of (is_val, text). Every doc is stored as <eos> + tokens."""
    train, val = [], []
    for is_val, text in batch:
        ids = [_tok.eos_id]
        ids.extend(_tok.encode(text))
        (val if is_val else train).append(np.asarray(ids, dtype=np.uint16))
    cat = lambda xs: np.concatenate(xs) if xs else np.empty(0, dtype=np.uint16)
    return cat(train), cat(val)


def interleaved_texts(paths, weights):
    """
    Walk all source files at once and always pull the next document from the source that is
    furthest behind its weighted share (measured in characters). A source that ends is dropped.
    """
    streams = [iter_texts(p) for p in paths]
    taken = [0.0] * len(paths)
    alive = [True] * len(paths)
    while any(alive):
        i = min((j for j in range(len(paths)) if alive[j]), key=lambda j: taken[j] / weights[j])
        try:
            text = next(streams[i])
        except StopIteration:
            alive[i] = False
            print(f"  source exhausted: {os.path.basename(paths[i])}", flush=True)
            continue
        taken[i] += len(text)
        yield text


def doc_batches(paths, max_chars=2_000_000):
    batch, chars, n = [], 0, 0
    for text in interleaved_texts(paths, C.SOURCE_WEIGHTS):
        batch.append((n % C.VAL_EVERY_NTH_DOC == 0, text))
        chars += len(text)
        n += 1
        if chars >= max_chars:
            yield batch
            batch, chars = [], 0
    if batch:
        yield batch


def tokenize_corpus(paths):
    print(f"[2/2] Tokenizing with {C.PREP_WORKERS} workers -> {C.TRAIN_TOKENS / 1e6:.0f}M train tokens", flush=True)
    t0 = time.time()
    n_train = n_val = 0
    next_report = 20_000_000
    window = 3 * C.PREP_WORKERS               # keep only a few batches in flight -> bounded RAM

    with open(C.TRAIN_BIN, "wb") as ftrain, open(C.VAL_BIN, "wb") as fval, \
            Pool(C.PREP_WORKERS, initializer=_init_worker, initargs=(C.TOKENIZER_PATH,)) as pool:

        def absorb(result):
            nonlocal n_train, n_val, next_report
            tr, va = result.get()
            if n_val < C.VAL_TOKENS and len(va):
                va = va[: C.VAL_TOKENS - n_val]
                fval.write(va.tobytes())
                n_val += len(va)
            if n_train < C.TRAIN_TOKENS + 1 and len(tr):
                tr = tr[: C.TRAIN_TOKENS + 1 - n_train]     # +1 so the last block has a target
                ftrain.write(tr.tobytes())
                n_train += len(tr)
            if n_train >= next_report:
                rate = n_train / (time.time() - t0)
                eta = (C.TRAIN_TOKENS - n_train) / max(rate, 1) / 60
                print(f"  {n_train / 1e6:6.0f}M tokens | {rate / 1e3:.0f}K tok/s | eta {eta:.1f} min", flush=True)
                next_report += 20_000_000

        pending = deque()
        for batch in doc_batches(paths):
            pending.append(pool.apply_async(_encode_batch, (batch,)))
            if len(pending) >= window:
                absorb(pending.popleft())
            if n_train >= C.TRAIN_TOKENS + 1:
                break
        while pending and n_train < C.TRAIN_TOKENS + 1:
            absorb(pending.popleft())
        pool.terminate()

    if n_train < C.TRAIN_TOKENS:
        print(f"  WARNING: corpus only yielded {n_train:,} tokens; training will loop over it.", flush=True)
    meta = {"train_tokens": n_train, "val_tokens": n_val, "vocab_size": C.VOCAB_SIZE, "dtype": "uint16"}
    with open(C.META_PATH, "w") as f:
        json.dump(meta, f)
    print(f"  finished in {(time.time() - t0) / 60:.1f} min: {n_train:,} train / {n_val:,} val tokens", flush=True)


# ------------------------------------------------------------------- hub
def mirror_to_hub():
    api = HfApi()
    api.create_repo(C.HF_MODEL_REPO, token=TOKEN, private=C.HF_PRIVATE, exist_ok=True)
    # meta goes last: its presence on the hub means the whole upload finished
    for local in (C.TOKENIZER_PATH, C.VAL_BIN, C.TRAIN_BIN, C.META_PATH):
        name = "data/" + os.path.basename(local)
        print(f"  uploading {name} ...", flush=True)
        api.upload_file(path_or_fileobj=local, path_in_repo=name, repo_id=C.HF_MODEL_REPO, token=TOKEN)


def main():
    os.makedirs(C.DATA_DIR, exist_ok=True)
    have_bins = all(os.path.exists(p) for p in (C.TRAIN_BIN, C.VAL_BIN, C.META_PATH, C.TOKENIZER_PATH))
    if have_bins:
        print("Tokenizer and token files already present - nothing to do.")
        return
    paths = local_train_files()
    if not os.path.exists(C.TOKENIZER_PATH):
        train_tokenizer(paths)
    tokenize_corpus(paths)
    print("Mirroring to Hugging Face ...", flush=True)
    mirror_to_hub()
    print("Data prep complete.")


if __name__ == "__main__":
    main()
