"""Builds the four Phase-4 notebooks (court judgment prediction), one per Phase-1 model, from the tested sources here."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAPSTONE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CAPSTONE, "phase2_src"))
import build_notebooks as P2          # noqa: E402  (tokenizer + Phase-1 model copies are shared with Phase 2/3)

SPECIFIC = {
    "arch1": {"sites": '["W_o", "down"]', "micro": 4, "accum": 4, "eval_batch": 16},
    "arch2": {"sites": '["proj", "down"]', "micro": 4, "accum": 4, "eval_batch": 16},
    "arch3": {"sites": '["proj", "down"]', "micro": 4, "accum": 4, "eval_batch": 16},
    "arch4": {"sites": '["out", "down"]', "micro": 2, "accum": 8, "eval_batch": 8},
}

CONFIG_TEMPLATE = '''"""Every knob of this Phase-4 (court judgment prediction) notebook."""
import os
import json

ARCH = "{arch}"
WORK_DIR = f"/kaggle/working/phase4_{{ARCH}}"
DATA_DIR = os.path.join(WORK_DIR, "data")
CACHE_DIR = os.path.join(WORK_DIR, "phase1")        # Phase-1 files downloaded from Hugging Face land here
OUT_DIR = os.path.join(WORK_DIR, "runs")             # one folder per experiment

HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
RAW = {{s: os.path.join(DATA_DIR, "Phase_4_CJPE", "Test" if s == "test" else "Train", f"{{s}}.json")
       for s in ("train", "dev", "test")}}

# ------------------------------------------------------------------ the Phase-1 model
# Looked up in this order: an attached Kaggle dataset whose path mentions "{arch}", then these HF repos.
{block}
KAGGLE_HINT = "{arch}"

CJPE_REPO = "D-240357/legal-cjpe-{arch}"         # fine-tuned checkpoints, results and submission zips go here

# ------------------------------------------------------------------ experiments
# Each one is a separate submission. "full" is the strong reference and the distillation teacher;
# "kd" waits for it to finish.
METHODS = ["full", "lorma", "bitfit", "adapter", "unfreeze", "kd"]
KD_TEACHER = "full"
TRAIN_MINUTES = 35             # wall-clock training budget per experiment (the LR schedule ends exactly on it)
MAX_EPOCHS = 2                 # upper bound if the GPU is fast enough to get through the data

# ------------------------------------------------------------------ how a case is read
CHUNK_TOKENS = 512
HEAD_CHUNKS = 1                # opening of the judgment (parties, facts)
TAIL_CHUNKS = 3                # its last part, right before the removed verdict - the most telling section in ILDC
CHARS_PER_TOKEN = 8            # only this much text is tokenized from each end (keeps the preprocessing fast)
{bidir_line}
POOL_DIM = 256                 # attention pooling across chunks
HEAD_DROPOUT = 0.1

# ------------------------------------------------------------------ optimisation
MICRO_DOCS = {micro}                 # cases per forward pass (each case = up to 4 chunks)
GRAD_ACCUM = {accum}
EVAL_BATCH = {eval_batch}
WARMUP_STEPS = 50
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.01
HEAD_LR = 1e-3
FULL_LR = 5e-5                 # top layer; each layer below gets LLRD times the one above it
LLRD = 0.9
ADAPTER_LR = 5e-4
BITFIT_LR = 1e-3
ADAPTER_RANK = 8
LORMA_ALPHA = 8
ADAPTER_DROPOUT = 0.05
ADAPTER_TARGETS = {targets}       # LoRMA: attention projections
ADAPTER_SITES = {sites}           # Houlsby adapters: after the attention output and the feed-forward output
ADAPTER_BOTTLENECK = 64
UNFREEZE_BY = 0.5              # gradual unfreezing: every layer is trainable after half the budget

KD_TEMPERATURE = 2.0
KD_ALPHA = 0.5                 # weight of the gold-label loss; the rest goes to matching the teacher
KD_HIDDEN_WEIGHT = 0.5         # also align the student's document vector with the teacher's (cosine)
STUDENT_LAYERS = {student}

# ------------------------------------------------------------------ evaluation
EVAL_EVERY_STEPS = 150         # quick check on DEV_SUBSET cases; the best checkpoint is kept
DEV_SUBSET = 600
DEV_FINAL_LIMIT = None         # the final threshold is tuned on the whole dev set (3,505 cases)
TRAIN_DOC_LIMIT = None         # set to e.g. 400 (and TRAIN_MINUTES = 3) for a quick smoke test
PREP_WORKERS = max(1, os.cpu_count() or 2)

SEED = 13

_override = os.environ.get("P4_CFG_OVERRIDE")
if _override:
    globals().update(json.loads(_override))
'''


def read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read().rstrip("\n")


def sources_for(arch):
    a, s = P2.ARCHS[arch], SPECIFIC[arch]
    p2 = P2.sources_for(arch)
    bidir = ("BIDIRECTIONAL = True            # lift the causal mask while fine-tuning (Dukic & Snajder 2024; LLM2Vec)"
             if a["bidir"] else "BIDIRECTIONAL = True            # the encoder is bidirectional anyway")
    block = "\n".join(l for l in a["block"].splitlines() if not l.startswith(("SPLIT_PUNCT", "SPLIT_DIGITS")))
    return {
        "config.py": CONFIG_TEMPLATE.format(arch=arch, block=block, bidir_line=bidir, micro=s["micro"], accum=s["accum"],
                                            eval_batch=s["eval_batch"], targets=a["targets"], sites=s["sites"],
                                            student=a["student"]).rstrip("\n"),
        "tokenizer.py": p2["tokenizer.py"],
        "phase1_model.py": p2["phase1_model.py"],
        "hub_files.py": read(HERE, "common", "hub_files.py"),
        "backbone.py": read(CAPSTONE, "phase3_src", arch, "backbone.py"),
        "cjpe_core.py": read(HERE, "common", "cjpe_core.py"),
        "run_cjpe.py": read(HERE, "common", "run_cjpe.py"),
    }


def build(arch, out_path):
    a = P2.ARCHS[arch]
    work = f"/kaggle/working/phase4_{arch}"
    files = sources_for(arch)
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": text.strip("\n").splitlines(keepends=True)})

    def writefile(name):
        code(f"%%writefile {work}/{name}\n{files[name]}")

    md(f"""
# Phase 4 - Court judgment prediction with {a['title']}

Starts from the Phase-1 model ({a['about']}) and predicts the outcome (0 / 1) of a case. Metric: macro-F1.

**Reading a long case.** Judgments average ~20K characters (up to 2.5M). Each case becomes four 512-token chunks: the
opening of the judgment plus its **last three chunks** - in ILDC (Malik et al. 2021, the source of this data) the
closing part, right before the removed verdict, carried the most signal. The Phase-1 backbone encodes every chunk
(causal mask lifted for the decoder-only models), chunks are mean-pooled, tagged with the section they came from, and
merged by **learned attention pooling** into one document vector - a hierarchical chunk -> document representation.

**Class balance.** The training set is 59% class 0 but dev and test are 50/50, so the loss is class-balanced and the
decision threshold is tuned on the dev set (it is the same distribution as the test set).

**Experiments** (each is its own submission; all from the Phase-4 list)

| Experiment | What trains |
|---|---|
| `full` | every weight, layer-wise LR decay - strong reference and the distillation teacher |
| `lorma` | **LoRMA** (compulsory): `(alpha/r BA + I) W x` on the attention projections (Bihany, Patel & Modi 2025) |
| `bitfit` | only bias terms (Ben Zaken et al. 2022); bias-free layers get a zero-initialised bias |
| `adapter` | Houlsby bottleneck adapters after the attention and feed-forward outputs (Houlsby et al. 2019) |
| `unfreeze` | gradual unfreezing: the head first, then one more backbone layer at a time from the top (Howard & Ruder 2018) |
| `kd` | distillation: the fine-tuned `full` model teaches a half-depth student via its logits and its document vector (Hinton et al. 2015) |

There are 37K training cases - more than a Kaggle session can read in full - so every experiment trains for a fixed
**time budget** (`TRAIN_MINUTES`) and the learning-rate schedule is fitted to that budget.

**Note:** the public `test.json` also contains a `label` field. It is never read here - only `id` and `text`.

**Kaggle settings:** GPU T4 x2, Internet on, secret `HF` holding your Hugging Face token. Then Run All.
""")

    md("## 1. Setup")
    code(f'''
import os, sys, json, glob, shutil, time, gc
import numpy as np
import torch
from huggingface_hub import login, hf_hub_download
from kaggle_secrets import UserSecretsClient

HF_TOKEN = UserSecretsClient().get_secret("HF")
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)

WORK_DIR = "{work}"
os.makedirs(WORK_DIR, exist_ok=True)
sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)

print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
''')

    md("## 2. Configuration\nEverything you might want to change lives here.")
    writefile("config.py")

    md("## 3. Code\nThe Phase-1 tokenizer and model (unchanged, so the trained weights load), plus the judgment-prediction pieces.")
    for name in ("tokenizer.py", "phase1_model.py", "hub_files.py", "backbone.py", "cjpe_core.py", "run_cjpe.py"):
        writefile(name)

    md("""
## 4. Data: download and turn every case into chunks
The cases are tokenized once with all CPU cores and stored as compact arrays, so each experiment starts instantly.
""")
    code('''
import importlib
import config as C
importlib.reload(C)
import backbone as BB
import cjpe_core as K
importlib.reload(K)

for split in ("train", "dev", "test"):
    folder = "Test" if split == "test" else "Train"
    hf_hub_download(C.HF_DATASET_REPO, f"Phase_4_CJPE/{folder}/{split}.json", repo_type="dataset",
                    local_dir=C.DATA_DIR, token=HF_TOKEN)

tok = BB.load_tokenizer()                       # caches the tokenizer before the worker processes start
chunk_cfg = {"W": C.CHUNK_TOKENS, "head": C.HEAD_CHUNKS, "tail": C.TAIL_CHUNKS,
             "chars_per_token": C.CHARS_PER_TOKEN, "pad": tok.pad_id}

for split in ("train", "dev", "test"):
    t0 = time.time()
    ids, texts, labels = K.load_cases(C.RAW[split], with_labels=split != "test")
    chunk_ids, chunk_lens = K.prepare_split(texts, chunk_cfg, C.PREP_WORKERS)
    arrays = {"ids": chunk_ids, "lens": chunk_lens}
    if labels is not None:
        arrays["labels"] = np.asarray(labels, dtype=np.int64)
    np.savez(os.path.join(C.DATA_DIR, f"{split}.npz"), **arrays)
    if split == "test":
        json.dump(ids, open(os.path.join(C.DATA_DIR, "test_ids.json"), "w"))
    used = (chunk_lens > 0).sum(1)
    extra = f" | class 1: {np.mean(labels):.0%}" if labels is not None else ""
    print(f"{split}: {len(ids):,} cases, chunks used per case {used.mean():.2f} (max {chunk_lens.shape[1]}) "
          f"| {time.time() - t0:.0f}s{extra}")
    del texts, chunk_ids, chunk_lens, arrays
    gc.collect()
''')

    md("## 5. Load the Phase-1 weights once\nThis checks the model end to end and caches the files for the experiment processes.")
    code('''
probe = K.CJPEModel(BB.build_backbone(), C)
dev = np.load(os.path.join(C.DATA_DIR, "dev.npz"))
with torch.no_grad():
    logits, doc, w = probe(torch.from_numpy(dev["ids"][:2].astype(np.int64)), torch.from_numpy(dev["lens"][:2].astype(np.int64)))
print(f"model ready: {sum(p.numel() for p in probe.parameters()) / 1e6:.1f}M parameters | logits {tuple(logits.shape)} | "
      f"chunk attention of case 0: {np.round(w[0].numpy(), 2)}")
del probe, dev
''')

    md(f"""
## 6. Run the experiments
Each experiment is its own process (`run_cjpe.py --method ...`); with two GPUs, two run at once, so the six take about
three rounds of ~`TRAIN_MINUTES` + 10 minutes. `kd` starts as soon as `full` (its teacher) is done.
Results, checkpoints and zips are pushed to `D-240357/legal-cjpe-{arch}` as they finish.
""")
    code('''
import subprocess, threading

pending = list(C.METHODS)
done, failed = set(), set()
lock = threading.Lock()


def next_job():
    with lock:
        if "kd" in pending and C.KD_TEACHER in failed:
            pending.remove("kd")
            print(f"skipping kd: its teacher '{C.KD_TEACHER}' failed")
        for m in pending:
            if m != "kd" or C.KD_TEACHER in done or C.KD_TEACHER not in C.METHODS:
                pending.remove(m)
                return m
        return "wait" if pending else None


def worker(gpu):
    while True:
        job = next_job()
        if job is None:
            return
        if job == "wait":
            time.sleep(15)
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        proc = subprocess.Popen([sys.executable, "run_cjpe.py", "--method", job], cwd=WORK_DIR, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(f"[gpu{gpu}] {line}", end="")
        with lock:
            (done if proc.wait() == 0 else failed).add(job)


t0 = time.time()
threads = [threading.Thread(target=worker, args=(g,)) for g in range(max(1, torch.cuda.device_count()))]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"\\nfinished in {(time.time() - t0) / 60:.0f} min | ok: {sorted(done)} | failed: {sorted(failed)}")
''')

    md("## 7. Results\nDev macro-F1 (threshold tuned on dev), what the threshold bought, error types and the cost of each method.")
    code('''
import pandas as pd

rows = []
for path in sorted(glob.glob(os.path.join(C.OUT_DIR, "*", "results.json"))):
    r = json.load(open(path))
    rows.append({"method": r["method"], "dev macro-F1": r["dev_macro_f1"], "at 0.5": r["dev_macro_f1_at_0.5"],
                 "threshold": r["threshold"], "macro P": r["dev_macro_p"], "macro R": r["dev_macro_r"],
                 "false pos": r["dev_false_pos"], "false neg": r["dev_false_neg"],
                 "cases seen": r["cases_seen"], "params (M)": r["params_total"] / 1e6,
                 "trainable %": 100 * r["params_trainable"] / r["params_total"], "minutes": r["minutes"],
                 "peak GPU GB": r["peak_gpu_gb"], "dev infer s": r["dev_inference_seconds"]})
table = pd.DataFrame(rows).sort_values("dev macro-F1", ascending=False).round(4)
display(table)
''')

    md("""
## 8. Pick a submission
Every experiment wrote `runs/<method>/submission_<arch>_<method>.zip`: a single `prediction.json` at the root holding a
JSON list of `{"id": ..., "pred": 0 or 1}` for all 1,517 test cases. The best on dev is also copied to
`/kaggle/working/submission.zip`. Upload the zips exactly as they are.
""")
    code('''
best = table.iloc[0]["method"]
shutil.copy(glob.glob(os.path.join(C.OUT_DIR, best, "submission_*.zip"))[0], "/kaggle/working/submission.zip")
print(f"best on dev: {best} -> /kaggle/working/submission.zip")
for z in sorted(glob.glob(os.path.join(C.OUT_DIR, "*", "submission_*.zip"))):
    print("  ", z)
''')

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)


if __name__ == "__main__":
    for arch, a in P2.ARCHS.items():
        out = os.path.join(P2.DESKTOP, a["folder"], f"phase4_{arch}_cjpe.ipynb")
        build(arch, out)
        print("wrote", out)
