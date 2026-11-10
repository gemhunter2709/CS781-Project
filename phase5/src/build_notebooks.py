"""Builds the four Phase-5 notebooks (statute identification), one per Phase-1 model, from the tested sources here."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAPSTONE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CAPSTONE, "phase2_src"))
import build_notebooks as P2          # noqa: E402  (tokenizer + Phase-1 model copies are shared with Phases 2-4)

SPECIFIC = {
    "arch1": {"sites": '["W_o", "down"]', "micro": 3, "accum": 6, "eval_batch": 12},
    "arch2": {"sites": '["proj", "down"]', "micro": 3, "accum": 6, "eval_batch": 12},
    "arch3": {"sites": '["proj", "down"]', "micro": 3, "accum": 6, "eval_batch": 12},
    "arch4": {"sites": '["out", "down"]', "micro": 1, "accum": 16, "eval_batch": 6},
}

CONFIG_TEMPLATE = '''"""Every knob of this Phase-5 (legal statute identification) notebook."""
import os
import json

ARCH = "{arch}"
WORK_DIR = f"/kaggle/working/phase5_{{ARCH}}"
DATA_DIR = os.path.join(WORK_DIR, "data")
CACHE_DIR = os.path.join(WORK_DIR, "phase1")        # Phase-1 files downloaded from Hugging Face land here
OUT_DIR = os.path.join(WORK_DIR, "runs")             # one folder per experiment

HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
RAW = {{s: os.path.join(DATA_DIR, "Phase_5_LSI", "Test" if s == "test" else "Train", f"{{s}}.json")
       for s in ("train", "dev", "test")}}

# ------------------------------------------------------------------ the Phase-1 model
# Looked up in this order: an attached Kaggle dataset whose path mentions "{arch}", then these HF repos.
{block}
KAGGLE_HINT = "{arch}"

LSI_REPO = "D-240357/legal-lsi-{arch}"           # fine-tuned checkpoints, results and submission zips go here

# ------------------------------------------------------------------ experiments
# Each one is a separate submission. "full" is the strong reference and the distillation teacher;
# "kd" waits for it to finish.
METHODS = ["full", "lorma", "adapter", "partial", "prompt", "kd"]
KD_TEACHER = "full"
TRAIN_MINUTES = 40             # wall-clock training budget per experiment (the LR schedule ends exactly on it)
MAX_EPOCHS = 3

# ------------------------------------------------------------------ how a document is read
CHUNK_TOKENS = 512
MAX_CHUNKS = 6                 # long documents are sampled at 6 evenly spaced places, first and last included
CHAR_CAP = 150_000             # text beyond this is not even tokenized (99% of documents are < 50K characters)
{bidir_line}
LWAN_DIM = 256                 # label-wise attention size
HEAD_DROPOUT = 0.1

# ------------------------------------------------------------------ loss and thresholds
LOSS = "asl"                   # "asl" (asymmetric loss, Ridnik et al. 2021) or "bce"
ASL_GAMMA_NEG = 2.0
ASL_GAMMA_POS = 0.0
ASL_CLIP = 0.05
THRESHOLDS = "auto"            # "per_label", "global" or "auto" (whichever is better on dev)
MIN_DEV_POSITIVES = 20         # statutes rarer than this on dev use the global threshold

# ------------------------------------------------------------------ optimisation
MICRO_DOCS = {micro}                 # documents per forward pass (each document = up to 6 chunks)
GRAD_ACCUM = {accum}
EVAL_BATCH = {eval_batch}
WARMUP_STEPS = 50
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.01
HEAD_LR = 1e-3
FULL_LR = 5e-5                 # top layer; each layer below gets LLRD times the one above it
LLRD = 0.9
PARTIAL_LR = 1e-4
PARTIAL_TOP_LAYERS = {partial}
ADAPTER_LR = 5e-4
ADAPTER_RANK = 8
LORMA_ALPHA = 8
ADAPTER_DROPOUT = 0.05
ADAPTER_TARGETS = {targets}       # LoRMA: attention projections
ADAPTER_SITES = {sites}           # Houlsby adapters: after the attention output and the feed-forward output
ADAPTER_BOTTLENECK = 64
PROMPT_LEN = 16
PROMPT_LR = 1e-2

KD_TEMPERATURE = 2.0
KD_ALPHA = 0.5                 # weight of the gold-label loss; the rest goes to matching the teacher
KD_HIDDEN_WEIGHT = 0.5         # also align the student's document representation with the teacher's (cosine)
STUDENT_LAYERS = {student}

# ------------------------------------------------------------------ evaluation
EVAL_EVERY_STEPS = 150         # quick check on DEV_SUBSET documents; the best checkpoint is kept
DEV_SUBSET = 1500
DEV_FINAL_LIMIT = None         # thresholds are tuned on the whole dev set (10,181 documents)
TRAIN_DOC_LIMIT = None         # set to e.g. 400 (and TRAIN_MINUTES = 3) for a quick smoke test
PREP_WORKERS = max(1, os.cpu_count() or 2)

SEED = 13

_override = os.environ.get("P5_CFG_OVERRIDE")
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
                                            partial=a["partial"], student=a["student"]).rstrip("\n"),
        "tokenizer.py": p2["tokenizer.py"],
        "phase1_model.py": p2["phase1_model.py"],
        "hub_files.py": read(HERE, "common", "hub_files.py"),
        "backbone.py": read(CAPSTONE, "phase3_src", arch, "backbone.py"),
        "lsi_core.py": read(HERE, "common", "lsi_core.py"),
        "run_lsi.py": read(HERE, "common", "run_lsi.py"),
    }


def build(arch, out_path):
    a = P2.ARCHS[arch]
    work = f"/kaggle/working/phase5_{arch}"
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
# Phase 5 - Statute identification with {a['title']}

Starts from the Phase-1 model ({a['about']}) and predicts **every** applicable statute (100 possible IDs, ~3.8 per
document) - a multi-label task scored by macro-F1, so rare statutes count as much as common ones. Statute mentions in
the text are masked (`<SECTION>`, `<ACT>`), so they have to be inferred from the facts.

**The model**
* Up to 6 chunks of 512 tokens per document, spread evenly from its start to its end - statutes can be implied anywhere.
  The Phase-1 backbone encodes each chunk (causal mask lifted for the decoder-only models) and every token is tagged
  with the chunk it came from.
* **Label-wise attention** (CAML, Mullenbach et al. 2018; LWAN for legal text, Chalkidis et al. 2019): each statute has
  its own attention over all tokens of the document and its own scorer, so the evidence for statute 34 is collected
  separately from the evidence for statute 5.
* **Asymmetric loss** (Ridnik et al. 2021) for the heavy positive/negative imbalance (4 positives vs 96 negatives).
* **Per-statute thresholds** tuned on the dev set (macro-F1 is the average of per-statute F1s); statutes too rare on
  dev fall back to a global threshold, and every document gets at least one statute.

**Experiments** (each is its own submission; all from the Phase-5 list)

| Experiment | What trains |
|---|---|
| `full` | every weight, layer-wise LR decay - strong reference and the distillation teacher |
| `lorma` | **LoRMA** (compulsory): `(alpha/r BA + I) W x` on the attention projections (Bihany, Patel & Modi 2025) |
| `adapter` | Houlsby bottleneck adapters after the attention and feed-forward outputs (Houlsby et al. 2019) |
| `partial` | only the top backbone layers + final norm (layer-wise / partial fine-tuning) |
| `prompt` | prompt tuning: 16 learned virtual tokens in front of every chunk (Lester et al. 2021) |
| `kd` | distillation: the fine-tuned `full` model teaches a half-depth student through its per-statute probabilities and its document representation (Hinton et al. 2015) |

The label-wise attention head trains in every experiment. 42K training documents are more than one session can
read many times over, so each experiment trains for a fixed **time budget** (`TRAIN_MINUTES`).

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

    md("## 3. Code\nThe Phase-1 tokenizer and model (unchanged, so the trained weights load), plus the statute-identification pieces.")
    for name in ("tokenizer.py", "phase1_model.py", "hub_files.py", "backbone.py", "lsi_core.py", "run_lsi.py"):
        writefile(name)

    md("""
## 4. Data: download and turn every document into chunks
Tokenized once with all CPU cores and stored as compact arrays, so each experiment starts instantly.
""")
    code('''
import importlib
import config as C
importlib.reload(C)
import backbone as BB
import lsi_core as L
importlib.reload(L)

for split in ("train", "dev", "test"):
    folder = "Test" if split == "test" else "Train"
    hf_hub_download(C.HF_DATASET_REPO, f"Phase_5_LSI/{folder}/{split}.json", repo_type="dataset",
                    local_dir=C.DATA_DIR, token=HF_TOKEN)

tok = BB.load_tokenizer()                       # caches the tokenizer before the worker processes start
chunk_cfg = {"W": C.CHUNK_TOKENS, "K": C.MAX_CHUNKS, "char_cap": C.CHAR_CAP, "pad": tok.pad_id}

for split in ("train", "dev", "test"):
    t0 = time.time()
    ids, texts, labels = L.load_docs(C.RAW[split], with_labels=split != "test")
    chunk_ids, chunk_lens = L.prepare_split(texts, chunk_cfg, C.PREP_WORKERS)
    arrays = {"ids": chunk_ids, "lens": chunk_lens}
    if labels is not None:
        arrays["y"] = L.multi_hot(labels)
        assert all(0 <= l < L.N_LABELS for ls in labels for l in ls)
    np.savez(os.path.join(C.DATA_DIR, f"{split}.npz"), **arrays)
    if split == "test":
        json.dump(ids, open(os.path.join(C.DATA_DIR, "test_ids.json"), "w"))
    extra = f" | statutes per doc {arrays['y'].sum(1).mean():.2f}" if labels is not None else ""
    print(f"{split}: {len(ids):,} docs, chunks per doc {(chunk_lens > 0).sum(1).mean():.2f} of {C.MAX_CHUNKS} "
          f"| {time.time() - t0:.0f}s{extra}")
    del texts, chunk_ids, chunk_lens, arrays
    gc.collect()
''')

    md("## 5. Load the Phase-1 weights once\nThis checks the model end to end and caches the files for the experiment processes.")
    code('''
probe = L.LSIModel(BB.build_backbone(), C)
dev = np.load(os.path.join(C.DATA_DIR, "dev.npz"))
with torch.no_grad():
    logits, _ = probe(torch.from_numpy(dev["ids"][:2].astype(np.int64)), torch.from_numpy(dev["lens"][:2].astype(np.int64)))
print(f"model ready: {sum(p.numel() for p in probe.parameters()) / 1e6:.1f}M parameters | logits {tuple(logits.shape)}")
del probe, dev
''')

    md(f"""
## 6. Run the experiments
Each experiment is its own process (`run_lsi.py --method ...`); with two GPUs, two run at once, so the six take about
three rounds of `TRAIN_MINUTES` + ~10 minutes. `kd` starts as soon as `full` (its teacher) is done.
Results, checkpoints and zips are pushed to `D-240357/legal-lsi-{arch}` as they finish.
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
        proc = subprocess.Popen([sys.executable, "run_lsi.py", "--method", job], cwd=WORK_DIR, env=env,
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

    md("## 7. Results\nDev macro-F1 (the competition metric), what per-statute thresholds bought, micro-F1, subset accuracy, cost, and the hardest statutes.")
    code('''
import pandas as pd

rows, worst = [], {}
for path in sorted(glob.glob(os.path.join(C.OUT_DIR, "*", "results.json"))):
    r = json.load(open(path))
    rows.append({"method": r["method"], "dev macro-F1": r["dev_macro_f1"], "global thr": r["dev_macro_f1_global_thr"],
                 "per-statute thr": r["dev_macro_f1_per_label_thr"], "micro-F1": r["dev_micro_f1"],
                 "subset acc": r["dev_subset_acc"], "statutes/test doc": r["test_labels_per_doc"],
                 "docs seen": r["docs_seen"], "params (M)": r["params_total"] / 1e6,
                 "trainable %": 100 * r["params_trainable"] / r["params_total"], "minutes": r["minutes"],
                 "peak GPU GB": r["peak_gpu_gb"], "dev infer s": r["dev_inference_seconds"]})
    worst[r["method"]] = [(w["label"], round(w["f1"], 2), w["dev_support"]) for w in r["worst_labels"][:6]]
table = pd.DataFrame(rows).sort_values("dev macro-F1", ascending=False).round(4)
display(table)
print("hardest statutes per method (id, dev F1, dev support):")
for m, w in worst.items():
    print(f"  {m:8s} {w}")
''')

    md("""
## 8. Pick a submission
Every experiment wrote `runs/<method>/submission_<arch>_<method>.zip`: a single `prediction.json` at the root holding a
JSON list of `{"id": ..., "pred": [statute ids]}` for all 13,019 test documents (unique, valid IDs). The best on dev is
also copied to `/kaggle/working/submission.zip`. Upload the zips exactly as they are.
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
        out = os.path.join(P2.DESKTOP, a["folder"], f"phase5_{arch}_lsi.ipynb")
        build(arch, out)
        print("wrote", out)
