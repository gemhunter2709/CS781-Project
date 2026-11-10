"""Builds the four Phase-7 notebooks (prior case retrieval), one per Phase-1 model, from the tested sources here."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAPSTONE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CAPSTONE, "phase2_src"))
import build_notebooks as P2          # noqa: E402  (tokenizer + Phase-1 model copies are shared with Phases 2-6)

SPECIFIC = {
    "arch1": {"ia3_out": '["W_qkv"]', "ia3_in": '["down"]', "micro": 4, "accum": 2, "train_chunks": 2, "eval_batch": 16},
    "arch2": {"ia3_out": '["qkv"]', "ia3_in": '["down"]', "micro": 4, "accum": 2, "train_chunks": 2, "eval_batch": 16},
    "arch3": {"ia3_out": '["qkv"]', "ia3_in": '["down"]', "micro": 4, "accum": 2, "train_chunks": 2, "eval_batch": 16},
    "arch4": {"ia3_out": '["kv_up"]', "ia3_in": '["down"]', "micro": 2, "accum": 4, "train_chunks": 1, "eval_batch": 6},
}

CONFIG_TEMPLATE = '''"""Every knob of this Phase-7 (prior case retrieval) notebook."""
import os
import json

ARCH = "{arch}"
WORK_DIR = f"/kaggle/working/phase7_{{ARCH}}"
DATA_DIR = os.path.join(WORK_DIR, "data")
CACHE_DIR = os.path.join(WORK_DIR, "phase1")        # Phase-1 files downloaded from Hugging Face land here
OUT_DIR = os.path.join(WORK_DIR, "runs")             # one folder per experiment

HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
SPLIT_FOLDERS = {{"train": "Train", "dev": "Dev", "test": "Test"}}
RAW = {{s: (os.path.join(DATA_DIR, "Phase_7_PCR", f, f"{{s}}.json"), os.path.join(DATA_DIR, "Phase_7_PCR", f, f"{{s}}_candidates.json"))
       for s, f in SPLIT_FOLDERS.items()}}

# ------------------------------------------------------------------ the Phase-1 model
# Looked up in this order: an attached Kaggle dataset whose path mentions "{arch}", then these HF repos.
{block}
KAGGLE_HINT = "{arch}"

PCR_REPO = "D-240357/legal-pcr-{arch}"           # fine-tuned checkpoints, results and submission zips go here

# ------------------------------------------------------------------ experiments
# Each one is a separate submission. "full" is the strong reference and the distillation teacher;
# "kd" waits for it to finish.
METHODS = ["full", "lorma", "partial", "ia3", "prompt", "kd"]
KD_TEACHER = "full"
TRAIN_MINUTES = 30             # wall-clock training budget per experiment (the LR schedule ends exactly on it)
MAX_EPOCHS = 12

# ------------------------------------------------------------------ how a case is read
CHUNK_TOKENS = 512
MAX_CHUNKS = 3                 # three evenly spaced 512-token windows (opening, middle, end) are mean-pooled
TRAIN_CHUNKS = {train_chunks}               # random subset of them read per case while training (saves memory)
CHAR_CAP = 300_000             # text beyond this is not even tokenized
{bidir_line}
EMB_DIM = 256                  # size of the case embedding

# ------------------------------------------------------------------ BM25 (also gives the hard negatives)
BM25_MIN_IDF = 1.0             # the query keeps only words that are rare enough to say something ...
BM25_MAX_TERMS = 300           # ... and at most its 300 most telling ones (tf x idf)
HARD_NEG_POOL = 30             # hard negatives are drawn from the top-30 BM25 candidates that are not relevant
FUSION_GRID = [round(0.05 * i, 2) for i in range(21)]   # final score = a z(dense) + (1 - a) z(BM25), a tuned on dev

# ------------------------------------------------------------------ optimisation
TEMPERATURE = 0.05             # InfoNCE temperature
MICRO_QUERIES = {micro}              # queries per forward pass: each brings one positive and one hard negative
GRAD_ACCUM = {accum}
EVAL_BATCH = {eval_batch}
WARMUP_STEPS = 30
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.01
HEAD_LR = 1e-3
FULL_LR = 2e-5
PARTIAL_LR = 5e-5
PARTIAL_TOP_LAYERS = {partial}
ADAPTER_LR = 5e-4
ADAPTER_RANK = 8
LORMA_ALPHA = 8
ADAPTER_DROPOUT = 0.05
ADAPTER_TARGETS = {targets}       # LoRMA: attention projections
IA3_OUT_TARGETS = {ia3_out}       # key/value projections
IA3_IN_TARGETS = {ia3_in}         # feed-forward down projections
PROMPT_LEN = 16
PROMPT_LR = 1e-2

KD_TEMPERATURE = 2.0
KD_ALPHA = 0.5                 # weight of the contrastive loss; the rest goes to matching the teacher's similarities
KD_HIDDEN_WEIGHT = 0.5         # also align the student's case embeddings with the teacher's (cosine)
STUDENT_LAYERS = {student}

# ------------------------------------------------------------------ evaluation
EVAL_EVERY_STEPS = 100         # full dev ranking (118 queries against 1,023 candidates); the best checkpoint is kept
TRAIN_QUERY_LIMIT = None       # set to e.g. 40 (and TRAIN_MINUTES = 2) for a quick smoke test
PREP_WORKERS = max(1, os.cpu_count() or 2)

SEED = 13

_override = os.environ.get("P7_CFG_OVERRIDE")
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
                                            train_chunks=s["train_chunks"], eval_batch=s["eval_batch"], targets=a["targets"],
                                            ia3_out=s["ia3_out"], ia3_in=s["ia3_in"], partial=a["partial"],
                                            student=a["student"]).rstrip("\n"),
        "tokenizer.py": p2["tokenizer.py"],
        "phase1_model.py": p2["phase1_model.py"],
        "hub_files.py": read(HERE, "common", "hub_files.py"),
        "backbone.py": read(CAPSTONE, "phase3_src", arch, "backbone.py"),
        "pcr_core.py": read(HERE, "common", "pcr_core.py"),
        "run_pcr.py": read(HERE, "common", "run_pcr.py"),
    }


def build(arch, out_path):
    a = P2.ARCHS[arch]
    work = f"/kaggle/working/phase7_{arch}"
    files = sources_for(arch)
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": text.strip("\n").splitlines(keepends=True)})

    def writefile(name):
        code(f"%%writefile {work}/{name}\n{files[name]}")

    reader = ("its bidirectional encoder" if arch == "arch2" else "the decoder with its causal mask lifted")
    md(f"""
# Phase 7 - Prior case retrieval with {a['title']}

Starts from the Phase-1 model ({a['about']}) and, for every query judgment, ranks the cases in its pool and returns
the **top 7** most likely to be cited. Metric: micro-F1@7.

**What the data asks for.** Train has 827 queries against a pool of 4,320 cases (dev 118 / 1,023, test 237 / 1,727);
each query cites ~6-8 of them. Judgments are long and names, organisations and citations are masked (`<NAME>`,
`<ORG>`, `<CITATION>`), so the signal is the legal content itself. Every query also sits in its own pool - it is never
allowed to retrieve itself.

**Approach: hybrid dense + lexical retrieval.**
* **Dense bi-encoder** (Karpukhin et al. 2020, DPR): {reader} reads three evenly spaced 512-token windows of a case
  (opening, middle, end - MaxP/FirstP-style passage evidence, Dai & Callan 2019); the windows are mean-pooled and
  projected to one normalised 256-d vector. Queries and candidates share the encoder.
* **Contrastive training**: InfoNCE with in-batch negatives plus **BM25 hard negatives** (Karpukhin et al. 2020;
  Xiong et al. 2021, ANCE); candidates relevant to another query in the batch are masked rather than punished.
* **BM25** (Robertson & Zaragoza 2009), written from scratch, on rare query terms only - this alone is a strong
  baseline for legal PCR (Joshi et al. 2023, IL-PCR / U-CREAT).
* **Fusion**: `a * z(dense) + (1 - a) * z(BM25)`, with `a` tuned on dev (convex score fusion, Bruch et al. 2023).

**Experiments** (each is its own submission; all from the Phase-7 list)

| Experiment | What trains |
|---|---|
| `full` | every weight - strong reference and the distillation teacher |
| `lorma` | **LoRMA** (compulsory): `(alpha/r BA + I) W x` on the attention projections (Bihany, Patel & Modi 2025) |
| `partial` | only the top {a['partial']} layers (+ final norm and projection) |
| `ia3` | IA3: learned rescaling of keys/values and feed-forward activations (Liu et al. 2022) |
| `prompt` | prompt tuning: 16 learned virtual tokens in front of every window (Lester et al. 2021) |
| `kd` | distillation (Hinton et al. 2015): a half-depth student learns the fine-tuned `full` model's query-candidate similarity distribution and its embeddings |

Every experiment trains for a fixed **time budget** (`TRAIN_MINUTES`); the checkpoint with the best dev micro-F1@7 is kept.

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

    md("## 3. Code\nThe Phase-1 tokenizer and model (unchanged, so the trained weights load), plus the retrieval pieces.")
    for name in ("tokenizer.py", "phase1_model.py", "hub_files.py", "backbone.py", "pcr_core.py", "run_pcr.py"):
        writefile(name)

    md("""
## 4. Data: download, tokenize, BM25
Done once with all CPU cores. Each split is stored as token windows for queries and pool, plus the BM25 scores
(dev / test) or the BM25 hard negatives (train). The BM25-only dev score printed here is the baseline to beat.
""")
    code('''
import importlib
import config as C
importlib.reload(C)
import backbone as BB
import pcr_core as R
importlib.reload(R)

for split, folder in C.SPLIT_FOLDERS.items():
    for name in (f"{split}.json", f"{split}_candidates.json"):
        hf_hub_download(C.HF_DATASET_REPO, f"Phase_7_PCR/{folder}/{name}", repo_type="dataset",
                        local_dir=C.DATA_DIR, token=HF_TOKEN)

tok = BB.load_tokenizer()                       # caches the tokenizer before the worker processes start
chunk_cfg = {"W": C.CHUNK_TOKENS, "K": C.MAX_CHUNKS, "char_cap": C.CHAR_CAP, "pad": tok.pad_id}

for split in ("train", "dev", "test"):
    t0 = time.time()
    q_ids, q_text, rel, c_ids, c_text = R.load_split(*C.RAW[split])
    qi, ql = R.prepare_texts(q_text, chunk_cfg, C.PREP_WORKERS)
    ci, cl = R.prepare_texts(c_text, chunk_cfg, C.PREP_WORKERS)
    bm25 = R.bm25_matrix(q_text, c_text, C)
    arrays = {"q_ids": qi, "q_lens": ql, "c_ids": ci, "c_lens": cl}
    pos = {c: j for j, c in enumerate(c_ids)}
    extra = ""
    if split == "train":
        hard = np.full((len(q_ids), C.HARD_NEG_POOL), -1, dtype=np.int32)
        for i, q in enumerate(q_ids):
            banned = {pos[c] for c in rel[i] if c in pos} | ({pos[q]} if q in pos else set())
            top = [j for j in np.argsort(-bm25[i])[:C.HARD_NEG_POOL + len(banned)] if j not in banned][:C.HARD_NEG_POOL]
            hard[i, :len(top)] = top
        arrays["hard_neg"] = hard
    else:
        arrays["bm25"] = bm25.astype(np.float32)
    if split != "test" and any(rel):
        extra = f" | relevant per query {np.mean([len(r) for r in rel]):.1f} | BM25-only micro-F1@7 " \\
                f"{R.micro_f1_at_7(R.rank(bm25, q_ids, c_ids), rel)['f1']:.4f}"
    np.savez(os.path.join(C.DATA_DIR, f"{split}.npz"), **arrays)
    json.dump({"q_ids": q_ids, "c_ids": c_ids, "rel": rel}, open(os.path.join(C.DATA_DIR, f"{split}_meta.json"), "w"))
    print(f"{split}: {len(q_ids)} queries, pool {len(c_ids):,} | windows per case {(cl > 0).sum(1).mean():.2f} of "
          f"{C.MAX_CHUNKS} | {time.time() - t0:.0f}s{extra}")
    del q_text, c_text, arrays, bm25
    gc.collect()
''')

    md("## 5. Load the Phase-1 weights once\nThis checks the encoder end to end and caches the files for the experiment processes.")
    code('''
probe = R.Retriever(BB.build_backbone(bidirectional=C.BIDIRECTIONAL), C)
dev = np.load(os.path.join(C.DATA_DIR, "dev.npz"))
with torch.no_grad():
    e = probe(torch.from_numpy(dev["q_ids"][:2].astype(np.int64)), torch.from_numpy(dev["q_lens"][:2].astype(np.int64)))
print(f"model ready: {sum(p.numel() for p in probe.parameters()) / 1e6:.1f}M parameters | embeddings {tuple(e.shape)}, "
      f"norms {e.norm(dim=-1).tolist()}")
del probe, dev, e
''')

    md(f"""
## 6. Run the experiments
Each experiment is its own process (`run_pcr.py --method ...`); with two GPUs, two run at once, so the six take about
three rounds of `TRAIN_MINUTES` + a few minutes. `kd` starts as soon as `full` (its teacher) is done.
Results, checkpoints and zips are pushed to `D-240357/legal-pcr-{arch}`.
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
        proc = subprocess.Popen([sys.executable, "run_pcr.py", "--method", job], cwd=WORK_DIR, env=env,
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

    md("## 7. Results\nDev micro-F1@7 of the fused ranking, next to dense-only and BM25-only, and the cost of each experiment.")
    code('''
import pandas as pd

rows = []
for path in sorted(glob.glob(os.path.join(C.OUT_DIR, "*", "results.json"))):
    r = json.load(open(path))
    rows.append({"method": r["method"], "dev F1@7": r["dev_f1"], "P": r["dev_p"], "R": r["dev_r"], "alpha": r["alpha"],
                 "dense only": r["dev_dense_only_f1"], "BM25 only": r["dev_bm25_only_f1"], "best step": r["best_step"],
                 "params (M)": r["params_total"] / 1e6, "trainable %": 100 * r["params_trainable"] / r["params_total"],
                 "dev encode s": r["dev_encode_seconds"], "minutes": r["minutes"], "peak GPU GB": r["peak_gpu_gb"]})
table = pd.DataFrame(rows).sort_values("dev F1@7", ascending=False).round(4)
display(table)
best = table.iloc[0]["method"]
''')

    md("""
## 8. Pick a submission
Every experiment wrote `runs/<method>/submission_<arch>_<method>.zip`: a single `prediction.json` at the root mapping
each of the 237 test query ids to 7 distinct candidate ids from the test pool (never the query itself). The best on
dev is also copied to `/kaggle/working/submission.zip`. Upload the zips exactly as they are.
""")
    code('''
shutil.copy(glob.glob(os.path.join(C.OUT_DIR, best, "submission_*.zip"))[0], "/kaggle/working/submission.zip")
print(f"best on dev micro-F1@7: {best} -> /kaggle/working/submission.zip")
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
        out = os.path.join(P2.DESKTOP, a["folder"], f"phase7_{arch}_pcr.ipynb")
        build(arch, out)
        print("wrote", out)
