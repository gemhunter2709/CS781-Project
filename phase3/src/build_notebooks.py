"""Builds the four Phase-3 notebooks (rhetorical roles), one per Phase-1 model, from the tested sources here."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAPSTONE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CAPSTONE, "phase2_src"))
import build_notebooks as P2          # noqa: E402  (tokenizer + Phase-1 model copies are shared with Phase 2)

SPECIFIC = {
    "arch1": {"ia3_out": '["W_qkv"]', "ia3_in": '["down"]', "span": 96, "eval_batch": 16},
    "arch2": {"ia3_out": '["qkv"]', "ia3_in": '["down"]', "span": 96, "eval_batch": 16},
    "arch3": {"ia3_out": '["qkv"]', "ia3_in": '["down"]', "span": 96, "eval_batch": 16},
    "arch4": {"ia3_out": '["kv_up"]', "ia3_in": '["down"]', "span": 48, "eval_batch": 8},
}

CONFIG_TEMPLATE = '''"""Every knob of this Phase-3 (rhetorical role) notebook."""
import os
import json

ARCH = "{arch}"
WORK_DIR = f"/kaggle/working/phase3_{{ARCH}}"
DATA_DIR = os.path.join(WORK_DIR, "data")
CACHE_DIR = os.path.join(WORK_DIR, "phase1")        # Phase-1 files downloaded from Hugging Face land here
OUT_DIR = os.path.join(WORK_DIR, "runs")             # one folder per experiment

HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
_P3 = os.path.join(DATA_DIR, "Phase_3_RR")
FILES = {{f"{{dom}}_{{split}}": os.path.join(_P3, "Test" if split == "test" else "Train", f"{{dom}}_{{split}}.json")
         for dom in ("CL", "IT") for split in ("train", "dev", "test")}}

# ------------------------------------------------------------------ the Phase-1 model
# Looked up in this order: an attached Kaggle dataset whose path mentions "{arch}", then these HF repos.
{block}
KAGGLE_HINT = "{arch}"

RR_REPO = "D-240357/legal-rr-{arch}"             # fine-tuned checkpoints, results and submission zips go here

# ------------------------------------------------------------------ experiments
# Each one is a separate submission. "full" is the strong reference and the distillation teacher;
# "kd" waits for it to finish.
METHODS = ["full", "lorma", "lora", "ia3", "prompt", "kd"]
KD_TEACHER = "full"

# ------------------------------------------------------------------ input / context
MAX_SENT_TOKENS = 96           # longer sentences are cut (median sentence is ~40 tokens)
PACK_TOKENS = 512              # consecutive sentences share one sequence, so each one is read in context
PACK_CONTEXT = True            # False = every sentence encoded alone (the "sentence-only" ablation)
DOC_CONTEXT = "bilstm"         # "none" = no document-level layer (another ablation)
{bidir_line}
SPAN_SENTS = {span}               # consecutive sentences per training example (whole documents at test time)
TRAIN_ON_DEV = False           # True for a final run on train+dev once the settings are chosen
TRAIN_DOC_LIMIT = None         # set to e.g. 8 for a quick end-to-end smoke test

# ------------------------------------------------------------------ head and losses
CONTEXT_DIM = 512
HEAD_DROPOUT = 0.2
CE_WEIGHT = 0.5                # class-weighted cross-entropy next to the CRF loss (helps the rare roles)
CLASS_WEIGHT_POWER = 0.5       # weight = (1 / frequency) ** power
SHIFT_WEIGHT = 0.3             # auxiliary "does the role change here?" loss (Malik et al. 2022)
TAU_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]   # prior correction tried on dev at the end (Menon et al. 2021)

# ------------------------------------------------------------------ optimisation
EPOCHS = 20                    # Phase-2 runs peaked in their last epochs, and one epoch here is only ~1-2 min
PATIENCE = 5
GRAD_ACCUM = 2
EVAL_PACK_BATCH = {eval_batch}
WARMUP_FRAC = 0.1
WEIGHT_DECAY = 0.01
HEAD_LR = 1e-3
FULL_LR = 1e-4                 # top layer; each layer below gets LLRD times the one above it
                               # (in Phase 2 a higher LR on the top layers beat a timid full fine-tune)
LLRD = 0.85
ADAPTER_LR = 5e-4
ADAPTER_RANK = 8
LORA_ALPHA = 16
LORMA_ALPHA = 8
ADAPTER_DROPOUT = 0.05
ADAPTER_TARGETS = {targets}
IA3_OUT_TARGETS = {ia3_out}       # key/value projections
IA3_IN_TARGETS = {ia3_in}         # feed-forward down projections
PROMPT_LEN = 16
PROMPT_LR = 1e-2

KD_TEMPERATURE = 2.0
KD_ALPHA = 0.5                 # weight of the task loss; the rest goes to matching the teacher
KD_HIDDEN_WEIGHT = 1.0         # also match the teacher's sentence representations
STUDENT_LAYERS = {student}

SEED = 13

_override = os.environ.get("P3_CFG_OVERRIDE")
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
        "config.py": CONFIG_TEMPLATE.format(arch=arch, block=block, bidir_line=bidir, span=s["span"],
                                            eval_batch=s["eval_batch"], targets=a["targets"], ia3_out=s["ia3_out"],
                                            ia3_in=s["ia3_in"], student=a["student"]).rstrip("\n"),
        "tokenizer.py": p2["tokenizer.py"],
        "phase1_model.py": p2["phase1_model.py"],
        "hub_files.py": read(HERE, "common", "hub_files.py"),
        "backbone.py": read(HERE, arch, "backbone.py"),
        "rr_core.py": read(HERE, "common", "rr_core.py"),
        "run_rr.py": read(HERE, "common", "run_rr.py"),
    }


def build(arch, out_path):
    a = P2.ARCHS[arch]
    work = f"/kaggle/working/phase3_{arch}"
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
# Phase 3 - Rhetorical roles with {a['title']}

Starts from the Phase-1 model ({a['about']}) and labels every sentence of a judgment with one of 13 rhetorical
roles. Primary metric: macro-F1. CL and IT documents are trained together; a domain embedding tells them apart and
scores are reported per domain.

**The model**
1. The Phase-1 backbone reads *packs* of consecutive sentences (up to 512 tokens), so each sentence is encoded next to
   its neighbours (Cohan et al. 2019, *Pretrained LMs for sequential sentence classification*). For the decoder-only
   models the causal mask is lifted (Dukic & Snajder 2024; LLM2Vec). Each sentence is mean-pooled.
2. Relative-position and domain embeddings are added: the role depends strongly on where a sentence sits.
3. A BiLSTM reads the whole document's sentence vectors, then a CRF picks the label sequence - roles come in long blocks
   (94% of CL and 75% of IT sentences repeat the previous label). (Bhattacharya et al. 2019; Kalamkar et al. 2022.)
4. Auxiliary *label-shift* head: "does the role change at this sentence?" (Malik et al. 2022, the paper behind this dataset).
5. Macro-F1 rewards rare roles (Issue, None, RulingByPresentCourt...): class-weighted loss, and at the end a prior
   correction (logit adjustment, Menon et al. 2021) whose strength - and CRF vs argmax - is chosen on the dev set.

**Experiments** (each one is its own submission)

| Experiment | Trained parameters |
|---|---|
| `full` | everything, with layer-wise LR decay - strong reference and the distillation teacher |
| `lorma` | **LoRMA** (compulsory): `(alpha/r BA + I) W x` on the attention projections (Bihany, Patel & Modi 2025) |
| `lora` | LoRA on the same projections (Hu et al. 2021) |
| `ia3` | IA3: learned rescaling of keys/values and feed-forward activations (Liu et al. 2022) |
| `prompt` | prompt tuning: 16 learned virtual tokens in front of every pack (Lester et al. 2021) |
| `kd` | distillation: the fine-tuned `full` model teaches a half-depth student through soft labels **and** its sentence representations (Hinton et al. 2015) |

In every PEFT run the backbone is frozen; only the adapter parameters and the task head (BiLSTM, CRF, embeddings) train.

**Kaggle settings:** GPU T4 x2, Internet on, secret `HF` holding your Hugging Face token. Then Run All.
""")

    md("## 1. Setup")
    code(f'''
import os, sys, json, glob, shutil, time
import torch
from huggingface_hub import login, hf_hub_download
from kaggle_secrets import UserSecretsClient

HF_TOKEN = UserSecretsClient().get_secret("HF")
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)

WORK_DIR = "{work}"
os.makedirs(WORK_DIR, exist_ok=True)
sys.path.insert(0, WORK_DIR)

print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
''')

    md("## 2. Configuration\nEverything you might want to change lives here.")
    writefile("config.py")

    md("## 3. Code\nThe Phase-1 tokenizer and model (unchanged, so the trained weights load), plus the rhetorical-role pieces.")
    for name in ("tokenizer.py", "phase1_model.py", "hub_files.py", "backbone.py", "rr_core.py", "run_rr.py"):
        writefile(name)

    md("## 4. Phase-3 data")
    code('''
import importlib
import config as C
importlib.reload(C)

for dom in ("CL", "IT"):
    for name in (f"Train/{dom}_train.json", f"Train/{dom}_dev.json", f"Test/{dom}_test.json"):
        hf_hub_download(C.HF_DATASET_REPO, f"Phase_3_RR/{name}", repo_type="dataset", local_dir=C.DATA_DIR, token=HF_TOKEN)
hf_hub_download(C.HF_DATASET_REPO, "Phase_3_RR/Train/label_vocab.json", repo_type="dataset", local_dir=C.DATA_DIR, token=HF_TOKEN)

import rr_core as R
importlib.reload(R)
vocab = json.load(open(os.path.join(C.DATA_DIR, "Phase_3_RR/Train/label_vocab.json")))
assert sorted(vocab, key=vocab.get) == R.LABELS, "label vocabulary changed - update LABELS in rr_core.py"

from collections import Counter
for dom in R.DOMAINS:
    tr, dv, te = (R.load_split(C.FILES[f"{dom}_{s}"], dom) for s in ("train", "dev", "test"))
    counts = Counter(R.LABELS[l] for d in tr for l in d["labels"])
    sticky = sum(a == b for d in tr for a, b in zip(d["labels"], d["labels"][1:])) / sum(len(d["labels"]) - 1 for d in tr)
    print(f"{dom}: train {len(tr)} docs / {sum(len(d['sents']) for d in tr):,} sentences | dev {len(dv)} | test {len(te)} docs "
          f"| same role as previous sentence: {sticky:.0%}")
    print("   ", dict(counts.most_common()))
''')

    md("## 5. Load the Phase-1 weights once\nThis checks the model and caches the files for the experiment processes.")
    code('''
import backbone as BB
importlib.reload(BB)
tok = BB.load_tokenizer()
probe = BB.build_backbone()
sample = R.tokenize_docs(R.load_split(C.FILES["CL_dev"], "CL")[:1], tok, C.MAX_SENT_TOKENS)[0]
packs, where = R.pack(sample["tok"][:40], C.PACK_TOKENS)
ids, mask, _ = R.batch_packs(packs, where, tok.pad_id, torch.device("cpu"))
with torch.no_grad():
    h = probe(ids, mask)
print(f"backbone ready: {sum(p.numel() for p in probe.parameters()) / 1e6:.1f}M parameters | "
      f"40 sentences -> {len(packs)} packs -> hidden states {tuple(h.shape)}")
del probe, h
''')

    md("""
## 6. Run the experiments
Each experiment is its own process (`run_rr.py --method ...`); with two GPUs, two run at once.
`kd` starts as soon as `full` (its teacher) is done. Results, checkpoints and zips are pushed to your HF repo as they finish.
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
        proc = subprocess.Popen([sys.executable, "run_rr.py", "--method", job], cwd=WORK_DIR, env=env,
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

    md("## 7. Results\nDev macro-F1 overall and per domain, cost of each method, per-role scores and the most frequent confusions.")
    code('''
import pandas as pd

rows, per_label, confusions = [], {}, {}
for path in sorted(glob.glob(os.path.join(C.OUT_DIR, "*", "results.json"))):
    r = json.load(open(path))
    m = r["method"]
    rows.append({"method": m, "dev macro-F1": r["dev_macro"]["f1"], "CL F1": r["dev_by_domain"]["CL"]["f1"],
                 "IT F1": r["dev_by_domain"]["IT"]["f1"], "macro P": r["dev_macro"]["p"], "macro R": r["dev_macro"]["r"],
                 "decoder": r["decoder"], "tau": r["tau"], "best epoch": r["best_epoch"],
                 "params (M)": r["params_total"] / 1e6, "trainable %": 100 * r["params_trainable"] / r["params_total"],
                 "train min": r["train_minutes"], "peak GPU GB": r["peak_gpu_gb"], "dev infer s": r["dev_inference_seconds"]})
    per_label[m] = {lab: v["f1"] for lab, v in r["dev_per_label"].items()}
    confusions[m] = r["confusions"]

table = pd.DataFrame(rows).sort_values("dev macro-F1", ascending=False).round(4)
display(table)
display(pd.DataFrame(per_label).round(3))
best = table.iloc[0]["method"]
print(f"most frequent confusions for '{best}' (gold -> predicted, count):")
for g, p, n in confusions[best]:
    print(f"   {g:24s} -> {p:24s} {n}")
''')

    md("""
## 8. Pick a submission
Every experiment wrote `runs/<method>/submission_<arch>_<method>.zip`: a single `prediction.json` at the root with one
label per sentence for all CL and IT test documents. The best on dev is also copied to `/kaggle/working/submission.zip`.
""")
    code('''
src = glob.glob(os.path.join(C.OUT_DIR, best, "submission_*.zip"))[0]
shutil.copy(src, "/kaggle/working/submission.zip")
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
        out = os.path.join(P2.DESKTOP, a["folder"], f"phase3_{arch}_rr.ipynb")
        build(arch, out)
        print("wrote", out)
