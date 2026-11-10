"""Builds the four Phase-2 notebooks (one per Phase-1 model) from the tested sources in this folder."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CAPSTONE = os.path.dirname(HERE)
DESKTOP = os.path.expanduser("~/Desktop")

ARCHS = {
    "arch1": {
        "folder": "First Model", "title": "Arch 1 - GPT-2 style decoder (word-level tokenizer)",
        "model_src": None,
        "block": '''PHASE1_REPOS = ["D-240357/legal-gpt2-decoder"]
WEIGHT_FILES = ["model_state_dict.pt"]
CONFIG_FILES = ["config.json"]
TOKENIZER_FILES = ["custom_tokenizer.json"]
TOKENIZER_REPOS = PHASE1_REPOS
SPLIT_DIGITS = True            # split "Pradesh1" into "Pradesh" + "1" (see the alignment check in the notebook)''',
        "targets": '["W_qkv", "W_o"]', "student": "[0, 2, 4, 6, 8, 10]", "partial": 4,
        "micro": 8, "accum": 2, "eval_batch": 16, "bidir": True,
        "about": "12-layer GPT-2 style decoder (111M), learned positions, word-level lowercased vocabulary.",
    },
    "arch2": {
        "folder": "Second Model", "title": "Arch 2 - Encoder-decoder (encoder used for tagging)",
        "model_src": "arch2_src/model.py",
        "block": '''PHASE1_REPOS = ["D-240357/legal-arch2-encdec"]
WEIGHT_FILES = ["model_final.pt", "model_best.pt"]
TOKENIZER_FILES = ["tokenizer.json", "data/tokenizer.json"]
TOKENIZER_REPOS = ["D-240357/legal-arch2-encdec", "D-240357/legal-arch3-decoder"]   # same BPE file as arch 3
SPLIT_PUNCT = True             # one punctuation character per piece (see the alignment check in the notebook)''',
        "targets": '["qkv", "proj"]', "student": "[0, 2, 4]", "partial": 2,
        "micro": 8, "accum": 2, "eval_batch": 16, "bidir": None,
        "about": "T5-style encoder-decoder (109.5M); only its 6-layer bidirectional encoder is needed for tagging.",
    },
    "arch3": {
        "folder": "Third Model", "title": "Arch 3 - Llama-style decoder",
        "model_src": "arch3_src/model.py",
        "block": '''PHASE1_REPOS = ["D-240357/legal-arch3-decoder"]
WEIGHT_FILES = ["model_final.pt", "model_best.pt"]
TOKENIZER_FILES = ["tokenizer.json", "data/tokenizer.json"]
TOKENIZER_REPOS = PHASE1_REPOS
SPLIT_PUNCT = True             # one punctuation character per piece (see the alignment check in the notebook)''',
        "targets": '["qkv", "proj"]', "student": "[0, 2, 4, 6, 8, 10]", "partial": 4,
        "micro": 8, "accum": 2, "eval_batch": 16, "bidir": True,
        "about": "12-layer Llama-style decoder (100.1M): RMSNorm, RoPE, grouped-query attention, SwiGLU, byte-level BPE.",
    },
    "arch4": {
        "folder": "Fourth Model ", "title": "Arch 4 - DeepSeek-style MoE decoder",
        "model_src": "arch4_src/model.py",
        "block": '''# The 600M-token run (v2) is preferred; the earlier 100M-token run is the fallback.
PHASE1_REPOS = ["D-240357/legal-arch4-moe-v2", "D-240357/legal-arch4-moe"]
WEIGHT_FILES = ["model_final.pt", "model_best.pt"]
TOKENIZER_FILES = ["tokenizer.json", "data/tokenizer.json"]
TOKENIZER_REPOS = PHASE1_REPOS + ["D-240357/legal-arch3-decoder"]
SPLIT_PUNCT = True             # one punctuation character per piece (see the alignment check in the notebook)''',
        "targets": '["q_down", "q_up", "kv_down", "kv_up", "out"]', "student": "[0, 2, 4, 6, 8, 10]", "partial": 4,
        "micro": 4, "accum": 4, "eval_batch": 8, "bidir": True,
        "about": "12-layer MoE decoder (222.6M total, ~81M active): MLA attention, 32 routed + 1 shared expert per layer.",
    },
}

CONFIG_TEMPLATE = '''"""Every knob of this Phase-2 (legal NER) notebook."""
import os
import json

ARCH = "{arch}"
WORK_DIR = f"/kaggle/working/phase2_{{ARCH}}"
DATA_DIR = os.path.join(WORK_DIR, "data")
CACHE_DIR = os.path.join(WORK_DIR, "phase1")        # Phase-1 files downloaded from Hugging Face land here
OUT_DIR = os.path.join(WORK_DIR, "runs")             # one folder per experiment

HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
_P2 = os.path.join(DATA_DIR, "Phase_2_LNER")
FOLD_TEXT = [os.path.join(_P2, "Train", "fold1-text.json"), os.path.join(_P2, "Train", "fold2-text.json")]
FOLD_GOLD = [os.path.join(_P2, "Train", "fold1-gold.json"), os.path.join(_P2, "Train", "fold2-gold.json")]
TEST_TEXT = os.path.join(_P2, "Test", "fold3-text.json")

# ------------------------------------------------------------------ the Phase-1 model
# Looked up in this order: an attached Kaggle dataset whose path mentions "{arch}", then these HF repos.
{block}
KAGGLE_HINT = "{arch}"

NER_REPO = "D-240357/legal-ner-{arch}"           # fine-tuned checkpoints, results and submission zips go here

# ------------------------------------------------------------------ experiments
# Each one is a separate submission. "kd" distils the fine-tuned KD_TEACHER into a half-depth student,
# so it waits for that run to finish.
METHODS = ["full", "lorma", "lora", "partial", "kd"]
KD_TEACHER = "full"

# ------------------------------------------------------------------ data
DEV_DOCS_PER_FOLD = 4          # 8 of the 70 training documents are held out to pick the best epoch / decoder
WINDOW = 512                   # tokens per window; documents average ~7K tokens
STRIDE = 384                   # 128-token overlap; every token is labelled from the window where it is most central
{bidir_line}
TRAIN_DOC_LIMIT = None         # set to e.g. 6 for a quick end-to-end smoke test

# ------------------------------------------------------------------ optimisation
EPOCHS = 10
PATIENCE = 3                   # stop after 3 epochs without a better dev F1
MICRO_BATCH = {micro}
GRAD_ACCUM = {accum}
EVAL_BATCH = {eval_batch}
WARMUP_FRAC = 0.1
WEIGHT_DECAY = 0.01
HEAD_DROPOUT = 0.1
HEAD_LR = 1e-3

FULL_LR = 1e-4                 # top layer; each layer below gets LLRD times the one above it
LLRD = 0.85
PARTIAL_LR = 2e-4
PARTIAL_TOP_LAYERS = {partial}
ADAPTER_LR = 5e-4
ADAPTER_RANK = 8
LORA_ALPHA = 16
LORMA_ALPHA = 8
ADAPTER_DROPOUT = 0.05
ADAPTER_TARGETS = {targets}       # attention projections, as in the LoRA and LoRMA papers
CLASS_WEIGHTING = "none"       # or "sqrt_inv" to up-weight rare tags

KD_TEMPERATURE = 2.0
KD_ALPHA = 0.5                 # weight of the gold-label loss; the rest goes to matching the teacher
STUDENT_LAYERS = {student}

SEED = 13

_override = os.environ.get("P2_CFG_OVERRIDE")
if _override:
    globals().update(json.loads(_override))
'''

PATCHES = {
    "arch3": [
        ("        self.q_dim, self.kv_dim = q_dim, kv_dim\n",
         "        self.q_dim, self.kv_dim = q_dim, kv_dim\n        self.causal = True                         # Phase 2 turns this off to let tokens look right\n"),
        ("out = F.scaled_dot_product_attention(q, k, v, is_causal=True)",
         "out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)"),
    ],
    "arch4": [
        ("        self.out = nn.Linear(self.H * self.vd, cfg.d_model, bias=False)\n",
         "        self.out = nn.Linear(self.H * self.vd, cfg.d_model, bias=False)\n        self.causal = True                         # Phase 2 turns this off to let tokens look right\n"),
        ("o = F.scaled_dot_product_attention(q, k, v, is_causal=True)",
         "o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)"),
        ("        self.cap_factor = cfg.capacity_factor\n",
         "        self.cap_factor = cfg.capacity_factor\n        self.dropless = False                      # Phase 2 fine-tuning routes every token\n"),
        ("        if self.training:\n            cap = int(",
         "        if self.training and not self.dropless:\n            cap = int("),
    ],
    "arch2": [],
}

NOTE = '"""\nPhase-1 model code, unchanged except for the small switches marked "Phase 2", so the saved weights load as-is.\n"""\n'


def read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read().rstrip("\n")


def sources_for(arch):
    a = ARCHS[arch]
    files = {}
    if arch == "arch1":
        files["tokenizer.py"] = read(HERE, "arch1", "tokenizer.py")
        files["phase1_model.py"] = read(HERE, "arch1", "phase1_model.py")
    else:
        files["tokenizer.py"] = read(CAPSTONE, "arch3_src", "bpe_tokenizer.py") + "\n" + read(HERE, "common", "bpe_ner_addon.py")
        model = read(CAPSTONE, a["model_src"])
        for old, new in PATCHES[arch]:
            assert model.count(old) == 1, (arch, old)
            model = model.replace(old, new)
        files["phase1_model.py"] = NOTE + model
    bidir = ("BIDIRECTIONAL = True            # lift the causal mask while fine-tuning (Dukic & Snajder 2024; LLM2Vec)"
             if a["bidir"] else "BIDIRECTIONAL = True            # the encoder is bidirectional anyway")
    files["config.py"] = CONFIG_TEMPLATE.format(arch=arch, block=a["block"], bidir_line=bidir, micro=a["micro"],
                                                accum=a["accum"], eval_batch=a["eval_batch"], partial=a["partial"],
                                                targets=a["targets"], student=a["student"]).rstrip("\n")
    files["backbone.py"] = read(HERE, arch, "backbone.py")
    files["hub_files.py"] = read(HERE, "common", "hub_files.py")
    files["ner_core.py"] = read(HERE, "common", "ner_core.py")
    files["run_ner.py"] = read(HERE, "common", "run_ner.py")
    return files


def build(arch, out_path):
    a = ARCHS[arch]
    work = f"/kaggle/working/phase2_{arch}"
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
# Phase 2 - Legal NER with {a['title']}

Starts from the Phase-1 model ({a['about']}) and fine-tunes it to tag the 12 legal entity types with
**exact character offsets** (strict macro-F1).

Five experiments run back to back, spread over both T4s; each one produces its own submission zip:

| Experiment | What is trained |
|---|---|
| `full` | every weight, with layer-wise learning-rate decay |
| `lorma` | **LoRMA** (compulsory): multiplicative low-rank update `(alpha/r * BA + I) W x` on the attention projections |
| `lora` | LoRA: additive low-rank update on the same projections |
| `partial` | only the top layers + final norm |
| `kd` | knowledge distillation: the fine-tuned `full` model teaches a half-depth student |

How the pieces fit:
* **Tagging**: BIO tags over tokens, decoded back to character spans. Punctuation is split one character per piece so
  gold boundaries line up with token boundaries (checked below - an "oracle" decode of the gold tags shows the ceiling).
* **Long documents** (avg. ~31K characters): 512-token windows with 128-token overlap; each token keeps the prediction
  from the window where it has the most context on both sides.
* **Decoder-only models** see the whole window during fine-tuning (causal mask lifted), following Dukic & Snajder (2024)
  *Looking Right is Sometimes Right* and BehnamGhader et al. (2024) *LLM2Vec*.
* **Decoding**: plain argmax, or a Viterbi pass that only allows valid BIO sequences; the better one on the dev
  documents is used for the test predictions.

References: Hu et al. 2021 (LoRA); Bihany, Patel & Modi 2025 (LoRMA); Hinton et al. 2015 (distillation);
Sanh et al. 2019 (DistilBERT layer selection); Howard & Ruder 2018 and Clark et al. 2020 (layer-wise LR decay).

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

    md("## 3. Code\nThe Phase-1 tokenizer and model (unchanged, so the trained weights load), plus the NER pieces.")
    writefile("tokenizer.py")
    writefile("phase1_model.py")
    writefile("hub_files.py")
    writefile("backbone.py")
    writefile("ner_core.py")
    writefile("run_ner.py")

    md("## 4. Phase-2 data")
    code('''
import importlib
import config as C
importlib.reload(C)

for name in ["Train/fold1-text.json", "Train/fold1-gold.json", "Train/fold2-text.json",
             "Train/fold2-gold.json", "Train/labels.txt", "Test/fold3-text.json"]:
    hf_hub_download(C.HF_DATASET_REPO, f"Phase_2_LNER/{name}", repo_type="dataset", local_dir=C.DATA_DIR, token=HF_TOKEN)

import ner_core as N
importlib.reload(N)
folds = [N.load_docs(C.FOLD_TEXT[i], C.FOLD_GOLD[i]) for i in range(2)]
test_docs = N.load_docs(C.TEST_TEXT)
with open(os.path.join(C.DATA_DIR, "Phase_2_LNER/Train/labels.txt")) as f:
    assert f.read().split() == N.LABELS, "label list changed - update LABELS in ner_core.py"

n_ents = sum(len(d["ents"]) for fold in folds for d in fold)
print(f"train docs {sum(map(len, folds))} | test docs {len(test_docs)} | gold entities {n_ents}")
''')

    md("""
## 5. Tokenizer alignment and offset self-test
Strict scoring needs exact character offsets. Here every training document is encoded, the gold entities are turned into
BIO tags, and those tags are decoded straight back into spans. The resulting "oracle" F1 is the best score this
tokenization allows, and it doubles as a test of the whole offset pipeline.
""")
    code('''
import backbone as BB
importlib.reload(BB)

tok = BB.load_tokenizer()
all_docs = [d for fold in folds for d in fold]
enc = [N.encode_doc(d, tok) for d in all_docs]
oracle = {d["id"]: N.oracle_spans(d) for d in enc}
macro, per = N.strict_scores(oracle, {d["id"]: d["ents"] for d in enc})
print(f"oracle strict macro-F1 = {macro['f1']:.4f}  (the ceiling for this tokenizer)")
print("per label:", {lab: round(v["f1"], 3) for lab, v in per.items()})

n_tok = [len(d["ids"]) for d in enc]
print(f"tokens per document: mean {sum(n_tok) / len(n_tok):,.0f}, max {max(n_tok):,}")
if hasattr(tok, "unk_id"):
    unk = sum(int((d["ids"] == tok.unk_id).sum()) for d in enc) / sum(n_tok)
    print(f"unknown-word tokens: {100 * unk:.1f}%")
assert macro["f1"] > 0.95, "offset pipeline looks broken - do not submit"
''')

    md("## 6. Load the Phase-1 weights once\nThis also caches them for the experiment processes.")
    code('''
probe = BB.build_backbone()
n = sum(p.numel() for p in probe.parameters())
ids = torch.tensor([enc[0]["ids"][:64].tolist()])
with torch.no_grad():
    h = probe(ids, torch.ones_like(ids, dtype=torch.bool))
print(f"backbone ready: {n / 1e6:.1f}M parameters, hidden states {tuple(h.shape)}")
del probe, h
''')

    md("""
## 7. Run the experiments
Each experiment is its own process (`run_ner.py --method ...`); with two GPUs, two run at the same time.
`kd` starts once `full` (its teacher) has finished. Logs are prefixed with the GPU they run on.
Fine-tuned weights, results and submission zips are pushed to your NER repo on Hugging Face as they finish.
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
        proc = subprocess.Popen([sys.executable, "run_ner.py", "--method", job], cwd=WORK_DIR, env=env,
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

    md("## 8. Results\nDev scores (strict, the competition metric), cost of each method, and where the errors are.")
    code('''
import pandas as pd

rows, per_label, errors = [], {}, {}
for path in sorted(glob.glob(os.path.join(C.OUT_DIR, "*", "results.json"))):
    r = json.load(open(path))
    m = r["method"]
    rows.append({"method": m, "dev macro-F1": r["dev_macro"]["f1"], "dev P": r["dev_macro"]["p"],
                 "dev R": r["dev_macro"]["r"], "decoder": r["decoder"], "best epoch": r["best_epoch"],
                 "params (M)": r["params_total"] / 1e6, "trainable %": 100 * r["params_trainable"] / r["params_total"],
                 "train min": r["train_minutes"], "peak GPU GB": r["peak_gpu_gb"], "dev infer s": r["dev_inference_seconds"]})
    per_label[m] = {lab: v["f1"] for lab, v in r["dev_per_label"].items()}
    errors[m] = r["dev_errors"]

table = pd.DataFrame(rows).sort_values("dev macro-F1", ascending=False).round(4)
display(table)
display(pd.DataFrame(per_label).round(3))
display(pd.DataFrame(errors).T)
''')

    md("""
## 9. Pick a submission
Every experiment already wrote `runs/<method>/submission_<arch>_<method>.zip` (a single `prediction.json` at the
root, every test document id present). The best one on dev is copied to `/kaggle/working/submission.zip` too.
Submitting several of them counts as different methods.
""")
    code('''
best = table.iloc[0]["method"]
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
    return files


if __name__ == "__main__":
    for arch, a in ARCHS.items():
        folder = os.path.join(DESKTOP, a["folder"])
        out = os.path.join(folder, f"phase2_{arch}_ner.ipynb")
        build(arch, out)
        print("wrote", out)
