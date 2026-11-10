"""Builds the four Phase-6 notebooks (abstractive summarization), one per Phase-1 model, from the tested sources here."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAPSTONE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CAPSTONE, "phase2_src"))
import build_notebooks as P2          # noqa: E402  (tokenizer + Phase-1 model copies are shared with Phases 2-5)

SPECIFIC = {
    "arch1": {"targets": '["W_qkv", "W_o"]', "houlsby": '["W_o", "down"]', "pfeiffer": '["down"]', "tgt": 1024,
              "max_new": 1024, "micro": 2, "accum": 4, "gen_batch": 8, "case": True},
    "arch2": {"targets": '["qkv", "proj", "q", "kv"]', "houlsby": '["proj", "down"]', "pfeiffer": '["down"]', "tgt": 1000,
              "max_new": 1000, "micro": 2, "accum": 4, "gen_batch": 8, "case": False},
    "arch3": {"targets": '["qkv", "proj"]', "houlsby": '["proj", "down"]', "pfeiffer": '["down"]', "tgt": 1024,
              "max_new": 1024, "micro": 2, "accum": 4, "gen_batch": 8, "case": False},
    "arch4": {"targets": '["q_down", "q_up", "kv_down", "kv_up", "out"]', "houlsby": '["out", "down"]', "pfeiffer": '["down"]',
              "tgt": 1024, "max_new": 1024, "micro": 1, "accum": 8, "gen_batch": 4, "case": False},
}

CONFIG_TEMPLATE = '''"""Every knob of this Phase-6 (abstractive summarization) notebook."""
import os
import json

ARCH = "{arch}"
WORK_DIR = f"/kaggle/working/phase6_{{ARCH}}"
DATA_DIR = os.path.join(WORK_DIR, "data")
CACHE_DIR = os.path.join(WORK_DIR, "phase1")        # Phase-1 files downloaded from Hugging Face land here
OUT_DIR = os.path.join(WORK_DIR, "runs")             # one folder per experiment

HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
RAW = {{"train": os.path.join(DATA_DIR, "Phase_6_Summ", "Train", "train.json"),
       "test": os.path.join(DATA_DIR, "Phase_6_Summ", "Test", "test.json")}}

# ------------------------------------------------------------------ the Phase-1 model
# Looked up in this order: an attached Kaggle dataset whose path mentions "{arch}", then these HF repos.
{block}
KAGGLE_HINT = "{arch}"

SUM_REPO = "D-240357/legal-sum-{arch}"           # fine-tuned checkpoints, results and submission zips go here

# ------------------------------------------------------------------ experiments
# Each one is a separate submission. "full" is the strong reference and the distillation teacher;
# "kd" waits for it to finish. "pfeiffer" (adapters after the feed-forward block only) is available too.
METHODS = ["full", "lorma", "lora", "houlsby", "prompt", "kd"]
KD_TEACHER = "full"
TRAIN_MINUTES = 40             # wall-clock training budget per experiment (the LR schedule ends exactly on it)
MAX_EPOCHS = 3

# ------------------------------------------------------------------ input and output lengths
SRC_LEN = 1000                 # source tokens the model reads
HEAD_FRAC = 0.4                # 40% from the opening of the judgment (parties, facts) ...
N_SPREAD = 3                   # ... and 3 evenly spaced pieces of the rest, the last ending at the final token
TGT_LEN = {tgt}                # summary tokens used for training (90% of references are shorter than this)
CHAR_CAP = 400_000             # text beyond this is not even tokenized
MAX_NEW_TOKENS = {max_new}         # hard ceiling at generation time (the model's context limit)
LENGTH_RATIO = "auto"          # summary words = document words / ratio (auto = training median, ~4.6) ...
MIN_SUMMARY_WORDS = 150        # ... clipped to these bounds
MAX_SUMMARY_WORDS = 900
LENGTH_SLACK = 0.15            # generation must reach 85% of the target length and stops by 115%

# ------------------------------------------------------------------ decoding
REPETITION_PENALTY = 1.15      # Keskar et al. 2019
NO_REPEAT_NGRAM = 4            # never emit the same 4-gram twice
SENTENCE_CASE = {case}         # arch 1 has a lower-case vocabulary: re-capitalise sentence starts
GEN_BATCH = {gen_batch}

# ------------------------------------------------------------------ optimisation
MICRO_DOCS = {micro}
GRAD_ACCUM = {accum}
WARMUP_STEPS = 50
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
FULL_LR = 3e-5
ADAPTER_LR = 5e-4
ADAPTER_RANK = 8
LORA_ALPHA = 16
LORMA_ALPHA = 8
ADAPTER_DROPOUT = 0.05
ADAPTER_TARGETS = {targets}       # LoRA / LoRMA: attention projections
HOULSBY_SITES = {houlsby}         # after the attention output and the feed-forward output
PFEIFFER_SITES = {pfeiffer}       # after the feed-forward output only
ADAPTER_BOTTLENECK = 64
PROMPT_LEN = 16
PROMPT_LR = 1e-2

KD_TEMPERATURE = 2.0
KD_ALPHA = 0.5                 # weight of the reference-summary loss; the rest goes to matching the teacher
STUDENT_LAYERS = {student}

# ------------------------------------------------------------------ evaluation
DEV_DOCS = 150                 # held out from train (there is no official dev set)
DEV_LOSS_DOCS = 100            # teacher-forced loss on these picks the best checkpoint
DEV_GEN_DOCS = 40              # these are actually summarised to measure ROUGE and length
EVAL_EVERY_STEPS = 150
TRAIN_DOC_LIMIT = None         # set to e.g. 200 (and TRAIN_MINUTES = 3) for a quick smoke test
PREP_WORKERS = max(1, os.cpu_count() or 2)

SEED = 13

_override = os.environ.get("P6_CFG_OVERRIDE")
if _override:
    globals().update(json.loads(_override))
'''


def read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read().rstrip("\n")


def sources_for(arch):
    a, s = P2.ARCHS[arch], SPECIFIC[arch]
    p2 = P2.sources_for(arch)
    block = "\n".join(l for l in a["block"].splitlines() if not l.startswith(("SPLIT_PUNCT", "SPLIT_DIGITS")))
    return {
        "config.py": CONFIG_TEMPLATE.format(arch=arch, block=block, tgt=s["tgt"], max_new=s["max_new"], case=s["case"],
                                            gen_batch=s["gen_batch"], micro=s["micro"], accum=s["accum"], targets=s["targets"],
                                            houlsby=s["houlsby"], pfeiffer=s["pfeiffer"], student=a["student"]).rstrip("\n"),
        "tokenizer.py": p2["tokenizer.py"],
        "phase1_model.py": p2["phase1_model.py"],
        "hub_files.py": read(HERE, "common", "hub_files.py"),
        "gen_core.py": read(HERE, "common", "gen_core.py"),
        "lm.py": read(HERE, arch, "lm.py"),
        "run_sum.py": read(HERE, "common", "run_sum.py"),
    }


def build(arch, out_path):
    a = P2.ARCHS[arch]
    work = f"/kaggle/working/phase6_{arch}"
    files = sources_for(arch)
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": text.strip("\n").splitlines(keepends=True)})

    def writefile(name):
        code(f"%%writefile {work}/{name}\n{files[name]}")

    how = ("The encoder reads the source and the decoder writes the summary, attending to it through cross-attention."
           if arch == "arch2" else
           "The model reads `[source] <eos> [summary]` and is trained only on the summary tokens; at test time it "
           "continues after `<eos>` with a KV cache" + (" (added here - Phase 1 never needed one)." if arch == "arch1" else "."))
    md(f"""
# Phase 6 - Abstractive summarization with {a['title']}

Starts from the Phase-1 model ({a['about']}) and writes a summary of each judgment.
Metric: BERTScore F1 (plus ROUGE-1 / ROUGE-L). {how}

**What the data asks for.** Reference summaries are long (median ~640 words, 90% under 1,500) and about **4.6 times
shorter than their document**; documents run from ~1,300 to 140,000 words.
* **Source selection**: 1,000 tokens - 40% from the opening of the judgment (parties, facts) plus three evenly spaced
  pieces of the rest, the last ending at the final word, so the whole case is represented.
* **Length control**: each summary's length budget follows its document's length (document words / 4.6, clipped to
  150-900 words); generation has to reach 85% of it and stops by 115% - length matters a lot for BERTScore and ROUGE.
* **Redundancy control**: greedy decoding with a repetition penalty (Keskar et al. 2019) and no repeated 4-grams.

**Experiments** (each is its own submission; all from the Phase-6 list)

| Experiment | What trains |
|---|---|
| `full` | every weight - strong reference and the distillation teacher |
| `lorma` | **LoRMA** (compulsory): `(alpha/r BA + I) W x` on the attention projections (Bihany, Patel & Modi 2025) |
| `lora` | LoRA on the same projections (Hu et al. 2021) |
| `houlsby` | Houlsby bottleneck adapters after the attention and feed-forward outputs (Houlsby et al. 2019) |
| `prompt` | prompt tuning: 16 learned virtual tokens in front of the source (Lester et al. 2021) |
| `kd` | word-level distillation (Hinton et al. 2015; Kim & Rush 2016): a half-depth student learns the fine-tuned `full` model's next-token distributions |

There is no dev set, so 150 training documents are held out: teacher-forced loss on them picks the best checkpoint,
and 40 of them are actually summarised to report ROUGE-1 / ROUGE-L and length. Every experiment trains for a fixed
**time budget** (`TRAIN_MINUTES`).

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

    md("## 3. Code\nThe Phase-1 tokenizer and model (unchanged, so the trained weights load), plus the summarization pieces.")
    for name in ("tokenizer.py", "phase1_model.py", "hub_files.py", "gen_core.py", "lm.py", "run_sum.py"):
        writefile(name)

    md("## 4. Data: download, select the source, tokenize\nDone once with all CPU cores and stored as compact arrays.")
    code('''
import importlib
import config as C
importlib.reload(C)
import lm as LM
import gen_core as G
importlib.reload(G)

for split, folder in (("train", "Train"), ("test", "Test")):
    hf_hub_download(C.HF_DATASET_REPO, f"Phase_6_Summ/{folder}/{split}.json", repo_type="dataset",
                    local_dir=C.DATA_DIR, token=HF_TOKEN)

tok = LM.load_tokenizer()                       # caches the tokenizer before the worker processes start
prep_cfg = {"src_len": C.SRC_LEN, "tgt_len": C.TGT_LEN, "char_cap": C.CHAR_CAP, "head_frac": C.HEAD_FRAC,
            "n_spread": C.N_SPREAD, "pad": tok.pad_id}

ids, docs, sums = G.load_split(C.RAW["train"], with_summary=True)
arrays = G.prepare_split(docs, sums, prep_cfg, C.PREP_WORKERS)
np.savez(os.path.join(C.DATA_DIR, "train.npz"), **arrays)
json.dump(sums, open(os.path.join(C.DATA_DIR, "train_summaries.json"), "w"))
full = arrays["tgt_len"] < C.TGT_LEN
meta = {"tokens_per_word": float(np.median(arrays["tgt_len"][full] / np.maximum(arrays["sum_words"][full], 1))),
        "length_ratio": float(np.median(arrays["doc_words"] / np.maximum(arrays["sum_words"], 1)))}
json.dump(meta, open(os.path.join(C.DATA_DIR, "meta.json"), "w"))
print(f"train: {len(ids):,} docs | source {arrays['src_len'].mean():.0f} tokens | summary {arrays['tgt_len'].mean():.0f} tokens "
      f"({100 * (~full).mean():.0f}% cut at {C.TGT_LEN}) | {meta['tokens_per_word']:.2f} tokens/word | "
      f"document/summary ratio {meta['length_ratio']:.2f}")
del docs, sums, arrays
gc.collect()

ids, docs, _ = G.load_split(C.RAW["test"], with_summary=False)
arrays = G.prepare_split(docs, None, prep_cfg, C.PREP_WORKERS)
np.savez(os.path.join(C.DATA_DIR, "test.npz"), **arrays)
json.dump(ids, open(os.path.join(C.DATA_DIR, "test_ids.json"), "w"))
print(f"test: {len(ids)} docs | source {arrays['src_len'].mean():.0f} tokens | document words median {np.median(arrays['doc_words']):.0f}")
''')

    md("## 5. Load the Phase-1 weights once\nA quick generation from the untouched Phase-1 model checks the whole decoding path and caches the files.")
    code('''
probe = LM.build_lm().eval()
test = np.load(os.path.join(C.DATA_DIR, "test.npz"))
with torch.no_grad():
    out = probe.generate([test["src"][0][:200].tolist()], [10], [20], tok, C, torch.device("cpu"))
print(f"model ready: {sum(p.numel() for p in probe.parameters()) / 1e6:.1f}M parameters | sample continuation: {tok.decode(out[0])!r}")
del probe, test
''')

    md(f"""
## 6. Run the experiments
Each experiment is its own process (`run_sum.py --method ...`); with two GPUs, two run at once, so the six take about
three rounds of `TRAIN_MINUTES` + ~15 minutes (training, then summarising 40 held-out and 100 test documents).
`kd` starts as soon as `full` (its teacher) is done. Results, checkpoints and zips are pushed to `D-240357/legal-sum-{arch}`.
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
        proc = subprocess.Popen([sys.executable, "run_sum.py", "--method", job], cwd=WORK_DIR, env=env,
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

    md("## 7. Results\nHeld-out ROUGE-1 / ROUGE-L, length against the references, and cost. A generated summary is shown next to its reference.")
    code('''
import pandas as pd

rows = []
for path in sorted(glob.glob(os.path.join(C.OUT_DIR, "*", "results.json"))):
    r = json.load(open(path))
    rows.append({"method": r["method"], "ROUGE-1": r["dev_rouge1_f1"], "ROUGE-L": r["dev_rougeL_f1"],
                 "held-out loss": r["heldout_loss"], "words (gen)": r["dev_words_generated"],
                 "words (ref)": r["dev_words_reference"], "docs seen": r["docs_seen"],
                 "params (M)": r["params_total"] / 1e6, "trainable %": 100 * r["params_trainable"] / r["params_total"],
                 "s / summary": r["seconds_per_summary"], "minutes": r["minutes"], "peak GPU GB": r["peak_gpu_gb"]})
table = pd.DataFrame(rows).sort_values("ROUGE-L", ascending=False).round(4)
display(table)
best = table.iloc[0]["method"]
sample = json.load(open(os.path.join(C.OUT_DIR, best, "dev_samples.json")))[0]
print(f"--- reference ---\\n{sample['reference'][:800]}\\n\\n--- {best} ---\\n{sample['generated'][:800]}")
''')

    md("""
## 8. Pick a submission
Every experiment wrote `runs/<method>/submission_<arch>_<method>.zip`: a single `prediction.json` at the root holding a
JSON list of `{"id": ..., "summary": ...}` for all 100 test documents. The best on held-out ROUGE-L is also copied to
`/kaggle/working/submission.zip`. Upload the zips exactly as they are.
""")
    code('''
shutil.copy(glob.glob(os.path.join(C.OUT_DIR, best, "submission_*.zip"))[0], "/kaggle/working/submission.zip")
print(f"best on held-out ROUGE-L: {best} -> /kaggle/working/submission.zip")
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
        out = os.path.join(P2.DESKTOP, a["folder"], f"phase6_{arch}_sum.ipynb")
        build(arch, out)
        print("wrote", out)
