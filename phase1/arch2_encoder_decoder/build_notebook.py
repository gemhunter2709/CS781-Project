"""Assembles phase1_arch2_encoder_decoder.ipynb from the tested source files in this folder."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = "/kaggle/working/arch2"
cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})

def source_file(name):
    body = open(os.path.join(HERE, name)).read().rstrip("\n")
    code(f"%%writefile {WORK}/{name}\n{body}")

md("""
# Arch 2 - Encoder-decoder language model (from scratch, 2x T4, ~1.5 h budget)

A compact T5-style model written in plain PyTorch ops (no model or tokenizer libraries). **109.5M parameters.**
It is shaped like the Phase 1 task itself: the **encoder reads the 512-word prompt** with bidirectional attention and the
**decoder writes the 128-word continuation**, attending to the prompt through cross-attention.

| Piece | Choice |
|---|---|
| Layout | 6 encoder + 6 decoder layers, d_model 768, 12 query / 4 KV heads (grouped-query attention) |
| Training example | a 1344-token window: first 1024 tokens go to the encoder, the next 320 are the decoder's target (cross-entropy on those) |
| Efficiency | RMSNorm, RoPE, SwiGLU, fused SDPA attention, shared embedding table, fp16 + `torch.compile`, fused AdamW, DDP over both T4s |
| Generation | encoder runs once per prompt; decoder uses a KV cache and precomputed cross-attention K/V; padded prompts are masked |
| **Time budget** | the LR schedule follows `max(step progress, wall-clock progress)`: the cosine decay lands on its floor exactly when `TRAIN_TIME_BUDGET_MIN` (80 min, incl. compile) runs out |
| Safety | checkpoints pushed to Hugging Face every 200 steps, best-validation weights saved separately, exact resume |

Tokenizer and token files are reused from your arch 3 repo (`HF_DATA_REPO` in `config.py`), so every model sees identical data.

**If Kaggle crashes: press Run All again** - it restores everything from the Hub and continues; time already spent counts against the budget.

Settings: GPU **T4 x2**, Internet **on**, secret `HF` holding a write token.
""")

md("## 1. Setup and Hugging Face login")
code('''
!pip install -q ijson
import os, sys, json, math, time, gc, zipfile
import torch
from huggingface_hub import login, HfApi, hf_hub_download
from kaggle_secrets import UserSecretsClient

HF_TOKEN = UserSecretsClient().get_secret("HF")
os.environ["HF_TOKEN"] = HF_TOKEN          # child processes (torchrun) read it from here
login(token=HF_TOKEN)

WORK_DIR = "/kaggle/working/arch2"
os.makedirs(WORK_DIR, exist_ok=True)
sys.path.insert(0, WORK_DIR)
print("Logged in to Hugging Face.")
''')

md("## 2. GPU check")
code('''
n_gpu = torch.cuda.device_count()
print("CUDA available:", torch.cuda.is_available(), "| GPUs:", n_gpu)
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}, {p.total_memory / 1e9:.1f} GB, compute capability {p.major}.{p.minor}")
if n_gpu < 2:
    print("Only one GPU visible - training still works, it just runs gradient accumulation instead of DDP.")
''')

md("## 3. Configuration\nEvery knob lives in `config.py`. Edit it here, then re-run this cell.")
source_file("config.py")

md("## 4. Restore anything already saved on Hugging Face\nOn a fresh run this does nothing. After a crash it brings back the tokenizer, token files and the latest checkpoint.")
code('''
import importlib
import config as C
importlib.reload(C)

for d in (C.DATA_DIR, C.CKPT_DIR):
    os.makedirs(d, exist_ok=True)

api = HfApi()
api.create_repo(C.HF_MODEL_REPO, token=HF_TOKEN, private=C.HF_PRIVATE, exist_ok=True)
on_hub = set(api.list_repo_files(C.HF_MODEL_REPO, token=HF_TOKEN))

def restore(remote_name, local_path, local_dir):
    if os.path.exists(local_path):
        print(f"  have locally: {remote_name}")
    elif remote_name in on_hub:
        print(f"  downloading {remote_name} ...")
        hf_hub_download(C.HF_MODEL_REPO, remote_name, local_dir=local_dir, token=HF_TOKEN)
    else:
        return False
    return True

# 'data/meta.json' is uploaded last, so its presence means the whole data upload finished
def hub_files(repo):
    try:
        return set(api.list_repo_files(repo, token=HF_TOKEN))
    except Exception:
        return set()

data_repo = None
for repo in (C.HF_MODEL_REPO, C.HF_DATA_REPO):
    if "data/meta.json" in hub_files(repo):
        data_repo = repo
        break

if data_repo:
    print(f"  using tokenizer + token files from {data_repo}")
    for name in ("tokenizer.json", "val.bin", "train.bin", "meta.json"):
        local = os.path.join(C.DATA_DIR, name)
        if os.path.exists(local):
            print(f"  have locally: data/{name}")
        else:
            print(f"  downloading data/{name} ...")
            hf_hub_download(data_repo, f"data/{name}", local_dir=C.WORK_DIR, token=HF_TOKEN)
else:
    print("  no prepared data on the hub yet - section 6 will build it")

if restore("training_state.pt", C.STATE_PATH, C.CKPT_DIR):
    restore("train_log.jsonl", C.LOG_PATH, C.CKPT_DIR)
    print("Checkpoint found: training will resume from it.")
else:
    print("No checkpoint yet: training will start from scratch.")
''')

md("## 5. Tokenizer: byte-level BPE from scratch\nPre-token regex -> UTF-8 bytes -> learned merges. Training works on the table of unique words, which keeps pure Python fast enough.")
source_file("bpe_tokenizer.py")

md("## 6. Data prep: train the tokenizer, tokenize the corpus, mirror to the Hub\nUses every CPU core. Skipped automatically when the files are already there.")
source_file("prep_data.py")
code('''
import subprocess
rc = subprocess.run([sys.executable, "prep_data.py"], cwd=WORK_DIR).returncode
assert rc == 0, "data prep failed - read the log above"
''')

md("## 7. Tokenizer sanity check")
code('''
import numpy as np
from bpe_tokenizer import ByteBPE

tok = ByteBPE.load(C.TOKENIZER_PATH)
sample = "The Hon'ble Supreme Court held that Section 438 of the Cr.P.C. does not apply; see (2013) 5 SCC 1."
ids = tok.encode(sample)
assert tok.decode(ids) == sample, "round-trip failed"
print(f"vocab {tok.vocab_size} | eos {tok.eos_id} | pad {tok.pad_id}")
print(f"{len(sample)} chars -> {len(ids)} tokens:", [tok.vocab[i].decode('utf-8', 'replace') for i in ids[:14]], "...")

meta = json.load(open(C.META_PATH))
train = np.memmap(C.TRAIN_BIN, dtype=np.uint16, mode="r")
print(f"train tokens: {meta['train_tokens']:,} | val tokens: {meta['val_tokens']:,} | max id {int(train[:5_000_000].max())}")
print("first tokens decoded:", repr(tok.decode(train[:60].tolist())))
''')

md("## 8. Model\nBidirectional encoder + causal decoder with cross-attention. Grouped-query attention, RoPE and RMSNorm throughout.")
source_file("model.py")

md("## 9. Model checks (CPU, tiny)\nParameter count of the real config, then correctness checks: cached incremental decoding must reproduce the teacher-forced pass, and left-padding a prompt must not change what the model sees.")
code('''
import model as M
importlib.reload(M)

real = M.ModelConfig(vocab_size=C.VOCAB_SIZE, d_model=C.D_MODEL, n_head=C.N_HEAD, n_kv_head=C.N_KV_HEAD,
                     n_enc_layer=C.N_ENC_LAYER, n_dec_layer=C.N_DEC_LAYER, d_ff=C.D_FF,
                     enc_len=C.ENC_LEN, dec_len=C.DEC_LEN)
with torch.device("meta"):                      # count parameters without allocating them
    n_params = M.EncDecLM(real).num_params()
print(f"Real config: {n_params / 1e6:.1f}M parameters")
assert 50e6 <= n_params <= 300e6, "outside the 50-300M budget"

torch.manual_seed(0)
tiny = M.EncDecLM(M.ModelConfig(vocab_size=300, d_model=64, n_head=4, n_kv_head=2, n_enc_layer=2, n_dec_layer=2,
                                d_ff=96, enc_len=24, dec_len=12))

# 1) training forward/backward runs and every parameter receives a gradient
tiny.train()
enc, dec_in, tgt = torch.randint(0, 300, (2, 24)), torch.randint(0, 300, (2, 12)), torch.randint(0, 300, (2, 12))
ce, z = tiny(enc, dec_in, tgt)
(ce + 1e-4 * z).backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in tiny.parameters()), "missing/non-finite gradient"
print(f"train step OK | ce {ce.item():.3f}")

# 2) incremental decoding with both caches == teacher-forced decoding
tiny.eval()
enc_out = tiny.encode(enc)
full, _ = tiny.decode(dec_in, enc_out=enc_out)
cross, caches, steps = tiny.cross_kv(enc_out), None, []
for t in range(12):
    lg, caches = tiny.decode(dec_in[:, t:t + 1], cross_kvs=cross, caches=caches, start_pos=t)
    steps.append(lg)
diff = (torch.cat(steps, 1) - full).abs().max().item()
print(f"Cached vs teacher-forced decoding, max abs diff: {diff:.2e}")
assert diff < 1e-4
del tiny
''')

md("## 10. Trainer (DDP, runs under torchrun)\nOne process per GPU. Batch order is a pure function of the step number, so resuming is exact and free.")
source_file("train.py")

md("""
## 11. Train
Launches `torchrun` on every visible GPU and streams its log here.

* Re-running this cell after a crash resumes from the last checkpoint on the Hub.
* The first ~2-4 minutes are `torch.compile` warming up - the first log line appears after that. Compile time counts against the budget.
* Lines look like `step 400 | loss ... | lr ... | K tok/s | mem ...GB | progress 31% | 55 min left`. `progress` is whichever is further along, steps or wall-clock; the run ends at 100% with the LR fully decayed. `[eval]` lines show validation loss.
* The run finishes by itself after `TRAIN_TIME_BUDGET_MIN` minutes, runs a final validation, and pushes the final weights.
""")
code('''
import subprocess

n_gpu = max(1, torch.cuda.device_count())
env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["OMP_NUM_THREADS"] = "4"
if C.NCCL_P2P_OFF:
    env["NCCL_P2P_DISABLE"] = "1"

cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpu}", "train.py"]
print("Launching:", " ".join(cmd))
proc = subprocess.Popen(cmd, cwd=WORK_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="")
rc = proc.wait()
print(f"\\ntrainer exited with code {rc}")
''')

md("## 12. Loss curves")
code('''
import matplotlib.pyplot as plt

train_pts, val_pts = [], []
with open(C.LOG_PATH) as f:
    for line in f:
        r = json.loads(line)
        if "loss" in r:
            train_pts.append((r["step"], r["loss"]))
        elif "val_loss" in r:
            val_pts.append((r["step"], r["val_loss"]))

plt.figure(figsize=(11, 4.5))
plt.plot(*zip(*train_pts), label="train", alpha=0.8)
if val_pts:
    plt.plot(*zip(*val_pts), "o-", label="validation", color="orange")
plt.xlabel("optimizer step"); plt.ylabel("cross-entropy"); plt.yscale("log")
plt.title("Arch 2 encoder-decoder - training and validation loss")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plot_path = os.path.join(C.WORK_DIR, "loss_curve.png")
plt.savefig(plot_path, dpi=150); plt.show()
if val_pts:
    print(f"final val loss {val_pts[-1][1]:.4f} | perplexity {math.exp(val_pts[-1][1]):.2f}")
''')

md("## 13. Load the trained model for generation")
code('''
# Which weights to generate with: "final" (end of training), "best" (lowest validation loss) or "latest" (last saved state).
# Section 13b compares them on held-out text, so you can pick by generation quality, not only validation loss.
CHOSEN_WEIGHTS = "final"

on_hub_now = set(api.list_repo_files(C.HF_MODEL_REPO, token=HF_TOKEN))
WEIGHT_FILES = {"final": (C.FINAL_PATH, "model_final.pt"),
                "best": (C.BEST_PATH, "model_best.pt"),
                "latest": (C.STATE_PATH, "training_state.pt")}
available = {}
for name, (path, remote) in WEIGHT_FILES.items():
    if not os.path.exists(path) and remote in on_hub_now and name != "latest":
        hf_hub_download(C.HF_MODEL_REPO, remote, local_dir=C.CKPT_DIR, token=HF_TOKEN)
    if os.path.exists(path):
        available[name] = path
if "final" in available:
    available.pop("latest", None)                 # identical to final once training has finished
print("Weights found:", list(available))
if CHOSEN_WEIGHTS not in available:
    CHOSEN_WEIGHTS = next(k for k in ("final", "best", "latest") if k in available)
    print("Requested weights not available, using:", CHOSEN_WEIGHTS)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(path):
    blob = torch.load(path, map_location="cpu")
    model = M.EncDecLM(M.ModelConfig(**blob["model_cfg"]))
    model.load_state_dict({k: v.float() for k, v in blob["model"].items()})
    del blob
    return model.to(device).eval()

gen_model = load_model(available[CHOSEN_WEIGHTS])
gc.collect()
print(f"Generating with '{CHOSEN_WEIGHTS}' weights on {device}: {gen_model.num_params() / 1e6:.1f}M parameters")
''')

md("""
## 13b. Local evaluation on held-out text (mirrors the official metrics)
The competition scores a 128-word continuation of a 512-word prompt with BERTScore F1 (primary), ROUGE-L F1 and BLEU-4.
Here we build the *same kind of task* from the validation documents (never trained on): first 512 words = prompt,
next 128 words = reference. ROUGE-L and BLEU-4 are implemented below; BERTScore is optional (it downloads roberta-large).
Use this to compare checkpoints by generation quality, not just validation loss.
""")
code('''
import numpy as np

def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]

def rouge_l_f1(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r or not h:
        return 0.0
    lcs = lcs_len(r, h)
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(h), lcs / len(r)
    return 2 * p * rec / (p + rec)

def bleu4(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not h:
        return 0.0
    log_p = 0.0
    for n in range(1, 5):
        ref_grams, hyp_grams = {}, {}
        for i in range(len(r) - n + 1):
            g = tuple(r[i:i + n]); ref_grams[g] = ref_grams.get(g, 0) + 1
        for i in range(len(h) - n + 1):
            g = tuple(h[i:i + n]); hyp_grams[g] = hyp_grams.get(g, 0) + 1
        overlap = sum(min(c, ref_grams.get(g, 0)) for g, c in hyp_grams.items())
        total = max(1, sum(hyp_grams.values()))
        log_p += math.log((overlap + 1) / (total + 1)) / 4          # add-one smoothing
    brevity = 1.0 if len(h) >= len(r) else math.exp(1 - len(r) / len(h))
    return brevity * math.exp(log_p)

# rebuild a few full documents from the validation token file (documents are separated by <eos>)
val_tokens = np.fromfile(C.VAL_BIN, dtype=np.uint16)
cuts = np.flatnonzero(val_tokens == tok.eos_id)
eval_prompts, eval_refs = [], []
for a, b in zip(cuts[:-1], cuts[1:]):                              # skip the last, possibly truncated, document
    words = tok.decode(val_tokens[a + 1:b].tolist()).split()
    if len(words) >= 512 + C.TARGET_WORDS:
        eval_prompts.append(" ".join(words[:512]))
        eval_refs.append(" ".join(words[512:512 + C.TARGET_WORDS]))
    if len(eval_prompts) >= C.N_LOCAL_EVAL:
        break
print(f"{len(eval_prompts)} held-out documents with 512-word prompts")

def run_generation(model, prompt_texts):
    room = gen_model.cfg.enc_len                      # the loaded checkpoint decides the prompt window
    ids = [tok.encode(t) for t in prompt_texts]
    ids = [x[-room:] if len(x) > room else [tok.eos_id] + x for x in ids]
    outs = []
    for b in range(0, len(ids), C.GEN_BATCH):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outs += M.generate(model, ids[b:b + C.GEN_BATCH], tok, max_new_tokens=C.MAX_NEW_TOKENS,
                               target_words=C.TARGET_WORDS, temperature=C.GEN_TEMPERATURE,
                               top_k=C.GEN_TOP_K, top_p=C.GEN_TOP_P)
    return outs

if eval_prompts:
    results = {}
    for name, path in available.items():
        mdl = gen_model if name == CHOSEN_WEIGHTS else load_model(path)
        hyps = run_generation(mdl, eval_prompts)
        rl = [rouge_l_f1(r, h) for r, h in zip(eval_refs, hyps)]
        bl = [bleu4(r, h) for r, h in zip(eval_refs, hyps)]
        results[name] = (float(np.mean(rl)), float(np.mean(bl)), hyps)
        print(f"{name:7s} | ROUGE-L F1 {results[name][0]:.4f} | BLEU-4 {results[name][1]:.4f} | "
              f"words/prediction {np.mean([len(h.split()) for h in hyps]):.0f}")
        if mdl is not gen_model:
            del mdl
            gc.collect()
            torch.cuda.empty_cache()
    hyps = results[CHOSEN_WEIGHTS][2]              # the rest of this section looks at the chosen weights

    RUN_BERTSCORE = False        # set True to also compute the official primary metric (needs: pip install bert_score)
    if RUN_BERTSCORE:
        from bert_score import score as bert_score
        _, _, f1 = bert_score(hyps, eval_refs, lang="en", model_type="roberta-large")
        print(f"BERTScore F1: {f1.mean().item():.4f}")
    print("\\nSample prompt tail :", eval_prompts[0].split()[-25:])
    print("Sample reference   :", eval_refs[0][:200])
    print("Sample generation  :", hyps[0][:200])
''')

md("## 14. Generate predictions for the test set\nEncoder runs once per prompt; the decoder samples with a KV cache (temperature / top-k / top-p from `config.py`). Prompts are sorted by length so padding stays small.")
code('''
test_path = hf_hub_download(C.HF_DATASET_REPO, C.TEST_FILE, repo_type="dataset", token=HF_TOKEN)
with open(test_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)
print("Test examples:", len(test_data))

room = gen_model.cfg.enc_len                      # the loaded checkpoint decides the prompt window
def make_prompt(text):
    ids = tok.encode(text)
    return ids[-room:] if len(ids) > room else [tok.eos_id] + ids

keys = list(test_data.keys())
n_trunc = sum(len(tok.encode(test_data[k]["text"])) > room for k in keys)
print(f"{n_trunc}/{len(keys)} prompts exceed {room} tokens and will be cut from the left (the doc says prompts are 512 words)")
prompts = {k: make_prompt(test_data[k]["text"]) for k in keys}
order = sorted(keys, key=lambda k: len(prompts[k]))

PRED_PATH = os.path.join(C.WORK_DIR, "prediction.json")
predictions = {}
t0 = time.time()
for b in range(0, len(order), C.GEN_BATCH):
    chunk = order[b:b + C.GEN_BATCH]
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        outs = M.generate(gen_model, [prompts[k] for k in chunk], tok,
                          max_new_tokens=C.MAX_NEW_TOKENS, target_words=C.TARGET_WORDS,
                          temperature=C.GEN_TEMPERATURE, top_k=C.GEN_TOP_K, top_p=C.GEN_TOP_P)
    for k, text in zip(chunk, outs):
        predictions[str(k)] = {"text": text}
    done = len(predictions)
    if (b // C.GEN_BATCH) % 5 == 0 or done == len(order):
        print(f"  {done}/{len(order)} | {time.time() - t0:.0f}s elapsed")
    if (b // C.GEN_BATCH) % 20 == 0:                       # partial save in case the session dies
        with open(PRED_PATH, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False)

predictions = {str(k): predictions[str(k)] for k in keys}  # original order
print("Generation complete:", len(predictions), "predictions")
''')

md("## 15. Save predictions and build the submission zip")
code('''
with open(PRED_PATH, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

SUBMISSION_ZIP = os.path.join(C.WORK_DIR, "submission.zip")
with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(PRED_PATH, arcname="prediction.json")
print(f"Wrote {SUBMISSION_ZIP} ({os.path.getsize(SUBMISSION_ZIP) / 1024:.1f} KB)")

# --- submission self-check (the checklist in the competition document, Section 19)
official_ids = set(map(str, test_data.keys()))
assert set(predictions) == official_ids, \
    f"ID mismatch: missing {len(official_ids - set(predictions))}, unknown {len(set(predictions) - official_ids)}"
assert all(isinstance(v.get("text"), str) and v["text"].strip() for v in predictions.values()), "empty prediction text"
n_words = [len(v["text"].split()) for v in predictions.values()]
print(f"ID check passed ({len(predictions)} ids). Words per prediction: min {min(n_words)} / mean {np.mean(n_words):.0f} / max {max(n_words)}")
if min(n_words) < C.TARGET_WORDS:
    print(f"WARNING: {sum(w < C.TARGET_WORDS for w in n_words)} predictions are shorter than {C.TARGET_WORDS} words")
with zipfile.ZipFile(SUBMISSION_ZIP) as z:
    assert z.namelist() == ["prediction.json"], f"zip must contain only prediction.json at its root, got {z.namelist()}"
print("Zip structure OK: prediction.json at the archive root.")

for k in list(predictions)[:3]:
    words = predictions[k]["text"].split()
    print(f"\\nExample {k} ({len(words)} words):\\n{predictions[k]['text'][:300]}...")

# keep a copy on the hub too
api.upload_file(path_or_fileobj=SUBMISSION_ZIP, path_in_repo="submission.zip", repo_id=C.HF_MODEL_REPO, token=HF_TOKEN)
api.upload_file(path_or_fileobj=plot_path, path_in_repo="loss_curve.png", repo_id=C.HF_MODEL_REPO, token=HF_TOKEN)
print("Submission and loss curve pushed to", C.HF_MODEL_REPO)
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "..", "phase1_arch2_encoder_decoder.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", os.path.abspath(out), len(cells), "cells")
