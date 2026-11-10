"""
Single place for every knob in the arch-3 pipeline.
All scripts (tokenizer, data prep, model, trainer) import this file.
"""
import os
import json

# ---------------------------------------------------------------- hub / data
HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
HF_MODEL_REPO = "D-240357/legal-arch4-moe-v2"    # NEW repo for the longer run, so nothing from the 100M-token run can be restored by mistake
HF_DATA_REPO = "D-240357/legal-arch3-decoder"    # arch 3 already built the tokenizer + token files: reuse them (same data, fair comparison)
HF_PRIVATE = True

TRAIN_FILES = [
    "Phase_1_Pretraining/Train/train_acts.json",
    "Phase_1_Pretraining/Train/train_high_courts.json",
    "Phase_1_Pretraining/Train/train_supreme_court.json",
    "Phase_1_Pretraining/Train/train_tribunals.json",
]
TEST_FILE = "Phase_1_Pretraining/Test/test.json"

# Relative share of the training text taken from each file above (same order). The corpus is
# interleaved by these weights, so 600M tokens cover all four sources instead of just the first
# files in the list. Weights are by characters; a source that runs out is simply skipped.
SOURCE_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

# ------------------------------------------------------------------- folders
WORK_DIR = "/kaggle/working/arch4"
DATA_DIR = os.path.join(WORK_DIR, "data")
CKPT_DIR = os.path.join(WORK_DIR, "ckpt")
CODE_DIR = WORK_DIR

TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")
META_PATH = os.path.join(DATA_DIR, "meta.json")
STATE_PATH = os.path.join(CKPT_DIR, "training_state.pt")
FINAL_PATH = os.path.join(CKPT_DIR, "model_final.pt")
LOG_PATH = os.path.join(CKPT_DIR, "train_log.jsonl")
BEST_PATH = os.path.join(CKPT_DIR, "model_best.pt")     # weights with the lowest validation loss so far

# ----------------------------------------------------------------- tokenizer
VOCAB_SIZE = 32000            # 256 bytes + merges + 2 specials (32000 = 500 * 64, tensor-core friendly)
TOK_SAMPLE_MB_PER_FILE = 40   # text used to learn the merges, taken from each train file
TOK_SAMPLE_STRIDE = 5         # take every 5th doc so the sample spans the whole file

# ---------------------------------------------------------------------- data
# Total tokens to train on (from scratch). The prepared file holds 600M tokens, so anything up to that is a single pass
# with no repeated data. At ~17.9K tok/s on 2xT4: 100M ~ 1.6 h, 400M ~ 6.2 h, 600M ~ 9.3 h (inside the 10.5 h crash guard).
TRAIN_TOKENS = 600_000_000
VAL_TOKENS = 3_000_000
VAL_EVERY_NTH_DOC = 100       # every 100th document is held out for validation
MIN_DOC_CHARS = 100
PREP_WORKERS = max(1, (os.cpu_count() or 2))

# --------------------------------------------------------------------- model
N_LAYER = 12
N_HEAD = 12
D_MODEL = 768
MAX_SEQ_LEN = 2048
ROPE_BASE = 10000.0
NORM_EPS = 1e-5

# Multi-head Latent Attention: per head q/k = 48 (content) + 16 (rotary) dims, v = 64
Q_LORA = 384                  # query bottleneck
KV_LORA = 192                 # KV latent size -> cache holds 192 + 16 numbers per token per layer
QK_NOPE = 48
QK_ROPE = 16
V_DIM = 64

# DeepSeekMoE
N_DENSE_LAYERS = 1            # first layer stays a plain SwiGLU FFN (as in DeepSeek)
D_FF_DENSE = 2048
N_ROUTED = 32                 # small routed experts per MoE layer ...
TOP_K = 4                     # ... of which each token uses 4 ...
EXPERT_FF = 192
SHARED_FF = 512               # ... plus one shared expert every token passes through
CAPACITY_FACTOR = 1.25        # training-time buffer size per expert (eval/generation is drop-free)

# load balancing (aux-loss-free, DeepSeek-V3 style)
BIAS_UPDATE_SPEED = 0.01      # how fast expert biases chase the mean load
BIAS_FREEZE_FRAC = 0.9        # freeze biases for the last 10% of training
SEQ_AUX_ALPHA = 1e-4          # tiny sequence-wise balance loss as a safety net

# multi-token prediction
USE_MTP = True
MTP_FF = 1024
MTP_WEIGHT_EARLY = 0.3
MTP_WEIGHT_LATE = 0.1
MTP_SWITCH_FRAC = 0.67        # fraction of training after which the weight drops to MTP_WEIGHT_LATE

# ------------------------------------------------------------------ training
SEED = 42
MICRO_BATCH = 2               # sequences per GPU per forward pass. 4 ran out of memory on a 15 GB T4 (two fp32 logits tensors + expert buffers)
TOKENS_PER_STEP = 65_536      # global batch; grad-accum is derived from this and the GPU count
PEAK_LR = 4e-4
MIN_LR_RATIO = 0.1
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
MAX_GRAD_NORM = 1.0
Z_LOSS = 1e-4                 # tiny penalty on logsumexp^2, keeps fp16 logits tame

COMPILE = True
ENFORCE_PARAM_BUDGET = True   # capstone rule: 50M-300M parameters
LOG_EVERY = 20
EVAL_EVERY = 250
EVAL_SEQS = 64                # validation sequences per eval (split across GPUs)
CKPT_EVERY = 250              # steps between local save + HF push
SQUASH_EVERY_N_PUSHES = 8     # collapse HF git history now and then so old 1.3 GB copies get freed
TIME_LIMIT_HOURS = 10.5       # counted from trainer start; saves + exits cleanly before Kaggle's 12h wall, just re-run to continue
NCCL_P2P_OFF = True           # P2P between the two T4s hangs on some Kaggle images; comm cost is tiny here

# --------------------------------------------------------------- generation
TARGET_WORDS = 128
MAX_NEW_TOKENS = 384
GEN_TEMPERATURE = 0.7
GEN_TOP_K = 50
GEN_TOP_P = 0.92
N_LOCAL_EVAL = 48              # held-out validation docs used for the local generation check
GEN_BATCH = 8                 # smaller than arch 3: drop-free MoE prefill buffers are big

# Handy for smoke tests: ARCH3_CFG_OVERRIDE='{"N_LAYER": 2, ...}' patches any value above.
_override = os.environ.get("ARCH3_CFG_OVERRIDE")
if _override:
    globals().update(json.loads(_override))
