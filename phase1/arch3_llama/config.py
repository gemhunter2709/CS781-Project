"""
Single place for every knob in the arch-3 pipeline.
All scripts (tokenizer, data prep, model, trainer) import this file.
"""
import os
import json

# ---------------------------------------------------------------- hub / data
HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
HF_MODEL_REPO = "D-240357/legal-arch3-decoder"   # everything (data bins, ckpts, final model) is mirrored here
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
WORK_DIR = "/kaggle/working/arch3"
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
TRAIN_TOKENS = 600_000_000
VAL_TOKENS = 3_000_000
VAL_EVERY_NTH_DOC = 100       # every 100th document is held out for validation
MIN_DOC_CHARS = 100
PREP_WORKERS = max(1, (os.cpu_count() or 2))

# --------------------------------------------------------------------- model
N_LAYER = 12
N_HEAD = 12
N_KV_HEAD = 4                 # grouped-query attention: 3 query heads share each KV head
D_MODEL = 768
D_FF = 2048                   # SwiGLU hidden size (~8/3 * d_model, rounded to a multiple of 64)
MAX_SEQ_LEN = 2048
ROPE_BASE = 10000.0
NORM_EPS = 1e-5

# ------------------------------------------------------------------ training
SEED = 42
MICRO_BATCH = 4               # sequences per GPU per forward pass (raise if memory is < 10 GB)
TOKENS_PER_STEP = 65_536      # global batch; grad-accum is derived from this and the GPU count
PEAK_LR = 6e-4
MIN_LR_RATIO = 0.1
WARMUP_STEPS = 300
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
MAX_GRAD_NORM = 1.0
Z_LOSS = 1e-4                 # tiny penalty on logsumexp^2, keeps fp16 logits tame

COMPILE = True
ENFORCE_PARAM_BUDGET = True   # capstone rule: 50M-300M parameters
LOG_EVERY = 20
EVAL_EVERY = 500
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
GEN_BATCH = 16

# Handy for smoke tests: ARCH3_CFG_OVERRIDE='{"N_LAYER": 2, ...}' patches any value above.
_override = os.environ.get("ARCH3_CFG_OVERRIDE")
if _override:
    globals().update(json.loads(_override))
