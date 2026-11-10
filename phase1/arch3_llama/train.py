"""
Distributed trainer for arch 3. Launched with torchrun, one process per GPU:

    torchrun --standalone --nproc_per_node=2 train.py

Design notes
  * Data: memory-mapped uint16 token file. Block order is a seeded permutation, so the batch for
    step N is a pure function of N -> resuming costs nothing (no fast-forwarding the stream).
  * LR is computed from the step number (no scheduler object whose state could drift).
  * Every CKPT_EVERY steps rank 0 saves the full training state and pushes it to Hugging Face
    from a background thread, so a Kaggle crash/timeout loses at most a few minutes.
  * When the wall-clock limit is near, all ranks agree to save, push and exit cleanly.
"""
import os
import sys
import json
import math
import time
import queue
import threading

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from model import GPT, GPTConfig

# If Inductor ever fails to compile some part of the graph, run that part eagerly instead of crashing the whole job.
torch._dynamo.config.suppress_errors = True

T_START = time.time()


# ------------------------------------------------------------------ dist setup
rank = int(os.environ.get("RANK", 0))
world = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
use_cuda = torch.cuda.is_available()
distributed = world > 1

if distributed:
    dist.init_process_group("nccl" if use_cuda else "gloo")
if use_cuda:
    torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
is_main = rank == 0

torch.manual_seed(C.SEED)
if use_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def say(*a):
    if is_main:
        print(*a, flush=True)


# ------------------------------------------------------------------------ data
class TokenFile:
    """Random-access blocks of (x, y) from a flat uint16 token file."""

    def __init__(self, path, block, seed=None):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block = block
        self.n_blocks = (len(self.data) - 1) // block
        self.order = np.random.default_rng(seed).permutation(self.n_blocks) if seed is not None else None

    def get(self, seq_ids):
        B = self.block
        rows = []
        for s in seq_ids:
            b = int(s) % self.n_blocks
            if self.order is not None:
                b = int(self.order[b])
            rows.append(self.data[b * B: b * B + B + 1])
        arr = np.stack(rows).astype(np.int64)
        return torch.from_numpy(arr[:, :-1]), torch.from_numpy(arr[:, 1:])


class Prefetcher:
    """Background thread that keeps a few ready-to-go micro-batches in a queue."""

    def __init__(self, stream, first_seq, micro, step_stride, start_step, total_steps, accum):
        self.q = queue.Queue(maxsize=6)
        self.args = (stream, first_seq, micro, step_stride, start_step, total_steps, accum)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        stream, first_seq, micro, stride, start, total, accum = self.args
        for step in range(start, total):
            for j in range(accum):
                base = step * stride + (j * world + rank) * micro
                x, y = stream.get(range(base, base + micro))
                if use_cuda:
                    x, y = x.pin_memory(), y.pin_memory()
                self.q.put((x, y))
        self.q.put(None)

    def next(self):
        item = self.q.get()
        x, y = item
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# --------------------------------------------------------------------- hub sync
class HubSync:
    """Uploads files in a background thread; never blocks the training loop for long."""

    def __init__(self):
        self.thread = None
        self.pushes = 0
        self.api = None
        if is_main and os.environ.get("HF_TOKEN"):
            from huggingface_hub import HfApi
            self.api = HfApi()
            self.api.create_repo(C.HF_MODEL_REPO, token=os.environ["HF_TOKEN"], private=C.HF_PRIVATE, exist_ok=True)

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def push(self, files, message, block=False):
        if self.api is None:
            return
        self.wait()                        # the previous upload must be done before files get rewritten

        def work():
            try:
                for local, remote in files:
                    self.api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=C.HF_MODEL_REPO,
                                         token=os.environ["HF_TOKEN"], commit_message=message)
                self.pushes += 1
                print(f"  -> pushed to HF: {message}", flush=True)
                if self.pushes % C.SQUASH_EVERY_N_PUSHES == 0:
                    self.api.super_squash_history(repo_id=C.HF_MODEL_REPO, token=os.environ["HF_TOKEN"])
            except Exception as e:                                   # a failed push must not kill training
                print(f"  -> HF push failed ({type(e).__name__}: {e}); will retry at the next checkpoint", flush=True)

        self.thread = threading.Thread(target=work)
        self.thread.start()
        if block:
            self.wait()


# ----------------------------------------------------------------------- helpers
def lr_at(step, total):
    if step < C.WARMUP_STEPS:
        return C.PEAK_LR * (step + 1) / C.WARMUP_STEPS
    progress = (step - C.WARMUP_STEPS) / max(1, total - C.WARMUP_STEPS)
    floor = C.PEAK_LR * C.MIN_LR_RATIO
    return floor + 0.5 * (C.PEAK_LR - floor) * (1 + math.cos(math.pi * min(1.0, progress)))


def make_optimizer(model):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": C.WEIGHT_DECAY},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=C.PEAK_LR, betas=tuple(C.BETAS), eps=1e-8, fused=use_cuda)


def save_training_state(raw_model, optimizer, scaler, step, tokens_seen, best_val):
    os.makedirs(C.CKPT_DIR, exist_ok=True)
    tmp = C.STATE_PATH + ".tmp"
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val": best_val,
        "model_cfg": raw_model.cfg.__dict__,
    }, tmp)
    os.replace(tmp, C.STATE_PATH)           # atomic: a crash never leaves a half-written checkpoint


@torch.no_grad()
def evaluate(model, val_stream, micro):
    model.eval()
    per_rank = max(1, C.EVAL_SEQS // (micro * world))
    total = torch.zeros(2, device=device)
    for i in range(per_rank):
        base = (i * world + rank) * micro
        x, y = val_stream.get(range(base, base + micro))
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
            ce, _ = model(x, y)
        total[0] += ce.detach()
        total[1] += 1
    if distributed:
        dist.all_reduce(total)
    model.train()
    return (total[0] / total[1]).item()


# ------------------------------------------------------------------------- main
def main():
    with open(C.META_PATH) as f:
        meta = json.load(f)

    seqs_per_step = C.TOKENS_PER_STEP // C.MAX_SEQ_LEN
    accum = seqs_per_step // (C.MICRO_BATCH * world)
    assert accum >= 1 and accum * C.MICRO_BATCH * world == seqs_per_step, \
        f"TOKENS_PER_STEP must split evenly over {world} GPUs x MICRO_BATCH {C.MICRO_BATCH}"
    total_steps = C.TRAIN_TOKENS // C.TOKENS_PER_STEP

    cfg = GPTConfig(vocab_size=C.VOCAB_SIZE, n_layer=C.N_LAYER, n_head=C.N_HEAD, n_kv_head=C.N_KV_HEAD,
                    d_model=C.D_MODEL, d_ff=C.D_FF, max_seq_len=C.MAX_SEQ_LEN, rope_base=C.ROPE_BASE,
                    norm_eps=C.NORM_EPS, z_loss=C.Z_LOSS)
    raw_model = GPT(cfg).to(device)
    n_params = raw_model.num_params()
    assert not C.ENFORCE_PARAM_BUDGET or 50e6 <= n_params <= 300e6, f"{n_params / 1e6:.1f}M params is outside the 50-300M budget"

    optimizer = make_optimizer(raw_model)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    step, tokens_seen, best_val = 0, 0, float("inf")
    if os.path.exists(C.STATE_PATH):
        ckpt = torch.load(C.STATE_PATH, map_location="cpu")
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step, tokens_seen, best_val = ckpt["step"], ckpt["tokens_seen"], ckpt["best_val"]
        del ckpt
        say(f"Resumed from checkpoint: step {step} / {total_steps}  ({tokens_seen / 1e6:.0f}M tokens)")
    else:
        say("No checkpoint found - starting from scratch.")

    say(f"GPUs: {world} | params: {n_params / 1e6:.1f}M | tokens/step: {C.TOKENS_PER_STEP:,} "
        f"| micro-batch {C.MICRO_BATCH} x accum {accum} per GPU | total steps: {total_steps} "
        f"| remaining: {max(0, total_steps - step)}")
    if meta["train_tokens"] < C.TRAIN_TOKENS:
        say(f"Note: only {meta['train_tokens']:,} tokens on disk, the data will be revisited (multiple epochs).")

    model = torch.compile(raw_model) if (C.COMPILE and use_cuda) else raw_model
    if distributed:
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)
    model.train()

    train_stream = TokenFile(C.TRAIN_BIN, C.MAX_SEQ_LEN, seed=C.SEED)
    val_stream = TokenFile(C.VAL_BIN, C.MAX_SEQ_LEN)
    feeder = Prefetcher(train_stream, 0, C.MICRO_BATCH, seqs_per_step, step, total_steps, accum)
    hub = HubSync()

    def sync_ctx(is_last_micro):
        # skip the gradient all-reduce on all but the last micro-batch of a step
        if distributed and not is_last_micro:
            return model.no_sync()
        return _null()

    log_file = open(C.LOG_PATH, "a") if is_main else None
    t_log = time.time()
    tok_since_log = 0
    loss_acc = torch.zeros((), device=device)
    steps_in_acc = 0
    grad_norm = torch.zeros((), device=device)
    time_limit_s = C.TIME_LIMIT_HOURS * 3600
    stop = False

    while step < total_steps:
        lr = lr_at(step, total_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        for j in range(accum):
            x, y = feeder.next()
            with sync_ctx(j == accum - 1):
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                    ce, z = model(x, y)
                    loss = (ce + C.Z_LOSS * z) / accum
                scaler.scale(loss).backward()
            loss_acc += ce.detach() / accum

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), C.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        steps_in_acc += 1
        tokens_seen += C.TOKENS_PER_STEP
        tok_since_log += C.TOKENS_PER_STEP

        if step % C.LOG_EVERY == 0 or step == total_steps:
            avg = (loss_acc / steps_in_acc).item()
            loss_acc.zero_()
            steps_in_acc = 0
            dt = time.time() - t_log
            speed = tok_since_log / dt
            eta_h = (total_steps - step) * C.TOKENS_PER_STEP / max(speed, 1) / 3600
            mem = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
            say(f"step {step:5d}/{total_steps} | loss {avg:.4f} | lr {lr:.2e} | gnorm {float(grad_norm):.2f} "
                f"| {speed / 1e3:.1f}K tok/s | mem {mem:.1f}GB | eta {eta_h:.2f}h")
            if is_main:
                log_file.write(json.dumps({"step": step, "loss": avg, "lr": lr, "tok_s": speed}) + "\n")
                log_file.flush()
            t_log, tok_since_log = time.time(), 0

        if step % C.EVAL_EVERY == 0 or step == total_steps:
            val = evaluate(model, val_stream, C.MICRO_BATCH)
            improved = val < best_val
            best_val = min(best_val, val)
            say(f"  [eval] step {step} | val loss {val:.4f} | val ppl {math.exp(min(val, 20)):.2f}")
            if is_main:
                log_file.write(json.dumps({"step": step, "val_loss": val}) + "\n")
                log_file.flush()
                if improved:                          # keep the best-validation weights, not just the latest
                    torch.save({"model": {k: v.half() for k, v in raw_model.state_dict().items()},
                                "model_cfg": cfg.__dict__, "step": step, "val_loss": val}, C.BEST_PATH)
                    hub.push([(C.BEST_PATH, "model_best.pt")], f"best val {val:.4f} at step {step}")

        # wall-clock guard: every rank must take the same decision, so rank 0 decides and broadcasts
        if step % 10 == 0:
            flag = torch.tensor(1.0 if (time.time() - T_START) > time_limit_s else 0.0, device=device)
            if distributed:
                dist.broadcast(flag, src=0)
            stop = bool(flag.item())

        if step % C.CKPT_EVERY == 0 or step == total_steps or stop:
            if is_main:
                hub.wait()
                save_training_state(raw_model, optimizer, scaler, step, tokens_seen, best_val)
                hub.push([(C.STATE_PATH, "training_state.pt"), (C.LOG_PATH, "train_log.jsonl")],
                         f"checkpoint step {step}", block=(step == total_steps or stop))
            if distributed:
                dist.barrier()
        if stop:
            say(f"Time limit reached at step {step}. State saved and pushed - re-run the training cell to continue.")
            break

    if step >= total_steps and is_main:
        torch.save({"model": {k: v.half() for k, v in raw_model.state_dict().items()},
                    "model_cfg": cfg.__dict__}, C.FINAL_PATH)
        hub.push([(C.FINAL_PATH, "model_final.pt"), (C.TOKENIZER_PATH, "tokenizer.json")],
                 "final model", block=True)
        say(f"Training complete in this session: {(time.time() - T_START) / 3600:.2f} h. Final weights pushed.")
    hub.wait()
    if distributed:
        dist.destroy_process_group()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
