"""Arch 4 (DeepSeek-style MoE decoder) as a summarizer: reads [source] <eos> [summary] and writes the summary."""
import os

import torch
import torch.nn as nn

import config as C
from hub_files import locate, fetch_from_hub  # noqa: F401  (fetch_from_hub is used by run_sum)
from phase1_model import MoEGPT, ModelConfig
from tokenizer import NerTokenizer
from gen_core import DecoderLM


def load_tokenizer():
    path, _ = locate(C.TOKENIZER_FILES, C.TOKENIZER_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)
    tok = NerTokenizer.load(path)
    tok.sep_id = tok.eos_id
    return tok


class Arch4LM(DecoderLM):
    def __init__(self, gpt, keep_layers):
        super().__init__()
        self.cfg = gpt.cfg
        self.tok_emb = gpt.tok_emb
        self.layers = nn.ModuleList(gpt.blocks if keep_layers is None else [gpt.blocks[i] for i in keep_layers])
        self.final_norm = gpt.final_norm
        self.register_buffer("rope_cos", gpt.rope_cos.clone(), persistent=False)
        self.register_buffer("rope_sin", gpt.rope_sin.clone(), persistent=False)
        self.max_len = gpt.cfg.max_seq_len
        for blk in self.layers:
            if blk.use_moe:
                blk.ffn.dropless = True          # every token is routed while fine-tuning; routing biases stay frozen

    def add_positions(self, x, pos):
        return x

    def run_blocks(self, x, pos, caches, key_mask):
        cos, sin = self.rope_cos[pos][:, None], self.rope_sin[pos][:, None]
        new = []
        for i, blk in enumerate(self.layers):
            x, present, _ = blk(x, cos, sin, None if caches is None else caches[i], key_mask)
            new.append(present)
        return x, new


def build_lm(keep_layers=None):
    path, source = locate(C.WEIGHT_FILES, C.PHASE1_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)
    blob = torch.load(path, map_location="cpu")
    cfg = dict(blob["model_cfg"], mtp=False)             # the multi-token head is a pre-training-only extra
    gpt = MoEGPT(ModelConfig(**cfg))
    gpt.load_state_dict({k: v.float() for k, v in blob["model"].items() if not k.startswith("mtp_head.")})
    if os.environ.get("P2_VERBOSE", "1") == "1":
        print(f"Phase-1 arch 4 weights: {path} ({source})", flush=True)
    return Arch4LM(gpt, keep_layers)


def student_layers():
    return list(C.STUDENT_LAYERS)
