"""Loads the Phase-1 arch 3 weights (Llama-style decoder) and exposes them as a token encoder for NER."""
import os

import torch
import torch.nn as nn

import config as C
from hub_files import locate, fetch_from_hub  # noqa: F401  (fetch_from_hub is used by run_ner)
from phase1_model import GPT, GPTConfig
from tokenizer import NerTokenizer


def load_tokenizer():
    path, _ = locate(C.TOKENIZER_FILES, C.TOKENIZER_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)
    tok = NerTokenizer.load(path)
    tok.split_punct = C.SPLIT_PUNCT
    return tok


class Backbone(nn.Module):
    def __init__(self, gpt, keep_layers, bidirectional):
        super().__init__()
        self.d_model = gpt.cfg.d_model
        self.tok_emb = gpt.tok_emb
        blocks = gpt.blocks if keep_layers is None else [gpt.blocks[i] for i in keep_layers]
        self.layers = nn.ModuleList(blocks)
        self.final_norm = gpt.final_norm
        self.register_buffer("rope_cos", gpt.rope_cos.clone(), persistent=False)
        self.register_buffer("rope_sin", gpt.rope_sin.clone(), persistent=False)
        self.bidirectional = bidirectional
        for blk in self.layers:
            blk.attn.causal = not bidirectional

    def forward(self, ids, mask):
        T = ids.size(1)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        key_mask = mask[:, None, None, :] if (self.bidirectional and not bool(mask.all())) else None
        x = self.tok_emb(ids)
        for blk in self.layers:
            x, _ = blk(x, cos, sin, None, key_mask)
        return self.final_norm(x)


def build_backbone(keep_layers=None, bidirectional=True):
    path, source = locate(C.WEIGHT_FILES, C.PHASE1_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)
    blob = torch.load(path, map_location="cpu")
    gpt = GPT(GPTConfig(**blob["model_cfg"]))
    gpt.load_state_dict({k: v.float() for k, v in blob["model"].items()})
    if os.environ.get("P2_VERBOSE", "1") == "1":
        print(f"Phase-1 arch 3 weights: {path} ({source})", flush=True)
    return Backbone(gpt, keep_layers, bidirectional)


def student_layers():
    return list(C.STUDENT_LAYERS)
