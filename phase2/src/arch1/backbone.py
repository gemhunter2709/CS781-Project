"""Loads the Phase-1 arch 1 weights and exposes them as a token encoder for the NER tagger."""
import json
import os

import torch
import torch.nn as nn

import config as C
from hub_files import locate, fetch_from_hub  # noqa: F401  (fetch_from_hub is used by run_ner)
from phase1_model import GPT2LM
from tokenizer import NerTokenizer


def _find(names, repos=None):
    return locate(names, repos or C.PHASE1_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)


def load_tokenizer():
    path, _ = _find(C.TOKENIZER_FILES, C.TOKENIZER_REPOS)
    tok = NerTokenizer()
    tok.load(path)
    tok.split_digits = C.SPLIT_DIGITS
    return tok


class Backbone(nn.Module):
    def __init__(self, gpt, keep_layers, bidirectional):
        super().__init__()
        self.d_model = gpt.token_emb.embedding_dim
        self.tok_emb = gpt.token_emb
        self.pos_emb = gpt.pos_emb
        self.emb_drop = gpt.emb_drop
        blocks = gpt.layers if keep_layers is None else [gpt.layers[i] for i in keep_layers]
        self.layers = nn.ModuleList(blocks)
        self.final_norm = gpt.ln_f
        self.bidirectional = bidirectional
        for blk in self.layers:
            blk.attn.causal = not bidirectional

    def forward(self, ids, mask):
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        h = self.emb_drop(self.tok_emb(ids) + self.pos_emb(pos))
        key_mask = mask[:, None, None, :] if (self.bidirectional and not bool(mask.all())) else None
        for blk in self.layers:
            h = blk(h, key_mask)
        return self.final_norm(h)


def build_backbone(keep_layers=None, bidirectional=True):
    cfg_path, _ = _find(C.CONFIG_FILES)
    weights_path, source = _find(C.WEIGHT_FILES)
    with open(cfg_path) as f:
        cfg = json.load(f)
    gpt = GPT2LM(cfg["vocab_size"], cfg["d_model"], cfg["n_head"], cfg["n_layer"], cfg["d_ff"],
                 cfg["dropout"], cfg["max_len"])
    state = torch.load(weights_path, map_location="cpu")
    state = {k.replace("_orig_mod.", ""): v.float() for k, v in state.items()}
    gpt.load_state_dict(state)
    if os.environ.get("P2_VERBOSE", "1") == "1":
        print(f"Phase-1 arch 1 weights: {weights_path} ({source})", flush=True)
    return Backbone(gpt, keep_layers, bidirectional)


def student_layers():
    return list(C.STUDENT_LAYERS)
