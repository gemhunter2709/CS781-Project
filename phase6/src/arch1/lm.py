"""Arch 1 (GPT-2 style decoder, word-level vocabulary) as a summarizer.
Phase 1 never needed a KV cache, so the attention step below re-uses the block's own projections with one; adapters
wrapped around those projections therefore apply during generation too."""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from hub_files import locate, fetch_from_hub  # noqa: F401  (fetch_from_hub is used by run_sum)
from phase1_model import GPT2LM
from tokenizer import NerTokenizer
from gen_core import DecoderLM


def load_tokenizer():
    path, _ = locate(C.TOKENIZER_FILES, C.TOKENIZER_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)
    tok = NerTokenizer()
    tok.load(path)
    tok.sep_id = tok.eos_id
    return tok


def attention(attn, x, past, key_mask, training):
    B, T, D = x.shape
    qkv = attn.W_qkv(x).reshape(B, T, 3, attn.n_head, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if past is not None:
        k = torch.cat([past[0], k], dim=2)
        v = torch.cat([past[1], v], dim=2)
    p = attn.dropout_p if training else 0.0
    if key_mask is None:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=p)
    out = out.permute(0, 2, 1, 3).reshape(B, T, D)
    return attn.resid_drop(attn.W_o(out)), (k, v)


class Arch1LM(DecoderLM):
    def __init__(self, gpt, keep_layers):
        super().__init__()
        self.tok_emb = gpt.token_emb
        self.pos_emb = gpt.pos_emb
        self.emb_drop = gpt.emb_drop
        self.layers = nn.ModuleList(gpt.layers if keep_layers is None else [gpt.layers[i] for i in keep_layers])
        self.final_norm = gpt.ln_f
        self.max_len = gpt.pos_emb.num_embeddings

    def add_positions(self, x, pos):
        return self.emb_drop(x + self.pos_emb(pos))      # learned absolute positions, per row

    def run_blocks(self, x, pos, caches, key_mask):
        new = []
        for i, blk in enumerate(self.layers):
            a, present = attention(blk.attn, blk.ln1(x), None if caches is None else caches[i], key_mask, self.training)
            x = x + a
            x = x + blk.ffn(blk.ln2(x))
            new.append(present)
        return x, new


def build_lm(keep_layers=None):
    cfg_path, _ = locate(C.CONFIG_FILES, C.PHASE1_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)
    path, source = locate(C.WEIGHT_FILES, C.PHASE1_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)
    with open(cfg_path) as f:
        cfg = json.load(f)
    gpt = GPT2LM(cfg["vocab_size"], cfg["d_model"], cfg["n_head"], cfg["n_layer"], cfg["d_ff"], cfg["dropout"], cfg["max_len"])
    gpt.load_state_dict({k.replace("_orig_mod.", ""): v.float() for k, v in torch.load(path, map_location="cpu").items()})
    if os.environ.get("P2_VERBOSE", "1") == "1":
        print(f"Phase-1 arch 1 weights: {path} ({source})", flush=True)
    return Arch1LM(gpt, keep_layers)


def student_layers():
    return list(C.STUDENT_LAYERS)
