"""Loads the Phase-1 arch 4 weights (DeepSeek-style MoE decoder) and exposes them as a sentence encoder for rhetorical roles."""
import os

import torch
import torch.nn as nn

import config as C
from hub_files import locate, fetch_from_hub  # noqa: F401  (fetch_from_hub is used by run_rr)
from phase1_model import MoEGPT, ModelConfig
from tokenizer import NerTokenizer


def add_prompt(x, mask, prompt):
    """Prompt tuning (Lester et al. 2021): prepend learned vectors to the token embeddings."""
    if prompt is None:
        return x, mask, 0
    P = prompt.size(0)
    x = torch.cat([prompt.to(x.dtype).unsqueeze(0).expand(x.size(0), -1, -1), x], dim=1)
    mask = torch.cat([mask.new_ones(mask.size(0), P), mask], dim=1)
    return x, mask, P


def load_tokenizer():
    path, _ = locate(C.TOKENIZER_FILES, C.TOKENIZER_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)
    tok = NerTokenizer.load(path)
    tok.sep_id = tok.eos_id               # marks the start of every sentence inside a pack
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
            if blk.use_moe:
                blk.ffn.dropless = True        # never drop tokens while fine-tuning; routing biases stay frozen

    def forward(self, ids, mask, prompt=None):
        x, mask, P = add_prompt(self.tok_emb(ids), mask, prompt)
        T = x.size(1)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        key_mask = mask[:, None, None, :] if (self.bidirectional and not bool(mask.all())) else None
        for blk in self.layers:
            x, _, _ = blk(x, cos, sin, None, key_mask)
        return self.final_norm(x)[:, P:]


def build_backbone(keep_layers=None, bidirectional=True):
    path, source = locate(C.WEIGHT_FILES, C.PHASE1_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)
    blob = torch.load(path, map_location="cpu")
    cfg = dict(blob["model_cfg"], mtp=False)               # the multi-token head is a pre-training-only extra
    gpt = MoEGPT(ModelConfig(**cfg))
    state = {k: v.float() for k, v in blob["model"].items() if not k.startswith("mtp_head.")}
    gpt.load_state_dict(state)
    if os.environ.get("P2_VERBOSE", "1") == "1":
        print(f"Phase-1 arch 4 weights: {path} ({source})", flush=True)
    return Backbone(gpt, keep_layers, bidirectional)


def student_layers():
    return list(C.STUDENT_LAYERS)
