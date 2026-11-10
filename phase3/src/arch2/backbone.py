"""Loads the Phase-1 arch 2 weights (encoder-decoder) and uses its bidirectional ENCODER for rhetorical roles."""
import os

import torch
import torch.nn as nn

import config as C
from hub_files import locate, fetch_from_hub  # noqa: F401  (fetch_from_hub is used by run_rr)
from phase1_model import EncDecLM, ModelConfig
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
    """The decoder is dropped: tagging only needs the encoder, which already attends both ways."""

    def __init__(self, model, keep_layers):
        super().__init__()
        self.d_model = model.cfg.d_model
        self.tok_emb = model.tok_emb
        layers = model.enc_layers if keep_layers is None else [model.enc_layers[i] for i in keep_layers]
        self.layers = nn.ModuleList(layers)
        self.final_norm = model.enc_norm
        self.register_buffer("rope_cos", model.rope_cos.clone(), persistent=False)
        self.register_buffer("rope_sin", model.rope_sin.clone(), persistent=False)

    def forward(self, ids, mask, prompt=None):
        x, mask, P = add_prompt(self.tok_emb(ids), mask, prompt)
        T = x.size(1)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        key_mask = None if bool(mask.all()) else mask[:, None, None, :]
        for layer in self.layers:
            x = layer(x, cos, sin, key_mask)
        return self.final_norm(x)[:, P:]


def build_backbone(keep_layers=None, bidirectional=True):
    path, source = locate(C.WEIGHT_FILES, C.PHASE1_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)
    blob = torch.load(path, map_location="cpu")
    model = EncDecLM(ModelConfig(**blob["model_cfg"]))
    model.load_state_dict({k: v.float() for k, v in blob["model"].items()})
    if os.environ.get("P2_VERBOSE", "1") == "1":
        print(f"Phase-1 arch 2 weights: {path} ({source})", flush=True)
    return Backbone(model, keep_layers)


def student_layers():
    return list(C.STUDENT_LAYERS)
