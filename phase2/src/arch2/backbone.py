"""Loads the Phase-1 arch 2 weights (encoder-decoder) and uses its bidirectional ENCODER for NER."""
import os

import torch
import torch.nn as nn

import config as C
from hub_files import locate, fetch_from_hub  # noqa: F401  (fetch_from_hub is used by run_ner)
from phase1_model import EncDecLM, ModelConfig
from tokenizer import NerTokenizer


def load_tokenizer():
    path, _ = locate(C.TOKENIZER_FILES, C.TOKENIZER_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)
    tok = NerTokenizer.load(path)
    tok.split_punct = C.SPLIT_PUNCT
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

    def forward(self, ids, mask):
        T = ids.size(1)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        key_mask = None if bool(mask.all()) else mask[:, None, None, :]
        x = self.tok_emb(ids)
        for layer in self.layers:
            x = layer(x, cos, sin, key_mask)
        return self.final_norm(x)


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
