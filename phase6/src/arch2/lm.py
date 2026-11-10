"""Arch 2 (encoder-decoder) as a summarizer: the encoder reads the selected source, the decoder writes the summary."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from hub_files import locate, fetch_from_hub  # noqa: F401  (fetch_from_hub is used by run_sum)
from phase1_model import EncDecLM, ModelConfig
from tokenizer import NerTokenizer
from gen_core import IGNORE, DecodeState


def load_tokenizer():
    path, _ = locate(C.TOKENIZER_FILES, C.TOKENIZER_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)
    tok = NerTokenizer.load(path)
    tok.sep_id = tok.eos_id
    return tok


class Arch2LM(nn.Module):
    is_encoder_decoder = True

    def __init__(self, model, keep_layers):
        super().__init__()
        self.inner_cfg = model.cfg
        self.tok_emb = model.tok_emb
        pick = (lambda ls: ls) if keep_layers is None else (lambda ls: [ls[i] for i in keep_layers if i < len(ls)])
        self.enc_layers = nn.ModuleList(pick(list(model.enc_layers)))
        self.dec_layers = nn.ModuleList(pick(list(model.dec_layers)))
        self.enc_norm = model.enc_norm
        self.dec_norm = model.dec_norm
        self.register_buffer("rope_cos", model.rope_cos.clone(), persistent=False)
        self.register_buffer("rope_sin", model.rope_sin.clone(), persistent=False)
        self.max_len = model.rope_cos.size(0)
        self.prompt = None

    def encode(self, src, valid):
        x = self.tok_emb(src)
        if self.prompt is not None:                      # prompt tuning: virtual tokens in front of the source
            P = self.prompt.size(0)
            x = torch.cat([self.prompt.to(x.dtype).unsqueeze(0).expand(x.size(0), -1, -1), x], dim=1)
            valid = torch.cat([valid.new_ones(valid.size(0), P), valid], dim=1)
        T = x.size(1)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        key_mask = None if bool(valid.all()) else valid[:, None, None, :]
        for layer in self.enc_layers:
            x = layer(x, cos, sin, key_mask)
        return self.enc_norm(x), valid

    def decode(self, dec_ids, start, enc_out, cross, enc_valid, caches):
        T = dec_ids.size(1)
        cos, sin = self.rope_cos[start:start + T], self.rope_sin[start:start + T]
        x = self.tok_emb(dec_ids)
        cross_mask = None if bool(enc_valid.all()) else enc_valid[:, None, None, :]
        new = []
        for i, layer in enumerate(self.dec_layers):
            x, present = layer(x, cos, sin, enc_out=enc_out, cross_kv=None if cross is None else cross[i],
                               enc_mask=cross_mask, past=None if caches is None else caches[i])
            new.append(present)
        return self.dec_norm(x), new

    def make_batch(self, src, src_len, tgt, tgt_len, tok):
        S, T = int(max(src_len)), int(max(tgt_len)) + 1
        enc = torch.full((len(src), S), tok.pad_id, dtype=torch.long)
        valid = torch.zeros((len(src), S), dtype=torch.bool)
        dec_in = torch.full((len(src), T), tok.pad_id, dtype=torch.long)
        labels = torch.full((len(src), T), IGNORE, dtype=torch.long)
        for i, (s, sl, t, tl) in enumerate(zip(src, src_len, tgt, tgt_len)):
            enc[i, :sl] = torch.from_numpy(s[:sl].astype("int64"))
            valid[i, :sl] = True
            summary = list(t[:tl]) + ([tok.eos_id] if tl < len(t) else [])
            dec_in[i, :len(summary)] = torch.tensor([tok.eos_id] + summary[:-1])
            labels[i, :len(summary)] = torch.tensor(summary)
        return enc, valid, dec_in, labels

    def teacher_forced(self, batch, device):
        enc, valid, dec_in, labels = (t.to(device) for t in batch)
        enc_out, valid = self.encode(enc, valid)
        h, _ = self.decode(dec_in, 0, enc_out, None, valid, None)
        keep = labels != IGNORE
        return F.linear(h[keep], self.tok_emb.weight).float(), labels[keep]

    @torch.no_grad()
    def generate(self, sources, min_new, max_new, tok, cfg, device):
        B, S = len(sources), max(len(s) for s in sources)
        enc = torch.full((B, S), tok.pad_id, dtype=torch.long)
        valid = torch.zeros((B, S), dtype=torch.bool)
        for i, s in enumerate(sources):
            enc[i, S - len(s):] = torch.tensor(list(s))
            valid[i, S - len(s):] = True
        enc_out, valid = self.encode(enc.to(device), valid.to(device))
        cross = [layer.cross_attn.project_kv(enc_out) for layer in self.dec_layers]
        cur = torch.full((B, 1), tok.eos_id, dtype=torch.long, device=device)
        h, caches = self.decode(cur, 0, None, cross, valid, None)
        state = DecodeState(B, self.tok_emb.num_embeddings, min_new, max_new, tok, cfg, device)
        step = 0
        while not state.done():
            nxt = state.step(F.linear(h[:, -1], self.tok_emb.weight).float())
            step += 1
            h, caches = self.decode(nxt[:, None], step, None, cross, valid, caches)
        return state.outputs()


def build_lm(keep_layers=None):
    path, source = locate(C.WEIGHT_FILES, C.PHASE1_REPOS, C.KAGGLE_HINT, C.CACHE_DIR)
    blob = torch.load(path, map_location="cpu")
    model = EncDecLM(ModelConfig(**blob["model_cfg"]))
    model.load_state_dict({k: v.float() for k, v in blob["model"].items()})
    if os.environ.get("P2_VERBOSE", "1") == "1":
        print(f"Phase-1 arch 2 weights: {path} ({source})", flush=True)
    return Arch2LM(model, keep_layers)


def student_layers():
    return list(C.STUDENT_LAYERS)
