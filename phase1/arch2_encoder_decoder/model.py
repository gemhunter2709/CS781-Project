"""
Arch 2: a compact encoder-decoder (T5-style) language model, written from scratch.

It is shaped like the task itself: the ENCODER reads the 512-word prompt with full bidirectional
attention, and the DECODER writes the continuation, attending to the prompt through cross-attention.

Same efficiency toolbox as the other architectures:
  * RMSNorm (pre-norm), no biases, SwiGLU feed-forward
  * RoPE inside self-attention (cross-attention needs none: the encoder output already carries position)
  * grouped-query attention everywhere (fewer KV heads than query heads)
  * everything runs through F.scaled_dot_product_attention (fused kernels, no T x T matrix)
  * one embedding table shared by encoder input, decoder input and the output head
  * KV-cache decoding with the cross-attention K/V computed once per prompt
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_head: int = 12
    n_kv_head: int = 4
    n_enc_layer: int = 6
    n_dec_layer: int = 6
    d_ff: int = 2048
    enc_len: int = 1024            # prompt tokens the encoder sees
    dec_len: int = 320             # continuation tokens the decoder is trained on
    rope_base: float = 10000.0
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight).to(x.dtype)


def build_rope_cache(head_dim, max_len, base):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


def repeat_kv(x, group):
    return x if group == 1 else x.repeat_interleave(group, dim=1)


class SelfAttention(nn.Module):
    """Grouped-query self-attention with RoPE. `causal` picks decoder (True) or encoder (False) behaviour."""

    def __init__(self, cfg, causal):
        super().__init__()
        self.causal = causal
        self.n_head, self.n_kv = cfg.n_head, cfg.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.group = cfg.n_head // cfg.n_kv_head
        self.q_dim = self.n_head * self.head_dim
        self.kv_dim = self.n_kv * self.head_dim
        self.qkv = nn.Linear(cfg.d_model, self.q_dim + 2 * self.kv_dim, bias=False)
        self.proj = nn.Linear(self.q_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, past=None, key_mask=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = apply_rope(q.view(B, T, self.n_head, self.head_dim).transpose(1, 2), cos, sin)
        k = apply_rope(k.view(B, T, self.n_kv, self.head_dim).transpose(1, 2), cos, sin)
        v = v.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)
        k, v = repeat_kv(k, self.group), repeat_kv(v, self.group)

        if key_mask is not None:                       # padded prompt (inference) or single-step decoding
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal and past is None and T > 1)
        return self.proj(o.transpose(1, 2).reshape(B, T, self.q_dim)), present


class CrossAttention(nn.Module):
    """Decoder queries attend to the encoder output. No positional rotation here."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.n_kv = cfg.n_head, cfg.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.group = cfg.n_head // cfg.n_kv_head
        self.q_dim = self.n_head * self.head_dim
        self.kv_dim = self.n_kv * self.head_dim
        self.q = nn.Linear(cfg.d_model, self.q_dim, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * self.kv_dim, bias=False)
        self.proj = nn.Linear(self.q_dim, cfg.d_model, bias=False)

    def project_kv(self, enc_out):
        B, S, _ = enc_out.shape
        k, v = self.kv(enc_out).split([self.kv_dim, self.kv_dim], dim=-1)
        return (k.view(B, S, self.n_kv, self.head_dim).transpose(1, 2),
                v.view(B, S, self.n_kv, self.head_dim).transpose(1, 2))

    def forward(self, x, enc_out=None, cross_kv=None, enc_mask=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k, v = cross_kv if cross_kv is not None else self.project_kv(enc_out)
        k, v = repeat_kv(k, self.group), repeat_kv(v, self.group)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=enc_mask)
        return self.proj(o.transpose(1, 2).reshape(B, T, self.q_dim))


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = SelfAttention(cfg, causal=False)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin, key_mask=None):
        a, _ = self.attn(self.attn_norm(x), cos, sin, key_mask=key_mask)
        x = x + a
        return x + self.ffn(self.ffn_norm(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = SelfAttention(cfg, causal=True)
        self.cross_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.cross_attn = CrossAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin, enc_out=None, cross_kv=None, enc_mask=None, past=None, key_mask=None):
        a, present = self.self_attn(self.self_norm(x), cos, sin, past=past, key_mask=key_mask)
        x = x + a
        x = x + self.cross_attn(self.cross_norm(x), enc_out=enc_out, cross_kv=cross_kv, enc_mask=enc_mask)
        return x + self.ffn(self.ffn_norm(x)), present


class EncDecLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)          # shared by encoder, decoder and output head
        self.enc_layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_enc_layer)])
        self.enc_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.dec_layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_dec_layer)])
        self.dec_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        cos, sin = build_rope_cache(cfg.d_model // cfg.n_head, max(cfg.enc_len, 2 * cfg.dec_len), cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        n_res = cfg.n_enc_layer + 2 * cfg.n_dec_layer                      # residual writers: shrink them with depth
        scale = 0.02 / math.sqrt(n_res)
        for layer in self.enc_layers:
            nn.init.normal_(layer.attn.proj.weight, std=scale)
            nn.init.normal_(layer.ffn.down.weight, std=scale)
        for layer in self.dec_layers:
            nn.init.normal_(layer.self_attn.proj.weight, std=scale)
            nn.init.normal_(layer.cross_attn.proj.weight, std=scale)
            nn.init.normal_(layer.ffn.down.weight, std=scale)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def active_params(self):                                              # dense model: everything is active
        return self.num_params()

    # ---- pieces ------------------------------------------------------------
    def encode(self, enc_ids, enc_mask=None):
        """enc_mask: optional (B, S) bool, True for real tokens (only needed for padded batches)."""
        B, S = enc_ids.shape
        cos, sin = self.rope_cos[:S], self.rope_sin[:S]
        key_mask = None if enc_mask is None else enc_mask[:, None, None, :]
        x = self.tok_emb(enc_ids)
        for layer in self.enc_layers:
            x = layer(x, cos, sin, key_mask)
        return self.enc_norm(x)

    def hidden_states(self, idx):
        """Encoder states (B, T, d_model): what a Phase 2-7 task head sits on top of."""
        return self.encode(idx)

    def cross_kv(self, enc_out):
        return [layer.cross_attn.project_kv(enc_out) for layer in self.dec_layers]

    def decode(self, dec_ids, enc_out=None, cross_kvs=None, enc_mask=None, caches=None, start_pos=0, last_only=False):
        B, T = dec_ids.shape
        cos, sin = self.rope_cos[start_pos:start_pos + T], self.rope_sin[start_pos:start_pos + T]
        cross_mask = None if enc_mask is None else enc_mask[:, None, None, :]
        x = self.tok_emb(dec_ids)
        new_caches = []
        for i, layer in enumerate(self.dec_layers):
            x, present = layer(x, cos, sin, enc_out=enc_out,
                               cross_kv=None if cross_kvs is None else cross_kvs[i], enc_mask=cross_mask,
                               past=None if caches is None else caches[i])
            new_caches.append(present)
        x = self.dec_norm(x)
        if last_only:
            x = x[:, -1:, :]
        return F.linear(x, self.tok_emb.weight), new_caches

    # ---- training forward --------------------------------------------------
    def forward(self, enc_ids, dec_ids, targets):
        """Teacher-forced: returns (cross-entropy over the decoder tokens, z-loss)."""
        enc_out = self.encode(enc_ids)
        logits, _ = self.decode(dec_ids, enc_out=enc_out)
        logits = logits.float()
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return ce, z


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _sample(logits, temperature, top_k, top_p):
    logits = logits / max(temperature, 1e-5)
    k = min(top_k, logits.size(-1)) if top_k > 0 else logits.size(-1)
    vals, idx = torch.topk(logits, k, dim=-1)
    probs = F.softmax(vals, dim=-1)
    if top_p < 1.0:
        before = probs.cumsum(-1) - probs
        vals = vals.masked_fill(before >= top_p, float("-inf"))
        probs = F.softmax(vals, dim=-1)
    return idx.gather(-1, torch.multinomial(probs, 1)).squeeze(-1)


@torch.no_grad()
def generate(model, prompts, tokenizer, max_new_tokens=320, target_words=128,
             temperature=0.7, top_k=50, top_p=0.92, check_every=8, suppress_eos=True):
    """
    prompts: list of token-id lists (<= enc_len each). The prompts are left-padded into one batch and the
    padding is masked out of both the encoder and the cross-attention. Returns one continuation string each.
    """
    model.eval()
    dev = next(model.parameters()).device
    B = len(prompts)
    S = max(len(p) for p in prompts)
    assert S <= model.cfg.enc_len, "prompt longer than the encoder context"
    assert max_new_tokens <= 2 * model.cfg.dec_len, "continuation longer than the cached RoPE table"
    pad, eos = tokenizer.pad_id, tokenizer.eos_id

    ids = torch.full((B, S), pad, dtype=torch.long)
    valid = torch.zeros((B, S), dtype=torch.bool)
    for r, p in enumerate(prompts):
        ids[r, S - len(p):] = torch.tensor(p, dtype=torch.long)
        valid[r, S - len(p):] = True
    ids, valid = ids.to(dev), valid.to(dev)

    enc_out = model.encode(ids, valid)
    cross_kvs = model.cross_kv(enc_out)

    cur = torch.full((B, 1), eos, dtype=torch.long, device=dev)            # decoder start token
    out = torch.empty((B, 0), dtype=torch.long, device=dev)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    caches = None
    for step in range(max_new_tokens):
        logits, caches = model.decode(cur, cross_kvs=cross_kvs, enc_mask=valid, caches=caches,
                                      start_pos=step, last_only=True)
        step_logits = logits[:, -1].float()
        step_logits[:, pad] = float("-inf")
        if suppress_eos:
            step_logits[:, eos] = float("-inf")
        nxt = _sample(step_logits, temperature, top_k, top_p)
        nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
        out = torch.cat([out, nxt[:, None]], dim=1)
        finished |= nxt == eos
        cur = nxt[:, None]

        if (step + 1) % check_every == 0:
            for r in torch.nonzero(~finished).flatten().tolist():
                if len(tokenizer.decode(out[r].tolist()).split()) >= target_words:
                    finished[r] = True
        if bool(finished.all()):
            break

    texts = []
    for r in range(B):
        toks = out[r].tolist()
        if eos in toks:
            toks = toks[:toks.index(eos)]
        toks = [t for t in toks if t != pad]
        texts.append(" ".join(tokenizer.decode(toks).split()[:target_words]))
    return texts
