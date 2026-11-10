"""
Arch 3: a compact Llama-style decoder, written from scratch in plain PyTorch ops.

  * RMSNorm (pre-norm), no biases anywhere
  * Rotary position embeddings (RoPE) instead of a learned position table
  * Grouped-query attention: fewer KV heads than query heads -> smaller KV cache, fewer params
  * Attention goes through F.scaled_dot_product_attention, so the T x T score matrix is never
    materialised (fused flash / memory-efficient kernel is picked automatically)
  * SwiGLU feed-forward
  * Tied input/output embeddings
  * KV-cache generation, batched, with left-padded prompts
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    d_model: int = 768
    d_ff: int = 2048
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    z_loss: float = 1e-4


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # statistics in fp32 even when the surrounding math is fp16
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight).to(x.dtype)


def build_rope_cache(head_dim, max_len, base):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)       # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x: (B, heads, T, head_dim); rotate the two halves of every head vector
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.group = cfg.n_head // cfg.n_kv_head
        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv * self.head_dim
        self.qkv = nn.Linear(cfg.d_model, q_dim + 2 * kv_dim, bias=False)   # one fused matmul
        self.proj = nn.Linear(q_dim, cfg.d_model, bias=False)
        self.q_dim, self.kv_dim = q_dim, kv_dim

    def forward(self, x, cos, sin, past=None, key_mask=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past is not None:                       # decoding: stash un-expanded K/V for reuse
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)

        if self.group > 1:                         # share each KV head across `group` query heads
            k = k.repeat_interleave(self.group, dim=1)
            v = v.repeat_interleave(self.group, dim=1)

        if key_mask is None:
            # training path: plain causal attention, fastest kernel
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)

        out = out.transpose(1, 2).reshape(B, T, self.q_dim)
        return self.proj(out), present


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)     # fused gate + up
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, past=None, key_mask=None):
        a, present = self.attn(self.attn_norm(x), cos, sin, past, key_mask)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        cos, sin = build_rope_cache(cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the layers that write into the residual stream so the
        # variance does not grow with depth.
        scale = 0.02 / math.sqrt(2 * cfg.n_layer)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, std=scale)
            nn.init.normal_(blk.ffn.down.weight, std=scale)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())        # tied weights are counted once

    def hidden_states(self, idx):
        """Final-layer hidden states (B, T, d_model) - what a Phase 2-7 task head sits on top of."""
        B, T = idx.shape
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x, _ = blk(x, cos, sin)
        return self.final_norm(x)

    def forward(self, idx, targets=None, caches=None, start_pos=0, key_mask=None, last_only=False):
        """
        Training:   model(idx, targets)            -> (ce_loss, z_loss)
        Inference:  model(idx, caches=..., ...)    -> (logits, new_caches)
        """
        B, T = idx.shape
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]

        x = self.tok_emb(idx)
        new_caches = []
        for i, blk in enumerate(self.blocks):
            past = caches[i] if caches is not None else None
            x, present = blk(x, cos, sin, past, key_mask)
            new_caches.append(present)
        x = self.final_norm(x)
        if last_only:
            x = x[:, -1:, :]

        logits = F.linear(x, self.tok_emb.weight)               # tied head

        if targets is None:
            return logits, new_caches

        logits = logits.float()
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return ce, z


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _sample(logits, temperature, top_k, top_p):
    """logits: (B, V) fp32 -> next token ids (B,)"""
    logits = logits / max(temperature, 1e-5)
    k = min(top_k, logits.size(-1)) if top_k > 0 else logits.size(-1)
    vals, idx = torch.topk(logits, k, dim=-1)                   # already sorted, descending
    probs = F.softmax(vals, dim=-1)
    if top_p < 1.0:
        before = probs.cumsum(-1) - probs                       # mass strictly before each token
        vals = vals.masked_fill(before >= top_p, float("-inf"))
        probs = F.softmax(vals, dim=-1)
    choice = torch.multinomial(probs, 1)
    return idx.gather(-1, choice).squeeze(-1)


@torch.no_grad()
def generate(model, prompts, tokenizer, max_new_tokens=384, target_words=128,
             temperature=0.7, top_k=50, top_p=0.92, check_every=8, suppress_eos=True):
    """
    prompts: list of token-id lists (any lengths). Returns a list of generated strings
    (the continuation only, trimmed to `target_words` words).
    Prompts are left-padded to a common length; padded keys are masked out, and because RoPE
    is relative, the shift does not change what the real tokens see.
    """
    model.eval()
    dev = next(model.parameters()).device
    B = len(prompts)
    L = max(len(p) for p in prompts)
    assert L + max_new_tokens <= model.cfg.max_seq_len, "prompt + continuation exceeds context"
    pad, eos = tokenizer.pad_id, tokenizer.eos_id

    ids = torch.full((B, L), pad, dtype=torch.long)
    valid = torch.zeros((B, L), dtype=torch.bool)
    for r, p in enumerate(prompts):
        ids[r, L - len(p):] = torch.tensor(p, dtype=torch.long)
        valid[r, L - len(p):] = True
    ids, valid = ids.to(dev), valid.to(dev)

    # prefill: causal mask AND key-validity; diagonal always allowed so pad rows never go all -inf
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    mask = (causal[None, None] & valid[:, None, None, :]) | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
    logits, caches = model(ids, caches=None, start_pos=0, key_mask=mask, last_only=True)

    out = torch.empty((B, 0), dtype=torch.long, device=dev)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    for step in range(max_new_tokens):
        step_logits = logits[:, -1].float()
        step_logits[:, pad] = float("-inf")
        if suppress_eos:                          # the task wants a full-length continuation: stop on word count only
            step_logits[:, eos] = float("-inf")
        nxt = _sample(step_logits, temperature, top_k, top_p)
        nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
        out = torch.cat([out, nxt[:, None]], dim=1)
        finished |= nxt == eos

        if (step + 1) % check_every == 0:
            for r in torch.nonzero(~finished).flatten().tolist():
                if len(tokenizer.decode(out[r].tolist()).split()) >= target_words:
                    finished[r] = True
        if bool(finished.all()):
            break

        valid = torch.cat([valid, torch.ones((B, 1), dtype=torch.bool, device=dev)], dim=1)
        step_mask = valid[:, None, None, :]
        logits, caches = model(nxt[:, None], caches=caches, start_pos=L + step,
                               key_mask=step_mask, last_only=True)

    texts = []
    for r in range(B):
        toks = out[r].tolist()
        if eos in toks:
            toks = toks[:toks.index(eos)]
        toks = [t for t in toks if t != pad]
        texts.append(" ".join(tokenizer.decode(toks).split()[:target_words]))
    return texts
