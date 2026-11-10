"""
Arch 4: a small DeepSeek-V3-flavoured mixture-of-experts decoder, written from scratch.

Ideas borrowed from the DeepSeek papers (scaled down to ~217M total / ~80M active parameters):

  * MLA - Multi-head Latent Attention. Keys/values are squeezed through a small latent vector,
    so the KV cache holds `kv_lora + qk_rope` numbers per token instead of full K and V.
    RoPE lives on a small separate slice of each head ("decoupled RoPE") so the latent stays
    position-free.
  * DeepSeekMoE - many small routed experts plus one shared expert that every token uses.
  * Auxiliary-loss-free load balancing (V3) - a per-expert bias is added to the routing scores
    *only to pick the top-k*; after every optimizer step it is nudged up for under-used experts
    and down for over-used ones. Gate weights come from the un-biased sigmoid scores. A tiny
    sequence-wise balance loss is kept only as a safety net.
  * Multi-token prediction (MTP) - during training an extra light module predicts the token two
    steps ahead. It shares the embedding / output head and is simply unused at inference.

Our own adaptation: expert dispatch uses fixed-size capacity buffers and one batched matmul over
all experts. Every tensor shape is static, so torch.compile can fuse it and there is no Python
loop over experts. In eval / generation the capacity is raised so that no token is ever dropped.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    # MLA
    q_lora: int = 384
    kv_lora: int = 192
    qk_nope: int = 48
    qk_rope: int = 16
    v_dim: int = 64
    # MoE
    n_dense_layers: int = 1
    d_ff_dense: int = 2048
    n_routed: int = 32
    top_k: int = 4
    expert_ff: int = 192
    shared_ff: int = 512
    capacity_factor: float = 1.25
    # multi-token prediction
    mtp: bool = True
    mtp_ff: int = 1024


# ---------------------------------------------------------------------------
# small building blocks
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight).to(x.dtype)


def build_rope_cache(dim, max_len, base):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


# ---------------------------------------------------------------------------
# Multi-head Latent Attention
# ---------------------------------------------------------------------------
class MLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H = cfg.n_head
        self.nope, self.rope, self.vd = cfg.qk_nope, cfg.qk_rope, cfg.v_dim
        self.kv_lora = cfg.kv_lora

        self.q_down = nn.Linear(cfg.d_model, cfg.q_lora, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora, cfg.norm_eps)
        self.q_up = nn.Linear(cfg.q_lora, self.H * (self.nope + self.rope), bias=False)

        # one projection yields the compressed KV latent AND the (shared) rotary key slice
        self.kv_down = nn.Linear(cfg.d_model, cfg.kv_lora + self.rope, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora, cfg.norm_eps)
        self.kv_up = nn.Linear(cfg.kv_lora, self.H * (self.nope + self.vd), bias=False)

        self.out = nn.Linear(self.H * self.vd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, past=None, key_mask=None):
        B, T, _ = x.shape
        H = self.H

        q = self.q_up(self.q_norm(self.q_down(x))).view(B, T, H, self.nope + self.rope).transpose(1, 2)
        q_nope, q_rope = q.split([self.nope, self.rope], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)

        kv = self.kv_down(x)
        latent, k_rope = kv.split([self.kv_lora, self.rope], dim=-1)
        latent = self.kv_norm(latent)                                   # (B, T, kv_lora)
        k_rope = apply_rope(k_rope.unsqueeze(1), cos, sin)              # (B, 1, T, rope) shared by all heads

        if past is not None:                                            # cache = latent + rope key only
            latent = torch.cat([past[0], latent], dim=1)
            k_rope = torch.cat([past[1], k_rope], dim=2)
        present = (latent, k_rope)

        S = latent.size(1)
        kv_full = self.kv_up(latent).view(B, S, H, self.nope + self.vd).transpose(1, 2)
        k_nope, v = kv_full.split([self.nope, self.vd], dim=-1)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope.expand(-1, H, -1, -1)], dim=-1)

        if key_mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        o = o.transpose(1, 2).reshape(B, T, H * self.vd)
        return self.out(o), present


# ---------------------------------------------------------------------------
# DeepSeekMoE layer with auxiliary-loss-free balancing and static-shape dispatch
# ---------------------------------------------------------------------------
@torch.compiler.disable
def _slot_positions(eid, E):
    """
    For every (token, expert) pair: its running position inside that expert's buffer, plus the
    one-hot assignment matrix. This is integer bookkeeping on ~33K elements, so it runs eagerly
    on purpose: comparing an index tensor with arange(E) trips an Inductor/Triton codegen bug on
    some torch builds (seen on Kaggle). Everything heavy around it stays compiled.
    """
    onehot = (eid.unsqueeze(1) == torch.arange(E, device=eid.device).unsqueeze(0)).to(torch.int32)
    pos = onehot.cumsum(0).gather(1, eid.unsqueeze(1)).squeeze(1) - 1
    return onehot, pos


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.E, self.K = cfg.n_routed, cfg.top_k
        self.cap_factor = cfg.capacity_factor
        D, h = cfg.d_model, cfg.expert_ff

        self.router = nn.Linear(D, self.E, bias=False)
        self.w_gate_up = nn.Parameter(torch.empty(self.E, D, 2 * h))    # all experts stacked -> one bmm
        self.w_down = nn.Parameter(torch.empty(self.E, h, D))
        self.shared = SwiGLU(D, cfg.shared_ff)

        nn.init.normal_(self.w_gate_up, std=0.02)
        nn.init.normal_(self.w_down, std=0.02)

        self.register_buffer("bias", torch.zeros(self.E))               # saved: it is learned state
        self.register_buffer("load", torch.zeros(self.E), persistent=False)
        self.register_buffer("dropped", torch.zeros(()), persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        N, E, K = B * T, self.E, self.K
        flat = x.reshape(N, D)

        # --- routing: sigmoid affinities, biased top-k, normalised gate weights
        scores = torch.sigmoid(F.linear(flat.float(), self.router.weight.float()))      # (N, E)
        _, idx = torch.topk(scores.detach() + self.bias, K, dim=-1)                     # bias steers selection only
        gate = scores.gather(1, idx)
        gate = gate / (gate.sum(-1, keepdim=True) + 1e-9)

        # --- slot assignment: position of every (token, expert) pair inside its expert's buffer
        eid = idx.reshape(-1)                                                           # token-major, (N*K,)
        onehot, pos = _slot_positions(eid, E)

        if self.training:
            cap = int(math.ceil(N * K / E * self.cap_factor / 8) * 8)
        else:
            cap = N                                                                     # drop-free for eval / generation
        keep = pos < cap
        slot = torch.where(keep, eid * cap + pos, torch.full_like(eid, E * cap))        # overflow -> spare row

        # --- dispatch -> one batched matmul over all experts -> combine
        dt = torch.get_autocast_dtype("cuda") if (torch.is_autocast_enabled() and x.is_cuda) else x.dtype
        xs = flat.to(dt).repeat_interleave(K, dim=0)
        buf = xs.new_zeros(E * cap + 1, D).index_put((slot,), xs)
        hid = buf[: E * cap].view(E, cap, D)
        g, u = torch.bmm(hid, self.w_gate_up.to(dt)).chunk(2, dim=-1)
        y = torch.bmm(F.silu(g) * u, self.w_down.to(dt))                                # (E, cap, D)
        y = torch.cat([y.reshape(E * cap, D), y.new_zeros(1, D)], dim=0)
        routed = (y[slot].view(N, K, D) * gate.to(y.dtype).unsqueeze(-1)).sum(1)

        out = (routed.to(x.dtype) + self.shared(flat)).view(B, T, D)

        if not self.training:
            return out, x.new_zeros((), dtype=torch.float32)

        # --- bookkeeping for balancing (no gradient) and the small sequence-wise safety loss
        with torch.no_grad():
            self.load += onehot.sum(0).float()
            self.dropped += (~keep).sum().float()
        p = (scores / (scores.sum(-1, keepdim=True) + 1e-9)).view(B, T, E).mean(1)      # mean affinity per sequence
        f = E * onehot.view(B, T * K, E).float().mean(1)                                # share of assignments per expert
        aux = (f * p).sum(-1).mean()
        return out, aux


# ---------------------------------------------------------------------------
# transformer block, MTP head, full model
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg, use_moe):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = MLA(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.use_moe = use_moe
        self.ffn = MoE(cfg) if use_moe else SwiGLU(cfg.d_model, cfg.d_ff_dense)

    def forward(self, x, cos, sin, past=None, key_mask=None):
        a, present = self.attn(self.attn_norm(x), cos, sin, past, key_mask)
        x = x + a
        if self.use_moe:
            f, aux = self.ffn(self.ffn_norm(x))
        else:
            f, aux = self.ffn(self.ffn_norm(x)), x.new_zeros((), dtype=torch.float32)
        return x + f, present, aux


class MTPHead(nn.Module):
    """Predicts token t+2 from (trunk hidden state at t, embedding of token t+1)."""

    def __init__(self, cfg):
        super().__init__()
        self.norm_h = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_e = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = MLA(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.mtp_ff)

    def forward(self, h, next_emb, cos, sin):
        z = self.proj(torch.cat([self.norm_h(h), self.norm_e(next_emb)], dim=-1))
        a, _ = self.attn(self.attn_norm(z), cos, sin)
        z = z + a
        return z + self.ffn(self.ffn_norm(z))


class MoEGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, use_moe=i >= cfg.n_dense_layers) for i in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mtp_head = MTPHead(cfg) if cfg.mtp else None

        cos, sin = build_rope_cache(cfg.qk_rope, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        scale = 0.02 / math.sqrt(2 * cfg.n_layer)                       # GPT-2 style residual-branch scaling
        for blk in self.blocks:
            nn.init.normal_(blk.attn.out.weight, std=scale)
            if blk.use_moe:
                nn.init.normal_(blk.ffn.w_down, std=scale)
                nn.init.normal_(blk.ffn.shared.down.weight, std=scale)
            else:
                nn.init.normal_(blk.ffn.down.weight, std=scale)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    # ---- bookkeeping -------------------------------------------------------
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def active_params(self):
        """Parameters touched per token: everything except the routed experts that were not picked."""
        total = self.num_params()
        routed = sum(b.ffn.w_gate_up.numel() + b.ffn.w_down.numel() for b in self.blocks if b.use_moe)
        inactive = routed * (1 - self.cfg.top_k / self.cfg.n_routed)
        mtp = sum(p.numel() for p in self.mtp_head.parameters()) if self.mtp_head is not None else 0
        return int(total - inactive - mtp)                              # MTP is training-only

    @torch.no_grad()
    def update_balance(self, speed):
        """
        Aux-loss-free balancing step (call once per optimizer step on every rank):
        push each expert's routing bias toward the mean load, then reset the counters.
        Returns (max load violation, fraction of assignments dropped) for logging.
        """
        moes = [b.ffn for b in self.blocks if b.use_moe]
        load = torch.stack([m.load for m in moes])
        dropped = torch.stack([m.dropped for m in moes])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load)
            dist.all_reduce(dropped)
        mean = load.mean(dim=-1, keepdim=True)
        if speed > 0:
            step = speed * torch.sign(mean - load)                      # under-used experts get a boost
            for m, s in zip(moes, step):
                m.bias += s
        maxvio = ((load - mean).abs().amax(dim=-1) / mean.squeeze(-1).clamp(min=1)).mean()
        drop_frac = dropped.sum() / load.sum().clamp(min=1)
        for m in moes:
            m.load.zero_()
            m.dropped.zero_()
        return maxvio, drop_frac

    def hidden_states(self, idx):
        """Final-layer hidden states (B, T, d_model) - what a Phase 2-7 task head sits on top of."""
        B, T = idx.shape
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x, _, _ = blk(x, cos, sin)
        return self.final_norm(x)

    # ---- forward -----------------------------------------------------------
    def forward(self, idx, targets=None, caches=None, start_pos=0, key_mask=None, last_only=False):
        """
        Training:   model(idx, targets) -> (ce, mtp_ce, seq_aux, z_loss)
        Inference:  model(idx, caches=..., ...) -> (logits, new_caches)
        """
        B, T = idx.shape
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]

        x = self.tok_emb(idx)
        new_caches, aux_total = [], x.new_zeros((), dtype=torch.float32)
        for i, blk in enumerate(self.blocks):
            past = caches[i] if caches is not None else None
            x, present, aux = blk(x, cos, sin, past, key_mask)
            new_caches.append(present)
            aux_total = aux_total + aux

        hidden = x
        x = self.final_norm(x)
        if last_only:
            x = x[:, -1:, :]
        logits = F.linear(x, self.tok_emb.weight)

        if targets is None:
            return logits, new_caches

        V = logits.size(-1)
        logits = logits.float()
        ce = F.cross_entropy(logits.view(-1, V), targets.reshape(-1))
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()

        mtp_ce = ce.new_zeros(())
        if self.mtp_head is not None and self.training:
            z2 = self.mtp_head(hidden, self.tok_emb(targets), cos, sin)         # embedding of token t+1
            lg2 = F.linear(self.final_norm(z2), self.tok_emb.weight).float()
            shifted = torch.cat([targets[:, 1:], torch.full_like(targets[:, :1], -100)], dim=1)   # token t+2
            mtp_ce = F.cross_entropy(lg2.view(-1, V), shifted.reshape(-1), ignore_index=-100)
        return ce, mtp_ce, aux_total / max(1, len(self.blocks) - self.cfg.n_dense_layers), z


# ---------------------------------------------------------------------------
# Generation (same batched, left-padded KV-cache sampler as arch 3)
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
def generate(model, prompts, tokenizer, max_new_tokens=384, target_words=128,
             temperature=0.7, top_k=50, top_p=0.92, check_every=8, suppress_eos=True):
    """prompts: list of token-id lists. Returns the generated continuations as strings."""
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
        logits, caches = model(nxt[:, None], caches=caches, start_pos=L + step,
                               key_mask=valid[:, None, None, :], last_only=True)

    texts = []
    for r in range(B):
        toks = out[r].tolist()
        if eos in toks:
            toks = toks[:toks.index(eos)]
        toks = [t for t in toks if t != pad]
        texts.append(" ".join(tokenizer.decode(toks).split()[:target_words]))
    return texts
