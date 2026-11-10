"""
Arch 1 (GPT-2 style decoder) exactly as trained in Phase 1, so the saved weights load unchanged.
Phase 2 adds one switch: `causal`, so attention can look right as well as left while fine-tuning for NER.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.resid_drop = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.causal = True

    def forward(self, x, key_mask=None):
        B, T, D = x.shape
        qkv = self.W_qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        p = self.dropout_p if self.training else 0.0
        if key_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=p)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=p)
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        return self.resid_drop(self.W_o(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.gelu(self.up(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout, max_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, key_mask=None):
        x = x + self.attn(self.ln1(x), key_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT2LM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, d_ff, dropout, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout, max_len)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.emb_drop(self.token_emb(x) + self.pos_emb(positions))
        for layer in self.layers:
            h = layer(h)
        return self.head(self.ln_f(h))
