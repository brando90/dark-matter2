"""DM-4 model: a minimal re-implementation of the OpenLM transformer Porian et al. trained, with the same parameter
names, so Porian's released checkpoints load into it unchanged (eval_porian.py checks that).

Recipe (read from OpenLM main and Porian's checkpoints/evaluating_checkpoint.py): pre-norm LayerNorm (gain + bias,
eps 1e-5), qk LayerNorm over the full width before the head split, rotary (base 10,000, rotate-half) on 4 heads,
SwiGLU FFN (hidden 256 ceil(int(8w/3)/256)), no linear biases, final LayerNorm, linear head. Porian's models are
UNTIED (weight_tying False in evaluating_checkpoint.py; output.weight != tok_embeddings.weight in the checkpoint).
Init as OpenLM: trunc-normal(1/sqrt(fan_in)) clipped at 3 sd, output projections scaled by 1/sqrt(2 (layer+1)).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_HEADS = 4      # Porian: n_heads 4 at every width


def ffn_hidden(w):
    return 256 * math.ceil(int(8 * w / 3) / 256)


def _tn(weight, std):
    nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std)


class Rotary(nn.Module):
    """OpenLM RotaryWithCast: cos/sin built in the dtype of k, then q, k cast to v's dtype."""

    def __init__(self, head_dim, seq_len):
        super().__init__()
        self.dim = head_dim
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)))
        self._cache = None

    def _tables(self, n, device, dtype):
        c = self._cache
        if c is None or c[0] < n or c[1].device != device or c[1].dtype != dtype:
            t = torch.arange(n, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            self._cache = (n, emb.cos()[None, :, None, :].to(dtype), emb.sin()[None, :, None, :].to(dtype))
        return self._cache[1][:, :n], self._cache[2][:, :n]

    @staticmethod
    def _rot(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, v):
        cos, sin = self._tables(k.shape[1], k.device, k.dtype)
        q = q * cos + self._rot(q) * sin
        k = k * cos + self._rot(k) * sin
        return q.to(v.dtype), k.to(v.dtype), v


class Attention(nn.Module):
    def __init__(self, layer_id, dim, seq_len):
        super().__init__()
        self.layer_id, self.dim, self.n_heads, self.head_dim = layer_id, dim, N_HEADS, dim // N_HEADS
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.pos_embed = Rotary(self.head_dim, seq_len)
        self.q_norm = nn.LayerNorm(dim, eps=1e-5)
        self.k_norm = nn.LayerNorm(dim, eps=1e-5)

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.dim)
        _tn(self.in_proj.weight, std)
        _tn(self.out_proj.weight, std / math.sqrt(2 * (self.layer_id + 1)))
        for n in (self.q_norm, self.k_norm):
            n.reset_parameters()

    def forward(self, x):
        b, s, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q.view(b, s, self.n_heads, self.head_dim)
        k = k.view(b, s, self.n_heads, self.head_dim)
        v = v.view(b, s, self.n_heads, self.head_dim)
        q, k, v = self.pos_embed(q, k, v)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.out_proj(o.transpose(1, 2).contiguous().view(b, s, -1))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Block(nn.Module):
    def __init__(self, layer_id, dim, seq_len):
        super().__init__()
        self.layer_id, self.dim, self.hidden = layer_id, dim, ffn_hidden(dim)
        self.attention = Attention(layer_id, dim, seq_len)
        self.feed_forward = SwiGLU(dim, self.hidden)
        self.attention_norm = nn.LayerNorm(dim, eps=1e-5)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-5)

    def reset_parameters(self):
        self.attention.reset_parameters()
        _tn(self.feed_forward.w12.weight, 1.0 / math.sqrt(self.dim))
        _tn(self.feed_forward.w3.weight, 1.0 / math.sqrt(self.hidden) / math.sqrt(2 * (self.layer_id + 1)))
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        return h + self.feed_forward(self.ffn_norm(h))


class Transformer(nn.Module):
    def __init__(self, width, depth, vocab, tied, seq_len=2048):
        super().__init__()
        self.width, self.depth, self.vocab, self.tied = width, depth, vocab, tied
        self.tok_embeddings = nn.Embedding(vocab, width)
        self.layers = nn.ModuleList([Block(i, width, seq_len) for i in range(depth)])
        self.norm = nn.LayerNorm(width, eps=1e-5)
        self.output = nn.Linear(width, vocab, bias=False)
        self.reset_parameters()
        if tied:                       # as OpenLM weight_tying: the embedding is the head's weight
            self.tok_embeddings.weight = self.output.weight

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.width)
        _tn(self.output.weight, std)
        _tn(self.tok_embeddings.weight, std)
        for layer in self.layers:
            layer.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, idx):
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x)).float()

    def counts(self):
        """N_nonemb as DM-2 (attention + FFN matrices; no norms), N_emb = unique embedding parameters."""
        nonemb = self.depth * (4 * self.width ** 2 + 3 * self.width * ffn_hidden(self.width))
        return dict(N_nonemb=nonemb, N_emb=self.vocab * self.width * (1 if self.tied else 2))
