"""Temporal attention over the core's own state history.

The core carries everything it knows in one `(B, N)` vector, so a fact needed
200 tokens later has to survive 200 overwrites of that vector. Attention gives
it a second route: each step's state queries a cache of earlier states, and what
comes back is added to the trajectory.

    q_t = h_t W_q,  k_i = h_i W_k,  v_i = h_i W_v
    h_t += softmax(q_t K^T / sqrt(d)) V W_o / sqrt(H*D)

Three decisions here are measurements rather than taste.

**Grouped-query, one KV head.** The cache is the memory term that decides
whether a batch fits, and `kv_heads=1` divides it by `heads` while the query's
head axis still reads `heads` different projections of the same keys.

**RoPE at absolute stream position, applied at write time.** Rotary inner
products depend only on the relative distance `t - s`, so rotating each key once
against a monotonically increasing counter means a cached key never has to be
re-rotated -- and attention reaches across chunk boundaries with no bookkeeping.

**The cache crosses calls detached.** The same contract the hidden state has:
value flows forward, gradient does not. So a `T=64` chunk still attends over the
previous `window` tokens, which is what makes a short chunk affordable -- and a
short chunk is what makes the whole model fast (see `aleph/core/core.py`).

`o_proj` is zero-initialized and the branch is divided by `sqrt(heads*head_dim)`,
so switching attention on does not change what the recurrence sees at step 0 and
widening it does not change the scale of what it eventually contributes.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class TemporalAttention(nn.Module):
    """Multi-query RoPE attention over the state trajectory, with cross-call carry.

    Args:
        width (int): State width `N`. Attention reads and writes the full state.
        heads (int): Query heads. Default 4.
        kv_heads (int): Key/value heads, shared across query heads. Must divide
            `heads`. Default 1 -- the cache, not the projections, is the cost.
        head_dim (int): Per-head width, even for RoPE. Default 16, which
            measured as good as 32 at half the parameters.
        window (int): Cached entries. Attention never looks further back, and
            the cache is truncated here. Default 128.
        rope_theta (float): RoPE base frequency. Default 10000.
        device: Torch device.
    """

    def __init__(self, width, heads=4, kv_heads=1, head_dim=16, window=128,
                 rope_theta=10000.0, device=None):
        super().__init__()
        if heads % kv_heads:
            raise ValueError(f"heads ({heads}) must be divisible by kv_heads ({kv_heads})")
        if head_dim % 2:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        self.width, self.heads, self.kv_heads = width, heads, kv_heads
        self.group = heads // kv_heads
        self.head_dim, self.window = head_dim, window
        self.out_scale = 1.0 / math.sqrt(heads * head_dim)

        self.norm = nn.RMSNorm(width, device=device)
        self.q_proj = nn.Linear(width, heads * head_dim, bias=False, device=device)
        self.k_proj = nn.Linear(width, kv_heads * head_dim, bias=False, device=device)
        self.v_proj = nn.Linear(width, kv_heads * head_dim, bias=False, device=device)
        self.o_proj = nn.Linear(heads * head_dim, width, bias=False, device=device)
        self.q_norm = nn.RMSNorm(head_dim, device=device)
        self.k_norm = nn.RMSNorm(head_dim, device=device)

        with torch.no_grad():
            for proj in (self.q_proj, self.k_proj, self.v_proj):
                nn.init.normal_(proj.weight, std=width ** -0.5)
            nn.init.zeros_(self.o_proj.weight)

        inv = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device,
                                                 dtype=torch.float32) / head_dim))
        self.register_buffer('inv_freq', inv, persistent=False)
        self.reset_cache()

    def reset_cache(self):
        self._k = self._v = None
        self._pos = 0

    def _rope(self, x, start):
        t = x.shape[2]
        pos = torch.arange(start, start + t, device=x.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)
        cos = torch.cos(freqs).to(x.dtype).repeat(1, 2)[None, None]
        sin = torch.sin(freqs).to(x.dtype).repeat(1, 2)[None, None]
        return x * cos + _rotate_half(x) * sin

    def forward(self, h_seq):
        """`(B, T, N) -> (B, T, N)`, the contribution to add to the trajectory."""
        b, t, _ = h_seq.shape
        if self._k is not None and self._k.shape[0] != b:
            self.reset_cache()

        x = self.norm(h_seq)
        shape = lambda z, n: z.view(b, t, n, self.head_dim).transpose(1, 2)
        q = self._rope(self.q_norm(shape(self.q_proj(x), self.heads)), self._pos)
        k = self._rope(self.k_norm(shape(self.k_proj(x), self.kv_heads)), self._pos)
        v = shape(self.v_proj(x), self.kv_heads)
        self._pos += t

        carry = self._k.shape[2] if self._k is not None else 0
        if carry:
            k = torch.cat([self._k, k], dim=2)
            v = torch.cat([self._v, v], dim=2)
        self._k = k[:, :, -self.window:].detach()
        self._v = v[:, :, -self.window:].detach()

        if self.group > 1:
            k = k.repeat_interleave(self.group, dim=1)
            v = v.repeat_interleave(self.group, dim=1)

        if carry:
            # Query i sees every carried key plus keys up to i: one causal mask
            # shifted right by however much carry survived truncation.
            mask = torch.ones(t, carry + t, dtype=torch.bool,
                              device=x.device).tril(diagonal=carry)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).reshape(b, t, self.heads * self.head_dim)
        return self.o_proj(out) * self.out_scale

    def extra_repr(self):
        return (f"width={self.width}, heads={self.heads}, kv_heads={self.kv_heads}, "
                f"head_dim={self.head_dim}, window={self.window}")
