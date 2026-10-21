"""Transformer building blocks.

Multi-axis rotary positions: the rotary dimensions of every head are split between three
position axes:

* ``token``  -- index in the sequence (classic RoPE)
* ``time``   -- musical time in 16th notes. In FIM order the middle comes *after* the suffix
                in the sequence, but musically it sits *between* prefix and suffix; on this
                axis attention sees the true musical distance. Repetition at "4 bars ago"
                is a fixed distance, which makes riff loops easy to attend to.
* ``phase``  -- position inside the bar. Two tokens on the same beat of different bars have
                distance 0, so a head can look at "what happened on this beat in other bars".

All axes are applied to q and k only, so the fast fused attention kernels still work.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class MultiAxisRotary(nn.Module):
    def __init__(self, head_dim: int, axes: dict, theta: float = 10000.0, time_period=(1.5, 4096.0),
                 phase_period=(1.5, 64.0)):
        super().__init__()
        assert head_dim % 2 == 0
        n_pairs = head_dim // 2
        shares = {k: float(v) for k, v in axes.items() if v > 0}
        total = sum(shares.values())
        counts, left = {}, n_pairs
        names = list(shares)
        for i, k in enumerate(names):
            c = left if i == len(names) - 1 else int(round(n_pairs * shares[k] / total))
            counts[k] = c
            left -= c
        self.axes = [k for k in names if counts[k] > 0]
        for k in self.axes:
            c = counts[k]
            if k == "token":
                inv = 1.0 / (theta ** (torch.arange(c, dtype=torch.float32) / c))
            else:
                lo, hi = time_period if k == "time" else phase_period
                periods = torch.exp(torch.linspace(math.log(lo), math.log(hi), c))
                inv = 2 * math.pi / periods
            self.register_buffer(f"inv_{k}", inv, persistent=False)

    def angles(self, pos: dict[str, torch.Tensor]) -> torch.Tensor:
        """pos[axis]: [B, T] -> angles [B, T, head_dim/2]."""
        parts = [pos[k].float().unsqueeze(-1) * getattr(self, f"inv_{k}") for k in self.axes]
        return torch.cat(parts, dim=-1)

    def cos_sin(self, pos: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.angles(pos)
        a = torch.cat([a, a], dim=-1).unsqueeze(1)   # [B, 1, T, D]
        return a.cos(), a.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return (x.float() * cos + rotate_half(x.float()) * sin).to(x.dtype)


class SelfAttention(nn.Module):
    def __init__(self, d: int, n_head: int, attn_dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.n_head = n_head
        self.hd = d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.attn_dropout = attn_dropout
        self.causal = causal

    def _qkv(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.hd).permute(2, 0, 3, 1, 4)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return q, k, v

    def forward(self, x, cos=None, sin=None, key_mask=None, kv_out: list | None = None):
        B, T, D = x.shape
        q, k, v = self._qkv(x, cos, sin)
        if kv_out is not None:
            kv_out.append((k, v))
        mask = None
        if key_mask is not None:  # [B, T] True = attend (bidirectional encoders)
            mask = key_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=self.causal and mask is None,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))

    def decode(self, x, cos, sin, cache: dict):
        """One new token per sequence; the prompt's K/V are shared by all candidates.

        cache: {"pk","pv": [1,H,Tp,hd] prompt, "gk","gv": [B,H,Tmax,hd] generated, "n": int}
        """
        B, T, D = x.shape
        q, k, v = self._qkv(x, cos, sin)
        n = cache["n"]
        cache["gk"][:, :, n:n + T] = k
        cache["gv"][:, :, n:n + T] = v
        gk, gv = cache["gk"][:, :, :n + T], cache["gv"][:, :, :n + T]
        scale = 1.0 / math.sqrt(self.hd)
        s_p = torch.einsum("bhqd,hkd->bhqk", q.float(), cache["pk"][0].float()) * scale
        s_g = torch.einsum("bhqd,bhkd->bhqk", q.float(), gk.float()) * scale
        if T > 1:
            causal = torch.ones(T, n + T, dtype=torch.bool, device=x.device).tril(n)
            s_g = s_g.masked_fill(~causal, float("-inf"))
        p = torch.softmax(torch.cat([s_p, s_g], dim=-1), dim=-1)
        Tp = s_p.shape[-1]
        y = torch.einsum("bhqk,hkd->bhqd", p[..., :Tp], cache["pv"][0].float()) \
            + torch.einsum("bhqk,bhkd->bhqd", p[..., Tp:], gv.float())
        return self.proj(y.to(x.dtype).transpose(1, 2).reshape(B, T, D))


class CrossAttention(nn.Module):
    """Tokens attend to the song memory (one vector per bar of the whole song)."""

    def __init__(self, d: int, d_mem: int, n_head: int, attn_dropout: float = 0.0):
        super().__init__()
        self.n_head = n_head
        self.hd = d // n_head
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d_mem, 2 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))   # starts closed: training begins as the plain decoder
        self.attn_dropout = attn_dropout

    def memory_kv(self, mem):
        B, M, _ = mem.shape
        k, v = self.kv(mem).view(B, M, 2, self.n_head, self.hd).permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x, mem=None, mem_mask=None, kv=None):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k, v = kv if kv is not None else self.memory_kv(mem)
        if k.shape[0] != B:
            k, v = k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)
            mem_mask = mem_mask.expand(B, -1) if mem_mask is not None else None
        mask = mem_mask[:, None, None, :] if mem_mask is not None else None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        return torch.tanh(self.gate) * self.proj(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.w12 = nn.Linear(d, 2 * d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, d: int, n_head: int, d_ff: int, dropout: float = 0.0, attn_dropout: float = 0.0,
                 causal: bool = True, d_mem: int | None = None):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = SelfAttention(d, n_head, attn_dropout, causal)
        self.n2 = RMSNorm(d)
        self.mlp = SwiGLU(d, d_ff)
        self.drop = nn.Dropout(dropout)
        self.cross = None
        if d_mem is not None:
            self.nc = RMSNorm(d)
            self.cross = CrossAttention(d, d_mem, n_head, attn_dropout)

    def forward(self, x, cos=None, sin=None, key_mask=None, mem=None, mem_mask=None, kv_out=None, mem_kv=None):
        x = x + self.drop(self.attn(self.n1(x), cos, sin, key_mask, kv_out))
        if self.cross is not None and (mem is not None or mem_kv is not None):
            x = x + self.drop(self.cross(self.nc(x), mem, mem_mask, mem_kv))
        return x + self.drop(self.mlp(self.n2(x)))

    def decode(self, x, cos, sin, cache, mem_kv=None, mem_mask=None):
        x = x + self.attn.decode(self.n1(x), cos, sin, cache)
        if self.cross is not None and mem_kv is not None:
            x = x + self.cross(self.nc(x), None, mem_mask, mem_kv)
        return x + self.mlp(self.n2(x))
