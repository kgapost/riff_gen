"""RiffGen models: one implementation, three configurations.

baseline  "tab language model": token embeddings + token-index RoPE (README D30, ~85M).
big       "music-aware": + embeddings of what each token *means* musically (pitch from the
          real tuning, pitch class, octave, fret, string-from-bottom, guitar, position in
          bar, beat strength, bar relative to the gap, segment) + multi-axis RoPE
          (token / musical time / position in bar), 8192 context (~260M).
huge      "structure-aware": big + a song memory. A bar encoder turns every bar of the
          *whole song* into a vector, a bidirectional song encoder relates the bars to each
          other (form, repeats, where the gap leads), and the token decoder reads that
          memory through gated cross-attention and a per-bar "plan" vector. Target bars are
          masked in the memory; auxiliary heads must predict their harmony (pitch classes)
          and rhythm (onset grid) from the context -- a plan -- and the section of every
          bar. A contrastive loss teaches the bar encoder that repeated bars (riffs) are
          the same thing (~380M).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from riffgen.dataset.serialize import MAX_FRET_FEAT, N_BAR_REL, N_BEAT, N_PHRASE, N_POS, N_SEG, NO_PITCH
from riffgen.model.config import ModelConfig
from riffgen.model.layers import Block, MultiAxisRotary, RMSNorm
from riffgen.tokens import sections as sections_mod

N_LABELS = len(sections_mod.labels())


class MusicEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d: int, music: bool):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.music = music
        if music:
            self.pitch = nn.Embedding(NO_PITCH + 1, d)
            self.pc = nn.Embedding(13, d)
            self.octave = nn.Embedding(12, d)
            self.fret = nn.Embedding(MAX_FRET_FEAT + 1, d)
            self.sfb = nn.Embedding(9, d)
            self.inst = nn.Embedding(6, d)
            self.pos = nn.Embedding(N_POS, d)
            self.beat = nn.Embedding(N_BEAT, d)
            self.bar_rel = nn.Embedding(N_BAR_REL, d)
            self.seg = nn.Embedding(N_SEG, d)
            self.phrase = nn.Embedding(N_PHRASE, d)

    def forward(self, f: dict) -> torch.Tensor:
        x = self.tok(f["ids"].long())
        if self.music:
            g = lambda k, hi: f[k].long().clamp(0, hi)  # noqa: E731
            pitch = g("pitch", NO_PITCH)
            none = pitch == NO_PITCH
            pc = torch.where(none, torch.full_like(pitch, 12), pitch % 12)
            octv = torch.where(none, torch.full_like(pitch, 11), (pitch // 12).clamp(0, 10))
            x = x + self.pitch(pitch) + self.pc(pc) + self.octave(octv) + self.fret(g("fret", MAX_FRET_FEAT)) \
                + self.sfb(g("sfb", 8)) + self.inst(g("inst", 5)) + self.pos(g("pos", N_POS - 1)) \
                + self.beat(g("beat", N_BEAT - 1))
            if "bar_rel" in f:
                x = x + self.bar_rel(g("bar_rel", N_BAR_REL - 1))
            if "seg" in f:
                x = x + self.seg(g("seg", N_SEG - 1))
            if "phrase" in f:
                x = x + self.phrase(g("phrase", N_PHRASE - 1))
        return x


class SongEncoder(nn.Module):
    """Whole song -> one memory vector per bar (+ planning heads)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        be, se = cfg.bar_encoder, cfg.song_encoder
        self.cfg = cfg
        self.bar_emb = MusicEmbedding(cfg.vocab_size, be.d_model, music=True)
        self.bar_blocks = nn.ModuleList([Block(be.d_model, be.n_head, be.d_ff, cfg.dropout, cfg.attn_dropout,
                                               causal=False) for _ in range(be.n_layer)])
        self.bar_rope = MultiAxisRotary(be.d_model // be.n_head, {"token": 0.5, "phase": 0.5},
                                        phase_period=cfg.phase_period)
        self.bar_norm = RMSNorm(be.d_model)
        self.pool_q = nn.Parameter(torch.randn(be.d_model) * 0.02)
        self.bar_out = nn.Linear(be.d_model, se.d_model, bias=False)
        self.mask_emb = nn.Embedding(4, se.d_model)       # visible / hidden target / unknown / pad
        self.rel_emb = nn.Embedding(129, se.d_model)
        self.tempo_emb = nn.Embedding(41, se.d_model)
        self.tsn_emb = nn.Embedding(33, se.d_model)
        self.tsd_emb = nn.Embedding(6, se.d_model)
        self.phrase_emb = nn.Embedding(N_PHRASE, se.d_model)
        self.song_blocks = nn.ModuleList([Block(se.d_model, se.n_head, se.d_ff, cfg.dropout, cfg.attn_dropout,
                                                causal=False) for _ in range(se.n_layer)])
        self.song_rope = MultiAxisRotary(se.d_model // se.n_head, {"token": 1.0}, theta=1000.0)
        self.song_norm = RMSNorm(se.d_model)
        self.null = nn.Parameter(torch.randn(se.d_model) * 0.02)
        self.head_pc = nn.Linear(se.d_model, 12)
        self.head_onset = nn.Linear(se.d_model, 48)
        self.head_section = nn.Linear(se.d_model, N_LABELS)
        self.proj_z = nn.Linear(be.d_model, 128, bias=False)

    def encode_bars(self, s: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Visible bars -> pooled vectors. Returns (pooled [N, d_bar], index (b, k) of each)."""
        vis = s["mask_kind"] == 0
        b_idx, k_idx = vis.nonzero(as_tuple=True)
        if len(b_idx) == 0:
            d = self.cfg.bar_encoder.d_model
            return s["bar_ids"].new_zeros((0, d), dtype=torch.float), b_idx, k_idx
        f = {"ids": s["bar_ids"][b_idx, k_idx]}
        for k in ("inst", "pitch", "fret", "sfb", "beat", "pos"):
            f[k] = s[f"bar_{k}"][b_idx, k_idx]
        lengths = s["bar_len"][b_idx, k_idx].clamp(min=1)
        Lb = f["ids"].shape[1]
        ar = torch.arange(Lb, device=lengths.device)
        key_mask = ar[None, :] < lengths[:, None]
        x = self.bar_emb(f)
        pos = {"token": ar[None, :].expand(len(lengths), -1), "phase": f["pos"].float() * (40.0 / 240.0)}
        cos, sin = self.bar_rope.cos_sin(pos)
        for blk in self.bar_blocks:
            if self.cfg.grad_checkpointing and self.training:
                x = checkpoint(blk, x, cos, sin, key_mask, use_reentrant=False)
            else:
                x = blk(x, cos, sin, key_mask)
        x = self.bar_norm(x)
        att = (x.float() @ self.pool_q.float()) / math.sqrt(x.shape[-1])
        att = att.masked_fill(~key_mask, float("-inf")).softmax(-1)
        pooled = torch.einsum("nl,nld->nd", att, x.float())
        return pooled, b_idx, k_idx

    def forward(self, s: dict) -> dict:
        B, NB = s["mask_kind"].shape
        pooled, b_idx, k_idx = self.encode_bars(s)
        se = self.cfg.song_encoder
        h = self.mask_emb(s["mask_kind"].clamp(0, 3)) + self.rel_emb(s["rel"].clamp(0, 128)) \
            + self.tempo_emb(s["tempo"].clamp(0, 40)) + self.tsn_emb(s["ts_num"].clamp(0, 32)) \
            + self.tsd_emb(s["ts_den"].clamp(0, 5))
        if "phrase" in s:
            h = h + self.phrase_emb(s["phrase"].clamp(0, N_PHRASE - 1))
        if len(b_idx):
            h = h.index_put((b_idx, k_idx), h[b_idx, k_idx] + self.bar_out(pooled).to(h.dtype))
        valid = s["mask_kind"] != 3
        pos = torch.arange(NB, device=h.device)[None].expand(B, -1)
        cos, sin = self.song_rope.cos_sin({"token": pos})
        for blk in self.song_blocks:
            if self.cfg.grad_checkpointing and self.training:
                h = checkpoint(blk, h, cos, sin, valid, use_reentrant=False)
            else:
                h = blk(h, cos, sin, valid)
        h = self.song_norm(h)
        mem = torch.cat([self.null.to(h.dtype)[None, None].expand(B, 1, se.d_model), h], dim=1)
        mem_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=h.device), valid], dim=1)
        return {"h": h, "mem": mem, "mem_mask": mem_mask, "pooled": pooled, "b_idx": b_idx, "k_idx": k_idx}

    def aux_losses(self, s: dict, out: dict) -> dict:
        h = out["h"].float()
        losses = {}
        m = s["aux_mask"]
        if m.any():
            losses["pc"] = F.binary_cross_entropy_with_logits(self.head_pc(h[m]), s["aux_pc"][m])
            losses["onset"] = F.binary_cross_entropy_with_logits(self.head_onset(h[m]), s["aux_onset"][m])
        sec = s["section"]
        sm = (sec >= 0) & (s["mask_kind"] != 3)
        if sm.any():
            losses["section"] = F.cross_entropy(self.head_section(h[sm]), sec[sm])
        pairs = s["pairs"]
        if len(pairs) and len(out["pooled"]):
            # index of each visible bar in the pooled list
            B, NB = s["mask_kind"].shape
            lookup = torch.full((B, NB), -1, dtype=torch.long, device=h.device)
            lookup[out["b_idx"], out["k_idx"]] = torch.arange(len(out["b_idx"]), device=h.device)
            a = lookup[pairs[:, 0], pairs[:, 1]]
            p = lookup[pairs[:, 0], pairs[:, 2]]
            ok = (a >= 0) & (p >= 0)
            if ok.any():
                z = F.normalize(self.proj_z(out["pooled"].to(self.proj_z.weight.dtype)).float(), dim=-1)
                a, p = a[ok][:512], p[ok][:512]
                logits = z[a] @ z.T / 0.1
                logits[torch.arange(len(a), device=h.device), a] = float("-inf")   # not itself
                losses["contrastive"] = F.cross_entropy(logits, p)
        return losses


class RiffGenModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.emb = MusicEmbedding(cfg.vocab_size, d, cfg.music_features)
        self.drop = nn.Dropout(cfg.dropout)
        d_mem = cfg.song_encoder.d_model if cfg.hierarchical else None
        self.blocks = nn.ModuleList([
            Block(d, cfg.n_head, cfg.d_ff, cfg.dropout, cfg.attn_dropout, causal=True,
                  d_mem=d_mem if (cfg.hierarchical and (i + 1) % cfg.cross_attn_every == 0) else None)
            for i in range(cfg.n_layer)])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.emb.tok.weight
        self.rope = MultiAxisRotary(cfg.head_dim, cfg.rope_axes, cfg.rope_theta, cfg.time_period, cfg.phase_period)
        if cfg.hierarchical:
            self.song = SongEncoder(cfg)
            self.plan_proj = nn.Linear(cfg.song_encoder.d_model, d, bias=False)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w3.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))
        if cfg.hierarchical:
            nn.init.zeros_(self.plan_proj.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    # ------------------------------------------------------------------
    def positions(self, f: dict, offset: int = 0) -> dict:
        B, T = f["ids"].shape
        pos = {"token": (torch.arange(T, device=f["ids"].device) + offset)[None].expand(B, -1)}
        if "time" in self.rope.axes:
            pos["time"] = f["time"]
        if "phase" in self.rope.axes:
            pos["phase"] = f["phase"]
        return pos

    def _plan(self, f: dict, song_out: dict) -> torch.Tensor:
        """Memory vector of each token's own bar (zeros for header tokens)."""
        tok_bar = f["song"]["tok_bar"] if "song" in f else f["tok_bar"]
        h = song_out["h"]
        idx = tok_bar.clamp(min=0)
        g = torch.gather(h, 1, idx[..., None].expand(-1, -1, h.shape[-1]))
        return g * (tok_bar >= 0)[..., None].to(g.dtype)

    def hidden(self, f: dict, song_out: dict | None = None, kv_out: list | None = None) -> torch.Tensor:
        x = self.emb(f)
        mem = mem_mask = None
        if self.cfg.hierarchical:
            if song_out is None:
                song_out = self.song(f["song"])
            x = x + self.plan_proj(self._plan(f, song_out).to(x.dtype))
            mem, mem_mask = song_out["mem"], song_out["mem_mask"]
        x = self.drop(x)
        cos, sin = self.rope.cos_sin(self.positions(f))
        for blk in self.blocks:
            if self.cfg.grad_checkpointing and self.training and kv_out is None:
                x = checkpoint(blk, x, cos, sin, None, mem, mem_mask, use_reentrant=False)
            else:
                x = blk(x, cos, sin, None, mem, mem_mask, kv_out)
        return self.norm(x)

    def forward(self, f: dict, per_token: bool = False) -> dict:
        song_out = self.song(f["song"]) if self.cfg.hierarchical else None
        x = self.hidden(f, song_out)
        targets = f["ids"][:, 1:]
        w = f["loss"][:, 1:]
        sel = w > 0
        h = x[:, :-1][sel]
        logits = self.lm_head(h).float()
        nll = F.cross_entropy(logits, targets[sel], reduction="none")
        out = {"loss_sum": (nll * w[sel]).sum(), "weight_sum": w[sel].sum(), "n_tokens": sel.sum()}
        if per_token:
            out["nll"] = nll.detach()
            out["targets"] = targets[sel]
            out["pred"] = logits.argmax(-1).detach()
            out["logits"] = logits.detach()
            out["seg"] = f["seg"][:, 1:][sel] if "seg" in f else None
        if self.cfg.hierarchical:
            out["aux"] = self.song.aux_losses(f["song"], song_out)
        return out

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def prefill(self, f: dict, n_cand: int, max_new: int) -> dict:
        """Run the prompt once (batch 1) and set up a K/V cache shared by n_cand candidates."""
        song_out = self.song(f["song"]) if self.cfg.hierarchical else None
        kv: list = []
        x = self.hidden(f, song_out, kv_out=kv)
        caches = []
        for (k, v) in kv:
            H, hd = k.shape[1], k.shape[3]
            caches.append({"pk": k, "pv": v, "n": 0,
                           "gk": torch.zeros(n_cand, H, max_new, hd, device=x.device, dtype=k.dtype),
                           "gv": torch.zeros(n_cand, H, max_new, hd, device=x.device, dtype=v.dtype)})
        state = {"caches": caches, "Tp": f["ids"].shape[1], "last_hidden": x[:, -1]}
        if self.cfg.hierarchical:
            state["song_h"] = song_out["h"]
            state["mem_kv"] = {i: blk.cross.memory_kv(song_out["mem"]) for i, blk in enumerate(self.blocks)
                               if blk.cross is not None}
            state["mem_mask"] = song_out["mem_mask"]
            state["bar_lo"] = int(f["song"]["bar_lo"][0])
        return state

    def prompt_logits(self, state: dict) -> torch.Tensor:
        return self.lm_head(state["last_hidden"]).float()

    @torch.no_grad()
    def decode_step(self, state: dict, f: dict) -> torch.Tensor:
        """f: features of the new token of every candidate, each [B, 1]. Returns logits [B, V]."""
        x = self.emb(f)
        if self.cfg.hierarchical:
            h = state["song_h"]
            local = (f["abs_bar"] - state["bar_lo"]).clamp(0, h.shape[1] - 1)
            plan = h[0, local[:, 0]][:, None]
            x = x + self.plan_proj(plan.to(x.dtype))
        n = state["caches"][0]["n"]
        pos = self.positions(f, offset=state["Tp"] + n)
        cos, sin = self.rope.cos_sin(pos)
        for i, blk in enumerate(self.blocks):
            mem_kv = state.get("mem_kv", {}).get(i)
            x = blk.decode(x, cos, sin, state["caches"][i], mem_kv,
                           state["mem_mask"] if mem_kv is not None else None)
        for c in state["caches"]:
            c["n"] += 1
        return self.lm_head(self.norm(x)[:, -1]).float()

    @torch.no_grad()
    def bar_embeddings(self, f: dict) -> torch.Tensor:
        """Mean final hidden state per example (style vectors for §3.9 option B)."""
        x = self.hidden(f)
        m = (f["loss"] > 0).float() if "loss" in f else torch.ones_like(f["ids"], dtype=torch.float)
        return (x.float() * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)


def build_model(cfg: ModelConfig) -> RiffGenModel:
    return RiffGenModel(cfg)
