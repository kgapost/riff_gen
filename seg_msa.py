"""Music structure analysis (MSA) experiments on top of riff.py's section segmenter (v3), compared on the same split.

    python seg_msa.py pretrain --run runs/msa/pre                       # Idea 2, stage 1: masked-bar pretraining
    python seg_msa.py train --run runs/msa/crf                          # Idea 2, stage 2: semi-Markov CRF
    python seg_msa.py train --run runs/msa/crf_pre --init runs/msa/pre/msa.pt
    python seg_msa.py train --run runs/msa/crf_pre_con --init runs/msa/pre/msa.pt --supcon 0.2   # + Idea 1 metric
    python seg_msa.py compare runs/segment/seg.pt runs/msa/crf/msa.pt ...   # + Idea 3 section bank, vs v3

The backbone is v3's: a bar is a bag of its tokens plus 40 bar features; a bidirectional transformer over the bars
with the bar self-similarity added to the attention scores. What is new:

- **Semi-Markov CRF** (Idea 2): the song is a sequence of sections (label, length). States are (label, bars so far,
  capped at R); a section starts with the bar's start score and a learned label->label transition, every bar adds its
  label score, and learned per-label duration scores decide when it ends. Trained end to end with the exact CRF
  likelihood (the forward algorithm over the gold lattice vs all lattices; bars whose label is unknown - 'other', or
  before the first marker - are summed over), decoded jointly with Viterbi: boundaries and labels in one pass.
  Posterior start probabilities (d logZ / d start score) give a second, threshold-tuned boundary read-out.
- **Masked-bar pretraining** (Idea 2, stage 1) on every tab but the val/test songs (with the 33k unmarked ones): 15 % of
  the bars are hidden; the model predicts their pitch classes, onset grid and density, and picks the hidden bar out of
  all hidden bars of the batch (InfoNCE; every identical bar of the same song is a positive): repetition, phrase
  continuation and texture without labels.
- **Label-aware bar metric** (Idea 1, as an auxiliary loss): supervised contrastive loss on bar embeddings, positives
  = same function, weighted up when they are the same section instance.
- **Section bank** (Idea 3): every labelled section of the training tabs, mean-pooled from the model's contextual bar
  states, is a prototype; a decoded section is relabelled by its nearest prototypes (fused with the model's own label
  scores, weight and temperature tuned on validation).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import riff as R
from riff import OTHER, SECTIONS, log

NEG = -1e9


# ---------------------------------------------------------------------------------------------------------- data

def item(ds: R.SegSet, sid, rng=None, max_bars=512) -> dict:
    """riff.seg_item ('other' bars unlabeled) plus the bar hashes (identical bars) for the contrastive losses."""
    view = ds.c.view(sid)
    lo = int(rng.integers(0, view.nb - max_bars + 1)) if rng is not None and view.nb > max_bars else 0
    hi = min(view.nb, lo + max_bars) if rng is not None else view.nb
    x = R.seg_item(ds.c.vocab, view, lo, hi, rng=rng, named=True)
    x["hash"] = np.asarray(view.f["hash"])[lo:hi].view(np.int64)
    return x


def batches(ds, ids, budget, rng=None):
    nbs = np.array([min(ds.c.songs[i]["nb"], 512 if rng is not None else 4096) for i in ids])
    order = np.argsort(nbs, kind="stable")
    chunks, cur = [], []
    for k in order:
        if cur and nbs[k] * (len(cur) + 1) > budget:
            chunks.append(cur)
            cur = []
        cur.append(ids[k])
    chunks += [cur] if cur else []
    if rng is not None:
        chunks = [chunks[i] for i in rng.permutation(len(chunks))]
    for ch in chunks:
        items = [item(ds, i, rng) for i in ch]
        b = R.seg_collate(items)
        b["hash"] = torch.zeros(b["mask"].shape, dtype=torch.long)
        for k, x in enumerate(items):
            b["hash"][k, :len(x["hash"])] = torch.as_tensor(x["hash"])
        yield b, int(sum(len(x["sec"]) for x in items)), ch


# ---------------------------------------------------------------------------------------------------------- model

class MSAModel(nn.Module):
    """v3's backbone + a semi-Markov CRF over (label, bars so far) + pretraining and metric heads."""

    def __init__(self, V, labels, d=256, layers=6, runs=32, drop=0.1):
        super().__init__()
        self.cfg = {"d": d, "layers": layers, "labels": list(labels), "runs": runs}
        K = len(labels)
        self.bag = nn.EmbeddingBag(2 * V.size, d, mode="mean")
        self.fin = nn.Sequential(nn.Linear(R.SEG_NF, d), nn.SiLU(), nn.Linear(d, d))
        self.body = R.Stack(d, layers, drop)
        self.ssm_w = nn.Parameter(torch.ones(d // 64))
        self.mask_emb = nn.Parameter(torch.zeros(d))
        self.lab, self.start = nn.Linear(d, K), nn.Linear(d, 1)
        self.trans, self.s0 = nn.Parameter(torch.zeros(K, K)), nn.Parameter(torch.zeros(K))
        self.dcont, self.dend = nn.Parameter(torch.zeros(K, runs)), nn.Parameter(torch.zeros(K, runs))
        self.pre = nn.Linear(d, 29)                          # hidden bar: 12 pitch classes, 16 onsets, density
        self.proj, self.proj_t = nn.Linear(d, 128), nn.Linear(d, 128)
        self.con = nn.Linear(d, 128)                         # the label-aware bar metric

    def encode(self, b, hide=None):
        B, T = b["mask"].shape
        e = self.bag(b["tok"], b["off"])
        x = torch.zeros(B * T, e.shape[1], device=e.device, dtype=e.dtype).index_copy(0, b["where"], e).view(B, T, -1)
        x = x + self.fin(b["feats"])
        ssm = b["ssm"]
        if hide is not None:                                  # hidden bars: no content, no similarity row
            x_in = torch.where(hide[..., None], self.mask_emb.expand_as(x), x)
            keep = (~hide).float()
            ssm = ssm * keep[:, :, None] * keep[:, None, :]
        else:
            x_in = x
        mask = (self.ssm_w.view(1, -1, 1, 1) * ssm[:, None]).masked_fill(~b["mask"][:, None, None], float("-inf"))
        h = self.body(x_in, torch.arange(T, device=e.device), mask=mask)
        return h, x

    def scores(self, h):
        return self.lab(h).float(), self.start(h)[..., 0].float()


def crf_logz(m: MSAModel, u, a, lens, allow=None):
    """log Z of the semi-Markov CRF (states: label x bars so far, capped). u (B,T,K) label scores, a (B,T) start
    scores; ``allow`` (B,T,K,R) restricts the lattice (the gold one). Bar 0 always starts a section."""
    B, T, K = u.shape
    Rn = m.cfg["runs"]
    tr, de, dc = m.trans, m.dend, m.dcont
    alpha = torch.full((B, K, Rn), NEG, device=u.device)
    alpha[:, :, 0] = m.s0 + u[:, 0] + a[:, :1]
    if allow is not None:
        alpha = alpha.masked_fill(~allow[:, 0], NEG)
    for i in range(1, T):
        leave = torch.logsumexp(alpha + de, 2)                                   # (B,K) a section ends
        st = torch.logsumexp(leave[:, :, None] + tr, 1) + a[:, i, None]          # (B,K) the next starts
        cont = alpha[:, :, :-1] + dc[:, :-1]
        last = torch.logaddexp(cont[:, :, -1], alpha[:, :, -1] + dc[:, -1])
        new = torch.cat([st[:, :, None], cont[:, :, :-1], last[:, :, None]], 2) + u[:, i, :, None]
        if allow is not None:
            new = new.masked_fill(~allow[:, i], NEG)
        alpha = torch.where((i < lens)[:, None, None], new, alpha)
    return torch.logsumexp((alpha + de).flatten(1), 1)


def gold_allow(m: MSAModel, b, remap):
    """The gold lattice: a known label fixes the label; a marker (or bar 0) means 'a section starts here', no marker
    means 'it goes on'; unknown labels ('other', before the first marker) are free."""
    sec, bnd, mask = b["sec"], b["bnd"] > 0, b["mask"]
    B, T = sec.shape
    K, Rn = len(m.cfg["labels"]), m.cfg["runs"]
    lab = remap[sec.clamp(min=0)]
    known = (sec >= 0) & (lab >= 0)
    labok = torch.ones(B, T, K, dtype=torch.bool, device=sec.device)
    labok[known] = F.one_hot(lab[known], K).bool()
    start = bnd.clone()
    start[:, 0] = True
    r0 = torch.zeros(Rn, dtype=torch.bool, device=sec.device)
    r0[0] = True
    runok = torch.where(start[..., None], r0, ~r0)                               # (B,T,R)
    allow = labok[..., None] & runok[:, :, None, :]
    return allow | ~mask[:, :, None, None]


def supcon(z, lab, inst, w_inst=1.0, tau=0.1, n=2048):
    """Supervised contrastive loss: positives share the label; the same section instance counts (1 + w_inst) x."""
    if len(z) > n:
        sel = torch.randperm(len(z), device=z.device)[:n]
        z, lab, inst = z[sel], lab[sel], inst[sel]
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / tau
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, NEG)
    w = ((lab[:, None] == lab[None]) & ~eye).float() * (1 + w_inst * (inst[:, None] == inst[None]).float())
    ok = w.sum(1) > 0
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    return (-(w * logp).sum(1)[ok] / w.sum(1)[ok]).mean() if ok.any() else z.sum() * 0


def pretrain_loss(m: MSAModel, b, rng_t):
    mask = b["mask"]
    hide = (torch.rand(mask.shape, generator=rng_t, device="cpu").to(mask.device) < 0.15) & mask
    if not hide.any():
        return None
    h, x = m.encode(b, hide)
    hv, xv = h[hide], x[hide]
    tgt = b["feats"][hide]
    p = m.pre(hv).float()
    rec = F.binary_cross_entropy_with_logits(p[:, :28], tgt[:, :28]) + F.mse_loss(p[:, 28], tgt[:, 28])
    song = torch.arange(mask.shape[0], device=mask.device)[:, None].expand_as(mask)[hide]
    hsh = b["hash"][hide]
    sel = torch.randperm(len(hv), device=hv.device)[:4096]
    zp, zt = F.normalize(m.proj(hv[sel]).float(), dim=-1), F.normalize(m.proj_t(xv[sel]).float(), dim=-1)
    sim = zp @ zt.T / 0.1
    pos = (song[sel][:, None] == song[sel][None]) & (hsh[sel][:, None] == hsh[sel][None])
    nce = (torch.logsumexp(sim, 1) - torch.logsumexp(sim.masked_fill(~pos, NEG), 1)).mean()
    with torch.no_grad():
        acc = float((sim.argmax(1)[:, None] == torch.arange(len(sel), device=sim.device)[:, None]).float().mean()
                    if len(sel) else 0)
        top1 = float(pos[torch.arange(len(sel)), sim.argmax(1)].float().mean())
    return rec + nce, {"rec": float(rec), "nce": float(nce), "top1": top1, "exact": acc}


def train_loss(m: MSAModel, b, remap, cls_w, supcon_w):
    h, _ = m.encode(b)
    u, a = m.scores(h)
    mask = b["mask"]
    lens = mask.sum(1)
    allow = gold_allow(m, b, remap)
    both = crf_logz(m, torch.cat([u, u]), torch.cat([a, a]), torch.cat([lens, lens]),
                    torch.cat([allow, torch.ones_like(allow)]))
    n = mask.sum()
    crf = (both[len(u):] - both[:len(u)]).sum() / n
    tgt = torch.where(b["sec"] >= 0, remap[b["sec"].clamp(min=0)], -1)
    ok = tgt >= 0
    ce = F.cross_entropy(u[ok], tgt[ok], weight=cls_w, label_smoothing=0.05) if ok.any() else u.sum() * 0
    bce = (F.binary_cross_entropy_with_logits(a, b["bnd"], reduction="none",
                                              pos_weight=torch.tensor(8.0, device=a.device)) * mask).sum() / n
    loss, st = crf + ce + bce, {"crf": float(crf), "ce": float(ce), "bce": float(bce)}
    if supcon_w and ok.any():
        inst = torch.cumsum(b["bnd"], 1).long() + 100000 * torch.arange(len(u), device=a.device)[:, None]
        sc = supcon(m.con(h[ok]).float(), tgt[ok], inst[ok])
        loss = loss + supcon_w * sc
        st["supcon"] = float(sc)
    return loss, st


# ---------------------------------------------------------------------------------------------------------- decoding

def crf_viterbi(m: MSAModel, u, a, lens, allowed_labels=None):
    """Joint boundaries + labels: the best (label, length) sequence per song. -> [(labels, starts)]"""
    B, T, K = u.shape
    Rn = m.cfg["runs"]
    tr, de, dc = m.trans.detach(), m.dend.detach(), m.dcont.detach()
    if allowed_labels is not None:
        u = u.masked_fill(~allowed_labels, NEG)
    alpha = torch.full((B, K, Rn), NEG, device=u.device)
    alpha[:, :, 0] = m.s0.detach() + u[:, 0] + a[:, :1]
    bk = torch.zeros(B, T, K, dtype=torch.long, device=u.device)
    br = torch.zeros(B, T, K, dtype=torch.long, device=u.device)
    bl = torch.zeros(B, T, K, dtype=torch.bool, device=u.device)
    for i in range(1, T):
        lv, lr_ = (alpha + de).max(2)
        cand = lv[:, :, None] + tr
        st, kk = cand.max(1)
        bk[:, i], br[:, i] = kk, lr_.gather(1, kk)
        cont = alpha[:, :, :-1] + dc[:, :-1]
        stay = alpha[:, :, -1] + dc[:, -1]
        bl[:, i] = stay > cont[:, :, -1]
        new = torch.cat([(st + a[:, i, None])[:, :, None], cont[:, :, :-1],
                         torch.maximum(cont[:, :, -1], stay)[:, :, None]], 2) + u[:, i, :, None]
        alpha = torch.where((i < lens)[:, None, None], new, alpha)
    end = (alpha + de).flatten(1).argmax(1)
    bk, br, bl, lens = bk.cpu().numpy(), br.cpu().numpy(), bl.cpu().numpy(), lens.cpu().numpy()
    out = []
    for s in range(B):
        n = int(lens[s])
        lab, r = divmod(int(end[s]), Rn)
        labels, starts = np.empty(n, np.int64), {0}
        for i in range(n - 1, -1, -1):
            labels[i] = lab
            if i == 0:
                break
            if r == 0:
                starts.add(i)
                lab, r = int(bk[s, i, lab]), int(br[s, i, lab])
            elif r < Rn - 1:
                r -= 1
            else:
                r = Rn - 1 if bl[s, i, lab] else Rn - 2
        out.append((labels, starts))
    return out


@torch.no_grad()
def predict_msa(m: MSAModel, ds, ids, dev, micro, allowed=None, marg=True, hidden=False) -> dict:
    """{sid: (labels, starts, start posteriors, label posteriors (n,K), hidden states)} with labels in m's label ids."""
    m.eval()
    out = {}
    for b, _, ch in batches(ds, ids, micro):
        b = R.to_dev(b, dev)
        h, _ = m.encode(b)
        u, a = m.scores(h)
        lens = b["mask"].sum(1)
        al = None
        if allowed is not None:
            al = torch.zeros(u.shape[-1], dtype=torch.bool, device=dev)
            al[allowed] = True
        dec = crf_viterbi(m, u, a, lens, al)
        pa = pu = None
        if marg:
            with torch.enable_grad():
                ua = (u.masked_fill(~al, NEG) if al is not None else u).detach().requires_grad_(True)
                aa = a.detach().requires_grad_(True)
                z = crf_logz(m, ua, aa, lens).sum()
                ga, gu = torch.autograd.grad(z, [aa, ua])
            pa, pu = ga.cpu().numpy(), gu.cpu().numpy()
        hs = h.half().cpu().numpy() if hidden else None
        for k, sid in enumerate(ch):
            n = int(lens[k])
            out[sid] = (dec[k][0], dec[k][1], pa[k, :n] if marg else None, pu[k, :n] if marg else None,
                        hs[k, :n] if hidden else None)
    return out


@torch.no_grad()
def predict_v3(model, ds, ids, dev, micro, hidden=False) -> dict:
    """riff.py's SegModel: {sid: (section probs, border probs, hidden states)}."""
    model.eval()
    out = {}
    for b, _, ch in batches(ds, ids, micro):
        b = R.to_dev(b, dev)
        B, T = b["mask"].shape
        e = model.bag(b["tok"], b["off"])
        x = torch.zeros(B * T, e.shape[1], device=dev).index_copy(0, b["where"], e).view(B, T, -1)
        mask = (model.ssm_w.view(1, -1, 1, 1) * b["ssm"][:, None]).masked_fill(~b["mask"][:, None, None], float("-inf"))
        h = model.body(x + model.fin(b["feats"]), torch.arange(T, device=dev), mask=mask)
        ls, lb = model(b)
        ps, pb = torch.softmax(ls.float(), -1).cpu().numpy(), torch.sigmoid(lb.float()).cpu().numpy()
        hs = h.half().cpu().numpy() if hidden else None
        for k, sid in enumerate(ch):
            n = int(b["mask"][k].sum())
            out[sid] = (ps[k, :n], pb[k, :n], hs[k, :n] if hidden else None)
    return out


# ---------------------------------------------------------------------------------------------------------- scoring

def truth(ds, sid):
    sec, bnd = R.seg_targets(ds.c.view(sid).f["sec"])
    return sec, set(np.where(bnd > 0)[0].tolist())


def score(ds, ids, dec: dict, peaks: dict | None = None) -> dict:
    """Named-bar accuracy and macro F1, pairwise-frame F (label grouping, named bars), section starts P/R/F within
    0/1/2 bars (from the decoded sections, and from ``peaks`` when given). dec: {sid: (labels, starts)} in SECTIONS ids."""
    conf = np.zeros((len(SECTIONS), len(SECTIONS)), np.int64)
    hits = {f"{k}_{t}": np.zeros(3, np.int64) for k in ("dec", "peaks") for t in (0, 1, 2)}
    pw = np.zeros(3)                                         # both-same, pred-same, gt-same pairs
    for sid in ids:
        sec, tb = truth(ds, sid)
        lab, st = dec[sid]
        ok = (sec >= 0) & (sec != OTHER)
        np.add.at(conf, (sec[ok], lab[ok]), 1)
        c = Counter(zip(sec[ok].tolist(), lab[ok].tolist()))
        pair = lambda v: v * (v - 1) / 2  # noqa: E731
        pw += [sum(pair(v) for v in c.values()), sum(pair(v) for v in Counter(lab[ok].tolist()).values()),
               sum(pair(v) for v in Counter(sec[ok].tolist()).values())]
        for t in (0, 1, 2):
            hits[f"dec_{t}"] += R.border_prf(st, tb, t)
            if peaks is not None:
                hits[f"peaks_{t}"] += R.border_prf(peaks[sid], tb, t)
    tp, fp, fn = np.diag(conf), conf.sum(0) - np.diag(conf), conf.sum(1) - np.diag(conf)
    f1 = [2 * tp[i] / max(2 * tp[i] + fp[i] + fn[i], 1) for i in range(len(SECTIONS)) if i != OTHER and conf[i].sum()]
    P, Rc = pw[0] / max(pw[1], 1), pw[0] / max(pw[2], 1)
    out = {"named_acc": float(np.trace(conf) / max(conf.sum(), 1)), "macro_f1": float(np.mean(f1)),
           "pairwise_F": float(2 * P * Rc / max(P + Rc, 1e-9))}
    for k, (h, p, t) in hits.items():
        if p or k.startswith("dec"):
            Pk, Rk = h / max(p, 1), h / max(t, 1)
            out[f"starts_{k}"] = {"P": round(Pk, 3), "R": round(Rk, 3), "F": round(2 * Pk * Rk / max(Pk + Rk, 1e-9), 3)}
    return out


def tune_peaks(ds, ids, post: dict) -> float:
    """Threshold of the start posteriors with the best start F1 within 1 bar on these songs."""
    best = (-1, 0.5)
    for t in R.TAUS:
        h = sum((np.array(R.border_prf(R.peak_borders(post[s], t), truth(ds, s)[1], 1)) for s in ids), np.zeros(3))
        P, Rc = h[0] / max(h[1], 1), h[0] / max(h[2], 1)
        best = max(best, (2 * P * Rc / max(P + Rc, 1e-9), t))
    return best[1]


# ---------------------------------------------------------------------------------------------------------- bank

def bank_from(ds, ids, hidden: dict, mapping) -> tuple[np.ndarray, np.ndarray]:
    """Idea 3's section bank: every labelled training section, mean-pooled and normalised, with its label."""
    vecs, labs = [], []
    for sid in ids:
        sec, tb = truth(ds, sid)
        edges = sorted(tb | {0}) + [len(sec)]
        h = hidden[sid]
        for s, e in zip(edges[:-1], edges[1:]):
            lab = mapping[sec[s]] if sec[s] >= 0 else -1
            if lab >= 0 and lab != OTHER:
                v = h[s:e].astype(np.float32).mean(0)
                vecs.append(v / (np.linalg.norm(v) + 1e-9))
                labs.append(lab)
    return np.stack(vecs).astype(np.float32), np.array(labs)


def bank_relabel(dec: dict, hidden: dict, own: dict, bank, lam: float, temp: float, k=32, dev="cpu") -> dict:
    """Relabel every decoded section: argmax over labels of log p_model(section) + lam * log p_bank(section), where
    p_bank is the softmax-weighted vote of the k nearest prototypes. own[sid]: (n, |SECTIONS|) label probs per bar."""
    vecs, labs = torch.as_tensor(bank[0], device=dev), torch.as_tensor(bank[1], device=dev)
    out = {}
    segs, keys = [], []
    for sid, (lab, st) in dec.items():
        edges = sorted(st) + [len(lab)]
        for s, e in zip(edges[:-1], edges[1:]):
            v = hidden[sid][s:e].astype(np.float32).mean(0)
            segs.append(v / (np.linalg.norm(v) + 1e-9))
            keys.append((sid, s, e))
    q = torch.as_tensor(np.stack(segs), device=dev)
    pk = []
    for i in range(0, len(q), 4096):
        sim = q[i:i + 4096] @ vecs.T
        top, idx = sim.topk(k, 1)
        w = torch.softmax(top / temp, 1)
        votes = torch.zeros(len(top), len(SECTIONS), device=dev).scatter_add_(1, labs[idx], w)
        pk.append(votes.cpu().numpy())
    pk = np.concatenate(pk)
    for (sid, s, e), v in zip(keys, pk):
        if sid not in out:
            out[sid] = (dec[sid][0].copy(), dec[sid][1])
        pm = own[sid][s:e].mean(0)
        sc = np.log(pm + 1e-6) + lam * np.log(v + 1e-3)
        sc[OTHER] = -1e9
        sc[pm <= 1e-6] = -1e9                              # labels the decoding may not use stay unused
        out[sid][0][s:e] = int(sc.argmax())
    return out


# ---------------------------------------------------------------------------------------------------------- commands

def load_msa(path, V, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = MSAModel(V, c["labels"], c["d"], c["layers"], c["runs"]).to(dev)
    m.load_state_dict(ck["model"])
    return m, ck


def cmd_pretrain(a):
    dev = torch.device("cuda")
    ds = R.SegSet(a.data)
    ids = list(ds.ids["train"]) + list(ds.unmarked)          # never the val/test songs
    labels = [i for i in range(len(SECTIONS)) if i != OTHER]
    m = MSAModel(ds.c.vocab, labels, a.d, a.layers).to(dev)
    nb = sum(ds.c.songs[i]["nb"] for i in ids)
    M = 72 * a.layers * a.d ** 2 + 6 * a.layers * a.d * 110
    C = M * a.epochs * nb
    lr, batch = min(0.3118 * C ** -0.125, 3e-3), int(0.2920 * C ** 0.3271)
    log(f"pretraining on {len(ids)} songs ({nb} bars), {a.epochs} epochs, lr {lr:.2e}, {batch} bars per step")
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    total, seen, t0 = a.epochs * nb, 0, time.time()
    g = torch.Generator().manual_seed(a.seed)
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        m.train()
        acc, stats, te = 0, Counter(), time.time()
        for b, n, _ in batches(ds, ids, a.micro, R.rng_for(a.seed, ep)):
            r = pretrain_loss(m, R.to_dev(b, dev), g)
            if r is None:
                continue
            loss, st = r
            (loss * n / batch).backward()
            stats.update({k: v * n for k, v in st.items()})
            stats["n"] += n
            acc += n
            if acc >= batch:
                seen += acc
                for gr in opt.param_groups:
                    gr["lr"] = R.lr_at(seen / total, lr)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                acc = 0
        log(f"epoch {ep + 1}: {time.time() - te:.0f}s " + " ".join(f"{k} {v / stats['n']:.3f}" for k, v in stats.items()
                                                                 if k != "n"))
        R.save_ckpt(run / "msa.pt", {"model": m.state_dict(), "cfg": m.cfg, "epoch": ep + 1, "pretrain": True})
    log(f"pretrained in {(time.time() - t0) / 60:.1f} min -> {run / 'msa.pt'}")


def cmd_train(a):
    dev = torch.device("cuda")
    ds = R.SegSet(a.data)
    used = np.zeros(len(SECTIONS), bool)
    for i in ds.ids["train"]:
        s = np.asarray(ds.c.view(i).f["sec"])
        used[s[s >= 0]] = True
    labels = [i for i in np.where(used)[0].tolist() if i != OTHER]
    if a.init:
        m, ck = load_msa(a.init, ds.c.vocab, dev)
        assert m.cfg["labels"] == labels or ck.get("pretrain"), "label sets differ"
        if m.cfg["labels"] != labels:                       # pretraining used every named section: re-head
            m.cfg["labels"] = labels
            K = len(labels)
            m.lab = nn.Linear(a.d, K).to(dev)
            m.trans, m.s0 = nn.Parameter(torch.zeros(K, K, device=dev)), nn.Parameter(torch.zeros(K, device=dev))
            m.dcont = nn.Parameter(torch.zeros(K, m.cfg["runs"], device=dev))
            m.dend = nn.Parameter(torch.zeros(K, m.cfg["runs"], device=dev))
        log(f"initialised from {a.init}")
    else:
        m = MSAModel(ds.c.vocab, labels, a.d, a.layers).to(dev)
    remap = torch.full((len(SECTIONS),), -1, dtype=torch.long, device=dev)
    remap[labels] = torch.arange(len(labels), device=dev)
    counts = np.zeros(len(SECTIONS))
    for i in ds.ids["train"][::10]:
        s, _ = R.seg_targets(ds.c.view(i).f["sec"])
        counts += np.bincount(s[s >= 0], minlength=len(SECTIONS))
    cls_w = torch.tensor((counts.sum() / np.maximum(counts, 1) / len(SECTIONS)) ** 0.5, dtype=torch.float32,
                         device=dev)[labels]
    ids = list(ds.ids["train"])
    nb = sum(ds.c.songs[i]["nb"] for i in ids)
    M = 72 * a.layers * a.d ** 2 + 6 * a.layers * a.d * 110
    C = M * a.epochs * nb
    lr, batch = min(0.3118 * C ** -0.125, 3e-3), int(0.2920 * C ** 0.3271)
    log(f"labels {[SECTIONS[i] for i in labels]}; {R.n_params(m) / 1e6:.1f}M parameters; lr {lr:.2e}, "
        f"{batch} bars per step; supcon {a.supcon}")
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    total, seen, best, t0 = a.epochs * nb, 0, -1.0, time.time()
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    hist = []
    for ep in range(a.epochs):
        m.train()
        acc, stats, te = 0, Counter(), time.time()
        for b, n, _ in batches(ds, ids, a.micro, R.rng_for(a.seed, ep)):
            loss, st = train_loss(m, R.to_dev(b, dev), remap, cls_w, a.supcon)
            (loss * n / batch).backward()
            stats.update({k: v * n for k, v in st.items()})
            stats["n"] += n
            acc += n
            if acc >= batch:
                seen += acc
                for gr in opt.param_groups:
                    gr["lr"] = R.lr_at(seen / total, lr)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                acc = 0
        pv = predict_msa(m, ds, ds.ids["val"], dev, a.micro, marg=False)
        lab_ids = np.array(labels)
        r = score(ds, ds.ids["val"], {s: (lab_ids[p[0]], p[1]) for s, p in pv.items()})
        hist.append({"epoch": ep + 1, **{k: v / stats["n"] for k, v in stats.items() if k != "n"}, "val": r})
        log(f"epoch {ep + 1}: {time.time() - te:.0f}s " + " ".join(f"{k} {v / stats['n']:.3f}" for k, v in stats.items()
                                                                 if k != "n")
            + f" | val named acc {r['named_acc']:.3f} macro F1 {r['macro_f1']:.3f} starts±1 {r['starts_dec_1']['F']:.3f}")
        if r["macro_f1"] > best:
            best = r["macro_f1"]
            R.save_ckpt(run / "msa.pt", {"model": m.state_dict(), "cfg": m.cfg, "epoch": ep + 1})
    (run / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    log(f"trained in {(time.time() - t0) / 60:.1f} min -> {run / 'msa.pt'}")


def cmd_compare(a):
    """Every model on the val/test songs of both label sets (9 named sections; 4 + other), with and without the
    section bank; v3 decoded as riff.py does (collapsed probabilities, learned-transition Viterbi)."""
    dev = torch.device("cuda")
    sets = {"9": R.SegSet(a.data), "5": R.SegSet(a.data5)}
    ident = np.arange(len(SECTIONS))
    merge = np.array([R.section_of(s) for s in SECTIONS])   # 9 -> current configs/sections.json
    res = {}
    for path in a.models:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        name = a.names[len(res)] if a.names else Path(path).parent.name
        res[name] = {}
        for key, ds in sets.items():
            mp = ident if key == "9" else merge
            t0 = time.time()
            if "runs" in ck["cfg"]:                          # an MSAModel
                m, _ = load_msa(path, ds.c.vocab, dev)
                lab_ids = np.array(m.cfg["labels"])
                allowed = None if key == "9" else [k for k, l in enumerate(lab_ids) if merge[l] != OTHER]
                P = {s: predict_msa(m, ds, ds.ids[s], dev, a.micro, allowed, hidden=True) for s in ("val", "test")}
                tr = predict_msa(m, ds, ds.ids["train"], dev, a.micro, None, marg=False, hidden=True)
                dec = {s: {sid: (mp[lab_ids[p[0]]], p[1]) for sid, p in P[s].items()} for s in P}
                own = {}
                for s in P:
                    for sid, p in P[s].items():
                        q = np.zeros((len(p[3]), len(SECTIONS)))
                        np.add.at(q.T, mp[lab_ids], p[3].T)       # label posteriors moved onto the label set
                        own[sid] = q / np.clip(q.sum(1, keepdims=True), 1e-9, None)
                tau = tune_peaks(ds, ds.ids["val"], {sid: p[2] for sid, p in P["val"].items()})
                peaks = {sid: R.peak_borders(p[2], tau) for sid, p in P["test"].items()}
                hid = {sid: p[4] for s in P for sid, p in P[s].items()} | {sid: p[4] for sid, p in tr.items()}
            else:                                            # riff.py's SegModel (v3)
                model, _ = R.load_seg(path, ds.c.vocab, dev)
                P = {s: predict_v3(model, ds, ds.ids[s], dev, a.micro, hidden=True) for s in ("val", "test")}
                tr = predict_v3(model, ds, ds.ids["train"], dev, a.micro, hidden=True)
                hmm = R.seg_transitions(ds)
                pr = {s: {sid: (p[0], p[1]) for sid, p in P[s].items()} for s in P}
                if key == "5":
                    pr = {s: R.collapse_preds(p) for s, p in pr.items()}
                dec = {s: {sid: R.seg_viterbi(ps, pb, *hmm) for sid, (ps, pb) in pr[s].items()} for s in pr}
                own = {sid: ps / np.clip(ps.sum(1, keepdims=True), 1e-9, None) for s in pr for sid, (ps, _) in pr[s].items()}
                for o in own.values():
                    o[:, OTHER] = 0
                tau = R.tune_tau(ds, ds.ids["val"], pr["val"])
                peaks = {sid: R.peak_borders(pb, tau) for sid, (_, pb) in pr["test"].items()}
                hid = {sid: p[2] for s in P for sid, p in P[s].items()} | {sid: p[2] for sid, p in tr.items()}
            base = score(ds, ds.ids["test"], dec["test"], peaks)
            bank = bank_from(ds, ds.ids["train"], hid, mp)
            best = (-1, 0, 1)
            for lam in (0.25, 0.5, 1.0, 2.0):
                for temp in (0.02, 0.05, 0.1):
                    v = score(ds, ds.ids["val"], bank_relabel(dec["val"], hid, own, bank, lam, temp, dev=dev))
                    best = max(best, (v["named_acc"], lam, temp))
            banked = score(ds, ds.ids["test"], bank_relabel(dec["test"], hid, own, bank, best[1], best[2], dev=dev))
            res[name][key] = {"test": base, "bank": {"lam": best[1], "temp": best[2], "bank_size": len(bank[1]),
                                                     "test": banked}}
            log(f"{name} [{key}] {time.time() - t0:.0f}s: named acc {base['named_acc']:.4f} F1 {base['macro_f1']:.4f} "
                f"pairF {base['pairwise_F']:.4f} starts {base['starts_dec_0']['F']}/{base['starts_dec_1']['F']} "
                f"peaks {base['starts_peaks_0']['F']}/{base['starts_peaks_1']['F']} | bank (lam {best[1]}, T {best[2]}): "
                f"named acc {banked['named_acc']:.4f} F1 {banked['macro_f1']:.4f} pairF {banked['pairwise_F']:.4f}")
            Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pretrain", "train"):
        p = sub.add_parser(name)
        p.add_argument("--data", default="data/encoded")
        p.add_argument("--run", required=True)
        p.add_argument("--d", type=int, default=256)
        p.add_argument("--layers", type=int, default=6)
        p.add_argument("--epochs", type=int, default=10 if name == "train" else 3)
        p.add_argument("--micro", type=int, default=8192)
        p.add_argument("--seed", type=int, default=0)
        if name == "train":
            p.add_argument("--init", default="")
            p.add_argument("--supcon", type=float, default=0.0)
    p = sub.add_parser("compare")
    p.add_argument("models", nargs="+")
    p.add_argument("--names", nargs="*")
    p.add_argument("--data", default="data/encoded")
    p.add_argument("--data5", default="data/encoded_map3")
    p.add_argument("--micro", type=int, default=8192)
    p.add_argument("--out", default="runs/msa/compare.json")
    a = ap.parse_args(argv)
    {"pretrain": cmd_pretrain, "train": cmd_train, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
