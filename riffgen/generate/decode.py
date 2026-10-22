"""Sampling loop: many candidates for one gap, under the constraints.

* The prompt (context around the gap) is run once; its K/V cache is shared by all
  candidates (memory for 32 candidates ~ 1 prompt + 32 short continuations).
* Every candidate has its own seed, ``derive(seed, gap, candidate_no)`` (D46): sampling uses
  Gumbel-max with a per-candidate generator, so candidate 7 can be regenerated exactly with
  the same model and settings, whatever the batch size.
* Temperature and top-p are applied to the *allowed* tokens only.
* We record the model's log-probability of what it wrote (for ranking) and how much
  probability it put on tokens the rules forbade ("invalid mass": a direct measure of how
  well the model learned the format).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch

from riffgen.dataset.serialize import Featurizer
from riffgen.generate.constraints import GapConstraint, GapSpec, VocabMasks
from riffgen.tokens.vocab import Vocab
from riffgen.utils.seed import derive_int


@dataclass
class Candidate:
    index: int
    seed: int
    tokens: list[int] = field(default_factory=list)
    logprob: float = 0.0          # sum of log p(token) under the constrained, T=1 distribution
    n_scored: int = 0             # tokens that were a real choice (not forced)
    invalid_mass: float = 0.0     # mean probability the model put on forbidden tokens
    finished: bool = False

    @property
    def mean_logprob(self) -> float:
        return self.logprob / max(1, self.n_scored)


def _filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_l, idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_l, dim=-1)
    cum = probs.cumsum(-1)
    remove = cum - probs > top_p
    sorted_l = sorted_l.masked_fill(remove, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, idx, sorted_l)


def _step_features(feats: list[Featurizer], toks: list[int], device) -> dict:
    rows = [f.step(t) for f, t in zip(feats, toks)]
    # (bar, tick, inst, pitch, fret, sfb, beat, seg, pos, time16, phase16)
    from riffgen.dataset.serialize import BAR_REL_MAX, N_BAR_REL
    ref = feats[0].ref_bar
    out = {
        "ids": torch.tensor(toks, dtype=torch.long, device=device)[:, None],
        "inst": torch.tensor([r[2] + 1 for r in rows], device=device)[:, None],
        "pitch": torch.tensor([r[3] for r in rows], device=device)[:, None],
        "fret": torch.tensor([r[4] for r in rows], device=device)[:, None],
        "sfb": torch.tensor([r[5] for r in rows], device=device)[:, None],
        "beat": torch.tensor([r[6] for r in rows], device=device)[:, None],
        "seg": torch.tensor([r[7] for r in rows], device=device)[:, None],
        "pos": torch.tensor([r[8] for r in rows], device=device)[:, None],
        "phrase": torch.tensor([r[11] for r in rows], device=device)[:, None],
        "bar_rel": torch.tensor([min(max(r[0] - ref, -BAR_REL_MAX), BAR_REL_MAX - 1) + BAR_REL_MAX for r in rows],
                                device=device)[:, None].clamp(0, N_BAR_REL - 1),
        "time": torch.tensor([r[9] for r in rows], dtype=torch.float32, device=device)[:, None],
        "phase": torch.tensor([r[10] for r in rows], dtype=torch.float32, device=device)[:, None],
        "abs_bar": torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)[:, None],
    }
    return out


@torch.no_grad()
def generate(model, vocab: Vocab, prompt: dict, featurizer: Featurizer, spec: GapSpec, n_candidates: int,
             seed: int, gap_key: str, temperature: float = 1.0, top_p: float = 0.95, batch_size: int = 16,
             max_new: int | None = None, amp=None, device=None, candidate_ids: list[int] | None = None) -> list[Candidate]:
    device = device or next(model.parameters()).device
    ctx = torch.autocast(device.type, dtype=amp) if amp is not None else nullcontext()
    masks = VocabMasks(vocab, spec)
    V_model = model.cfg.vocab_size
    max_new = max_new or (spec.n_bars * 520 + 8)
    with ctx:
        state = model.prefill(prompt, 1, 1)
        first_logits = model.prompt_logits(state)[0]
    if not torch.isfinite(first_logits).all():
        raise FloatingPointError("non-finite logits (fp16 overflow?)")
    ids = candidate_ids if candidate_ids is not None else list(range(n_candidates))
    out: list[Candidate] = []
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        B = len(chunk)
        cands = [Candidate(index=k, seed=derive_int(seed, "candidate", gap_key, k)) for k in chunk]
        gens = [torch.Generator().manual_seed(c.seed) for c in cands]
        cons = [GapConstraint(vocab, spec, masks) for _ in range(B)]
        feats = [featurizer_clone(featurizer) for _ in range(B)]
        # fresh per-candidate caches, shared prompt K/V
        for c in state["caches"]:
            H, hd = c["pk"].shape[1], c["pk"].shape[3]
            c["gk"] = torch.zeros(B, H, max_new, hd, device=c["pk"].device, dtype=c["pk"].dtype)
            c["gv"] = torch.zeros(B, H, max_new, hd, device=c["pk"].device, dtype=c["pk"].dtype)
            c["n"] = 0
        logits = first_logits[None].expand(B, -1).clone()
        invalid_sum = np.zeros(B)
        steps = np.zeros(B)
        for t in range(max_new):
            allowed = np.zeros((B, V_model), bool)
            for b, c in enumerate(cons):
                allowed[b, :len(vocab)] = c.allowed()
            allowed_t = torch.from_numpy(allowed).to(logits.device)
            lp_all = torch.log_softmax(logits.float(), dim=-1)
            p_invalid = (lp_all.exp() * (~allowed_t)).sum(-1).cpu().numpy()
            masked = logits.float().masked_fill(~allowed_t, float("-inf"))
            lp_masked = torch.log_softmax(masked, dim=-1)
            scaled = _filter_top_p(masked / max(temperature, 1e-5), top_p)
            gumbel = torch.stack([-torch.log(-torch.log(torch.rand(V_model, generator=g).clamp_(1e-10, 1.0)))
                                  for g in gens]).to(logits.device)
            choice = torch.argmax(scaled + gumbel, dim=-1).tolist()
            n_allowed = allowed.sum(1)
            toks = []
            for b, c in enumerate(cons):
                cand = cands[b]
                if c.done:
                    toks.append(vocab.eos_id)
                    continue
                tok = int(choice[b])
                if n_allowed[b] > 1:
                    cand.logprob += float(lp_masked[b, tok])
                    cand.n_scored += 1
                    invalid_sum[b] += p_invalid[b]
                    steps[b] += 1
                cand.tokens.append(tok)
                c.advance(tok)
                if c.done:
                    cand.finished = True
                toks.append(tok)
            if all(c.done for c in cons):
                break
            with ctx:
                f = _step_features(feats, toks, logits.device)
                logits = model.decode_step(state, f)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite logits (fp16 overflow?)")
        for b, cand in enumerate(cands):
            cand.invalid_mass = float(invalid_sum[b] / max(1, steps[b]))
        out += cands
    return out


def featurizer_clone(f: Featurizer) -> Featurizer:
    import copy
    return copy.copy(f)
