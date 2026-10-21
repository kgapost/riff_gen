"""Teacher-forced evaluation: loss on fixed examples, split by what the token is.

Per-category loss tells *what* the model finds hard: ``note`` (which string/fret = pitch
and shape), ``wait`` (rhythm), ``fx`` (palm mute, bends...), ``rest``, ``bar`` (bar lines
and time signatures), ``end``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from contextlib import nullcontext

import torch

from riffgen.dataset.loader import TokenBudgetBatcher
from riffgen.tokens.vocab import K, Vocab
from riffgen.utils.device import to_device

CATEGORIES = {
    "note": (K.NOTE,), "rest": (K.REST,), "wait": (K.WAIT,), "fx": (K.NFX, K.PARAM, K.BFX),
    "bar": (K.NEW_MEASURE, K.MEASURE, K.TS, K.SECTION), "end": (K.EOS, K.END),
}


def category_table(vocab: Vocab) -> torch.Tensor:
    cat = torch.full((len(vocab),), len(CATEGORIES), dtype=torch.long)
    for ci, kinds in enumerate(CATEGORIES.values()):
        for k in kinds:
            cat[torch.from_numpy(vocab.kind == k)] = ci
    return cat


@torch.no_grad()
def evaluate_examples(model, examples: list[dict], vocab: Vocab, device, amp=None, micro_batch_tokens: int = 16384) -> dict:
    if not examples:
        return {}
    model.eval()
    cat_of = category_table(vocab).to(device)
    names = list(CATEGORIES) + ["other"]
    nll_sum = defaultdict(float)
    cnt = defaultdict(float)
    correct = defaultdict(float)
    aux_sum, aux_n = defaultdict(float), defaultdict(int)
    loss_sum = weight_sum = 0.0
    ctx = torch.autocast(device.type, dtype=amp) if amp is not None else nullcontext()
    batches = TokenBudgetBatcher(iter(examples), micro_batch_tokens, vocab.pad_id, buffer_size=len(examples),
                                 shuffle=False)
    for batch in batches:
        batch = to_device(batch, device)
        with ctx:
            out = model(batch, per_token=True)
        loss_sum += float(out["loss_sum"])
        weight_sum += float(out["weight_sum"])
        t = out["targets"]
        c = cat_of[t]
        for ci, name in enumerate(names):
            m = c == ci
            if m.any():
                nll_sum[name] += float(out["nll"][m].sum())
                cnt[name] += float(m.sum())
                correct[name] += float((out["pred"][m] == t[m]).sum())
        for k, v in (out.get("aux") or {}).items():
            aux_sum[k] += float(v)
            aux_n[k] += 1
    res = {"loss": loss_sum / max(weight_sum, 1e-9)}
    res["ppl"] = math.exp(min(res["loss"], 20))
    total = sum(cnt.values())
    res["acc"] = sum(correct.values()) / max(total, 1)
    for name in names:
        if cnt[name]:
            res[f"nll_{name}"] = nll_sum[name] / cnt[name]
            res[f"acc_{name}"] = correct[name] / cnt[name]
    for k in aux_sum:
        res[f"aux_{k}"] = aux_sum[k] / aux_n[k]
    res["tokens"] = int(total)
    model.train()
    return res
