#!/usr/bin/env python
"""Will a bigger model, or a longer context, fit on this card? Measure, do not guess.

    .venv/bin/python runs/vram_probe.py

The current baseline is d=640 / 12 layers = 60M parameters trained on 8 x 501.5M = 4.01B tokens, which is
67 tokens per parameter. Compute-optimal is around 20, so the data would support roughly 200M parameters -
and that is the likeliest reason epoch 8 bought only 0.0005, below the 0.0036 seed-noise floor: the model
ran out of capacity, not out of data. Separately only 28.3 % of songs fit whole in a 4096-token window
against 53.8 % at 8192, so context is still binding.

Both changes cost memory, and this card has 24 GiB. So each configuration here gets a real forward and
backward with an AdamW step, at the micro-batch the training loop actually uses, and reports peak
allocation. A configuration is only usable if it leaves headroom: the dataloader, evaluation and
fragmentation all want room, and an OOM eleven hours into a Stage A run costs a day.

Activation checkpointing (cfg["ckpt"]) is on throughout, as it was for the 17.4-hour run - without it a
16384-token micro-batch stores every activation and even the 60M model OOMs on this card.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402
import torch  # noqa: E402

CONFIGS = [
    ("baseline  d=640 L=12", 640, 12),
    ("d=768 L=14", 768, 14),
    ("d=896 L=16", 896, 16),
    ("d=1024 L=16", 1024, 16),
]
CTXS = [4096, 8192]
MICRO = 16384


def probe(V, d, layers, ctx, micro, ckpt=True) -> tuple[float, float, str]:
    cfg = {"kind": "flat", "d": d, "layers": layers, "ctx": ctx, "epochs": 8,
           "micro_tokens": micro, "ckpt": ckpt, "drop": 0.0, "name": "probe"}
    dev = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = riff.build_model(cfg, V).to(dev)
        n = sum(p.numel() for p in model.parameters())
        params = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.1},
                  {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
        opt = torch.optim.AdamW(groups, lr=1e-4, betas=(0.9, 0.95), fused=True)
        # the widest row length bucketing will produce is the micro budget over one row; use the real
        # shape the trainer sees: rows x ctx-ish, totalling about micro tokens
        # flat_example fills 0.95*ctx minus the 16-token head, so rows get LONGER as ctx grows and the
        # row count falls to keep the micro budget. Capping L at 3891 regardless of ctx - as the first
        # version of this did - made ctx 8192 report exactly the same memory as 4096, which is how the
        # mistake showed itself.
        L = int(0.95 * ctx) - 16
        rows = max(1, micro // L)
        ids = torch.randint(0, V.size, (rows, L), device=dev)
        w = torch.ones((rows, L - 1), device=dev)
        for _ in range(2):                                # two steps: allocator reaches steady state
            with riff.autocast(dev):
                loss, _stats = model.loss({"ids": ids, "w": w, "ntok": rows * L})
            loss.backward()                               # same order as train_run: autocast fwd, bwd out
            opt.step()
            opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        del model, opt, ids, w
        torch.cuda.empty_cache()
        return n / 1e6, peak, "ok"
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return 0.0, 0.0, "OOM"
    except Exception as e:
        torch.cuda.empty_cache()
        return 0.0, 0.0, f"{type(e).__name__}: {str(e)[:60]}"


def main() -> int:
    c = riff.Corpus(riff.ROOT / "data_encoded" / "encoded")
    V = c.vocab
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"{torch.cuda.get_device_name(0)}, {total:.1f} GiB, micro_tokens {MICRO}, "
          f"activation checkpointing on\n")
    print(f"{'config':22s} {'params':>8s} {'ctx':>6s} {'micro':>7s} {'ckpt':>5s} {'peak GiB':>9s}  status")
    for name, d, layers in CONFIGS:
        for ctx in CTXS:
            n, peak, status = probe(V, d, layers, ctx, MICRO)
            row = f"{name:22s} {n:7.1f}M {ctx:6d} {MICRO:7d} {'on':>5s} "
            print(row + (f"{peak:9.2f}  ok" if status == "ok" else f"{'-':>9s}  {status}"), flush=True)

    # Activation checkpointing costs about a third of the speed, and there is clearly headroom now, so ask
    # whether it is still needed - and whether a larger micro-batch fits, which raises arithmetic intensity.
    print("\nis activation checkpointing still necessary, and does a bigger micro-batch fit?")
    print(f"{'config':22s} {'params':>8s} {'ctx':>6s} {'micro':>7s} {'ckpt':>5s} {'peak GiB':>9s}  status")
    for name, d, layers in CONFIGS:
        for ctx in CTXS:
            for micro, ck in ((MICRO, False), (MICRO * 2, True), (MICRO * 4, True)):
                n, peak, status = probe(V, d, layers, ctx, micro, ckpt=ck)
                row = (f"{name:22s} {n if n else 0:7.1f}M {ctx:6d} {micro:7d} "
                       f"{'on' if ck else 'off':>5s} ")
                print(row + (f"{peak:9.2f}  ok" if status == "ok" else f"{'-':>9s}  {status}"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
