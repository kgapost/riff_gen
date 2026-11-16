#!/usr/bin/env python
"""Repair label_best.pt files that were saved with lora=0 by the old label_best_watcher.py.

    .venv/bin/python runs/repair_label_best.py [--dry-run] [run-dir ...]

With no run directories it repairs every runs/*/label_best.pt.

The watcher used to set the rank with getattr(model, "_lora_rank", 0), and no such attribute exists, so
every file it wrote claims lora=0 while its state dict is a LoRA one (att.q.a / att.q.b / att.q.base.weight
instead of att.q.weight). riff.load_ckpt then builds a plain model and load_state_dict refuses it, so the
file cannot be opened at all. The WEIGHTS are fine; only the metadata is wrong, and the losses the watcher
logged to label_evals.jsonl were measured before saving and are unaffected.

The rank is recoverable from the weights: a LoRA pair is a (r x in) "a" and an (out x r) "b", so r is the
shared small dimension. We read it rather than trust an argument, and refuse to write if the ranks found in
one file disagree.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402
import torch  # noqa: E402


def rank_of(sd) -> int | None:
    """The LoRA rank implied by the state dict, or None if it is not a LoRA checkpoint."""
    ranks = set()
    for k, v in sd.items():
        if k.endswith(".a") and v.ndim == 2:
            ranks.add(min(v.shape))
    if not ranks:
        return None
    if len(ranks) > 1:
        raise ValueError(f"inconsistent LoRA ranks in one checkpoint: {sorted(ranks)}")
    return ranks.pop()


def main(argv) -> int:
    dry = "--dry-run" in argv
    args = [a for a in argv[1:] if not a.startswith("-")]
    runs = [Path(a) for a in args] or sorted((riff.ROOT / "runs").glob("*/"))
    paths = [r / "label_best.pt" for r in runs]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        print("no label_best.pt files found")
        return 0

    bad = fixed = 0
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        r = rank_of(ck["model"])
        have = ck.get("lora", 0)
        if r is None:
            print(f"{p}: not a LoRA checkpoint (lora={have}) - left alone")
            continue
        if have == r:
            print(f"{p}: already correct (lora={r})")
            continue
        bad += 1
        print(f"{p}: lora {have} -> {r}" + ("  [dry run]" if dry else ""))
        if not dry:
            ck["lora"] = r
            riff.save_ckpt(p, ck)      # atomic tmp+rename, same as riff.py's own writes
            fixed += 1

    print(f"\n{len(paths)} checked, {bad} wrong, {fixed} rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
