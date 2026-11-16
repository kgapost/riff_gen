#!/usr/bin/env python
"""Validation loss of one checkpoint on the val gaps of ONE label, for the stage B/C summaries.

    python runs/eval_label.py <ckpt> genre:prog
    python runs/eval_label.py <ckpt> album:hh

riff.py eval scores a whole split; the summaries need the subset of val gaps whose song carries the label, so
that "did the fine-tune help?" is asked about the genre it was tuned on and not about the other 27. This only
calls riff.gap_loss / riff.summarise - it changes nothing in riff.py.

The gap index handed to gap_loss stays the enumerate() position over the split's gaps, exactly as
riff.evaluate numbers them, because that index seeds the example builder (rng_for(0, j)): keeping it makes
these numbers comparable with the ones riff.py eval prints.

Prints one JSON object: {"label":…, "n_gaps":…, "loss":…, "acc":…, "loss_cmp":…, "n_gaps_cmp":…}.
"loss_cmp" is over the label's COMPARABLE gaps (nothing in the song repeats them) and is absent when the
label has none. n_gaps 0 means the split holds no gap for that label - the caller must treat it as missing,
not as zero loss.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    ckpt, label, split = argv[1], argv[2], (argv[3] if len(argv) > 3 else "val")
    kind, _, value = label.partition(":")
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, V, cfg, _ = riff.load_ckpt(ckpt, dev)
    c = riff.Corpus(riff.ROOT / "data_encoded" / "encoded")
    if c.vocab.tokens != V.tokens:
        sys.exit("the checkpoint was trained on another vocabulary")

    def has(sid):
        return c.songs[sid]["labels"].get(kind) == value

    sel = [(j, g) for j, g in enumerate(g for g in c.gaps if g["split"] == split) if has(g["song"])]
    out = {"label": label, "split": split, "n_gaps": len(sel)}
    if sel:
        model.eval()
        out |= riff.summarise(riff.gap_loss(model, c, cfg, sel, dev))
        cmp = [(j, g) for j, g in riff.comparable_gaps(c, split) if has(g["song"])]
        out["n_gaps_cmp"] = len(cmp)
        if cmp:
            out["loss_cmp"] = riff.summarise(riff.gap_loss(model, c, cfg, cmp, dev))["loss"]
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
