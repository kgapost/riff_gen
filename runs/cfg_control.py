#!/usr/bin/env python
"""Does the genre label actually change what gets GENERATED, and at what guidance weight?

    .venv/bin/python runs/cfg_control.py --song "data/tabs - own - cs - own/black.gp5" --gap 17-24

Nine genre fine-tuning runs asked "does conditioning on a genre help predict existing tabs?" and the
answer was no: top-1 genre classification is 4.6% against 3.6% chance, and knowing the true genre is
worth 0.0014 in loss against a 0.0036 seed-noise floor.

But that is the wrong question for the actual goal, which is to ask for a doom riff and receive one.
Prediction loss and generation style are different things, and classifier-free guidance exists precisely
to turn a weak conditional signal into a strong behavioural one: cmd_infer already builds an
unconditional row beside each labelled row (riff.py:2895) and cfg_combine extrapolates between them, so
--cfg does not need the label's raw effect to be large, it multiplies it.

So this fills ONE gap of ONE song many times over - same song, same bars, same seed - varying only the
genre label and the guidance weight, and measures how far the output moves. If divergence grows with the
weight, genre control works and this says what weight to use. If it stays flat, guidance cannot rescue
the label either, and conditional generation needs a training change rather than an inference knob.

Measured per generated candidate, read through the vocabulary's own kind/n_sf tables rather than by
parsing token names (guitar notes are "n:<string>:<fret>", bass is "bp:<midi>", techniques are "fx:*"):
    fret        12-bin histogram over frets 0-23, L1-compared between genres
    density     notes per bar
    string      mean string number - low strings are heavy, high strings are not
    pm / dead   share of notes carrying palm mute or a dead note, the two most genre-telling techniques
Divergence between two genres is the L1 distance of their fret histograms plus the normalised absolute
differences of the scalars, so 0 means "the label changed nothing at all".
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402
import torch  # noqa: E402
import guitarpro as gp  # noqa: E402

GENRES = ["doom_stoner_sludge", "classical", "jazz", "metal_xtreme", "acoustic"]
# riff.py infer defaults to --cfg 1.5, and the first run of this script showed 1.0, 2.0 and 3.0 giving
# BIT-IDENTICAL output: the genre logit difference is too small to flip any sampled token below about 5.
# So the sweep goes high, and the mean log-probability is tracked alongside, because guidance that is
# strong enough to change the style is also strong enough to push the output off-distribution.
WEIGHTS = [1.5, 5.0, 7.0, 10.0, 14.0]


def stats_of(bars, V):
    """fret histogram + scalars for one generated candidate, via V.kind / V.n_sf."""
    fret = np.zeros(12)
    strings, n_notes, pm, dead = [], 0, 0, 0
    for b in bars:
        for t in np.asarray(b).ravel().tolist():
            t &= riff.MASK                            # generated bars carry the guitar FLAG bit
            if t >= len(V.tokens):
                continue
            if V.kind[t] == riff.K_NOTE:
                s, f = V.n_sf[t]
                fret[min(int(f) // 2, 11)] += 1
                strings.append(int(s))
                n_notes += 1
            elif V.kind[t] == riff.K_FX:
                nm = V.tokens[t]
                pm += nm == "fx:pm"
                dead += nm == "fx:dead"
    tot = fret.sum()
    return {"fret": fret / tot if tot else fret, "density": n_notes / max(len(bars), 1),
            "string": float(np.mean(strings)) if strings else 0.0,
            "pm": pm / max(n_notes, 1), "dead": dead / max(n_notes, 1), "notes": n_notes}


def _rel(x, y):
    return abs(x - y) / max(abs(x) + abs(y), 1e-9)


def diverge(a, b):
    return (float(np.abs(a["fret"] - b["fret"]).sum())
            + _rel(a["density"], b["density"]) + _rel(a["string"], b["string"])
            + _rel(a["pm"], b["pm"]) + _rel(a["dead"], b["dead"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/final10/best.pt")
    ap.add_argument("--song", required=True)
    ap.add_argument("--gap", required=True, help="measures, e.g. 17-24")
    ap.add_argument("--cands", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, V, cfg, _ = riff.load_ckpt(a.ckpt, dev)
    model.eval()
    enc = riff.encode_song(gp.parse(a.song), V)
    view = riff.SongView.from_encoded(enc, {})
    lo, hi = (int(x) - 1 for x in a.gap.split("-"))
    target = (lo, hi - lo + 1)
    print(f"{Path(a.song).name}: {view.nb} bars, filling {lo + 1}-{hi + 1} ({target[1]} bars)\n")

    table = {}
    for w in WEIGHTS:
        for g in GENRES:
            rows = [[V.none] * 4]                    # unconditional row, as cmd_infer builds it
            r = [V.none] * 4
            r[riff.SLOTS.index("genre")] = V.id[f"lbl:genre:{g}"]
            rows.append(r)
            with torch.no_grad(), riff.autocast(dev):
                cands = riff.fill_gap(model, V, cfg, view, target, (), rows, [0.0, w],
                                      n_cand=a.cands, seed=a.seed)
            st = [stats_of(bars, V) for bars, _ in cands if bars]
            scores = [sc for bars, sc in cands if bars]
            if not st:
                print(f"  cfg {w}: {g} produced nothing")
                continue
            agg = {"fret": np.mean([s["fret"] for s in st], 0),
                   "logp": float(np.mean(scores)) if scores else 0.0}
            for k in ("density", "string", "pm", "dead", "notes"):
                agg[k] = float(np.mean([s[k] for s in st]))
            table[(w, g)] = agg
            print(f"  cfg {w:3.1f}  {g:20s} density {agg['density']:5.1f}  string {agg['string']:4.2f}  "
                  f"pm {agg['pm']:.3f}  notes {agg['notes']:5.1f}  logp {agg['logp']:+.3f}", flush=True)

    print("\n=== mean pairwise divergence between genres, by guidance weight")
    print("   (0 = the genre label changed nothing about the output)")
    for w in WEIGHTS:
        have = [g for g in GENRES if (w, g) in table]
        if len(have) < 2:
            continue
        ds = [diverge(table[(w, x)], table[(w, y)]) for x, y in itertools.combinations(have, 2)]
        print(f"   cfg {w:3.1f}:  mean {np.mean(ds):.3f}   max {np.max(ds):.3f}   ({len(ds)} pairs)")

    print("\n=== doom vs classical specifically (should differ most)")
    for w in WEIGHTS:
        if (w, "doom_stoner_sludge") in table and (w, "classical") in table:
            d = table[(w, "doom_stoner_sludge")]
            cl = table[(w, "classical")]
            print(f"   cfg {w:3.1f}:  divergence {diverge(d, cl):.3f}   "
                  f"string {d["string"]:.2f} vs {cl["string"]:.2f}   density {d['density']:.1f} vs {cl['density']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
