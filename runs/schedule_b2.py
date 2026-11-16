#!/usr/bin/env python
"""Stage B2 schedule: genre fine-tunes as LoRA adapters instead of full fine-tunes.

    .venv/bin/python runs/schedule_b2.py [MULT] [out.json]

Writes runs/schedule_b2.json (or the given path). MULT is the budget multiplier on T_g, default 20.

WHY THIS DIFFERS FROM runs/schedule.json (stage B, as first run)
---------------------------------------------------------------
Stage B full-fine-tuned every genre at lr 1e-4 and 26 of 28 genres came out WORSE on their own songs.
Cause: riff.lr_at(frac, peak, warm=0.02, floor=0.1) means stage A *ended* at 3.74e-04 * 0.1 = 3.74e-05,
and stage B then warmed back up to 1e-4 - 2.7x the converged rate. Every run's first evaluation was
already worse than the pretrained 0.1937 (acoustic 0.2012, rock 0.2025, prog 0.2007): the damage happened
during the warm-up, before any genre-specific learning could pay for it.

LoRA removes the failure mode rather than tuning around it. With --lora the base weights are frozen, so a
high learning rate cannot overwrite what pretraining learned; the adapter matrices start at zero and NEED
a real rate. That is why stage C's album runs were safe at 3.74e-04 while stage B was not safe at 1e-4.
So B2 uses --lora with the pretraining peak rate, exactly as stage C did.

BUDGET
------
The number that matters is not the budget but how many passes the TARGET genre's own tokens get:
budget * share / T_g, where share is the genre's fraction of the focus-weighted epoch. The first stage B
gave every genre ~4.0 passes. Zero-initialised adapters have to learn the genre from scratch, where a full
fine-tune starts from weights that already know it, so 4 passes is too few here. LoRA also trains ~1-2M
parameters instead of ~57M, which makes overfitting much harder, so a larger budget costs wall-clock
rather than quality. MULT=20 gives ~8 passes, MULT=30 ~12.

RANK
----
16 for most genres, 32 for the large ones (T_g >= 15M: metal_xtreme, rock, prog, punk, metal_classic,
rock_classic, metal_modern, metal_power, metal_thrash). Those cover more stylistic ground per label, and
rank 16 was fitted to albums of 50-140 songs, not genres of 4700.

EVAL DENSITY
------------
eval_every = (budget / epoch_tokens(corpus, focus)) / EVALS with EVALS=16, twice the first stage B's 8.
train_run writes last.pt only at an evaluation, so the evaluation count is exactly the number of
checkpoints runs/label_best_watcher.py gets to choose between. 8 was too coarse to select on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402

GENRES = ["acoustic", "blues_soul_rnb", "britpop", "classical", "doom_stoner_sludge", "electronic", "ethnic",
          "experimental", "funk", "funkrock", "grunge", "jazz", "metal_classic", "metal_groove",
          "metal_melodic_death", "metal_modern", "metal_nu", "metal_power", "metal_thrash", "metal_xtreme",
          "pop", "prog", "punk", "rock", "rock_classic", "rock_hard", "shred", "soundtracks"]

BIG_TOKENS = 15_000_000      # at or above this T_g, rank 32
EVALS = 16                   # evaluations (= checkpoints the watcher can choose from) per run
MIN_BUDGET = 20_000_000


def main() -> int:
    mult = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else riff.ROOT / "runs" / "schedule_b2.json"

    c = riff.Corpus(riff.ROOT / "data_encoded" / "encoded")
    train = [s for s in c.songs if s["split"] == "train"]
    T_all = sum(s["tokens"] for s in train)

    genres = {}
    for g in GENRES:
        sel = [s for s in train if s["labels"].get("genre") == g]
        T_g = sum(s["tokens"] for s in sel)
        if not T_g:
            print(f"skip {g}: no train tokens", file=sys.stderr)
            continue
        # W unchanged from stage B: the focus weight that lifts the genre to ~40% of the draws.
        W = min(100, max(1, round(0.67 * (T_all - T_g) / T_g)))
        budget = int(max(mult * T_g, MIN_BUDGET))
        focus = {f"genre:{g}": float(W), "src:early": 0.2}
        ep = riff.epoch_tokens(c, focus)
        # share: the fraction of a focus-weighted epoch that is this genre, so passes = budget*share/T_g
        share = (T_g * W) / ep
        genres[g] = {
            "n_train_songs": len(sel),
            "T_g": T_g,
            "W": W,
            "lora": 32 if T_g >= BIG_TOKENS else 16,
            "budget": budget,
            "lr": 3.74e-4,
            "epoch_tokens_focused": ep,
            "eval_every": round(budget / ep / EVALS, 5),
            "share": round(share, 3),
            "passes": round(budget * share / T_g, 2),
        }

    doc = {"mult": mult, "evals": EVALS, "T_all": T_all, "genres": genres,
           "total_budget": sum(v["budget"] for v in genres.values())}
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"{out}: {len(genres)} genres, total {doc['total_budget'] / 1e6:.0f}M tokens, mult {mult}")
    for g, v in sorted(genres.items(), key=lambda kv: -kv[1]["T_g"]):
        print(f"  {g:22s} T_g={v['T_g'] / 1e6:7.2f}M lora={v['lora']:2d} W={v['W']:3d} "
              f"budget={v['budget'] / 1e6:7.1f}M passes={v['passes']:5.2f} eval_every={v['eval_every']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
