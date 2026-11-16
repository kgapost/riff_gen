#!/usr/bin/env python
"""Train-split token totals per genre / album label, and the derived fine-tune schedule.

Reads only the prepared corpus metadata; writes runs/schedule.json. Nothing here trains.
The formulas are the ones the task fixed:

  stage B (genre, full fine-tune)   W = round(0.67 * (T_all - T_g) / T_g), capped 100
                                    budget = max(10 * T_g, 20e6)
  stage C (album, LoRA)             W = round((T_all - T_a) / T_a), capped 300
                                    budget = max(30 * T_a, 20e6)
  both                              eval_every = (budget / epoch_tokens(corpus, focus)) / 8

epoch_tokens is the FOCUS-WEIGHTED train total (riff.py:1633), so it has to be recomputed per run with that
run's focus dict - which is why eval_every differs per genre even when the budget does not.
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
ALBUMS = ["vc1", "vc2", "caffeine", "cs", "potha", "tme", "maphorion", "omni", "waves_black", "waves_white",
          "inner", "popper", "hh"]


def main() -> int:
    c = riff.Corpus(riff.ROOT / "data_encoded" / "encoded")
    train = [s for s in c.songs if s["split"] == "train"]
    T_all = sum(s["tokens"] for s in train)

    def tok(pred):
        sel = [s for s in train if pred(s["labels"])]
        return sum(s["tokens"] for s in sel), len(sel)

    out = {"T_all": T_all, "n_train": len(train), "genres": {}, "albums": {}}

    for g in GENRES:
        T_g, n = tok(lambda L, g=g: L.get("genre") == g)
        if not T_g:
            print(f"!! genre {g}: no train tokens, skipping")
            continue
        W = min(100, max(1, round(0.67 * (T_all - T_g) / T_g)))
        budget = max(10 * T_g, 20_000_000)
        focus = {f"genre:{g}": float(W), "src:early": 0.2}
        ep = riff.epoch_tokens(c, focus)
        out["genres"][g] = {"n_train_songs": n, "T_g": T_g, "W": W, "budget": int(budget),
                            "epoch_tokens_focused": ep, "eval_every": round((budget / ep) / 8, 5),
                            "share": round(W * T_g / (W * T_g + (T_all - T_g)), 3)}

    # the album's own genres, for the §13 replay term: the genres its insp/own tabs carry elsewhere in the
    # corpus. An album tab has no genre label of its own (labels_of gives album+src), so the replay genres are
    # taken from the same FAMILY appearing in a genre folder - that is what the 2946 cross-folder copies are.
    fam_genre = {}
    for s in train:
        g = s["labels"].get("genre")
        if g:
            fam_genre.setdefault(riff.family(Path(s["file"])), set()).add(g)

    for a in ALBUMS:
        T_a, n = tok(lambda L, a=a: L.get("album") == a)
        n_own = sum(1 for s in train if s["labels"].get("album") == a and s["labels"].get("src") == "own")
        n_insp = sum(1 for s in train if s["labels"].get("album") == a and s["labels"].get("src") == "insp")
        if not T_a:
            print(f"!! album {a}: no train tokens, skipping")
            continue
        W = min(300, max(1, round((T_all - T_a) / T_a)))
        budget = max(30 * T_a, 20_000_000)
        gens = {}
        for s in train:
            if s["labels"].get("album") == a:
                for g in fam_genre.get(riff.family(Path(s["file"])), ()):
                    gens[g] = gens.get(g, 0) + 1
        top = sorted(gens, key=lambda g: -gens[g])[:3]
        focus = {f"album:{a}": float(W)} | {f"genre:{g}": 2.0 for g in top}
        ep = riff.epoch_tokens(c, focus)
        out["albums"][a] = {"n_own": n_own, "n_insp": n_insp, "T_a": T_a, "W": W, "budget": int(budget),
                            "replay_genres": top, "epoch_tokens_focused": ep,
                            "eval_every": round((budget / ep) / 8, 5),
                            "share": round(W * T_a / (W * T_a + (T_all - T_a)), 3)}

    (riff.ROOT / "runs" / "schedule.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    gb = sum(v["budget"] for v in out["genres"].values())
    ab = sum(v["budget"] for v in out["albums"].values())
    print(f"T_all = {T_all / 1e6:.1f}M train tokens over {len(train)} songs\n")
    print(f"{'genre':24s} {'songs':>6s} {'T_g':>9s} {'W':>4s} {'budget':>9s} {'share':>6s} {'eval_every':>10s}")
    for g, v in out["genres"].items():
        print(f"{g:24s} {v['n_train_songs']:6d} {v['T_g'] / 1e6:8.1f}M {v['W']:4d} "
              f"{v['budget'] / 1e6:8.0f}M {v['share']:6.2f} {v['eval_every']:10.4f}")
    print(f"\n{'album':14s} {'own':>4s} {'insp':>5s} {'T_a':>9s} {'W':>4s} {'budget':>9s} {'share':>6s} replay")
    for a, v in out["albums"].items():
        print(f"{a:14s} {v['n_own']:4d} {v['n_insp']:5d} {v['T_a'] / 1e6:8.1f}M {v['W']:4d} "
              f"{v['budget'] / 1e6:8.0f}M {v['share']:6.2f} {','.join(v['replay_genres']) or '-'}")
    print(f"\nstage B total {gb / 1e6:.0f}M tokens, stage C total {ab / 1e6:.0f}M tokens, "
          f"together {(gb + ab) / 1e6:.0f}M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
