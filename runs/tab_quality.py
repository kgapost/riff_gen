#!/usr/bin/env python
"""Rank tabs by objective, content-based quality signals. CPU only; nothing here deletes anything.

    .venv/bin/python runs/tab_quality.py [--csv out.csv] [--top N]

Four signals, all from the per-bar arrays prepare already wrote (no GPU, no model):

  bars            s["nb"]. Under ~8 bars a file cannot be a song. fixed_gaps already refuses anything
                  under 12 bars as a validation gap, but such a file still TRAINS.
  distinct_bars   unique bar_hash values / nb. A tab whose bars are nearly all identical carries little
                  information per token.
  pitch_classes   popcount of the OR of bar_pc over the song: how many of the 12 pitch classes appear at
                  all. One or two means a single-note or power-chord-only transcription.
  notes_per_bar   mean of bar_gn.

READ THE CAVEAT BEFORE DELETING ANYTHING. A low distinct_bars ratio means REPETITIVE, and repetition is
often the music rather than a defect: the lowest-ratio songs in this corpus include Sunn O))) (drone),
The Chemical Brothers (loop-based electronic), a James Brown vamp and "Ice Ice Baby". Those are faithful
transcriptions of repetitive music. The signal separates "little information per bar" from "much", not
"bad" from "good", so the output is a REVIEW QUEUE, not a delete list.

Size on disk is deliberately NOT used as a quality signal. It correlates with length, not merit, and the
small files cluster in classical (326 under 5KB) and acoustic (277) where short pieces are normal -
classical being the one genre that has so far gained anything from fine-tuning. Cut on bars, never bytes.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402

ENC = riff.ROOT / "data_encoded" / "encoded"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    c = riff.Corpus(ENC)
    bh = np.load(ENC / "bar_hash.npy", mmap_mode="r")
    pc = np.load(ENC / "bar_pc.npy", mmap_mode="r")
    gn = np.load(ENC / "bar_gn.npy", mmap_mode="r")

    rows = []
    for s in c.songs:
        b0, n = s["bar0"], s["nb"]
        if n < 1:
            continue
        h = np.asarray(bh[b0:b0 + n])
        p = int(np.bitwise_or.reduce(np.asarray(pc[b0:b0 + n]))) & 0xFFF
        rows.append({"file": s["file"], "genre": s["labels"].get("genre", ""),
                     "bars": n, "tokens": s["tokens"],
                     "distinct_bars": round(len(np.unique(h)) / n, 4),
                     "pitch_classes": bin(p).count("1"),
                     "notes_per_bar": round(float(np.asarray(gn[b0:b0 + n]).mean()), 2)})

    def show(title, sel, key):
        sel = sorted(sel, key=key)
        print(f"\n=== {title}: {len(sel)} songs")
        for r in sel[:a.top]:
            print(f"   bars {r['bars']:4d}  distinct {r['distinct_bars']:.3f}  pc {r['pitch_classes']:2d}  "
                  f"{Path(r['file']).parent.name.replace('tabs - ', '')[:13]:13s} {Path(r['file']).stem[:44]}")

    # 1: too short to be a song - the one group that is safe to cut on its own
    show("under 8 bars - cannot be a song", [r for r in rows if r["bars"] < 8], lambda r: r["bars"])
    # 2: almost no harmonic content
    show("3 or fewer pitch classes and 24+ bars", [r for r in rows if r["pitch_classes"] <= 3 and r["bars"] >= 24],
         lambda r: r["pitch_classes"])
    # 3: long but nearly nothing distinct - REVIEW, often legitimately repetitive music
    show("24+ bars but under 12% distinct bars (review, not delete)",
         [r for r in rows if r["bars"] >= 24 and r["distinct_bars"] < 0.12], lambda r: r["distinct_bars"])
    # 4: album-length, worth splitting into tracks rather than cutting
    show("400+ bars - probably a whole album in one file",
         [r for r in rows if r["bars"] >= 400], lambda r: -r["bars"])

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nall {len(rows)} songs -> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
