#!/usr/bin/env python
"""Remove fragment tabs from the GENRE folders: under 10 bars, or 3 or fewer pitch classes.

    .venv/bin/python runs/cull_fragments.py            # dry run
    .venv/bin/python runs/cull_fragments.py --apply

Album folders ("tabs - own - *") are never touched - those are the user's own songs and their chosen
inspirations, and a short sketch there is deliberate.

Criteria, both measured from the encoded corpus by runs/tab_quality.py:
  bars < 10              a file this short cannot be a song. fixed_gaps already refuses anything under 12
                         bars as a validation gap, but it still trains.
  pitch_classes <= 3     the popcount of the OR of bar_pc over the whole song: how many of the twelve
                         pitch classes the song touches at all.

Removed files are MOVED to data/_culled_fragments_<date>/<source folder>/, not unlinked. prepare only
reads directories whose name starts with "tabs - " (riff.py:544), so a quarantined file is already
invisible to the corpus; purging the folder later is a separate, deliberate `rm`. 61,010 curated tabs took
real effort to assemble and the re-download needs the user's own Ultimate Guitar account, so an
irreversible delete is not worth the few seconds it saves.

PATHS: the song["file"] values in the corpus are from the last prepare and are stale for anything moved
since, so each song is resolved by BASENAME against what is on disk now. A basename that no longer exists,
or that now exists in more than one genre folder, is reported and skipped rather than guessed at.
"""
from __future__ import annotations

import argparse
import collections
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402

DATA = riff.ROOT / "data"
ENC = riff.ROOT / "data_encoded" / "encoded"
EXT = {".gp3", ".gp4", ".gp5"}
MIN_BARS = 10
MAX_PC = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-bars", type=int, default=MIN_BARS)
    ap.add_argument("--max-pc", type=int, default=MAX_PC)
    a = ap.parse_args()

    # what is on disk now, genre folders only
    folders = [d for d in DATA.iterdir()
               if d.is_dir() and d.name.lower().startswith("tabs - ")
               and not d.name.lower().startswith("tabs - own - ")]
    ondisk = collections.defaultdict(list)
    for d in folders:
        for p in d.rglob("*"):
            if p.suffix.lower() in EXT:
                ondisk[p.name].append(p)

    c = riff.Corpus(ENC)
    pc = np.load(ENC / "bar_pc.npy", mmap_mode="r")

    victims, ambiguous, gone = [], [], []
    seen = set()
    for s in c.songs:
        if s["file"].startswith("tabs - own - "):
            continue
        n = s["nb"]
        if n < 1:
            continue
        npc = bin(int(np.bitwise_or.reduce(np.asarray(pc[s["bar0"]:s["bar0"] + n]))) & 0xFFF).count("1")
        if not (n < a.min_bars or npc <= a.max_pc):
            continue
        name = Path(s["file"]).name
        if name in seen:
            continue
        seen.add(name)
        hits = ondisk.get(name, [])
        if not hits:
            gone.append(s["file"])
        elif len(hits) > 1:
            ambiguous.append((name, [str(h.parent.name) for h in hits]))
        else:
            victims.append((hits[0], n, npc))

    quar = DATA / f"_culled_fragments_{date.today():%Y%m%d}"
    by_reason = collections.Counter("bars<%d" % a.min_bars if n < a.min_bars else "pc<=%d" % a.max_pc
                                   for _, n, _ in victims)
    by_folder = collections.Counter(p.parent.name.replace("tabs - ", "") for p, _, _ in victims)
    print(f"{len(victims)} files to cull (bars < {a.min_bars} or pitch classes <= {a.max_pc})")
    print("  by reason: " + ", ".join(f"{k}={v}" for k, v in by_reason.most_common()))
    print("  by folder: " + ", ".join(f"{k}={v}" for k, v in by_folder.most_common()))
    if gone:
        print(f"  {len(gone)} corpus entries no longer on disk (moved or already culled); skipped")
    if ambiguous:
        print(f"  {len(ambiguous)} ambiguous basenames in >1 genre folder; SKIPPED, resolve by hand:")
        for nm, fs in ambiguous[:10]:
            print(f"     {nm[:52]:52s} in {', '.join(fs)}")

    if not a.apply:
        print("\ndry run; pass --apply. Smallest first:")
        for p, n, npc in sorted(victims, key=lambda v: (v[1], v[2]))[:12]:
            print(f"   {n:3d} bars, {npc:2d} pc  {p.parent.name.replace('tabs - ','')[:14]:14s} {p.stem[:44]}")
        return 0

    for p, _, _ in victims:
        d = quar / p.parent.name / p.name
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists():
            print(f"  already in quarantine, skipping: {p.name}")
            continue
        shutil.move(str(p), str(d))
    print(f"\nmoved {len(victims)} files to {quar}")
    print("they are out of the corpus now; `rm -rf` that folder when you want them gone for good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
