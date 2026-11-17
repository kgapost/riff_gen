#!/usr/bin/env python
"""Consolidate one artist's tabs into a single genre folder.

    .venv/bin/python runs/consolidate_artists.py              # dry run, prints the plan
    .venv/bin/python runs/consolidate_artists.py --apply

Stage B2 measured nine genres and only classical improved. One cause we could see in the data was the
shred folder holding ~122 film and game scores, and another was the same songs filed under two genres at
once: 83 Van Halen families in both shred and rock_hard, 51 Nevermore families in both metal_modern and
metal_xtreme. A song filed twice inflates T_g, and because fixed_gaps skips anything flagged "dup" it is
also dropped from validation - so the duplicates both distort the budget and shrink the thing we measure.

This moves every tab by a named artist into one chosen genre folder.

RULES
  - Genre folders only. Anything matching "tabs - own - *" is an album folder and is never touched: those
    hold the user's own songs and their chosen inspirations, and their src: labels matter.
  - Folders that are not genres are left alone too ("tabs - generic", "tabs - exercises", "tabs - errors").
    A Van Halen tab under "tabs - exercises" is plausibly a technique exercise, filed on purpose.
  - The artist is matched on the FILENAME'S ARTIST FIELD - the text before the first " - " - not with a
    substring search over the whole name, which would also catch a song whose TITLE mentions the artist.
  - A file whose name already exists in the destination is a duplicate: it is moved to
    data/_removed_duplicates_<date>/<source folder>/ rather than deleted, so nothing is lost and the move
    is reversible. prepare only reads directories whose name starts with "tabs - " (riff.py:544), so the
    quarantine folder is invisible to it even though it sits under data/.

Nothing here re-encodes anything: riff.py prepare has to be run afterwards for the corpus to change.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

DATA = Path("/home/kapost/riff_gen/data")
EXT = {".gp3", ".gp4", ".gp5"}

# artist field (lowercased) -> destination genre folder name as it appears on disk
PLAN = {
    "nevermore": "tabs - metal xtreme",
    "van halen": "tabs - shred",
    "allan holdsworth": "tabs - experimental",
    "frank zappa": "tabs - experimental",
    "opeth": "tabs - prog",
}
# genre folders that are not really genres; left alone
NOT_GENRES = {"tabs - generic", "tabs - exercises", "tabs - errors"}


def genre_folders() -> list[Path]:
    return sorted(d for d in DATA.iterdir()
                  if d.is_dir() and d.name.lower().startswith("tabs - ")
                  and not d.name.lower().startswith("tabs - own - ")
                  and d.name.lower() not in NOT_GENRES)


def artist_of(p: Path) -> str:
    return p.stem.split("-")[0].strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    quar = DATA / f"_removed_duplicates_{date.today():%Y%m%d}"
    folders = genre_folders()
    moves, dups, stay = [], [], []

    for art, destname in PLAN.items():
        dest = DATA / destname
        if not dest.is_dir():
            sys.exit(f"destination folder missing: {dest}")
        for d in folders:
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() not in EXT or artist_of(p) != art:
                    continue
                if d == dest:
                    stay.append(p)
                elif (dest / p.name).exists():
                    dups.append((p, quar / d.name / p.name))
                else:
                    moves.append((p, dest / p.name))

    for art in PLAN:
        n_m = sum(1 for s, _ in moves if artist_of(s) == art)
        n_d = sum(1 for s, _ in dups if artist_of(s) == art)
        n_s = sum(1 for s in stay if artist_of(s) == art)
        print(f"{art:18s} -> {PLAN[art]:22s} move {n_m:4d}, duplicate {n_d:4d}, already there {n_s:4d}")

    print(f"\n{len(moves)} moves, {len(dups)} duplicates to {quar.name}/, {len(stay)} untouched")
    by_src = {}
    for s, _ in moves + dups:
        by_src[s.parent.name] = by_src.get(s.parent.name, 0) + 1
    print("sources: " + ", ".join(f"{k} ({v})" for k, v in sorted(by_src.items(), key=lambda kv: -kv[1])))

    if not a.apply:
        print("\ndry run; pass --apply to perform it")
        for s, d in (moves + dups)[:10]:
            print(f"  {s.parent.name}/{s.name[:46]:46s} -> {d.parent.name}/")
        return 0

    for s, d in moves + dups:
        d.parent.mkdir(parents=True, exist_ok=True)
        assert s.is_file() and not d.exists(), f"{s} -> {d}"
        shutil.move(str(s), str(d))
    print(f"\napplied: {len(moves)} moved, {len(dups)} quarantined in {quar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
