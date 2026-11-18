#!/usr/bin/env python
"""Try splitting the long files that have no usable markers, on runs of EMPTY measures instead.

    .venv/bin/python runs/split_on_silence.py --review
    .venv/bin/python runs/split_on_silence.py --apply --min-silence 2

split_albums.py splits on marker titles and reaches only 23 of the 400+ bar files with confidence; 402
remain. Many album files separate their tracks with a couple of silent measures rather than naming them,
so this looks for runs of >= --min-silence consecutive measures in which NO track plays a note, and
treats them as track boundaries.

It reports before it writes, because the signal is weaker than a marker and the failure mode is the same
one that nearly shredded Gallowbraid's Ashen Eidolon: a song with a two-bar rest before the last chorus
looks exactly like two tracks with a gap between them. Guards:

  --min-silence   consecutive silent measures needed (2 by default, as asked)
  MIN_SEG         a segment under this many bars is not a plausible track, and a plan containing one is
                  rejected rather than trimmed
  MIN_TRACKS      at least this many segments, or there is nothing to split

"Silent" means no note in ANY track - guitars, bass and drums. A measure where only the drums play is not
silence and does not separate tracks, which matters because album files often run drums through a break.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import shutil
import sys
from datetime import date
from pathlib import Path

import guitarpro as gp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402

DATA = riff.ROOT / "data"
EXT = {".gp3", ".gp4", ".gp5"}
MIN_BARS = 400
MIN_SEG = 32
MIN_TRACKS = 2


def silent_runs(song, need: int) -> list[tuple[int, int]]:
    """[(lo, hi)] half-open runs of >= need consecutive measures with no note in any track."""
    n = len(song.measureHeaders)
    quiet = []
    for i in range(n):
        has = False
        for t in song.tracks:
            if i >= len(t.measures):
                continue
            for v in t.measures[i].voices:
                for b in v.beats:
                    if b.notes:
                        has = True
                        break
                if has:
                    break
            if has:
                break
        quiet.append(not has)
    runs, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if j - i >= need:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def plan_for(p: Path, need: int):
    try:
        song = gp.parse(str(p))
    except Exception as e:
        return "error", f"parse failed: {e}", []
    n = len(song.measureHeaders)
    if n < MIN_BARS:
        return "skip", f"only {n} raw measures", []
    runs = silent_runs(song, need)
    if not runs:
        return "skip", f"{n} bars, no run of {need}+ silent measures", []
    # boundaries are the silent runs; segments are the music between them
    bounds, prev = [], 0
    for lo, hi in runs:
        if lo > prev:
            bounds.append((prev, lo))
        prev = hi
    if prev < n:
        bounds.append((prev, n))
    segs = [(lo, hi) for lo, hi in bounds if hi > lo]
    if len(segs) < MIN_TRACKS:
        return "skip", f"{n} bars, {len(runs)} silent runs -> {len(segs)} segment(s)", []
    short = [hi - lo for lo, hi in segs if hi - lo < MIN_SEG]
    if short:
        return "review", (f"{n} bars, {len(segs)} segments but {len(short)} under {MIN_SEG} bars "
                          f"(shortest {min(short)}) - the silences are rests, not track breaks"), segs
    return "split", f"{n} bars, {len(runs)} silent runs -> {len(segs)} segments", segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-silence", type=int, default=2)
    a = ap.parse_args()

    qcsv = riff.ROOT / "runs" / "tab_quality.csv"
    ondisk = {}
    for d in DATA.iterdir():
        if (d.is_dir() and d.name.lower().startswith("tabs - ")
                and not d.name.lower().startswith("tabs - own - ")):
            for p in d.rglob("*"):
                if p.suffix.lower() in EXT:
                    ondisk.setdefault(p.name, p)
    cands = []
    with open(qcsv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["bars"]) >= MIN_BARS and (p := ondisk.get(Path(row["file"]).name)):
                cands.append(p)
    cands = sorted(set(cands))
    print(f"{len(cands)} candidates, silence run >= {a.min_silence} measures\n", flush=True)

    verdicts = collections.Counter()
    plans = []
    for p in cands:
        v, why, segs = plan_for(p, a.min_silence)
        verdicts[v] += 1
        plans.append((p, v, why, segs))
        if a.review or v == "error":
            print(f"[{v:6s}] {p.stem[:50]:50s} {why}", flush=True)

    print("\n" + ", ".join(f"{k}={v}" for k, v in verdicts.most_common()))
    sp = [x for x in plans if x[1] == "split"]
    if sp:
        print(f"\n{len(sp)} would split; segment counts: "
              f"{collections.Counter(len(x[3]) for x in sp).most_common()}")
        print("examples:")
        for p, _, why, segs in sp[:8]:
            print(f"   {p.stem[:46]:46s} -> {len(segs)} segs "
                  f"{[hi - lo for lo, hi in segs][:8]}")
    if not a.apply:
        print("\nnothing written; pass --apply")
        return 0

    quar = DATA / f"_split_silence_originals_{date.today():%Y%m%d}"
    made = 0
    for p, v, why, segs in plans:
        if v != "split":
            continue
        song = gp.parse(str(p))
        wrote = []
        for k, (lo, hi) in enumerate(segs, 1):
            out = copy.deepcopy(song)
            out.measureHeaders = out.measureHeaders[lo:hi]
            for t in out.tracks:
                t.measures = t.measures[lo:hi]
            for j, h in enumerate(out.measureHeaders):
                h.number = j + 1
            dst = p.parent / f"{p.stem} __part{k:02d}{p.suffix}"
            if dst.exists():
                continue
            gp.write(out, str(dst))
            wrote.append(dst)
        if len(wrote) < MIN_TRACKS:
            for w in wrote:
                w.unlink(missing_ok=True)
            print(f"  BACKED OUT {p.name}")
            continue
        d = quar / p.parent.name / p.name
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(d))
        made += len(wrote)
        print(f"  {p.stem[:46]:46s} -> {len(wrote)} parts")
    print(f"\n{made} part files written, originals in {quar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
