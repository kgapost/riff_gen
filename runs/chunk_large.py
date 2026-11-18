#!/usr/bin/env python
"""Guarantee no tab is longer than --max-bars, chunking at the best boundary available.

    .venv/bin/python runs/chunk_large.py --review
    .venv/bin/python runs/chunk_large.py --apply --max-bars 300 --target 150

Splitting on MEANING turned out to have a ceiling: markers are reliable and bought 25 albums, silence is
~60 % precise even at its best and bought 5 more by hand, and the remaining ~390 long files are genuinely
undecidable without listening - both signals wanted to cut up Jinjer's "Inhale Dont Breathe", a single
song. So this stops asking what a boundary MEANS and enforces a size cap instead, which was the actual
request: no very large tabs.

That is defensible for training rather than merely tidy. The context is 4096 tokens and these files run to
215,000 (Gojira) and 179,000 (Appetite For Destruction), so the model never sees one whole anyway. What it
does see is a window that never contains a song ending, which teaches it that music continues forever -
and 3.3 % of all tokens were doing that. A chunk boundary in the wrong place creates one false song end;
leaving the file whole creates thousands of windows with no true one.

Damage is minimised by snapping each cut to the best boundary within --snap measures of the target:
    1. a measure carrying a marker   (a real section or track name)
    2. the middle of a silent run    (no note in any track)
    3. the exact target measure      (last resort)

Album folders ARE included here, unlike the rest of this cleanup: a chunk keeps its parent folder, so the
album and src labels survive the split intact, and three 400+ bar inspirations sit under them.

Originals go to data/_chunked_originals_<date>/, not deleted; prepare ignores any directory whose name
does not start with "tabs - " (riff.py:544).
"""
from __future__ import annotations

import argparse
import collections
import copy
import shutil
import sys
from datetime import date
from pathlib import Path

import guitarpro as gp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402

DATA = riff.ROOT / "data"
EXT = {".gp3", ".gp4", ".gp5"}


def silent(song, i: int) -> bool:
    for t in song.tracks:
        if i >= len(t.measures):
            continue
        for v in t.measures[i].voices:
            for b in v.beats:
                if b.notes:
                    return False
    return True


def cut_points(song, target: int, snap: int) -> list[int]:
    """Measure indices to cut at, snapped to markers or silence where one is close enough."""
    n = len(song.measureHeaders)
    marks = {i for i, m in enumerate(song.measureHeaders) if m.marker}
    quiet = [silent(song, i) for i in range(n)]
    runs = []
    i = 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            runs.append((i + j) // 2)
            i = j
        else:
            i += 1
    cuts, at = [], target
    while at < n - target // 2:
        best = None
        for cand in sorted(marks | set(runs), key=lambda x: abs(x - at)):
            if abs(cand - at) <= snap and cand > (cuts[-1] if cuts else 0) + target // 2:
                best = cand
                break
        cuts.append(best if best is not None else at)
        at = cuts[-1] + target
    return cuts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bars", type=int, default=300, help="any file longer than this is chunked")
    ap.add_argument("--target", type=int, default=150, help="aim for chunks of about this many measures")
    ap.add_argument("--snap", type=int, default=40, help="how far to look for a marker or silence")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    files = []
    for d in sorted(DATA.iterdir()):
        if d.is_dir() and d.name.lower().startswith("tabs - "):
            files += [p for p in sorted(d.rglob("*")) if p.suffix.lower() in EXT]
    print(f"scanning {len(files)} files for raw measure count > {a.max_bars}", flush=True)

    plans, errs = [], 0
    for p in files:
        try:
            song = gp.parse(str(p))
        except Exception:
            errs += 1
            continue
        n = len(song.measureHeaders)
        if n <= a.max_bars:
            continue
        cuts = cut_points(song, a.target, a.snap)
        segs = []
        prev = 0
        for cpt in cuts + [n]:
            if cpt > prev:
                segs.append((prev, cpt))
            prev = cpt
        if len(segs) < 2:
            continue
        plans.append((p, n, segs))
        if a.review:
            print(f"  {n:5d} -> {len(segs):2d} chunks {[hi - lo for lo, hi in segs][:10]}  "
                  f"{p.parent.name.replace('tabs - ', '')[:13]:13s} {p.stem[:40]}", flush=True)

    tot = sum(len(s) for _, _, s in plans)
    longest = max((max(hi - lo for lo, hi in s) for _, _, s in plans), default=0)
    print(f"\n{len(plans)} files over {a.max_bars} bars -> {tot} chunks; "
          f"longest resulting chunk {longest} bars; {errs} unparseable")
    inalbum = sum(1 for p, _, _ in plans if p.parent.name.lower().startswith("tabs - own - "))
    print(f"{inalbum} of them sit in album folders (chunks keep the folder, so album/src labels survive)")
    if not a.apply:
        print("\nnothing written; pass --apply")
        return 0

    quar = DATA / f"_chunked_originals_{date.today():%Y%m%d}"
    made = 0
    for p, n, segs in plans:
        song = gp.parse(str(p))
        wrote = []
        for k, (lo, hi) in enumerate(segs, 1):
            out = copy.deepcopy(song)
            out.measureHeaders = out.measureHeaders[lo:hi]
            for t in out.tracks:
                t.measures = t.measures[lo:hi]
            for j, h in enumerate(out.measureHeaders):
                h.number = j + 1
            dst = p.parent / f"{p.stem} __c{k:02d}{p.suffix}"
            if dst.exists():
                continue
            gp.write(out, str(dst))
            wrote.append(dst)
        if len(wrote) < 2:
            for w in wrote:
                w.unlink(missing_ok=True)
            print(f"  BACKED OUT {p.name}")
            continue
        d = quar / p.parent.name / p.name
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(d))
        made += len(wrote)
    print(f"\n{made} chunks written from {len(plans)} files; originals in {quar}")
    print(f"{collections.Counter()}" if False else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
