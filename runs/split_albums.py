#!/usr/bin/env python
"""Split whole-album tab files into one file per track, at the markers the files already carry.

    .venv/bin/python runs/split_albums.py --review          # decide and explain, change nothing
    .venv/bin/python runs/split_albums.py --apply

423 files run 400+ bars - 0.69% of the files but 3.30% of all tokens. The context is 4096 tokens while
"Appetite For Destruction" is 179,212, so the model never sees a song boundary in these files: it learns
that music simply continues. Splitting them is the one dataset change that fixes something the model
provably gets wrong, rather than tidying labels it barely reads.

The split points are already in the files. Guitar Pro measure headers carry markers, and on real album
files those markers are the track names at the right bars:

    Guns N' Roses - Appetite For Destruction   1822 bars, 12 markers
        bar    1  Welcome To The Jungle
        bar  136  It's So Easy
        bar  668  Paradise City

THE TRAP, and why this script classifies rather than splits on every marker: a long SONG also carries
markers, and they are section names. "Mastodon - Hearts Alive" has 533 bars and 20 markers reading
Intro / Verse 1 / Chorus / Instrumental - splitting it would shred one song into twenty fragments. Of
the 423 long files, 167 have 20+ markers and many of those are single songs. "The Cure - Seventeen
Seconds" is worse: an album whose markers mix numbered track titles with section names.

So a marker counts as a track boundary only if its title does not look like a section name, and a file is
only treated as an album if the resulting segments are song-length. Everything else is reported for
review and left alone. Being wrong here costs real transcriptions, so the rule errs toward not splitting.

Originals are MOVED to data/_split_originals_<date>/ rather than deleted, and prepare ignores any
directory not starting with "tabs - " (riff.py:544), so the original leaves the corpus exactly when its
pieces enter it. Verified end to end before use: split -> gp.write -> gp.parse -> riff.encode_song gives
122 bars / 13,083 tokens / tempo 176 for one Dookie track.
"""
from __future__ import annotations

import argparse
import collections
import copy
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import guitarpro as gp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402

DATA = riff.ROOT / "data"
EXT = {".gp3", ".gp4", ".gp5"}
MIN_BARS = 400          # candidate threshold: 4x the 102-bar median song
MIN_SEG = 32            # a track shorter than this is implausible; such a split is rejected
MIN_TRACKS = 2

# A filename keyword is the only RELIABLE album signal. Bar statistics cannot tell a 4-track album from a
# 15-minute progressive song with four named sections - both are 500 bars with four markers 100+ bars
# apart. "Gallowbraid - Ashen Eidolon" and "Jinjer - Inhale Dont Breathe" are single songs that the
# segment-length rule happily proposed splitting into 4 and 7 "tracks". With --confident, only files whose
# NAME says they are a compilation get split automatically; the rest are listed for a human to judge.
ALBUMISH = re.compile(r"\b(medley|best of|part ?\d|pt\.? ?\d|album|discography|collection|trilogy|"
                      r"compilation|full album|anthology|greatest hits|side [ab]\b)\b", re.I)

SECTION = re.compile(
    r"^\W*(intro|outro|verse|chorus|pre[\s-]?chorus|post[\s-]?chorus|bridge|solo|riff|instrumental|"
    r"interlude|break(down)?|coda|refrain|ending|theme|vamp|hook|fill|link|tag|head|build|drop|"
    r"main|part|section|repeat|end|a|b|c|d|guitar|lead|rhythm|bass|drums|clean|heavy|slow|fast)"
    r"[\s\d'ivx.:#()\[\]-]*\W*$", re.I)


def is_section(title: str) -> bool:
    """True when a marker looks like a part of one song rather than the name of a track."""
    t = (title or "").strip()
    if not t:
        return True                      # blank markers carry no track name
    if SECTION.match(t):
        return True
    # "1 - "A Reflection"" or "No. 3" or "12. Title" are track names despite leading digits
    return False


def plan_for(p: Path):
    """(verdict, reason, [(lo, hi, title)]) for one file; lo/hi are 0-based half-open bar indices."""
    try:
        song = gp.parse(str(p))
    except Exception as e:
        return "error", f"parse failed: {e}", []
    n = len(song.measureHeaders)
    # The csv's bar counts come from the ENCODED corpus, where repeats are expanded, so a 107-measure file
    # can show 400+ encoded bars. Album-ness is a property of the written score, so the raw measure count
    # decides: without this check the plan happily "split" a 107-measure file into 2 "tracks".
    if n < MIN_BARS:
        return "skip", f"only {n} raw measures (the csv counts repeat-expanded bars)", []
    marks = [(i, (m.marker.title or "").strip()) for i, m in enumerate(song.measureHeaders) if m.marker]
    if not marks:
        return "skip", f"{n} bars, no markers", []
    track_marks = [(i, t) for i, t in marks if not is_section(t)]
    if len(track_marks) < MIN_TRACKS:
        return "skip", f"{n} bars, {len(marks)} markers but {len(track_marks)} look like track names", []
    # segments run from one track marker to the next
    bounds = [i for i, _ in track_marks]
    if bounds[0] > 0:                    # material before the first named track
        segs = [(0, bounds[0], "")] + [(bounds[k], bounds[k + 1] if k + 1 < len(bounds) else n, track_marks[k][1])
                                       for k in range(len(bounds))]
    else:
        segs = [(bounds[k], bounds[k + 1] if k + 1 < len(bounds) else n, track_marks[k][1])
                for k in range(len(bounds))]
    segs = [(lo, hi, t) for lo, hi, t in segs if t and hi - lo >= 1]
    if len(segs) < MIN_TRACKS:
        return "skip", f"{n} bars, only {len(segs)} usable segments", []
    short = [t for lo, hi, t in segs if hi - lo < MIN_SEG]
    if short:
        return "review", (f"{n} bars, {len(segs)} segments but {len(short)} under {MIN_SEG} bars "
                          f"(e.g. {short[0][:28]!r}) - looks like sections, not tracks"), segs
    return "split", f"{n} bars -> {len(segs)} tracks", segs


def safe(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r'[\\/:*?"<>|]', "_", name)).strip()[:90]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--confident", action="store_true",
                    help="only split files whose NAME says they are an album/medley/part (see ALBUMISH)")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    # Candidates come from runs/tab_quality.csv, which already holds every song's bar count. The first
    # version of this parsed all ~60,000 files with gp.parse just to find the long ones and was killed by
    # its own 50-minute timeout having printed nothing. Paths in that csv predate the moves and the cull,
    # so each is resolved by BASENAME against what is on disk now.
    qcsv = riff.ROOT / "runs" / "tab_quality.csv"
    if not qcsv.is_file():
        sys.exit(f"{qcsv} missing - run runs/tab_quality.py --csv {qcsv} first")
    ondisk = {}
    for d in DATA.iterdir():
        # Album folders are excluded, as everywhere else in this cleanup: those are the user's own songs
        # and their chosen inspirations, and splitting one would silently rewrite an album assignment.
        # Three candidates do sit there (Pain Of Salvation - Icon under vc2 - insp among them); they are
        # reported by --review and left alone.
        if (d.is_dir() and d.name.lower().startswith("tabs - ")
                and not d.name.lower().startswith("tabs - own - ")):
            for p in d.rglob("*"):
                if p.suffix.lower() in EXT:
                    ondisk.setdefault(p.name, p)
    import csv as _csv
    cands, missing = [], 0
    with open(qcsv, encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            if int(row["bars"]) < MIN_BARS:
                continue
            p = ondisk.get(Path(row["file"]).name)
            if p is None:
                missing += 1
            else:
                cands.append(p)
    cands = sorted(set(cands))
    if missing:
        print(f"({missing} long files from the csv are no longer on disk - moved or culled)")
    if a.limit:
        cands = cands[:a.limit]
    print(f"{len(cands)} files with {MIN_BARS}+ bars\n", flush=True)

    verdicts = collections.Counter()
    plans = []
    for p in cands:
        v, why, segs = plan_for(p)
        if v == "split" and a.confident and not ALBUMISH.search(p.stem):
            v, why = "review", why + " - but the name does not say album/medley, could be one long song"
        verdicts[v] += 1
        plans.append((p, v, why, segs))
        if a.review or v in ("review", "error"):
            print(f"[{v:6s}] {p.parent.name.replace('tabs - ', '')[:13]:13s} {p.stem[:44]:44s} {why}")

    print("\n" + ", ".join(f"{k}={v}" for k, v in verdicts.most_common()))
    if not a.apply:
        ex = next((x for x in plans if x[1] == "split"), None)
        if ex:
            print(f"\nexample plan - {ex[0].stem}:")
            for lo, hi, t in ex[3][:8]:
                print(f"   bars {lo + 1:5d}-{hi:5d}  {t[:50]}")
        print("\nnothing written; pass --apply")
        return 0

    quar = DATA / f"_split_originals_{date.today():%Y%m%d}"
    made = 0
    for p, v, why, segs in plans:
        if v != "split":
            continue
        artist = p.stem.split(" - ")[0].strip() if " - " in p.stem else p.stem
        song = gp.parse(str(p))
        wrote = []
        for lo, hi, title in segs:
            out = copy.deepcopy(song)
            out.measureHeaders = out.measureHeaders[lo:hi]
            for t in out.tracks:
                t.measures = t.measures[lo:hi]
            for k, h in enumerate(out.measureHeaders):
                h.number = k + 1
            if out.measureHeaders:
                out.measureHeaders[0].marker = None
            dst = p.parent / f"{safe(artist)} - {safe(title)}{p.suffix}"
            if dst.exists():
                dst = p.parent / f"{safe(artist)} - {safe(title)}__split{p.suffix}"
            if dst.exists():
                continue
            gp.write(out, str(dst))
            wrote.append(dst)
        if len(wrote) < MIN_TRACKS:                     # back out rather than leave a half-split album
            for w in wrote:
                w.unlink(missing_ok=True)
            print(f"  BACKED OUT {p.name}: only {len(wrote)} files written")
            continue
        d = quar / p.parent.name / p.name
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(d))
        made += len(wrote)
        print(f"  {p.stem[:48]:48s} -> {len(wrote)} tracks")
    print(f"\n{made} track files written; {verdicts['split']} albums moved to {quar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
