#!/usr/bin/env python
"""Flag tabs whose music looks more like another genre's than like the folder they sit in.

    .venv/bin/python runs/mislabel_scan.py [--top N] [--genre G] [--csv out.csv] [--mem-gb N]

Writes a ranked list of candidates per genre. Reads only the encoded corpus - no GPU, no training.

WHY: stage B2's shred run made the loss WORSE (0.5601 -> 0.5710) while classical improved by 0.018. The
shred folder turned out to hold ~122 film and game scores (Nobuo Uematsu, Akira Yamaoka, Henry Mancini,
John Williams, Nino Rota, Danny Elfman, Hans Zimmer) alongside the real shred artists, so ~10% of the
label's draws teach a different idiom. Hand-checking 28 folders of 61k tabs is not practical; this
narrows it to a few dozen candidates per genre to review by ear.

HOW: each song becomes a unigram distribution over the token vocabulary, Hellinger-transformed (sqrt, so
plain Euclidean distance is a sensible metric on distributions). Each genre gets a centroid from its
TRAIN songs. A song is flagged when some other genre's centroid is closer than its own, and the margin
(own distance minus best other distance) ranks the candidates.

MEMORY: two streaming passes, never an all-songs matrix. The first version here held every song's
histogram at once - 61k x 65536 float64 is 32GB - and the OOM killer took out the stage B2 trainer with
it on 2026-11-17, losing 81% of a run. Now only the 28 centroids are resident (a few MB) and each song's
histogram is discarded as soon as it has been used. --mem-gb additionally caps this process's address
space with RLIMIT_AS, so a future mistake here raises MemoryError in THIS process instead of letting the
kernel pick a victim; it is deliberately low enough to be harmless next to a training run.

WHAT THIS IS NOT: evidence. A unigram histogram knows nothing about structure - clustering WITHIN
acoustic scored a silhouette of only 0.09, so this signal is weak on its own. Nearest-centroid across
LABELLED genres is a better-posed question than unsupervised clustering, but it will still flag
legitimately unusual songs (an acoustic interlude on a metal album, a clean-toned intro). Treat every row
as "worth a listen", never as "move this file". The two checks that make a row credible are printed
alongside it: whether the ARTIST's other songs sit in the suggested genre, and how many of that artist's
songs got flagged the same way - a composer whose entire catalogue points elsewhere is a filing mistake,
one odd track is just an odd track.
"""
from __future__ import annotations

import argparse
import collections
import csv
import resource
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402

NV = 65536


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25, help="candidates to print per genre")
    ap.add_argument("--genre", help="only report this genre (all are still needed for the centroids)")
    ap.add_argument("--csv", help="write every candidate here")
    ap.add_argument("--mem-gb", type=float, default=4.0, help="hard cap on this process's memory")
    a = ap.parse_args()

    if a.mem_gb > 0:
        cap = int(a.mem_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    c = riff.Corpus(riff.ROOT / "data_encoded" / "encoded")

    def hist(i):
        arr = np.asarray(c.view(i).tok)
        h = np.bincount(arr, minlength=NV).astype(np.float32)[:NV]
        t = h.sum()
        if t:
            np.divide(h, t, out=h)
            np.sqrt(h, out=h)                        # Hellinger
        return h

    songs = [(i, s["labels"]["genre"], s["split"]) for i, s in enumerate(c.songs) if s["labels"].get("genre")]
    genres = sorted({g for _, g, _ in songs})
    gi = {g: j for j, g in enumerate(genres)}

    # pass 1: centroids from the train split only, accumulated in place
    acc = np.zeros((len(genres), NV), dtype=np.float64)
    cnt = np.zeros(len(genres), dtype=np.int64)
    print(f"pass 1/2: centroids over {len(songs)} labelled songs", file=sys.stderr)
    for n, (i, g, sp) in enumerate(songs):
        if sp != "train":
            continue
        acc[gi[g]] += hist(i)
        cnt[gi[g]] += 1
        if n % 10000 == 0:
            print(f"  {n}/{len(songs)}", file=sys.stderr)
    if (cnt == 0).any():
        bad = [genres[j] for j in np.nonzero(cnt == 0)[0]]
        print(f"genres with no train songs, skipped: {bad}", file=sys.stderr)
    C = (acc / np.maximum(cnt, 1)[:, None]).astype(np.float32)
    Cn = (C ** 2).sum(1)                             # for the squared-distance identity
    del acc

    # pass 2: nearest centroid per song
    print("pass 2/2: distances", file=sys.stderr)
    cands = collections.defaultdict(list)
    for n, (i, g, _sp) in enumerate(songs):
        h = hist(i)
        d = (h ** 2).sum() - 2.0 * (C @ h) + Cn      # squared distance to every centroid
        d[cnt == 0] = np.inf
        own = d[gi[g]]
        d[gi[g]] = np.inf
        j = int(np.argmin(d))
        if d[j] < own:
            cands[g].append((float(own - d[j]), genres[j], i))
        if n % 10000 == 0:
            print(f"  {n}/{len(songs)}", file=sys.stderr)

    # the two corroborating checks: where the artist's other songs live, and how often this artist is flagged
    art_gen = collections.defaultdict(collections.Counter)
    for i, g, _ in songs:
        art_gen[Path(c.songs[i]["file"]).stem.split("-")[0].strip()][g] += 1
    flagged = collections.Counter()
    for g, rows in cands.items():
        for _, _, i in rows:
            flagged[(Path(c.songs[i]["file"]).stem.split("-")[0].strip(), g)] += 1

    total = collections.Counter(g for _, g, _ in songs)
    out = []
    for g in genres:
        rows = sorted(cands[g], key=lambda r: -r[0])
        if not a.genre or g == a.genre:
            print(f"\n=== {g}: {len(rows)} of {total[g]} songs "
                  f"({100 * len(rows) / max(total[g], 1):.1f}%) sit closer to another genre")
            for margin, to, i in rows[:a.top]:
                art = Path(c.songs[i]["file"]).stem.split("-")[0].strip()
                print(f"  {margin:8.5f} -> {to:20s} {Path(c.songs[i]['file']).stem[:50]:50s} "
                      f"[artist: {art_gen[art].get(to, 0)} in {to}, "
                      f"{flagged[(art, g)]}/{art_gen[art][g]} of its {g} flagged]")
        for margin, to, i in rows:
            art = Path(c.songs[i]["file"]).stem.split("-")[0].strip()
            out.append({"genre": g, "suggest": to, "margin": round(margin, 6),
                        "file": c.songs[i]["file"], "artist": art,
                        "artist_songs_in_suggested": art_gen[art].get(to, 0),
                        "artist_songs_in_genre": art_gen[art][g],
                        "artist_flagged_in_genre": flagged[(art, g)]})

    if a.csv and out:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"\n{len(out)} candidates -> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
