"""Synthetic songs in DadaGP token format (and Drop A GP5 files) for tests and smoke runs.

Not music anyone should listen to, but with the structure the pipeline must handle:
repeating riffs, a non-repeating lead line, simple chug riffs, double-tracked guitars, drop
tunings written DadaGP-style (fret -2), 7-string songs, odd meters, repeat signs, silent
guitar stretches (gaps), bass and drums that must be removed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from riffgen.dataset.song import parse_dadagp, raw_to_song
from riffgen.gp import writer
from riffgen.tokens import tunings as T
from riffgen.tokens.vocab import split_wait

SIXTEENTH = 240


def _riff(rng, bars: int, ticks: int, low: int, n_strings: int, kind: str) -> list[list[tuple]]:
    """Per bar: list of (tick, [(string, fret), ...], fx list)."""
    out = []
    if kind == "simple":
        for _ in range(bars):
            out.append([(t, [(n_strings, low)], ["nfx:palm_mute"]) for t in range(0, ticks, 2 * SIXTEENTH)])
        return out
    if kind == "lead":
        for _ in range(bars):
            ev = []
            for t in range(0, ticks, SIXTEENTH):
                s = int(rng.integers(1, 4))
                ev.append((t, [(s, int(rng.integers(12, 20)))], ["nfx:vibrato"] if rng.random() < 0.2 else []))
            out.append(ev)
        return out
    # riff: a 1-bar or 2-bar motif repeated over the requested bars
    unit = 1 if rng.random() < 0.5 else 2
    motif = []
    for _ in range(unit):
        ev = []
        t = 0
        while t < ticks:
            step = int(rng.choice([SIXTEENTH, 2 * SIXTEENTH, 3 * SIXTEENTH]))
            if rng.random() < 0.3:
                f = low + int(rng.choice([0, 3, 5, 7]))
                notes = [(n_strings, f), (n_strings - 1, f + 2)]
                fx = []
            else:
                notes = [(n_strings, low + int(rng.choice([0, 0, 1, 3])))]
                fx = ["nfx:palm_mute"] if rng.random() < 0.6 else []
            ev.append((t, notes, fx))
            t += step
        motif.append(ev)
    for b in range(bars):
        out.append(motif[b % unit])
    return out


def synth_song_tokens(rng, *, n_strings: int = 6, drop: bool = False, downtune: int = 0, two_guitars: bool = True,
                      with_lead: bool = True, with_repeat: bool = True, gap_bars: int = 0,
                      odd_meter: bool = False) -> list[str]:
    tempo = int(rng.choice([90, 120, 140, 160, 180]))
    tok = ["artist:synth", f"downtune:{downtune}", f"tempo:{tempo}", "start"]
    plan = [("riff", 2), ("riff", 4), ("simple", 4), ("riff", 4)]
    if with_lead:
        plan.append(("lead", 4))
    plan += [("riff", 4), ("riff", 2)]
    sections = []
    cache = {}
    for i, (kind, bars) in enumerate(plan):
        key = (kind, i % 3) if kind == "riff" else (kind, i)
        if key not in cache or kind != "riff":
            ticks = 3360 if (odd_meter and kind == "riff" and i % 3 == 1) else 3840
            low = 0
            cache[key] = (_riff(rng, max(bars, 2), ticks, low, n_strings, kind), ticks)
        riff, ticks = cache[key]
        sections.append((kind, [riff[b % len(riff)] for b in range(bars)], ticks))
    bars_out = []
    for kind, bars, ticks in sections:
        for bar in bars:
            bars_out.append((kind, bar, ticks))
    if gap_bars:
        at = len(bars_out) // 2
        bars_out[at:at + gap_bars] = [("gap", [], 3840)] * gap_bars
    first = True
    for bi, (kind, bar, ticks) in enumerate(bars_out):
        tok.append("new_measure")
        if with_repeat and bi == 2:
            tok.append("measure:repeat_open")
        if with_repeat and bi == 3:
            tok.append("measure:repeat_close:1")
        events: dict[int, list[str]] = {}
        for (t, notes, fx) in bar:
            lst = events.setdefault(t, [])
            for s, f in notes:
                fret = f - 2 if (drop and s >= 6) else f
                lst.append(f"distorted0:note:s{s}:f{fret}")
                lst.extend(fx)
            if two_guitars and kind != "lead":
                for s, f in notes:
                    fret = f - 2 if (drop and s >= 6) else f
                    lst.append(f"distorted1:note:s{s}:f{fret}")
            if kind != "gap":
                lst.append(f"bass:note:s5:f{notes[0][1]}")
        for t in range(0, ticks, 960):
            events.setdefault(t, []).append("drums:note:36")
        if first:
            first = False
        cur = 0
        for t in sorted(events):
            if t > cur:
                tok += [f"wait:{w}" for w in split_wait(t - cur)]
                cur = t
            tok += events[t]
        tok += [f"wait:{w}" for w in split_wait(ticks - cur)]
    tok.append("end")
    return tok


def write_dadagp_corpus(root: str | Path, n_songs: int = 24, seed: int = 0) -> list[Path]:
    root = Path(root)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n_songs):
        kind = i % 4
        toks = synth_song_tokens(rng, n_strings=7 if kind == 3 else 6, drop=kind == 1, downtune=-2 if kind == 2 else 0,
                                 two_guitars=i % 2 == 0, with_lead=i % 3 == 0, with_repeat=i % 2 == 1,
                                 gap_bars=5 if i % 5 == 0 else 0, odd_meter=i % 6 == 0)
        artist = f"Band{i % 6}"
        p = root / artist[0] / artist / f"song{i:02d}.gp5.tokens.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(toks), encoding="utf-8")
        paths.append(p)
    return paths


def write_album(root: str | Path, n_songs: int = 8, seed: int = 1, unfinished: int = 1) -> dict:
    """Drop A GP5 files: <root>/own, <root>/favourites (with a __v2 copy), <root>/songs (with gaps)."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    tracks_tuning = {0: T.DROP_A7, 1: T.DROP_A7}
    files = {"own": [], "favourites": [], "songs": []}

    def write(path: Path, toks: list[str], marker_every: int = 0):
        raw = parse_dadagp(toks)
        song = raw_to_song(raw, path.stem, {s: tracks_tuning[s] for s in (0, 1)}, dadagp_normalised=False, kind="own")
        from riffgen.dataset.song import ts_from_ticks
        for k, bar in enumerate(song.bars):
            bar.ts = ts_from_ticks(bar.ticks)
            if marker_every and k % marker_every == 0:
                bar.marker = ["Intro", "Verse", "Chorus", "Bridge"][(k // marker_every) % 4]
        song.tempo = 140
        for s, tr in song.tracks.items():
            tr.name = "Rhythm" if s == 0 else "Lead"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer.write_bars(path, song.bars, song.tracks, song.tempo, title=path.stem)
        files[path.parent.name].append(path)

    for i in range(n_songs):
        toks = synth_song_tokens(rng, n_strings=7, two_guitars=True, with_lead=i % 2 == 0, with_repeat=False)
        folder = "own" if i < n_songs // 2 else "favourites"
        write(root / folder / f"track{i:02d}.gp5", toks, marker_every=8)
        if folder == "favourites" and i == n_songs - 1:
            write(root / folder / f"track{i:02d}__v2.gp5", toks, marker_every=8)
    for j in range(unfinished):
        toks = synth_song_tokens(rng, n_strings=7, two_guitars=True, with_lead=False, with_repeat=False, gap_bars=6)
        write(root / "songs" / f"unfinished{j:02d}.gp5", toks, marker_every=8)
    return files


NEEDLE = ("distorted0:note:s1:f13",)     # a note that only the intro riff uses


def synth_long_song_tokens(rng, n_bars: int = 160) -> list[str]:
    """A long song for the long-context tests: intro riff X (4 bars, with a unique note), verse/chorus/
    bridge sections of 8 bars that come back several times, and X again in the last 4 bars."""
    tempo = 140
    tok = ["artist:synth", "downtune:0", f"tempo:{tempo}", "start"]
    ticks = 3840
    x_bar = [(0, [(1, 13)], []), (960, [(6, 0)], ["nfx:palm_mute"]), (1440, [(6, 3)], ["nfx:palm_mute"]),
             (1920, [(1, 13), (6, 0)], []), (2880, [(6, 5)], ["nfx:palm_mute"])]
    parts = {name: _riff(rng, 8, ticks, low, 6, "riff") for name, low in (("V", 0), ("C", 5), ("B", 8))}
    form = ["X"]
    while sum(4 if f == "X" else 8 for f in form) < n_bars - 4:
        form += ["V", "C"] if len(form) % 5 else ["B", "V", "C"]
    form.append("X")
    bars = []
    for f in form:
        if f == "X":
            bars += [x_bar] * 4
        else:
            bars += parts[f]
    for bar in bars:
        tok.append("new_measure")
        events: dict[int, list[str]] = {}
        for t, notes, fx in bar:
            lst = events.setdefault(t, [])
            for st, fr in notes:
                lst.append(f"distorted0:note:s{st}:f{fr}")
                lst.extend(fx)
            lst.append(f"distorted1:note:s{notes[-1][0]}:f{notes[-1][1]}")
        cur = 0
        for t in sorted(events):
            if t > cur:
                tok += [f"wait:{w}" for w in split_wait(t - cur)]
                cur = t
            tok += events[t]
        tok += [f"wait:{w}" for w in split_wait(ticks - cur)]
    tok.append("end")
    return tok


def write_long_corpus(root: str | Path, n_songs: int = 4, n_bars: int = 160, seed: int = 5) -> list[Path]:
    root = Path(root)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n_songs):
        p = root / "L" / "Long" / f"long{i:02d}.gp5.tokens.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(synth_long_song_tokens(rng, n_bars)), encoding="utf-8")
        paths.append(p)
    return paths
