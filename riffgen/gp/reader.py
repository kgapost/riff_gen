"""Read Guitar Pro files (GP3/4/5) with PyGuitarPro 0.6.

Two jobs:

1. ``read_gp_meta``: metadata the DadaGP token format does not keep -- time signatures
   (D23), section markers (D25), track names and the real tuning of every
   guitar slot. Used for DadaGP songs when the original GP file is available.

2. ``encode_own_gp``: turn one of *your* GP5 files into tokens with the official DadaGP
   encoder. The encoder rejects 7-string Drop A (A E A D G B E is not one of its four
   tuning shapes), so we use a **tuning bypass**: non-guitar tracks are removed first (so a
   5-string bass in an odd tuning cannot break the encoder), guitar tracks are re-labelled
   as standard 6/7-string for the encoder only, and the real tuning is restored afterwards
   in our ``tuning:`` token. DadaGP writes string + fret (not pitch), so the tab itself is
   unchanged and frets stay physical.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

import guitarpro

from riffgen.gp import dadagp_bridge
from riffgen.tokens import tunings as tunings_mod

GUITAR_NAME_RE = re.compile(r"guit|gtr|git\b|rhythm|lead|riff|7[- ]?str", re.I)
CLEAN_NAME_RE = re.compile(r"clean|acoust|nylon|steel|ac\.|classical", re.I)


class SkipSong(Exception):
    """The song cannot be used; the message goes to the skip log."""


@dataclass
class GPMeta:
    n_measures: int
    ts: list[tuple[int, int]]
    markers: list[str | None]
    tempo: float
    slot_tunings: dict[int, tuple[int, ...]] = field(default_factory=dict)
    slot_names: dict[int, str] = field(default_factory=dict)
    slot_programs: dict[int, int] = field(default_factory=dict)
    slot_track_index: dict[int, int] = field(default_factory=dict)
    title: str = ""
    artist: str = ""


def parse(path: str | Path):
    try:
        return guitarpro.parse(str(path))
    except Exception as e:  # PyGuitarPro raises many different errors on odd files
        raise SkipSong(f"cannot read GP file: {type(e).__name__}: {e}") from e


def track_tuning(track) -> tuple[int, ...]:
    return tuple(s.value for s in sorted(track.strings, key=lambda s: s.number))


def _headers_meta(song) -> tuple[list[tuple[int, int]], list[str | None]]:
    ts, markers = [], []
    for h in song.measureHeaders:
        ts.append((int(h.timeSignature.numerator), int(h.timeSignature.denominator.value)))
        markers.append(h.marker.title if h.marker is not None else None)
    return ts, markers


def read_gp_meta(path: str | Path, song=None) -> GPMeta:
    """Metadata of a DadaGP song, with guitar slots assigned exactly like dadagp.py does."""
    song = song if song is not None else parse(path)
    ts, markers = _headers_meta(song)
    meta = GPMeta(n_measures=len(song.measureHeaders), ts=ts, markers=markers, tempo=float(song.tempo),
                  title=song.title or "", artist=song.artist or "")
    counters = {"distorted": 0, "clean": 0}
    for ti, track in enumerate(song.tracks):
        group = dadagp_bridge.instrument_group(track)
        if group not in counters:
            continue
        slot = counters[group] + (0 if group == "distorted" else 3)
        counters[group] += 1
        if slot > 4 or (group == "distorted" and slot > 2):
            continue
        meta.slot_tunings[slot] = track_tuning(track)
        meta.slot_names[slot] = track.name or ""
        meta.slot_programs[slot] = int(track.channel.instrument)
        meta.slot_track_index[slot] = ti
    return meta


def _is_guitar(track) -> bool:
    if track.isPercussionTrack:
        return False
    tuning = track_tuning(track)
    if len(tuning) not in (6, 7):
        return False
    group = dadagp_bridge.instrument_group(track)
    if group in ("distorted", "clean"):
        return True
    if tuning == tunings_mod.DROP_A7:
        return True
    return bool(GUITAR_NAME_RE.search(track.name or ""))


def _is_clean(track) -> bool:
    prog = int(track.channel.instrument)
    if 24 <= prog <= 28:
        return True
    if 29 <= prog <= 31:
        return False
    return bool(CLEAN_NAME_RE.search(track.name or ""))


def _track_signature(track) -> tuple:
    sig = []
    for m in track.measures:
        for v in m.voices:
            for b in v.beats:
                sig.append((b.duration.time, tuple((n.string, n.value, n.type.value) for n in b.notes)))
    return tuple(sig)


@dataclass
class OwnEncoding:
    tokens: list[str]
    meta: GPMeta
    slot_tunings: dict[int, tuple[int, ...]]
    warnings: list[str]
    removed_tracks: list[str]


def encode_own_gp(path: str | Path, require_tuning: str | None = tunings_mod.TARGET_TUNING_NAME,
                  max_distorted: int = 3, max_clean: int = 2) -> OwnEncoding:
    song = parse(path)
    ts, markers = _headers_meta(song)
    warnings: list[str] = []
    removed: list[str] = []

    guitars, others = [], []
    for ti, track in enumerate(song.tracks):
        (guitars if _is_guitar(track) else others).append((ti, track))
    removed += [f"{t.name or 'track %d' % (ti + 1)} (not a guitar)" for ti, t in others]
    if not guitars:
        raise SkipSong("no guitar track")

    if require_tuning:
        for ti, track in guitars:
            name = tunings_mod.name_of(track_tuning(track))
            if name != require_tuning:
                raise SkipSong(f"guitar track '{track.name}' is tuned {tunings_mod.describe(track_tuning(track))} "
                               f"({name}), expected {require_tuning} (D10: tabs are converted by hand, never automatically)")

    # remove exact duplicates first (double-tracked rhythm guitars), then enforce DadaGP's limits
    kept, seen = [], {}
    for ti, track in guitars:
        sig = _track_signature(track)
        if sig in seen and len(guitars) > max_distorted:
            removed.append(f"{track.name} (identical to {seen[sig]})")
            continue
        seen.setdefault(sig, track.name)
        kept.append((ti, track))
    distorted = [(ti, t) for ti, t in kept if not _is_clean(t)]
    clean = [(ti, t) for ti, t in kept if _is_clean(t)]
    while len(clean) > max_clean and len(distorted) < max_distorted:
        distorted.append(clean.pop())
        warnings.append(f"treating clean track '{distorted[-1][1].name}' as distorted (max {max_clean} clean)")
    for ti, t in distorted[max_distorted:] + clean[max_clean:]:
        removed.append(f"{t.name} (more than {max_distorted} distorted / {max_clean} clean guitars)")
    distorted, clean = distorted[:max_distorted], clean[:max_clean]
    distorted.sort(key=lambda x: x[0])
    clean.sort(key=lambda x: x[0])

    meta = GPMeta(n_measures=len(song.measureHeaders), ts=ts, markers=markers, tempo=float(song.tempo),
                  title=song.title or Path(path).stem, artist=song.artist or "")
    slot_tunings: dict[int, tuple[int, ...]] = {}
    new_tracks = []
    for group, items, base in (("distorted", distorted, 0), ("clean", clean, 3)):
        for k, (ti, track) in enumerate(items):
            slot = base + k
            real = track_tuning(track)
            slot_tunings[slot] = real
            meta.slot_tunings[slot] = real
            meta.slot_names[slot] = track.name or ""
            meta.slot_programs[slot] = int(track.channel.instrument)
            meta.slot_track_index[slot] = ti
            track.channel = copy.copy(track.channel)
            track.channel.instrument = 30 if group == "distorted" else 27
            std = tunings_mod.STD7 if len(real) == 7 else tunings_mod.STD6
            track.strings = [guitarpro.GuitarString(number=i + 1, value=v) for i, v in enumerate(std)]
            for m in track.measures:
                for v in m.voices:
                    for b in v.beats:
                        if b.effect.mixTableChange is not None:
                            b.effect.mixTableChange.instrument = None
            new_tracks.append((ti, track))
    # keep the original track order: DadaGP numbers distorted0.. in track order
    new_tracks.sort(key=lambda x: x[0])
    song.tracks = [t for _, t in new_tracks]
    try:
        tokens = dadagp_bridge.encode_song(song)
    except Exception as e:
        raise SkipSong(f"DadaGP encoder failed: {type(e).__name__}: {e}") from e
    return OwnEncoding(tokens=tokens, meta=meta, slot_tunings=slot_tunings, warnings=warnings, removed_tracks=removed)
