"""Turn event tokens into musical note events (onsets) for analysis and the checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from riffgen.dataset.song import Bar, SongData, parse_note

TECH_NFX = {"bend", "slide", "hammer", "vibrato", "harmonic", "trill", "tremoloPicking", "grace"}


@dataclass
class Note:
    string: int
    fret: int
    pitch: int
    fx: frozenset[str] = frozenset()

    @property
    def tie(self) -> bool:
        return "tie" in self.fx

    @property
    def dead(self) -> bool:
        return "dead" in self.fx


@dataclass
class Onset:
    tick: int                       # within the bar
    dur: int                        # until the next event of this guitar in the bar (or bar end)
    notes: list[Note] = field(default_factory=list)
    beat_fx: frozenset[str] = frozenset()
    rest: bool = False

    @property
    def pitches(self) -> tuple[int, ...]:
        return tuple(sorted(n.pitch for n in self.notes if not n.dead))

    @property
    def low_pitch(self) -> int | None:
        p = [n.pitch for n in self.notes if not n.dead]
        return min(p) if p else None

    @property
    def is_attack(self) -> bool:
        """A new sound starts here (not only tied continuations)."""
        return any(not n.tie for n in self.notes)

    @property
    def palm_mute(self) -> bool:
        return any("palm_mute" in n.fx for n in self.notes)

    @property
    def accent(self) -> bool:
        return any(("accentuated_note" in n.fx or "heavy_accentuated_note" in n.fx) for n in self.notes)


@lru_cache(maxsize=4096)
def _fx_name(tok: str) -> str:
    parts = tok.split(":")
    return parts[1] if len(parts) > 1 else tok


def event_to_onset(tokens: list[str], tick: int, tuning: tuple[int, ...]) -> Onset:
    on = Onset(tick=tick, dur=0)
    raw: list[tuple[int, int, int, set[str]]] = []
    cur_fx: set[str] | None = None
    bfx = set()
    for t in tokens:
        if t.endswith(":rest"):
            on.rest = True
            cur_fx = None
        elif ":note:" in t:
            p = parse_note(t)
            if p is None:
                cur_fx = None
                continue
            _, s, f = p
            pitch = (tuning[s - 1] + f) if 1 <= s <= len(tuning) else 0
            cur_fx = set()
            raw.append((s, f, pitch, cur_fx))
        elif t.startswith("nfx:") and cur_fx is not None:
            cur_fx.add(_fx_name(t))
        elif t.startswith("bfx:"):
            bfx.add(_fx_name(t))
    on.notes = [Note(s, f, p, frozenset(fx)) for s, f, p, fx in raw]
    on.beat_fx = frozenset(bfx)
    if on.notes:
        on.rest = False
    return on


def bar_onsets(bar: Bar, slot: int, tuning: tuple[int, ...]) -> list[Onset]:
    evs = [e for e in bar.events if e.track == slot]
    out = [event_to_onset(e.tokens, e.tick, tuning) for e in evs]
    for i, on in enumerate(out):
        nxt = out[i + 1].tick if i + 1 < len(out) else bar.ticks
        on.dur = max(0, nxt - on.tick)
    return out


def song_onsets(song: SongData) -> dict[int, list[list[Onset]]]:
    """slot -> per bar -> onsets."""
    return {slot: [bar_onsets(b, slot, tr.tuning) for b in song.bars] for slot, tr in song.tracks.items()}


# ---------------------------------------------------------------------------
# Fixed-size bar descriptors used by the similarity matrix

R_SLOTS = 48     # onset grid per bar
P_SLOTS = 16     # pitch-time grid per bar


def bar_vectors(onsets: list[Onset], ticks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """(rhythm[48], pitch-time[16*12], pitch-class histogram[12], has_notes)."""
    r = np.zeros(R_SLOTS, np.float32)
    p = np.zeros((P_SLOTS, 12), np.float32)
    h = np.zeros(12, np.float32)
    has = False
    ticks = max(ticks, 1)
    for on in onsets:
        if not on.notes:
            continue
        has = True
        lp = on.low_pitch
        if on.is_attack:
            r[min(R_SLOTS - 1, on.tick * R_SLOTS // ticks)] = 2.0 if on.accent else 1.0
            if lp is not None:
                p[min(P_SLOTS - 1, on.tick * P_SLOTS // ticks), lp % 12] = 1.0
        w = max(on.dur, 60) / 960.0
        for n in on.notes:
            if not n.dead:
                h[n.pitch % 12] += w
    return r, p.reshape(-1), h, has
