"""In-memory song representation and DadaGP token parsing.

A song is a list of bars. A bar has a length in ticks (quarter note = 960) and a list of
events. An event is "what one guitar does at one onset": the note tokens (with their
``nfx:``/``param:`` effect tokens) and beat effects, or a rest. Tempo changes are global
events (track -1), because we remove bass and drums and must not lose the tempo changes
that were attached to them.

Canonical form (applied to all data, so the model never has to learn arbitrary orderings):

* guitar slots are ``distorted0-2`` (0-2) and ``clean0-1`` (3-4); within an onset events
  are sorted by slot, notes inside an event by string (1 = highest first, like GP);
* frets are physical frets: DadaGP's drop-string normalisation (-2) is undone;
* rests are only kept where they end a sounding note (a rest at the start of a bar or
  after another rest carries no information in the DadaGP format);
* bad notes (fret < 0 or > 29, string beyond the guitar's strings, two notes on one
  string) are removed and counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from riffgen.tokens import tunings as tunings_mod
from riffgen.tokens.vocab import GUITAR_SLOT, GUITARS, MAX_FRET, OTHER_INSTRUMENTS

QUARTER = 960


@dataclass
class TrackInfo:
    slot: int
    tuning: tuple[int, ...]
    name: str = ""
    program: int = -1
    orig_index: int = -1

    @property
    def n_strings(self) -> int:
        return len(self.tuning)

    @property
    def tuning_name(self) -> str:
        return tunings_mod.name_of(self.tuning)

    def to_dict(self) -> dict:
        return {"slot": self.slot, "tuning": list(self.tuning), "name": self.name,
                "program": self.program, "orig_index": self.orig_index}

    @classmethod
    def from_dict(cls, d: dict) -> "TrackInfo":
        return cls(slot=d["slot"], tuning=tuple(d["tuning"]), name=d.get("name", ""),
                   program=d.get("program", -1), orig_index=d.get("orig_index", -1))


@dataclass
class Event:
    tick: int
    track: int          # guitar slot 0-4, or -1 for global events (tempo)
    tokens: list[str]

    @property
    def has_notes(self) -> bool:
        return any(":note:" in t for t in self.tokens)


@dataclass
class Bar:
    ticks: int
    events: list[Event] = field(default_factory=list)
    measure_tokens: list[str] = field(default_factory=list)
    ts: tuple[int, int] | None = None
    tempo: float = 0.0
    marker: str | None = None
    section: str | None = None
    orig: int = 0
    gap: bool = False
    ctx: bool = False        # simple riff: read as context, never trained to write (D17)
    drop: bool = False       # filter action "drop": unknown (gap:N) in context, never a target

    def has_guitar_notes(self) -> bool:
        return any(e.track >= 0 and e.has_notes for e in self.events)

    @property
    def triplet_feel(self) -> int:
        for t in self.measure_tokens:
            if t.startswith("measure:triplet_feel:"):
                return int(t.rsplit(":", 1)[1])
        return 0


@dataclass
class SongData:
    song_id: str
    tracks: dict[int, TrackInfo]
    bars: list[Bar]
    tempo: float
    artist: str = "unknown"
    title: str = ""
    family: str = ""
    source: str = ""
    kind: str = "dadagp"
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


# ---------------------------------------------------------------------------
# Raw DadaGP parsing (all instruments, tokens as written)

@dataclass
class RawEvent:
    tick: int
    inst: str
    tokens: list[str]


@dataclass
class RawBar:
    ticks: int = 0
    measure_tokens: list[str] = field(default_factory=list)
    events: list[RawEvent] = field(default_factory=list)


@dataclass
class RawSong:
    artist: str
    downtune: int
    tempo: int
    bars: list[RawBar]
    orphans: int = 0


def read_token_file(path) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [t.strip() for t in f.read().split("\n") if t.strip()]


def parse_dadagp(tokens: list[str]) -> RawSong:
    artist, downtune, tempo = "unknown", 0, 120
    i = 0
    while i < len(tokens) and tokens[i] != "start":
        t = tokens[i]
        if t.startswith("artist:"):
            artist = t.split(":", 1)[1]
        elif t.startswith("downtune:"):
            downtune = int(t.split(":", 1)[1])
        elif t.startswith("tempo:"):
            tempo = int(t.split(":", 1)[1])
        elif i == 0 and ":" not in t:
            artist = t
        i += 1
    body = tokens[i + 1:] if i < len(tokens) else tokens

    bars: list[RawBar] = []
    cur: RawBar | None = None
    clock = 0
    bar_start = 0
    current: RawEvent | None = None
    events_at: dict[tuple[int, str], RawEvent] = {}
    orphans = 0

    def close_bar():
        if cur is not None:
            cur.ticks = clock - bar_start

    for tok in body:
        if tok == "end":
            break
        if tok == "new_measure":
            close_bar()
            cur = RawBar()
            bars.append(cur)
            bar_start = clock
            current = None
            events_at = {}
            continue
        if cur is None:  # tokens before the first new_measure
            cur = RawBar()
            bars.append(cur)
            bar_start = clock
        head, _, rest = tok.partition(":")
        if head == "measure":
            cur.measure_tokens.append(tok)
        elif head == "wait":
            clock += int(rest)
            current = None
        elif head in ("nfx", "bfx", "param"):
            if current is None:
                orphans += 1
            else:
                current.tokens.append(tok)
        elif rest.startswith("note") or rest == "rest":
            key = (clock - bar_start, head)
            ev = events_at.get(key)
            if ev is None:
                ev = RawEvent(clock - bar_start, head, [])
                events_at[key] = ev
                cur.events.append(ev)
            ev.tokens.append(tok)
            current = ev
        else:
            orphans += 1
    close_bar()
    return RawSong(artist=artist, downtune=downtune, tempo=tempo, bars=bars, orphans=orphans)


# ---------------------------------------------------------------------------
# Canonicalisation

_NOTE_RE = re.compile(r"^([a-z]+\d?):note:s(\d+):f(-?\d+)$")


@lru_cache(maxsize=65536)
def parse_note(tok: str) -> tuple[str, int, int] | None:
    m = _NOTE_RE.match(tok)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


@dataclass
class CanonStats:
    bad_fret: int = 0
    bad_string: int = 0
    dup_string: int = 0
    orphan_fx: int = 0


def canon_event_tokens(tokens: list[str], slot: int, n_strings: int, drop_strings: frozenset[int],
                       stats: CanonStats) -> tuple[list[str], list[str], bool]:
    """Return (tokens, tempo_tokens, is_rest). ``tokens`` is empty if nothing is left."""
    notes: list[list[str]] = []
    beat_fx: list[list[str]] = []
    tempo: list[str] = []
    has_rest = False
    target: list[str] | None = None  # where param: tokens go
    group: list[str] | None = None   # current note group
    for t in tokens:
        head = t.split(":", 1)[0]
        if t.endswith(":rest") and head in GUITARS:
            has_rest = True
            group = None
            target = None
        elif ":note:" in t:
            p = parse_note(t)
            if p is None:
                group = None
                continue
            _, s, f = p
            if s in drop_strings:
                f += 2
            if s < 1 or s > n_strings:
                stats.bad_string += 1
                group = None
            elif f < 0 or f > MAX_FRET:
                stats.bad_fret += 1
                group = None
            else:
                group = [f"{GUITARS[slot]}:note:s{s}:f{f}"]
                notes.append(group)
            target = group
        elif head == "nfx":
            if group is not None:
                group.append(t)
                target = group
            else:
                stats.orphan_fx += 1
                target = None
        elif head == "bfx":
            name = t.split(":")[1] if t.count(":") >= 1 else ""
            if name in ("tempo_change", "tempo_interpolation"):
                tempo.append(t)
                target = None
            else:
                fx = [t]
                beat_fx.append(fx)
                target = fx
        elif head == "param":
            if target is not None:
                target.append(t)
            else:
                stats.orphan_fx += 1
    if notes:
        by_string: dict[int, list[str]] = {}
        for g in notes:
            s = parse_note(g[0])[1]
            if s in by_string:
                stats.dup_string += 1
                continue
            by_string[s] = g
        out = [tok for s in sorted(by_string) for tok in by_string[s]]
        out += [tok for fx in beat_fx for tok in fx]
        return out, tempo, False
    if has_rest:
        return [f"{GUITARS[slot]}:rest"] + [tok for fx in beat_fx for tok in fx], tempo, True
    return [], tempo, False


def drop_redundant_rests(bars: list[Bar]) -> int:
    """Keep a rest only if it ends a note of the same guitar inside the same bar."""
    removed = 0
    for bar in bars:
        sounding: dict[int, bool] = {}
        kept = []
        for ev in bar.events:
            if ev.track < 0:
                kept.append(ev)
                continue
            if ev.has_notes:
                sounding[ev.track] = True
                kept.append(ev)
            elif sounding.get(ev.track, False) and ev.tick > 0:
                sounding[ev.track] = False
                kept.append(ev)
            else:
                removed += 1
        bar.events = kept
    return removed


def sort_events(bar: Bar) -> None:
    bar.events.sort(key=lambda e: (e.tick, e.track))


def raw_to_song(raw: RawSong, song_id: str, slot_tunings: dict[int, tuple[int, ...]] | None = None,
                *, dadagp_normalised: bool = True, kind: str = "dadagp") -> SongData:
    """Keep guitars only, convert frets to physical frets, canonicalise.

    slot_tunings: real tuning per guitar slot (from the GP file). If None, the tuning is
    inferred from the tokens like the DadaGP decoder does.
    dadagp_normalised: True if drop strings in the tokens carry DadaGP's -2 shift. False
    for our own files, which are encoded with a standard tuning (frets already physical).
    """
    present: dict[int, dict] = {}
    for bar in raw.bars:
        for ev in bar.events:
            if ev.inst in GUITAR_SLOT:
                info = present.setdefault(GUITAR_SLOT[ev.inst], {"s7": False, "drop": False, "notes": 0})
                for t in ev.tokens:
                    p = parse_note(t)
                    if p:
                        info["notes"] += 1
                        if p[1] == 7:
                            info["s7"] = True
                        if p[1] in (6, 7) and p[2] in (-1, -2):
                            info["drop"] = True

    tracks: dict[int, TrackInfo] = {}
    drop_strings: dict[int, frozenset[int]] = {}
    warnings = []
    for slot, info in sorted(present.items()):
        if slot_tunings and slot in slot_tunings:
            tuning = tuple(slot_tunings[slot])
        else:
            tuning = tunings_mod.infer_dadagp_tuning(raw.downtune, 7 if info["s7"] else 6, info["drop"])
        tracks[slot] = TrackInfo(slot=slot, tuning=tuning)
        drop_strings[slot] = tunings_mod.dadagp_drop_strings(tuning) if dadagp_normalised else frozenset()

    stats = CanonStats()
    bars: list[Bar] = []
    for bi, rb in enumerate(raw.bars):
        bar = Bar(ticks=rb.ticks, measure_tokens=list(rb.measure_tokens), orig=bi)
        tempo_by_tick: dict[int, list[str]] = {}
        for ev in rb.events:
            if ev.inst in GUITAR_SLOT:
                slot = GUITAR_SLOT[ev.inst]
                toks, tempo, _ = canon_event_tokens(ev.tokens, slot, tracks[slot].n_strings, drop_strings[slot], stats)
                if toks:
                    bar.events.append(Event(ev.tick, slot, toks))
            else:
                tempo = [t for t in ev.tokens if t.startswith("bfx:tempo_")]
            if tempo:
                tempo_by_tick.setdefault(ev.tick, [])
                for t in tempo:
                    if t not in tempo_by_tick[ev.tick]:
                        tempo_by_tick[ev.tick].append(t)
        for tick, toks in tempo_by_tick.items():
            toks.sort(key=lambda t: 0 if t.startswith("bfx:tempo_change") else 1)
            bar.events.append(Event(tick, -1, toks))
        sort_events(bar)
        bars.append(bar)
    removed_rests = drop_redundant_rests(bars)

    song = SongData(song_id=song_id, tracks=tracks, bars=bars, tempo=float(raw.tempo), artist=raw.artist,
                    kind=kind, warnings=warnings)
    song.stats.update({"bad_fret": stats.bad_fret, "bad_string": stats.bad_string, "dup_string": stats.dup_string,
                       "orphan_fx": stats.orphan_fx + raw.orphans, "redundant_rests": removed_rests,
                       "other_instruments": sorted({ev.inst for b in raw.bars for ev in b.events
                                                    if ev.inst in OTHER_INSTRUMENTS})})
    if stats.bad_fret:
        song.warn(f"{stats.bad_fret} notes with fret outside 0-{MAX_FRET} removed")
    return song


# ---------------------------------------------------------------------------
# Small helpers used everywhere

def ts_from_ticks(ticks: int) -> tuple[int, int]:
    """Guess a time signature from a bar length (used only when the GP file is missing).

    Like DadaGP's decoder this cannot tell 3/4 from 6/8 (D23); we prefer quarter notes.
    """
    if ticks <= 0:
        return (4, 4)
    for den, unit in ((4, 960), (8, 480), (16, 240), (32, 120)):
        if ticks % unit == 0 and ticks // unit <= 32:
            return (ticks // unit, den)
    return (max(1, min(32, round(ticks / 120))), 32)


def ts_ticks(ts: tuple[int, int]) -> int:
    return int(ts[0] * QUARTER * 4 // ts[1])


def assign_tempi(song: SongData) -> None:
    """Fill bar.tempo from the initial tempo and global tempo-change events."""
    cur = song.tempo
    for bar in song.bars:
        bar.tempo = cur
        for ev in bar.events:
            if ev.track != -1:
                continue
            for t in ev.tokens:
                if t.startswith("bfx:tempo_change:"):
                    cur = float(t.rsplit(":", 1)[1])
                    if ev.tick == 0:
                        bar.tempo = cur


def mark_gaps(song: SongData, min_gap_bars: int = 4) -> list[tuple[int, int]]:
    """D15/D34: runs of >= min_gap_bars bars without any guitar note become gaps."""
    runs = []
    i, n = 0, len(song.bars)
    while i < n:
        if song.bars[i].has_guitar_notes():
            i += 1
            continue
        j = i
        while j < n and not song.bars[j].has_guitar_notes():
            j += 1
        if j - i >= min_gap_bars:
            runs.append((i, j))
            for k in range(i, j):
                song.bars[k].gap = True
        i = j
    return runs
