"""Write bars back to Guitar Pro 5.

Used for: generated candidates (spliced into *your original song file*, so bass, drums and
everything else stay exactly as you wrote them, D4), normalized GP5 exports (§3.6 step 8),
filter audit snippets (D18) and sampler previews.

Unlike the DadaGP decoder we never round a duration: a duration Guitar Pro cannot show as
one note is split into tied notes (or several rests), so every bar keeps its exact length.
Effect tokens are applied with DadaGP's own functions (format fidelity), plus tremolo
picking, which DadaGP's decoder drops.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import guitarpro as gp

from riffgen.dataset.song import Bar, Event, TrackInfo, parse_note, ts_from_ticks
from riffgen.gp import dadagp_bridge
from riffgen.gp.dadagp_bridge import DADAGP_DIR
from riffgen.tokens.vocab import SLOT_KIND

# Single-note durations Guitar Pro 5 can store: whole..64th, dotted, and tuplets.
# (DadaGP's list also has values below a 64th note, which GP5 cannot store.)
_TUPLETS = [(1, 1), (3, 2), (5, 4), (6, 4), (7, 4), (9, 8), (10, 8), (11, 8), (12, 8), (13, 8)]


def _duration_table() -> dict[int, tuple[float, tuple[int, bool, int, int]]]:
    table: dict[int, tuple[float, tuple[int, bool, int, int]]] = {}
    for value in (1, 2, 4, 8, 16, 32, 64):
        for dotted in (False, True):
            for enters, times in _TUPLETS:
                if dotted and (enters, times) != (1, 1):
                    continue
                d = gp.Duration(value=value, isDotted=dotted, tuplet=gp.Tuplet(enters=enters, times=times))
                t = d.time
                cost = 1.0 if (enters, times) == (1, 1) and not dotted else 1.15 if dotted else \
                    1.6 if (enters, times) in ((3, 2), (5, 4), (6, 4)) else 2.2
                if t not in table or cost < table[t][0]:
                    table[t] = (cost, (value, dotted, enters, times))
    return table


DURATIONS = _duration_table()
SUPPORTED_TIMES = sorted(DURATIONS)
MAX_DP = 32768
GRID = 20        # fallback grid: every multiple of 20 ticks >= 40 is exactly representable


def make_duration(ticks: int):
    value, dotted, enters, times = DURATIONS[ticks][1]
    return gp.Duration(value=value, isDotted=dotted, tuplet=gp.Tuplet(enters=enters, times=times))


@lru_cache(maxsize=1)
def _dp_tables():
    INF = float("inf")
    cost = [INF] * (MAX_DP + 1)
    choice = [0] * (MAX_DP + 1)
    cost[0] = 0.0
    pieces = [(t, DURATIONS[t][0]) for t in sorted(SUPPORTED_TIMES, reverse=True)]
    for x in range(1, MAX_DP + 1):
        best, arg = INF, 0
        for t, c in pieces:
            if t <= x and cost[x - t] + c < best:
                best, arg = cost[x - t] + c, t
        cost[x], choice[x] = best, arg
    return cost, choice


def exact(ticks: int) -> bool:
    if ticks <= 0:
        return ticks == 0
    if ticks > MAX_DP:
        return exact(ticks - 3840)
    return _dp_tables()[0][ticks] < float("inf")


@lru_cache(maxsize=4096)
def decompose(ticks: int) -> tuple[int, ...]:
    """Split a duration into Guitar Pro note values (longest first). Requires exact(ticks)."""
    if ticks <= 0:
        return ()
    if ticks in DURATIONS:
        return (ticks,)
    if ticks > MAX_DP:
        return (3840,) + decompose(ticks - 3840)
    cost, choice = _dp_tables()
    if cost[ticks] == float("inf"):
        raise ValueError(f"{ticks} ticks cannot be written exactly")
    out, x = [], ticks
    while x > 0:
        out.append(choice[x])
        x -= choice[x]
    return tuple(sorted(out, reverse=True))


def snap_ticks(ticks: list[int], bar_ticks: int) -> list[int | None]:
    """Fallback for odd timings: onsets to a 20-tick grid, at least 40 ticks apart (None = dropped)."""
    out: list[int | None] = []
    prev = None
    for t in ticks:
        q = int(round(t / GRID)) * GRID
        if prev is not None and q - prev < 2 * GRID:
            q = prev + 2 * GRID
        if q > bar_ticks - 2 * GRID and q != 0:
            out.append(None)
            continue
        out.append(q)
        prev = q
    return out


def _fx_groups(tokens: list[str]) -> list[dict]:
    groups: list[dict] = []
    for t in tokens:
        if t.startswith("param:"):
            if groups:
                groups[-1]["params"].append(t)
        else:
            groups.append({"token": t, "params": []})
    return groups


def _apply_note_fx(note, nfx: list[str]) -> None:
    groups = _fx_groups(nfx)
    plain = [g for g in groups if not g["token"].startswith("nfx:tremoloPicking")]
    dadagp_bridge.dadagp().tokens_to_note_effect(note, plain)
    for g in groups:
        if g["token"].startswith("nfx:tremoloPicking:duration"):
            t = int(g["token"].rsplit("duration", 1)[1])
            if t in DURATIONS:
                note.effect.tremoloPicking = gp.TremoloPickingEffect(duration=make_duration(t))


def _apply_beat_fx(beat, bfx: list[str]) -> None:
    groups = [g for g in _fx_groups(bfx) if g["token"].startswith("bfx:")]
    if groups:
        dadagp_bridge.dadagp().tokens_to_beat_effect(beat.effect, groups)


def _split_event(tokens: list[str]) -> tuple[list[tuple[str, list[str]]], list[str], bool]:
    notes: list[tuple[str, list[str]]] = []
    bfx: list[str] = []
    rest = False
    target = None
    for t in tokens:
        if t.endswith(":rest"):
            rest = True
            target = None
        elif ":note:" in t:
            notes.append((t, []))
            target = notes[-1][1]
        elif t.startswith("nfx:"):
            if notes:
                notes[-1][1].append(t)
                target = notes[-1][1]
        elif t.startswith("bfx:"):
            bfx.append(t)
            target = bfx
        elif t.startswith("param:") and target is not None:
            target.append(t)
    return notes, bfx, rest and not notes


SUSTAIN_FX = ("nfx:palm_mute", "nfx:let_ring", "nfx:vibrato")


def _make_beat(voice, dur: int, notes: list[tuple[str, list[str]]] | None, bfx: list[str] | None, tied: bool):
    beat = gp.Beat(voice)
    beat.duration = make_duration(dur)
    if not notes:
        beat.status = gp.BeatStatus.rest
        return beat
    beat.status = gp.BeatStatus.normal
    used = set()
    for tok, nfx in notes:
        p = parse_note(tok)
        if p is None or p[1] in used:
            continue
        used.add(p[1])
        note = gp.Note(beat)
        note.string = p[1]
        note.value = p[2]
        note.type = gp.NoteType.normal
        if tied:
            note.type = gp.NoteType.tie
            _apply_note_fx(note, [f for f in nfx if f in SUSTAIN_FX])
        else:
            _apply_note_fx(note, nfx)
        beat.notes.append(note)
    if bfx and not tied:
        _apply_beat_fx(beat, bfx)
    return beat


def fill_measure(measure, events: list[Event], bar_ticks: int) -> None:
    """Replace a measure's content with the events of one guitar (sorted by tick)."""
    for v in measure.voices:
        v.beats = []
    voice = measure.voices[0]
    evs = sorted((e for e in events if e.tokens and 0 <= e.tick < bar_ticks), key=lambda e: e.tick)
    bounds = [0] + [e.tick for e in evs] + [bar_ticks]
    if not all(exact(b - a) for a, b in zip(bounds, bounds[1:])):
        snapped = snap_ticks([e.tick for e in evs], bar_ticks)
        evs = [Event(q, e.track, e.tokens) for e, q in zip(evs, snapped) if q is not None]
    cursor = 0
    for i, ev in enumerate(evs):
        if ev.tick < cursor:
            continue
        if ev.tick > cursor:
            for d in decompose(ev.tick - cursor):
                voice.beats.append(_make_beat(voice, d, None, None, False))
        end = evs[i + 1].tick if i + 1 < len(evs) else bar_ticks
        end = min(max(end, ev.tick + 1), bar_ticks) if bar_ticks > ev.tick else ev.tick + 1
        notes, bfx, is_rest = _split_event(ev.tokens)
        for k, d in enumerate(decompose(end - ev.tick)):
            if is_rest or not notes:
                voice.beats.append(_make_beat(voice, d, None, bfx if k == 0 else None, False))
            else:
                voice.beats.append(_make_beat(voice, d, notes, bfx, tied=k > 0))
        cursor = end
    if cursor < bar_ticks:
        for d in decompose(bar_ticks - cursor):
            voice.beats.append(_make_beat(voice, d, None, None, False))


def _ts(header, ts: tuple[int, int]) -> None:
    header.timeSignature = gp.TimeSignature(numerator=int(ts[0]), denominator=gp.Duration(value=int(ts[1])))


def blank_song():
    return gp.parse(str(DADAGP_DIR / "blank.gp5"))


def new_song(bars: list[Bar], tracks: dict[int, TrackInfo], tempo: float, title: str = "riffgen",
             markers: bool = True):
    """A fresh GP5 song with one track per guitar slot."""
    song = blank_song()
    song.title = title
    song.artist = "RiffGen"
    song.tempo = int(round(tempo))
    song.measureHeaders = []
    song.tracks = []
    start = 960
    for i, bar in enumerate(bars):
        h = gp.MeasureHeader()
        h.number = i + 1
        h.start = start
        _ts(h, bar.ts or ts_from_ticks(bar.ticks))
        if markers and (bar.marker or bar.section):
            h.marker = gp.Marker(title=bar.marker or bar.section.capitalize())
        song.addMeasureHeader(h)
        start += bar.ticks
    for k, (slot, info) in enumerate(sorted(tracks.items())):
        t = gp.Track(song)
        t.number = k + 1
        t.name = info.name or ("Guitar %d" % (k + 1))
        t.strings = [gp.GuitarString(number=i + 1, value=v) for i, v in enumerate(info.tuning)]
        t.channel = gp.MidiChannel(channel=k if k < 9 else k + 1, effectChannel=min(15, 9 + k + 1),
                                   instrument=30 if SLOT_KIND[slot] == "distorted" else 27)
        t.measures = [gp.Measure(t, h) for h in song.measureHeaders]
        for bar, m in zip(bars, t.measures):
            fill_measure(m, [e for e in bar.events if e.track == slot], bar.ticks)
        song.tracks.append(t)
    _attach_tempo_changes(song, bars, sorted(tracks))
    return song


def _attach_tempo_changes(song, bars: list[Bar], slots: list[int]) -> None:
    if not song.tracks:
        return
    for m_idx, bar in enumerate(bars):
        for ev in bar.events:
            if ev.track != -1:
                continue
            for t in song.tracks:
                pos = 0
                for beat in t.measures[m_idx].voices[0].beats:
                    if pos == ev.tick:
                        _apply_beat_fx(beat, ev.tokens)
                        break
                    pos += beat.duration.time
                else:
                    continue
                break


def write_song(song, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    gp.write(song, str(path))


def write_bars(path: str | Path, bars: list[Bar], tracks: dict[int, TrackInfo], tempo: float,
               title: str = "riffgen") -> None:
    write_song(new_song(bars, tracks, tempo, title), path)


# ---------------------------------------------------------------------------
# splicing generated bars into the original song

def _first_notes(measure) -> dict[int, tuple[int, object]]:
    beats = measure.voices[0].beats
    if not beats:
        return {}
    return {n.string: (n.value, n) for n in beats[0].notes}


def _last_notes(measure) -> dict[int, int]:
    beats = [b for b in measure.voices[0].beats]
    if not beats:
        return {}
    return {n.string: n.value for n in beats[-1].notes}


def _fix_ties(prev_measure, next_measure) -> int:
    """A tie whose previous note does not exist (or differs) becomes a normal note."""
    fixed = 0
    last = _last_notes(prev_measure) if prev_measure is not None else {}
    for string, (fret, note) in _first_notes(next_measure).items():
        if note.type == gp.NoteType.tie and last.get(string) != fret:
            note.type = gp.NoteType.normal
            fixed += 1
    return fixed


def splice(song, measure_indices: list[int], bars: list[Bar], slot_track_index: dict[int, int],
           marker: str | None = None):
    """Put generated bars into ``song`` (a parsed GP file) at the given measure indices (0-based).

    Guitar slots the model left silent get rests; all other tracks are untouched.
    """
    song = copy.deepcopy(song)
    for m_idx, bar in zip(measure_indices, bars):
        if bar.ts is not None:
            h = song.measureHeaders[m_idx]
            if (h.timeSignature.numerator, h.timeSignature.denominator.value) != tuple(bar.ts):
                _ts(h, bar.ts)   # D33: the gap gets the time signature of the bar before it
        for slot, ti in slot_track_index.items():
            track = song.tracks[ti]
            fill_measure(track.measures[m_idx], [e for e in bar.events if e.track == slot], bar.ticks)
    if marker and measure_indices:
        song.measureHeaders[measure_indices[0]].marker = gp.Marker(title=marker)
    for ti in slot_track_index.values():
        ms = song.tracks[ti].measures
        first, last = measure_indices[0], measure_indices[-1]
        _fix_ties(ms[first - 1] if first > 0 else None, ms[first])
        if last + 1 < len(ms):
            _fix_ties(ms[last], ms[last + 1])
    return song
