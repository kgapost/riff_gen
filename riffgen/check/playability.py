"""Playability check (README §7.7, D41) and automatic re-fingering (--fix).

Works on physical string/fret positions and real time (``seconds = ticks/960 * 60/BPM``):
the same jump can be easy at 80 BPM and impossible at 220 BPM.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from riffgen import CONFIG_DIR
from riffgen.analysis.notes import Onset, bar_onsets
from riffgen.dataset.song import Event, SongData, parse_note
from riffgen.tokens import tunings as tunings_mod
from riffgen.utils.config import load_config


@dataclass
class Issue:
    severity: str        # error | warning
    kind: str
    bar: int             # 1-based, as Guitar Pro shows it
    beat: float          # 1-based beat inside the bar
    track: int
    msg: str


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    score: float = 100.0

    @property
    def errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


def load_profile(path=None):
    return load_config(path or CONFIG_DIR / "playability.yaml")


def max_span(fret: int, prof) -> int:
    ms = prof.get("max_span", {})
    if fret < 5:
        return int(ms.get("below_5", 4))
    if fret < 12:
        return int(ms.get("below_12", 5))
    return int(ms.get("above_12", 6))


def fingers_needed(notes: list[tuple[int, int]]) -> int:
    """Distinct fretted positions; a barre (same fret on neighbouring strings) counts once."""
    fretted = sorted((f, s) for s, f in notes if f > 0)
    n = 0
    by_fret: dict[int, list[int]] = {}
    for f, s in fretted:
        by_fret.setdefault(f, []).append(s)
    for f, strings in by_fret.items():
        strings.sort()
        groups = 1
        for a, b in zip(strings, strings[1:]):
            if b - a > 1:
                groups += 1
        n += 1 if groups == 1 else groups
    return n


def chord_ok(notes: list[tuple[int, int]], prof) -> bool:
    fretted = [f for _, f in notes if f > 0]
    if len({s for s, _ in notes}) != len(notes):
        return False
    if fretted and max(fretted) - min(fretted) > max_span(min(fretted), prof):
        return False
    return fingers_needed(notes) <= int(prof.get("max_fingers", 4))


HARMONIC_OK = {2, 3, 4, 5, 7, 9, 12, 16, 19, 24}


def check_song(song: SongData, bars: range | None = None, prof=None, require_tuning: str | None = "drop_a7") -> Report:
    prof = prof or load_profile()
    rep = Report()
    bar_idx = list(bars) if bars is not None else list(range(len(song.bars)))
    if require_tuning:
        for slot, tr in song.tracks.items():
            if tr.tuning_name != require_tuning:
                rep.issues.append(Issue("error", "tuning", 0, 0, slot,
                                        f"tuning {tunings_mod.describe(tr.tuning)} is not {require_tuning}"))
    for slot, tr in song.tracks.items():
        n_str = len(tr.tuning)
        prev = None   # (time_s, hand_pos, strings, single, onset, bar, notes)
        window: list[float] = []
        ringing: dict[int, float] = {}
        t_bar = sum(song.bars[b].ticks for b in range(bar_idx[0])) if bar_idx else 0
        for b in bar_idx:
            bar = song.bars[b]
            sec_per_tick = 60.0 / max(bar.tempo or song.tempo or 120, 1) / 960.0
            ons = bar_onsets(bar, slot, tr.tuning)
            for on in ons:
                if not on.notes:
                    continue
                beat = 1 + on.tick / 960.0
                t = (t_bar + on.tick) * sec_per_tick
                notes = [(n.string, n.fret) for n in on.notes]
                strings = [s for s, _ in notes]
                if len(set(strings)) != len(strings):
                    rep.issues.append(Issue("error", "same_string", b + 1, beat, slot, "two notes on one string"))
                for s, f in notes:
                    if not (1 <= s <= n_str) or not (0 <= f <= 24):
                        rep.issues.append(Issue("error", "range", b + 1, beat, slot, f"string {s} fret {f} out of range"))
                fretted = [f for _, f in notes if f > 0]
                if fretted:
                    span = max(fretted) - min(fretted)
                    lim = max_span(min(fretted), prof)
                    if span > lim:
                        rep.issues.append(Issue("error", "span", b + 1, beat, slot, f"hand span {span} frets > {lim}"))
                    elif span == lim:
                        rep.issues.append(Issue("warning", "span", b + 1, beat, slot, f"hand span {span} frets (at the limit)"))
                fn = fingers_needed(notes)
                if fn > int(prof.get("max_fingers", 4)):
                    rep.issues.append(Issue("error", "fingers", b + 1, beat, slot, f"needs {fn} fingers"))
                for n in on.notes:
                    if "bend" in n.fx and n.fret == 0:
                        rep.issues.append(Issue("error", "technique", b + 1, beat, slot, "bend on an open string"))
                    if "harmonic" in n.fx and n.fret not in HARMONIC_OK:
                        rep.issues.append(Issue("warning", "technique", b + 1, beat, slot,
                                                f"harmonic at fret {n.fret} (natural harmonics: 3,4,5,7,9,12,...)"))
                    if ringing.get(n.string, -1) > t and not n.tie:
                        rep.issues.append(Issue("warning", "let_ring", b + 1, beat, slot,
                                                f"string {n.string} still rings and is played again"))
                    if "let_ring" in n.fx:
                        ringing[n.string] = t + on.dur * sec_per_tick * 4
                hand = min(fretted) if fretted else None
                picked = any(not n.tie and "hammer" not in n.fx for n in on.notes)
                if picked:
                    window = [x for x in window if t - x < 1.0] + [t]
                    if len(window) > float(prof.get("max_notes_per_second", 16)):
                        rep.issues.append(Issue("warning", "speed", b + 1, beat, slot,
                                                f"{len(window)} picked notes in one second"))
                if prev is not None:
                    dt = t - prev[0]
                    sixteenth = 240 * sec_per_tick
                    if hand is not None and prev[1] is not None:
                        jump = abs(hand - prev[1])
                        if jump > int(prof.get("max_jump_frets_per_16th", 7)) and dt < sixteenth * 1.01:
                            rep.issues.append(Issue("warning", "jump", b + 1, beat, slot,
                                                    f"{jump}-fret jump in {dt * 1000:.0f} ms"))
                    if prev[3] and len(notes) == 1 and dt > 0:
                        skip = abs(strings[0] - prev[2][0]) - 1
                        if skip >= 2 and 1.0 / dt > float(prof.get("max_string_skip_speed", 12)):
                            rep.issues.append(Issue("warning", "string_skip", b + 1, beat, slot,
                                                    f"skips {skip} strings at {1.0 / dt:.0f} notes/s"))
                    # legato and slides need a next note on the same string with another fret
                    for n in prev[6]:
                        if ("hammer" in n.fx or "slide" in n.fx):
                            nxt = [m for m in on.notes if m.string == n.string]
                            if nxt and nxt[0].fret == n.fret:
                                rep.issues.append(Issue("error", "technique", prev[5] + 1, prev[4], slot,
                                                        "legato/slide to the same fret"))
                prev = (t, hand, strings, len(notes) == 1, beat, b, on.notes)
            t_bar += bar.ticks
    sc = prof.get("score", {})
    n_bars = max(1, len(bar_idx)) / 8.0
    rep.score = max(0.0, 100.0 - (float(sc.get("error", 15)) * rep.errors + float(sc.get("warning", 3)) * rep.warnings) / max(n_bars, 1.0))
    return rep


def gp_bar_length_issues(gp_song, track_indices: list[int]) -> list[Issue]:
    """Bars whose beats do not add up to the time signature (typical after manual edits)."""
    out = []
    for ti in track_indices:
        for m_idx, m in enumerate(gp_song.tracks[ti].measures):
            need = m.header.length
            for v_idx, voice in enumerate(m.voices):
                if not voice.beats:
                    continue
                have = sum(b.duration.time for b in voice.beats)
                if have != need:
                    out.append(Issue("error", "bar_length", m_idx + 1, 1, ti,
                                     f"voice {v_idx + 1} holds {have} ticks, the time signature needs {need}"))
    return out


# ---------------------------------------------------------------------------
# --fix: same pitches and rhythm, new strings/frets (dynamic programming)

def _positions(pitch: int, tuning: tuple[int, ...]) -> list[tuple[int, int]]:
    return [(s + 1, pitch - open_) for s, open_ in enumerate(tuning) if 0 <= pitch - open_ <= 24]


def _candidates(onset: Onset, tuning, prof, limit: int = 60) -> list[tuple[tuple[int, int], ...]]:
    pitches = [n.pitch for n in onset.notes]
    options = [_positions(p, tuning) for p in pitches]
    if any(not o for o in options):
        return []
    out = []
    for combo in itertools.product(*options):
        if chord_ok(list(combo), prof):
            out.append(tuple(combo))
            if len(out) >= limit:
                break
    return out


def refinger_track(song: SongData, slot: int, bars: list[int], prof) -> int:
    """Re-finger one guitar over the given bars. Returns the number of changed notes."""
    tr = song.tracks[slot]
    items = []   # (bar, event, onset)
    for b in bars:
        for ev in [e for e in song.bars[b].events if e.track == slot]:
            on = bar_onsets(song.bars[b], slot, tr.tuning)
            on = [o for o in on if o.tick == ev.tick]
            if on and on[0].notes and not any(n.tie for n in on[0].notes):
                items.append((b, ev, on[0]))
    if not items:
        return 0
    cands = [_candidates(on, tr.tuning, prof) or [tuple((n.string, n.fret) for n in on.notes)] for _, _, on in items]
    orig = [tuple((n.string, n.fret) for n in on.notes) for _, _, on in items]

    def local(c, o):
        fretted = [f for _, f in c if f > 0]
        span = (max(fretted) - min(fretted)) if fretted else 0
        changed = sum(1 for a, b in zip(c, o) if a != b)
        return 0.3 * span + 0.8 * changed + 0.02 * sum(f for _, f in c)

    def hand(c):
        fretted = [f for _, f in c if f > 0]
        return sum(fretted) / len(fretted) if fretted else None

    INF = float("inf")
    cost = [[local(c, orig[0]) for c in cands[0]]]
    back = [[-1] * len(cands[0])]
    for i in range(1, len(items)):
        row, brow = [], []
        for c in cands[i]:
            h = hand(c)
            best, arg = INF, 0
            for j, pc in enumerate(cands[i - 1]):
                ph = hand(pc)
                move = abs(h - ph) * 0.4 if (h is not None and ph is not None) else 0.0
                v = cost[i - 1][j] + move
                if v < best:
                    best, arg = v, j
            row.append(best + local(c, orig[i]))
            brow.append(arg)
        cost.append(row)
        back.append(brow)
    j = min(range(len(cost[-1])), key=lambda k: cost[-1][k])
    choice = [0] * len(items)
    for i in range(len(items) - 1, -1, -1):
        choice[i] = j
        j = back[i][j]
    changed = 0
    for (b, ev, on), c_idx, cand in zip(items, choice, cands):
        new = cand[c_idx]
        changed += _rewrite_event(ev, on, new)
    return changed


def _rewrite_event(ev: Event, on: Onset, new: tuple[tuple[int, int], ...]) -> int:
    """Replace string/fret of the event's notes in order; keep their effect tokens."""
    mapping = {(n.string, n.fret): pos for n, pos in zip(on.notes, new)}
    out, changed = [], 0
    groups: list[list[str]] = []
    tail: list[str] = []
    for t in ev.tokens:
        if ":note:" in t:
            groups.append([t])
        elif groups and (t.startswith("nfx:") or t.startswith("param:")) and not tail:
            groups[-1].append(t)
        else:
            tail.append(t)
    renamed = []
    for g in groups:
        inst, s, f = parse_note(g[0])
        s2, f2 = mapping.get((s, f), (s, f))
        if (s2, f2) != (s, f):
            changed += 1
        renamed.append((s2, [f"{inst}:note:s{s2}:f{f2}"] + g[1:]))
    renamed.sort(key=lambda x: x[0])
    for _, g in renamed:
        out += g
    ev.tokens = out + tail
    return changed


def fix_song(song: SongData, bars: range | None = None, prof=None) -> int:
    prof = prof or load_profile()
    idx = list(bars) if bars is not None else list(range(len(song.bars)))
    return sum(refinger_track(song, slot, idx, prof) for slot in song.tracks)
