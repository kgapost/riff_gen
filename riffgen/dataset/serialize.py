"""Bars -> token ids, and token ids -> musical features.

Serialization follows the DadaGP layout (D21): ``new_measure``, measure tokens, then for
every onset the events of each guitar (slot order), with ``wait:`` tokens between onsets
and up to the end of the bar. Our extra tokens: ``ts:`` (first bar of a segment, after a
gap, and when it changes), ``section:`` (at section starts), ``gap:N`` for unknown bars.

The ``Featurizer`` computes, for every token, where it sits musically. The *same* code runs
over training examples and, step by step, over generated tokens -- so the model sees the
same features in both. Features:

    bar      absolute bar index in the song (the bar the token belongs to)
    tick     onset inside the bar (wait tokens: the time they start from)
    time     absolute ticks from song start (for musical-time rotary positions)
    inst     guitar slot (notes, rests and their effect tokens)
    pitch    MIDI pitch of a note = tuning[string] + fret  (the *sound*, not the shape)
    fret     fret of a note
    sfb      string counted from the lowest string (1 = lowest): s7 on a 7-string and s6 on
             a 6-string are both "the low string", so drop-D riffs and Drop A riffs look
             alike to the model even though the string numbers differ
    beat     metrical strength of the onset (downbeat, beat, 8th, 16th, triplet, finer)
    seg      header / prefix / suffix / middle / song body
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from riffgen.dataset.store import SongView
from riffgen.tokens.vocab import K, Vocab, swap_table

SEG_HEADER, SEG_PREFIX, SEG_SUFFIX, SEG_MIDDLE, SEG_SONG = 0, 1, 2, 3, 4
N_SEG = 5
N_BEAT = 7          # 0 none, 1 downbeat, 2 beat, 3 8th, 4 16th, 5 triplet, 6 finer
POS_UNIT = 40       # tick resolution of the position-in-bar embedding (1/24 quarter)
N_POS = 385         # 0 = none, 1..384 = 16 quarters
BAR_REL_MAX = 64    # bar relative to the target: -64..63, 128 = none
N_BAR_REL = 129
NO_PITCH = 128
MAX_FRET_FEAT = 31  # 0..29 frets, 31 = none


@dataclass
class TrackPlan:
    """Which guitars appear in an example and under which slot (D44 augmentation)."""
    slot_map: dict[int, int]                       # original slot -> slot in the example
    tunings: dict[int, tuple[int, ...]]            # example slot -> tuning
    remap: np.ndarray | None = None                # token id remap for renamed slots
    dropped: list[int] = field(default_factory=list)
    swapped: bool = False

    @classmethod
    def identity(cls, view: SongView) -> "TrackPlan":
        tr = view.tracks
        return cls(slot_map={s: s for s in tr}, tunings={s: t.tuning for s, t in tr.items()})


def make_track_plan(view: SongView, vocab: Vocab, rng: np.random.Generator, swap_p: float, drop_p: float,
                    keep_slots: set[int] | None = None) -> TrackPlan:
    tracks = view.tracks
    present = sorted(tracks)
    dropped: list[int] = []
    if len(present) >= 2 and rng.random() < drop_p:
        dups = [tuple(p) for p in view.meta.get("dup_tracks", [])]
        cands = [b for a, b in dups if b in present]
        if not cands:
            cands = present[:]
        if keep_slots:
            cands = [c for c in cands if c not in keep_slots or len(keep_slots) > 1]
        if cands:
            d = int(cands[rng.integers(len(cands))])
            dropped = [d]
    kept = [s for s in present if s not in dropped]
    # compact numbering per kind, optionally shuffled within the kind (swap)
    slot_map: dict[int, int] = {}
    swapped = False
    for kind_slots, base in (([s for s in kept if s < 3], 0), ([s for s in kept if s >= 3], 3)):
        order = list(kind_slots)
        if len(order) >= 2 and rng.random() < swap_p:
            perm = rng.permutation(len(order))
            order = [order[i] for i in perm]
            swapped = swapped or any(i != j for i, j in enumerate(perm))
        for k, s in enumerate(order):
            slot_map[s] = base + k
    remap = None
    if any(k != v for k, v in slot_map.items()):
        remap = swap_table(vocab, slot_map)
    return TrackPlan(slot_map=slot_map, tunings={slot_map[s]: tracks[s].tuning for s in kept}, remap=remap,
                     dropped=dropped, swapped=swapped)


def serialize_bar(view: SongView, b: int, vocab: Vocab, plan: TrackPlan, *, with_ts: bool, with_section: bool,
                  strip_tempo: bool, triplet_ids: dict[int, int] | None = None) -> list[int]:
    bar = view.bars[b]
    out = [vocab.new_measure_id]
    trip = int(bar["triplet"])
    if trip and triplet_ids and trip in triplet_ids:
        out.append(triplet_ids[trip])
    if with_ts:
        out.append(vocab.ts_token(int(bar["ts_num"]), int(bar["ts_den"])))
    if with_section and bar["section"] >= 0:
        out.append(vocab.section_ids[_section_label(int(bar["section"]))])
    groups: dict[int, list[tuple[int, np.ndarray]]] = {}
    for ev in view.bar_events(b):
        tr = int(ev["track"])
        toks = view.event_tokens(ev)
        if tr >= 0:
            if tr not in plan.slot_map:
                continue
            slot = plan.slot_map[tr]
            if plan.remap is not None:
                toks = plan.remap[toks]
        else:
            if strip_tempo:
                continue
            slot = -1
        groups.setdefault(int(ev["tick"]), []).append((slot, toks))
    cur = 0
    for tick in sorted(groups):
        if tick > cur:
            out.extend(vocab.wait_tokens(tick - cur))
            cur = tick
        for _, toks in sorted(groups[tick], key=lambda x: x[0]):
            out.extend(int(t) for t in toks)
    end = int(bar["ticks"])
    if end > cur:
        out.extend(vocab.wait_tokens(end - cur))
    return out


_SECTION_LABELS: tuple[str, ...] | None = None


def _section_label(i: int) -> str:
    global _SECTION_LABELS
    if _SECTION_LABELS is None:
        from riffgen.tokens import sections
        _SECTION_LABELS = sections.labels()
    return _SECTION_LABELS[i]


def triplet_token_ids(vocab: Vocab) -> dict[int, int]:
    out = {}
    for t, i in vocab.index.items():
        if t.startswith("measure:triplet_feel:"):
            out[int(t.rsplit(":", 1)[1])] = i
    return out


def section_in_effect(view: SongView, b: int) -> int:
    for i in range(min(b, view.n_bars - 1), -1, -1):
        s = int(view.bars[i]["section"])
        if s >= 0:
            return s
    return -1


def header_tokens(vocab: Vocab, task: str, plan: TrackPlan, tempo: float, n_bars: int | None = None,
                  section: int = -1) -> list[int]:
    from riffgen.tokens import tunings as tunings_mod
    out = [vocab.bos_id, vocab.task_infill_id if task == "infill" else vocab.task_song_id]
    for slot in sorted(plan.tunings):
        out.append(int(vocab.track_id[slot]))
        out.append(vocab.tuning_token(tunings_mod.name_of(plan.tunings[slot])))
    out.append(vocab.tempo_token(tempo))
    if n_bars is not None:
        out.append(vocab.bars_ids[min(n_bars, max(vocab.bars_ids))])
    if section >= 0:
        out.append(vocab.section_ids[_section_label(section)])
    return out


# ---------------------------------------------------------------------------

def beat_class(tick: int, num: int, den: int) -> int:
    if tick == 0:
        return 1
    beat = 3840 // max(den, 1)
    if den == 8 and num % 3 == 0 and num > 3:
        beat = 1440  # compound meters: dotted quarter beat
    if tick % beat == 0:
        return 2
    if tick % 480 == 0:
        return 3
    if tick % 240 == 0:
        return 4
    if tick % 160 == 0 or tick % 320 == 0:
        return 5
    return 6


class Featurizer:
    """Step through token ids and compute musical features (training and generation)."""

    FIELDS = ("bar", "tick", "inst", "pitch", "fret", "sfb", "beat", "seg", "pos")

    def __init__(self, vocab: Vocab, bar_ticks: np.ndarray, bar_ts: np.ndarray, tunings: dict[int, tuple[int, ...]],
                 seg_start_bars: dict[int, int], ref_bar: int):
        self.v = vocab
        self.bar_ticks = np.asarray(bar_ticks, np.int64)
        self.bar_start = np.concatenate([[0], np.cumsum(self.bar_ticks)])
        self.bar_ts = bar_ts
        self.tunings = tunings
        self.seg_start_bars = seg_start_bars
        self.ref_bar = ref_bar
        self.ref_time = int(self.bar_start[min(max(ref_bar, 0), len(self.bar_start) - 1)])
        self.seg = SEG_HEADER
        self.bar = ref_bar
        self.tick = 0
        self.cur_inst = -1
        self.started = False

    def _enter(self, seg: int) -> None:
        self.seg = seg
        self.bar = self.seg_start_bars.get(seg, self.ref_bar) - 1
        self.tick = 0
        self.cur_inst = -1
        self.started = False

    def _bar_time(self, b: int) -> int:
        n = len(self.bar_ticks)
        if b < 0:
            return int(b * (self.bar_ticks[0] if n else 3840))
        if b >= n:
            last = int(self.bar_ticks[-1]) if n else 3840
            return int(self.bar_start[-1] + (b - n) * last)
        return int(self.bar_start[b])

    def _ts(self, b: int) -> tuple[int, int]:
        if len(self.bar_ts) == 0:
            return 4, 4
        b = min(max(b, 0), len(self.bar_ts) - 1)
        return int(self.bar_ts[b][0]), int(self.bar_ts[b][1])

    def step(self, tok: int) -> tuple[int, int, int, int, int, int, int, int, int, float, float]:
        """Features of ``tok`` (and advance the state).

        Returns (bar, tick, inst, pitch, fret, sfb, beat, seg, pos_bucket, time16, phase16).
        """
        v = self.v
        kind = int(v.kind[tok])
        none = (-1, NO_PITCH, MAX_FRET_FEAT, 0)
        # segment boundaries
        if kind in (K.FIM_PREFIX, K.FIM_SUFFIX, K.FIM_MIDDLE):
            self._enter({K.FIM_PREFIX: SEG_PREFIX, K.FIM_SUFFIX: SEG_SUFFIX, K.FIM_MIDDLE: SEG_MIDDLE}[kind])
            return self._out(self.bar + 1, 0, *none, 0, header=True)
        if self.seg == SEG_HEADER and kind in (K.START, K.NEW_MEASURE, K.GAP) and SEG_SONG in self.seg_start_bars:
            self._enter(SEG_SONG)
        if self.seg == SEG_HEADER:
            return self._out(self.ref_bar, 0, *none, 0, header=True)
        # inside a segment
        if kind == K.NEW_MEASURE:
            self.bar += 1
            self.tick = 0
            self.cur_inst = -1
            self.started = True
            return self._out(self.bar, 0, *none, 1)
        if kind == K.GAP:
            bar_of_token = self.bar + 1
            self.bar += int(v.n[tok])
            self.tick = 0
            self.cur_inst = -1
            return self._out(bar_of_token, 0, *none, 0)
        if not self.started:  # e.g. ``start`` before the first bar of a segment
            return self._out(self.bar + 1, 0, *none, 0, header=True)
        tick = self.tick
        inst, pitch, fret, sfb = none
        if kind == K.WAIT:
            self.tick += int(v.wait[tok])
            self.cur_inst = -1
        elif kind == K.NOTE:
            inst = int(v.inst[tok])
            self.cur_inst = inst
            tuning = self.tunings.get(inst)
            s, f = int(v.string[tok]), int(v.fret[tok])
            fret = min(max(f, 0), MAX_FRET_FEAT - 1)
            if tuning is not None and 1 <= s <= len(tuning):
                pitch = min(max(tuning[s - 1] + f, 0), 127)
                sfb = len(tuning) - s + 1
        elif kind == K.REST:
            inst = int(v.inst[tok])
            self.cur_inst = inst
        elif kind in (K.NFX, K.PARAM, K.BFX):
            inst = self.cur_inst
        num, den = self._ts(self.bar)
        beat = beat_class(tick, num, den) if kind in (K.NOTE, K.REST, K.WAIT) else 0
        return self._out(self.bar, tick, inst, pitch, fret, sfb, beat)

    def _out(self, bar, tick, inst, pitch, fret, sfb, beat, header: bool = False):
        pos = 0 if header else min(1 + tick // POS_UNIT, N_POS - 1)
        t = self._bar_time(bar) + tick
        time16 = (t - self.ref_time) / 240.0
        phase16 = tick / 240.0
        return (bar, tick, inst, pitch, fret, sfb, beat, self.seg, pos, time16, phase16)

    def run(self, ids) -> dict[str, np.ndarray]:
        rows = [self.step(int(t)) for t in ids]
        return rows_to_arrays(rows, self.ref_bar)


def rows_to_arrays(rows: list[tuple], ref_bar: int) -> dict[str, np.ndarray]:
    if not rows:
        rows_arr = np.zeros((0, 11), np.float64)
    else:
        rows_arr = np.array(rows, dtype=np.float64)
    bar = rows_arr[:, 0].astype(np.int32)
    rel = np.clip(bar - ref_bar, -BAR_REL_MAX, BAR_REL_MAX - 1) + BAR_REL_MAX
    seg = rows_arr[:, 7].astype(np.int8)
    rel = np.where(seg == SEG_HEADER, N_BAR_REL - 1, rel)
    inst = rows_arr[:, 2].astype(np.int8)
    return {
        "bar": bar,
        "bar_rel": rel.astype(np.int16),
        "inst": (inst + 1).astype(np.int8),               # 0 = none
        "pitch": rows_arr[:, 3].astype(np.int16),
        "fret": rows_arr[:, 4].astype(np.int8),
        "sfb": rows_arr[:, 5].astype(np.int8),
        "beat": rows_arr[:, 6].astype(np.int8),
        "seg": seg,
        "pos": rows_arr[:, 8].astype(np.int16),
        "time": rows_arr[:, 9].astype(np.float32),
        "phase": rows_arr[:, 10].astype(np.float32),
    }


FEATURE_KEYS = ("bar_rel", "inst", "pitch", "fret", "sfb", "beat", "seg", "pos", "time", "phase")
