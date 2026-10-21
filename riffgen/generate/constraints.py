"""Constrained decoding (D33): rules the model cannot break while it writes the gap.

Enforced at every token:

* exactly ``n_bars`` bars, each filled to exactly its length in ticks: ``wait`` tokens may
  not pass the bar end, and when the bar is full ``new_measure`` (or ``<eos>`` after the
  last bar) is forced -- so odd meters come out right;
* time signature = the one given (bar before the gap), no time-signature changes, no
  tempo changes, no section/gap/header tokens;
* only the guitars the song has, strings 1..7 and frets 0..24 (7-string Drop A);
* the canonical order the model was trained on: guitars in slot order inside an onset,
  notes by string, one note per string, effects only right after their note, params only
  after an effect that takes them, rests only where they end a sounding note, ties only
  onto the same string and fret.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from riffgen.tokens.vocab import K, Vocab

MAX_TOKENS_PER_BAR = 600
MAX_PARAMS = 16


@dataclass
class GapSpec:
    n_bars: int
    bar_ticks: list[int]
    bar_ts: list[tuple[int, int]]
    slots: dict[int, int]                         # allowed guitar slot -> number of strings
    force_ts_each_bar: bool = False               # --use-gap-signatures: ts token when it changes
    max_fret: int = 24
    prev_notes: dict[int, set] = field(default_factory=dict)   # last sounding (string, fret) per slot


class VocabMasks:
    """Precomputed boolean masks over the vocabulary."""

    def __init__(self, vocab: Vocab, spec: GapSpec):
        v = vocab
        self.v = v
        V = len(v)
        kind = v.kind
        self.wait_ok = kind == K.WAIT
        self.wait_val = v.wait.astype(np.int64)
        note_ok = np.zeros(V, bool)
        for slot, n_str in spec.slots.items():
            ids = v.note_id[slot, 1:n_str + 1, 0:spec.max_fret + 1]
            note_ok[ids[ids >= 0]] = True
        self.note_ok = note_ok
        self.note_slot = v.inst.astype(np.int64)
        self.note_string = v.string.astype(np.int64)
        self.note_fret = v.fret.astype(np.int64)
        rest_ok = np.zeros(V, bool)
        for slot in spec.slots:
            if v.rest_id[slot] >= 0:
                rest_ok[v.rest_id[slot]] = True
        self.rest_ok = rest_ok
        self.nfx = kind == K.NFX
        self.param = kind == K.PARAM
        self.bfx = (kind == K.BFX) & ~v.tempo_bfx
        self.fx_params = v.fx_params
        self.fx_name = np.array([t.split(":")[1] if t.startswith(("nfx:", "bfx:")) and t.count(":") >= 1 else ""
                                 for t in v.tokens], dtype=object)
        self.tie_id = v.index.get("nfx:tie", -1)
        self.V = V


class GapConstraint:
    """Grammar state of one candidate."""

    START, HEAD, BODY, DONE = range(4)

    def __init__(self, vocab: Vocab, spec: GapSpec, masks: VocabMasks | None = None):
        self.v = vocab
        self.spec = spec
        self.m = masks or VocabMasks(vocab, spec)
        self.phase = self.START
        self.bar = -1
        self.tick = 0
        self.bar_tokens = 0
        self.cur_slot = -1
        self.cur_is_rest = False
        self.cur_string = 0
        self.cur_note: tuple[int, int] | None = None
        self.note_fx: set[str] = set()
        self.beat_fx: set[str] = set()
        self.last_kind = -1
        self.params = 0
        self.param_ok = False
        self.sounding = {s: False for s in spec.slots}
        self.prev_notes = {s: set(spec.prev_notes.get(s, set())) for s in spec.slots}
        self.cur_notes: dict[int, set] = {s: set() for s in spec.slots}
        self.ts_pending = False

    def clone(self) -> "GapConstraint":
        c = copy.copy(self)
        c.note_fx = set(self.note_fx)
        c.beat_fx = set(self.beat_fx)
        c.sounding = dict(self.sounding)
        c.prev_notes = {k: set(v) for k, v in self.prev_notes.items()}
        c.cur_notes = {k: set(v) for k, v in self.cur_notes.items()}
        return c

    @property
    def done(self) -> bool:
        return self.phase == self.DONE

    @property
    def remaining(self) -> int:
        return self.spec.bar_ticks[self.bar] - self.tick if 0 <= self.bar < self.spec.n_bars else 0

    def _ts_id(self, b: int) -> int:
        return self.v.ts_token(*self.spec.bar_ts[b])

    def _needs_ts(self, b: int) -> bool:
        if b == 0:
            return True
        return self.spec.force_ts_each_bar and self.spec.bar_ts[b] != self.spec.bar_ts[b - 1]

    def allowed(self) -> np.ndarray:
        v, m = self.v, self.m
        mask = np.zeros(m.V, bool)
        if self.phase == self.DONE:
            mask[v.eos_id] = True
            return mask
        if self.phase == self.START:
            mask[v.new_measure_id] = True
            return mask
        if self.ts_pending:
            mask[self._ts_id(self.bar)] = True
            return mask
        rem = self.remaining
        if rem <= 0:
            mask[v.new_measure_id if self.bar + 1 < self.spec.n_bars else v.eos_id] = True
            return mask
        if self.bar_tokens >= MAX_TOKENS_PER_BAR:     # runaway bar: only waits from now on
            mask |= m.wait_ok & (m.wait_val <= rem)
            return mask
        # waits
        mask |= m.wait_ok & (m.wait_val <= rem) & (m.wait_val > 0)
        # notes: new slot after current, or more strings of the current slot
        notes = m.note_ok.copy()
        if self.cur_slot >= 0:
            later_slot = m.note_slot > self.cur_slot
            same = (m.note_slot == self.cur_slot) & (m.note_string > self.cur_string) & (not self.cur_is_rest) \
                & (not self.beat_fx)
            notes &= later_slot | same
        mask |= notes
        # rests: a fresh slot whose last event in this bar was a note
        if self.tick > 0:
            for slot, snd in self.sounding.items():
                if snd and slot > self.cur_slot and self.v.rest_id[slot] >= 0:
                    mask[self.v.rest_id[slot]] = True
        in_note = self.last_kind in (K.NOTE, K.NFX, K.PARAM) and self.cur_note is not None and not self.beat_fx
        if in_note:
            nfx = m.nfx.copy()
            for name in self.note_fx:
                nfx &= m.fx_name != name
            if m.tie_id >= 0 and self.cur_note not in self.prev_notes.get(self.cur_slot, set()):
                nfx[m.tie_id] = False
            mask |= nfx
        if self.param_ok and self.params < MAX_PARAMS:
            mask |= m.param
        if self.cur_slot >= 0 and self.last_kind in (K.NOTE, K.NFX, K.PARAM, K.REST, K.BFX):
            bfx = m.bfx.copy()
            for name in self.beat_fx:
                bfx &= m.fx_name != name
            mask |= bfx
        return mask

    def _close_onset(self) -> None:
        for slot, notes in self.cur_notes.items():
            if notes:
                self.prev_notes[slot] = set(notes)
        self.cur_notes = {s: set() for s in self.spec.slots}
        self.cur_slot = -1
        self.cur_is_rest = False
        self.cur_string = 0
        self.cur_note = None
        self.note_fx = set()
        self.beat_fx = set()
        self.param_ok = False

    def advance(self, tok: int) -> None:
        v, m = self.v, self.m
        kind = int(v.kind[tok])
        self.bar_tokens += 1
        if tok == v.eos_id:
            self.phase = self.DONE
            return
        if kind == K.NEW_MEASURE:
            self._close_onset()
            self.bar += 1
            self.tick = 0
            self.bar_tokens = 0
            self.sounding = {s: False for s in self.spec.slots}
            self.phase = self.BODY
            self.ts_pending = self._needs_ts(self.bar)
            self.last_kind = kind
            return
        if kind == K.TS:
            self.ts_pending = False
        elif kind == K.WAIT:
            self._close_onset()
            self.tick += int(v.wait[tok])
        elif kind == K.NOTE:
            slot = int(v.inst[tok])
            if slot != self.cur_slot:
                self.beat_fx = set()
            self.cur_slot = slot
            self.cur_is_rest = False
            self.cur_string = int(v.string[tok])
            self.cur_note = (self.cur_string, int(v.fret[tok]))
            self.cur_notes[slot].add(self.cur_note)
            self.sounding[slot] = True
            self.note_fx = set()
            self.param_ok = False
        elif kind == K.REST:
            slot = int(v.inst[tok])
            self.cur_slot = slot
            self.cur_is_rest = True
            self.cur_note = None
            self.sounding[slot] = False
            self.prev_notes[slot] = set()
            self.beat_fx = set()
            self.param_ok = False
        elif kind == K.NFX:
            self.note_fx.add(m.fx_name[tok])
            self.param_ok = bool(v.fx_params[tok])
            self.params = 0
        elif kind == K.BFX:
            self.beat_fx.add(m.fx_name[tok])
            self.cur_note = None
            self.param_ok = bool(v.fx_params[tok])
            self.params = 0
        elif kind == K.PARAM:
            self.params += 1
        self.last_kind = kind


def forced_token(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) == 1 else None
