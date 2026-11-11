#!/usr/bin/env python
"""RiffGen in one file: encode the tab corpus, train three models, evaluate, fill gaps, grid-search.

    python riff.py prepare                          # data/tabs - */, data/tabs - own - */  ->  data_encoded/encoded/
    python riff.py hparams                          # the calculated batch sizes and learning rates
    python riff.py train --model big                # runs/big/; evaluates every --eval-every epochs
    python riff.py eval --ckpt runs/big/best.pt     # loss and generation metrics on the fixed test gaps
    python riff.py infer --ckpt runs/big/best.pt --song my.gp5 --labels genre:prog,album:cs
    python riff.py grid                             # 3 models x 3 batch sizes x 3 learning rates (short runs)

Models (README §6): baseline = flat fill-in-the-middle decoder over a structure-aware 4,096-token window;
big = hierarchical bar model (bar encoder -> global model over every bar -> local decoder); huge = big
plus a pointer memory into the notes of related bars, planning heads and phrase positions.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

ROOT = Path(__file__).resolve().parent

# ============================================================================ tokens

TICKS = 13440                                   # a whole note; every duration below is a whole number of ticks
DUR_VALUES = (1, 2, 4, 8, 16, 32, 64)
TUPLETS = {1: 1, 3: 2, 5: 4, 6: 4, 7: 4}        # notes in the tuplet -> in the time of
EFFECTS = ("tie", "dead", "ghost", "pm", "ho", "slide", "bend", "harm", "vib", "ring", "acc", "stacc")
DRUMS = ("kick", "snare", "hat", "tom", "crash", "ride", "perc")
SECTIONS = ("intro", "verse", "prechorus", "chorus", "bridge", "solo", "breakdown", "interlude", "outro", "other")
RELATIONS = ("copy", "variation", "answer", "new", "chug")
SLOTS = ("genre", "album", "src", "rel")
DROP_A = (64, 59, 55, 50, 45, 40, 33)           # 7-string Drop A, string 1 (high E) .. string 7 (low A)
FRETS = 25                                      # frets 0..24
BASS = (24, 67)                                 # bass pitches kept (MIDI)
TS_DENS = (1, 2, 4, 8, 16, 32)
G_SLOTS = 4                                     # guitar tracks kept per song (README §2)
MAX_BAR = 384                                   # tokens in one bar (all tracks)
FLAG = 0x8000                                   # set on every stored token the model writes (guitar parts, end of bar)
MASK = 0x7FFF

# Labels come from the folder a tab is in (README §8). Edit these to match your folders.
#
# These constants ARE the label vocabulary: make_tokens builds the lbl: tokens from them and never from whatever
# labels one corpus happens to hold. That is what makes the token list reproducible - preparing the corpus again
# after writing a new riff, filling an album folder or adding a genre gives back the same list, so checkpoints
# stay loadable (train_run exits when --init was trained on another vocabulary). Adding a name here renumbers the
# label ids and invalidates existing checkpoints, so the lists are written out in full ahead of being filled:
# every album is registered while its folder is still empty, and GENRE_OTHER absorbs any folder not listed, so a
# new genre folder costs nothing. Music token ids never depend on any of this - labels go last.
EARLY = {"generic", "exercises", "metal_generic"}       # the early years: learned from, steered away from
ALBUMS = ("vc1", "vc2", "caffeine", "cs", "potha", "tme", "maphorion", "omni", "waves_black", "waves_white",
          "inner", "popper", "hh")
GENRES = ("acoustic", "blues_soul_rnb", "britpop", "classical", "doom_stoner_sludge", "electronic", "ethnic", "experimental",
          "funk", "funkrock", "grunge", "jazz", "metal_classic", "metal_groove", "metal_melodic_death", "metal_modern",
          "metal_nu", "metal_power", "metal_thrash", "metal_xtreme", "pop", "prog", "punk", "rock", "rock_classic",
          "rock_hard", "shred", "soundtracks")
GENRE_OTHER = "other"                                   # any genre folder not listed above
SOURCES = ("own", "insp", "early")                      # own: mine, the model may copy it. insp: chosen as
#                                                         inspiration, the model may learn from it but not
#                                                         reproduce it (configs/plagiarism.yaml tiers)


ALBUM_FOLDER = re.compile(r"^tabs - own - (.+) - (insp|own)$", re.I)     # the album and its source are in the folder name

# The token kinds. These sat next to the old ALBUM_SUBS and were deleted with it in 78cd570; nothing
# imports riff.py for them by name, so the module still loaded and only failed on first use - Vocab,
# token_loss and BarWriter all touch them, which breaks prepare, train, eval and infer alike.
K_DUR, K_NOTE, K_FX, K_TRK, K_BASS, K_DRUM, K_TS, K_SEC, K_TREM = range(1, 10)


def album_folder(top: str):
    """'tabs - own - cs - insp' -> ('cs', 'insp'); None when the folder is not an album folder."""
    m = ALBUM_FOLDER.match(top)
    return (re.sub(r"[^a-z0-9]+", "_", m.group(1).lower()).strip("_"), m.group(2).lower()) if m else None


def dur_name(v: int, dot: bool, tup: int) -> str:
    return f"d:{v}{'.' if dot else ''}{f't{tup}' if tup > 1 else ''}"


def dur_ticks(v: int, dot: bool, tup: int) -> int:
    t = TICKS // v * (3 if dot else 2) // 2
    return t * TUPLETS[tup] // tup


DUR_TABLE = {dur_name(v, d, u): (v, d, u) for v in DUR_VALUES for d in (False, True) for u in TUPLETS}


def label_tokens() -> list[str]:
    """The whole label vocabulary, from the constants above: the same list whatever a given corpus contains."""
    return [f"lbl:rel:{r}" for r in RELATIONS] + sorted(
        [f"lbl:album:{a}" for a in ALBUMS] + [f"lbl:genre:{g}" for g in (*GENRES, GENRE_OTHER)] +
        [f"lbl:src:{s}" for s in SOURCES])


def make_tokens() -> list[str]:
    t = ["<pad>", "<bos>", "<pre>", "<suf>", "<mid>", "<eob>", "<none>", "<unk_bar>", "<gap_start>"]
    t += [f"tempo:{i}" for i in range(20)] + [f"len:{i}" for i in range(1, 65)] + [f"gap:{i}" for i in range(1, 65)]
    t += [f"ts:{n}/{d}" for d in TS_DENS for n in range(1, 33)] + [f"sec:{s}" for s in SECTIONS]
    t += ["trk:dr", "trk:bs"] + [f"trk:g{i}" for i in range(G_SLOTS)] + list(DUR_TABLE) + ["trem"]
    t += [f"n:{s}:{f}" for s in range(1, 8) for f in range(FRETS)] + [f"fx:{e}" for e in EFFECTS]
    t += [f"bp:{p}" for p in range(BASS[0], BASS[1] + 1)] + [f"dr:{d}" for d in DRUMS]
    return t + label_tokens()                           # labels last: music ids never move


class Vocab:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.id = {t: i for i, t in enumerate(self.tokens)}
        V = self.size = len(self.tokens)
        for name in ("pad", "bos", "pre", "suf", "mid", "eob", "none", "unk_bar", "gap_start"):
            setattr(self, name, self.id[f"<{name}>"])
        self.kind = np.zeros(V, np.int64)
        self.dur = np.zeros(V, np.int64)
        self.ts_len = np.zeros(V, np.int64)
        self.n_sf = np.full((V, 2), -1, np.int64)
        heads = {"n": K_NOTE, "fx": K_FX, "trk": K_TRK, "bp": K_BASS, "dr": K_DRUM, "sec": K_SEC}
        for i, t in enumerate(self.tokens):
            head, _, rest = t.partition(":")
            if t in DUR_TABLE:
                self.kind[i], self.dur[i] = K_DUR, dur_ticks(*DUR_TABLE[t])
            elif head == "ts":
                n, d = map(int, rest.split("/"))
                self.kind[i], self.ts_len[i] = K_TS, TICKS * n // d
            elif t == "trem":
                self.kind[i] = K_TREM
            elif head in heads:
                self.kind[i] = heads[head]
                if head == "n":
                    self.n_sf[i] = list(map(int, rest.split(":")))
        self.tup = np.array([DUR_TABLE[t][2] if t in DUR_TABLE else 0 for t in self.tokens])   # tuplet kind, 1 = plain
        self.dotted_tup = np.array([t in DUR_TABLE and DUR_TABLE[t][1] and DUR_TABLE[t][2] > 1 for t in self.tokens])
        order = sorted((i for i in range(V) if self.kind[i] == K_DUR), key=lambda i: self.dur[i])
        self.durs_asc, self.ticks_asc = order, [int(self.dur[i]) for i in order]
        self.min_dur = self.ticks_asc[0]
        self.trem = self.id["trem"]
        self.fx = {e: self.id[f"fx:{e}"] for e in EFFECTS}
        self.trk_g = []                                 # as many guitar slots as this vocabulary holds, not as
        while f"trk:g{len(self.trk_g)}" in self.id:      # many as G_SLOTS: a corpus keeps the slots it was built
            self.trk_g.append(self.id[f"trk:g{len(self.trk_g)}"])       # with, and its checkpoints with it
        self.trk_dr, self.trk_bs = self.id["trk:dr"], self.id["trk:bs"]
        self.notes = np.array([[self.id[f"n:{s}:{f}"] for f in range(FRETS)] for s in range(1, 8)])
        self.shift = {}
        for sh in range(-3, 4):
            lut = np.arange(V, dtype=np.int64)
            for s in range(7):
                for f in range(FRETS):
                    lut[self.notes[s, f]] = self.notes[s, f + sh] if 0 <= f + sh < FRETS else -1
            for p in range(BASS[0], BASS[1] + 1):
                lut[self.id[f"bp:{p}"]] = self.id[f"bp:{p + sh}"] if BASS[0] <= p + sh <= BASS[1] else -1
            self.shift[sh] = lut

    def fit(self, rem: int):
        """The longest duration token that fits into ``rem`` ticks, or None."""
        j = bisect.bisect_right(self.ticks_asc, rem) - 1
        return self.durs_asc[j] if j >= 0 else None

    def slots(self, labels: dict) -> list[int]:
        return [self.id.get(f"lbl:{s}:{labels.get(s)}", self.none) if labels.get(s) else self.none for s in SLOTS]

    def tempo(self, bpm: float) -> int:
        b = round(math.log(max(bpm, 40) / 40) / math.log(300 / 40) * 19)
        return self.id[f"tempo:{min(max(b, 0), 19)}"]


def ghash(bar: np.ndarray) -> int:
    """Identity of a bar's guitar part (README §9: riffs, copies)."""
    g = np.ascontiguousarray(bar[(bar & FLAG) > 0], np.uint16)
    return int.from_bytes(hashlib.blake2b(g.tobytes(), digest_size=8).digest(), "little")


# ============================================================================ Guitar Pro -> tokens

class Skip(Exception):
    pass


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _letters(text: str) -> str:
    t = unicodedata.normalize("NFKD", (text or "").lower())
    return " ".join(re.sub(r"[^a-z]+", " ", "".join(c for c in t if not unicodedata.combining(c))).split())


_SECTION_MAP: list | None = None


def section_of(title: str) -> int:
    """Marker text -> section, by configs/sections.json (patterns in order; short ones as whole words)."""
    global _SECTION_MAP
    if _SECTION_MAP is None:
        data = json.loads((ROOT / "configs" / "sections.json").read_text(encoding="utf-8"))
        _SECTION_MAP = [(_letters(p), SECTIONS.index(lab)) for p, lab in data["patterns"]]
    text = _letters(title)
    for p, lab in _SECTION_MAP:
        if (f" {p} " in f" {text} ") if len(p) <= 3 else (p in text):
            return lab
    return SECTIONS.index("other")


def drum_class(n: int) -> int:
    for c, notes in enumerate(((35, 36), (37, 38, 39, 40), (42, 44, 46), (41, 43, 45, 47, 48, 50),
                               (49, 52, 55, 57), (51, 53, 59))):
        if n in notes:
            return c
    return 6


def track_kind(t) -> str | None:
    if t.isPercussionTrack:
        return "drums"
    n, prog, name = len(t.strings), t.channel.instrument, (t.name or "").lower()
    low = min((s.value for s in t.strings), default=0)
    if 32 <= prog <= 39 or "bass" in name or n in (4, 5) or (n and low < 31):
        return "bass"
    if n in (6, 7) and (24 <= prog <= 31 or re.search(r"guit|gtr|git|rhythm|lead|solo|dist|clean|acoust|riff", name)):
        return "guitar"
    return None


def open_strings(t) -> list[int]:
    return [s.value + t.offset for s in sorted(t.strings, key=lambda s: s.number)]     # string 1 (high) first


def tuning_plans(tracks):
    """README §3: one transposition k for the song, then per guitar a map of its strings onto Drop A.

    Standard-like tunings land on Drop A's six upper strings (E standard keeps every fret); a 6-string
    drop tuning lands on the six lowest, so its drop shapes stay drop shapes. Strings whose interval
    differs (the B string of a drop tuning, the low B of B standard) keep their pitch with a fret shift.
    """
    opens, tgts, diffs = [open_strings(t) for t in tracks], [], Counter()
    for op in opens:
        tgt = list(range(1, 7)) if len(op) == 6 and op[4] - op[5] == 7 else list(range(min(len(op), 7)))
        tgts.append(tgt)
        diffs.update(DROP_A[t] - op[i] for i, t in enumerate(tgt))
    k = max(diffs, key=lambda d: (diffs[d], -abs(d)))
    return k, [(k, tgt, [op[i] + k - DROP_A[t] for i, t in enumerate(tgt)], op) for op, tgt in zip(opens, tgts)]


def place_note(string: int, fret: int, plan, used: set):
    """(Drop A string index, fret) of a note, or None. Out-of-range frets move, at the same pitch, to the
    nearest free string that can play them."""
    k, tgt, delta, op = plan
    i = string - 1
    if i >= len(op):
        return None
    start = 6
    if i < len(tgt):
        t, f = tgt[i], fret + delta[i]
        if 0 <= f < FRETS and t not in used:
            return t, f
        start = t
    pitch = op[i] + fret + k
    for t in sorted(range(7), key=lambda x: abs(x - start)):
        f = pitch - DROP_A[t]
        if 0 <= f < FRETS and t not in used:
            return t, f
    return None


def note_fx(n, gp) -> list[str]:
    e, out = n.effect, []
    if n.type == gp.NoteType.tie:
        out.append("tie")
    elif n.type == gp.NoteType.dead:
        out.append("dead")
    for name, on in (("ghost", e.ghostNote), ("pm", e.palmMute), ("ho", e.hammer), ("slide", bool(e.slides)),
                     ("bend", e.bend is not None), ("harm", e.harmonic is not None), ("vib", e.vibrato),
                     ("ring", e.letRing), ("acc", e.accentuatedNote or e.heavyAccentuatedNote), ("stacc", e.staccato)):
        if on:
            out.append(name)
    return out


def track_bar(measure, kind: str, plan, V: Vocab, L: int):
    """Tokens of one track in one bar (README §4) and its guitar notes as (onset, string index, pitch).
    An empty bar gives no tokens; otherwise the beats fill the bar exactly (overfull bars are cut,
    underfull ones end with rests)."""
    import guitarpro as gp
    voice = max(measure.voices, key=lambda v: sum(len(b.notes) for b in v.beats))
    toks, notes, pos, any_note = [], [], 0, False
    for beat in voice.beats:
        d = beat.duration
        v = min(DUR_VALUES, key=lambda x: abs(math.log2(x) - math.log2(max(d.value, 1))))
        u = d.tuplet.enters if TUPLETS.get(d.tuplet.enters) == d.tuplet.times else 1
        did = V.id[dur_name(v, bool(d.isDotted), u)]
        if pos + V.dur[did] > L:
            did = V.fit(L - pos)
            if did is None:
                break
        played = [n for n in beat.notes if n.type != gp.NoteType.rest]
        body = []
        if kind == "guitar":
            used, placed = set(), []
            for n in played:
                p = place_note(n.string, n.value, plan, used)
                if p:
                    used.add(p[0])
                    placed.append((p[0], p[1], n))
            placed.sort(key=lambda x: -x[0])                     # low string first
            if any(n.effect.tremoloPicking is not None for _, _, n in placed):
                body.append(V.trem)
            for t, f, n in placed:
                body.append(int(V.notes[t, f]))
                body += [V.fx[e] for e in note_fx(n, gp)]
                notes.append((pos, t, DROP_A[t] + f))
        elif kind == "bass":
            k, op = plan
            pitches = set()
            for n in played:
                if n.string <= len(op):
                    p = op[n.string - 1] + n.value + k
                    while p < BASS[0]:
                        p += 12
                    while p > BASS[1]:
                        p -= 12
                    pitches.add(p)
            body = [V.id[f"bp:{p}"] for p in sorted(pitches)[:4]]
        else:
            body = [V.id[f"dr:{DRUMS[c]}"] for c in sorted({drum_class(n.value) for n in played})]
        any_note |= bool(body)
        toks += [did] + body
        pos += int(V.dur[did])
    if not any_note:
        return [], []
    while L - pos >= V.min_dur:
        did = V.fit(L - pos)
        toks.append(did)
        pos += int(V.dur[did])
    return toks, notes


def unroll(headers, max_bars: int = 2048) -> list[int]:
    """Measure indices in playing order: repeats (at most 4 passes) and alternative endings written out."""
    out, i, start, passno, done = [], 0, 0, 1, {}
    while i < len(headers) and len(out) < max_bars:
        h = headers[i]
        if h.isRepeatOpen and i != start:
            start, passno = i, 1
        alt = h.repeatAlternative
        if alt and not (alt >> (passno - 1)) & 1:
            i += 1
            continue
        out.append(i)
        if h.repeatClose > 0:
            if done.get(i, 0) < min(h.repeatClose, 3):
                done[i] = done.get(i, 0) + 1
                passno += 1
                i = start
                continue
            done[i] = 0
            start, passno = i + 1, 1
        i += 1
    return out


def encode_song(song, V: Vocab, max_bars: int = 2048) -> dict:
    """A parsed Guitar Pro song -> bars of tokens plus per-bar features (README §2-§5)."""
    tracks = song.tracks
    kinds = [track_kind(t) for t in tracks]
    counts = [sum(len(b.notes) for m in t.measures for v in m.voices for b in v.beats) for t in tracks]
    guitars, seen = [], set()
    for i in sorted((i for i, k in enumerate(kinds) if k == "guitar" and counts[i]), key=lambda i: -counts[i]):
        sig = hash(tuple((b.start, n.string, n.value) for m in tracks[i].measures for v in m.voices
                         for b in v.beats for n in b.notes))
        if sig not in seen and len(guitars) < len(V.trk_g):  # doubled tracks are kept once
            seen.add(sig)
            guitars.append(i)
    guitars.sort()
    if not guitars:
        raise Skip("no guitar track")
    first = lambda kind: next((i for i, k in enumerate(kinds) if k == kind and counts[i]), None)  # noqa: E731
    dr, bs = first("drums"), first("bass")
    k, plans = tuning_plans([tracks[i] for i in guitars])
    if abs(k) > 12:
        raise Skip("tuning out of range")
    ctx_parts = [("dr", dr, "drums", None), ("bs", bs, "bass", (k, open_strings(tracks[bs])) if bs is not None else None)]
    order = unroll(song.measureHeaders, max_bars)
    memo = {}

    def part(ti, kind, plan, m, L):
        if (ti, m) not in memo:
            memo[ti, m] = track_bar(tracks[ti].measures[m], kind, plan, V, L)
        return memo[ti, m]

    bars, feats = [], {n: [] for n in ("hash", "pc", "on", "low", "gn", "sec")}
    for m in order:
        h = song.measureHeaders[m]
        num, den = h.timeSignature.numerator, h.timeSignature.denominator.value
        if den not in TS_DENS or not 1 <= num <= 32:
            raise Skip("time signature")
        L = TICKS * num // den
        head = [V.id[f"ts:{num}/{den}"]]
        sec = section_of(h.marker.title) if h.marker is not None else -1
        if sec >= 0:
            head.append(V.id[f"sec:{SECTIONS[sec]}"])
        ctx = [[V.id[f"trk:{name}"]] + part(ti, kind, plan, m, L)[0] for name, ti, kind, plan in ctx_parts if ti is not None]
        gparts, gnotes = [], []
        for slot, (ti, plan) in enumerate(zip(guitars, plans)):
            t, n = part(ti, "guitar", plan, m, L)
            gparts.append([V.trk_g[slot]] + t)
            gnotes.append(n)
        room = lambda: len(head) + sum(map(len, ctx)) + sum(map(len, gparts)) + 1   # noqa: E731 - +1: <eob>
        while ctx and room() > MAX_BAR:
            ctx.pop(0)                                       # a too dense bar loses its drums, then its bass,
        while len(gparts) > 1 and room() > MAX_BAR:
            gparts.pop()                                     # then its last guitar, rather than the whole song
            gnotes.pop()
        g = [x for p in gparts for x in p] + [V.eob]
        notes = [x for n in gnotes for x in n]
        if len(head) + len(g) > MAX_BAR:
            raise Skip("bar too dense")
        bar = np.concatenate([np.array(head + [x for c in ctx for x in c], np.uint16), np.array(g, np.uint16) | FLAG])
        bars.append(bar)
        pc = on = low = 0
        for pos, t, p in notes:
            pc |= 1 << (p % 12)
            on |= 1 << min(15, pos * 16 // L)
            low += t >= 5
        feats["hash"].append(ghash(bar))
        feats["pc"].append(pc)
        feats["on"].append(on)
        feats["low"].append(int(100 * low / len(notes)) if notes else 0)
        feats["gn"].append(len(notes))
        feats["sec"].append(sec)
    if not bars:
        raise Skip("no bars")
    dt = {"hash": np.uint64, "pc": np.uint16, "on": np.uint16, "low": np.uint8, "gn": np.uint16, "sec": np.int8}
    return {"bars": bars, "feats": {n: np.array(v, dt[n]) for n, v in feats.items()}, "orig": order,
            "guitars": guitars, "plans": plans, "k": k, "tempo": float(song.tempo)}


# ============================================================================ 1) split, curate, encode

def family(path: Path) -> str:
    """Versions of one song form one family (README §15): 'Band - Song (ver 2)__id12.gp5' -> 'band_song'."""
    s = re.sub(r"__(v|id)\d+", "", path.stem)
    return slug(re.sub(r"\([^)]*\)|\[[^]]*\]", "", s))


def labels_of(rel: Path) -> dict:
    """Folder -> labels (README §8). The folders are the whole truth, so the same tab can sit in a genre folder
    and in any number of album folders and is labelled once per copy:

        tabs - own - <album> - own/...    album + src:own    mine - the model may copy it or come very close to it
        tabs - own - <album> - insp/...   album + src:insp   chosen as inspiration - it may inform, not be reproduced
        tabs - <genre>/...   genre              (an unlisted folder becomes genre:other)
        the EARLY folders    src:early          the early years: learned from, steered away from

    The album and the source are both in the folder name, so there is no subfolder to forget."""
    top = rel.parts[0]
    if (af := album_folder(top)) is not None:
        return {"album": af[0], "src": af[1]}
    name = slug(top[len("tabs - "):] if top.lower().startswith("tabs - ") else top)
    if name in EARLY:
        return {"src": "early"}
    return {"genre": name if name in GENRES else GENRE_OTHER}


def label_match(labels: dict, spec: str) -> bool:
    """``src:own`` matches that key and value; ``album:*`` matches any value of that key."""
    k, _, v = spec.partition(":")
    return k in labels and (v in ("*", "") or labels[k] == v)


_BASE: Vocab | None = None


def _prepare_one(job):
    global _BASE
    import guitarpro as gp
    path, root = Path(job[0]), Path(job[1])
    _BASE = _BASE or Vocab(make_tokens())
    try:
        enc = encode_song(gp.parse(str(path)), _BASE)
    except Skip as e:
        return "skip", str(e)
    except Exception as e:                                  # noqa: BLE001 - a damaged file is a skip, not a crash
        return "skip", f"unreadable ({type(e).__name__})"
    rel = path.relative_to(root)
    toks = np.concatenate(enc["bars"])
    return "ok", {"file": rel.as_posix(), "family": family(path), "labels": labels_of(rel), "tempo": enc["tempo"],
                  "holdout": "holdout" in rel.parts, "digest": hashlib.md5(toks.tobytes()).hexdigest(),
                  "lens": np.array([len(b) for b in enc["bars"]], np.int32), "tokens": toks, "feats": enc["feats"]}


def cmd_prepare(a) -> None:
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    folders = sorted(d for d in src.iterdir() if d.is_dir() and d.name.lower() not in ("tabs - errors", "__pycache__")
                     and d.name.lower().startswith("tabs - "))
    files = [p for d in folders for p in sorted(d.rglob("*")) if p.suffix.lower() in (".gp3", ".gp4", ".gp5")
             and not {"__erroneous", "__8string"} & set(p.parts)]
    # An album folder whose name is not registered in ALBUMS would label its tabs with a token that does not
    # exist, and Vocab.slots would quietly fall back to <none>: the tabs would train with no album label at all.
    # A misspelt folder is far likelier than a deliberate new album, so it stops here rather than an hour later.
    if stray := sorted({d.name for d in folders if d.name.lower().startswith("tabs - own - ") and (album_folder(d.name) is None or album_folder(d.name)[0] not in ALBUMS)}):
        sys.exit(f"album folder(s) not in ALBUMS: {', '.join(stray)}\n"
                 f"known albums: {', '.join(ALBUMS)}\n"
                 f"fix the folder name, or add the album to ALBUMS in riff.py - but adding one renumbers the "
                 f"label ids, so every existing checkpoint has to be pretrained again.")
    # The holdout songs are yours, so they belong in 'tabs - own - <album> - own/holdout/': under '... - insp' they would read
    # as src:insp and the model would be told it may not copy your own song. Worth stopping for - the mistake is invisible otherwise.
    if bad := sorted({p.relative_to(src).parts[0] for p in files
                      if (af := album_folder(p.relative_to(src).parts[0])) and af[1] == "insp" and "holdout" in p.relative_to(src).parts}):
        sys.exit(f"holdout songs under {', '.join(bad)}: move them to 'tabs - own - <album> - own/holdout/', "
                 f"otherwise they are labelled src:insp (inspiration) instead of src:own.")
    excl = [s.strip() for s in a.exclude.split(",") if s.strip()]
    if excl:
        kept = [p for p in files if not any(label_match(labels_of(p.relative_to(src)), s) for s in excl)]
        log(f"excluded {len(files) - len(kept)} files matching {', '.join(excl)}")
        files = kept
    if a.frac < 1.0:                                    # a fraction of each folder, before the train/test split
        by: dict[str, list] = {}
        for p in files:
            by.setdefault(p.relative_to(src).parts[0], []).append(p)
        sampled = []
        for folder, ps in sorted(by.items()):           # seeded per folder: one folder changing leaves the rest alone
            r = rng_for(a.seed, int(hashlib.md5(folder.encode()).hexdigest()[:8], 16))
            n = max(1, round(len(ps) * a.frac))
            sampled += [ps[i] for i in sorted(r.choice(len(ps), n, replace=False).tolist())]
        log(f"sampled {len(sampled)}/{len(files)} files ({a.frac:.0%} of each of the {len(by)} folders)")
        files = sampled
    if a.limit:
        keep = set(np.random.default_rng(a.seed).choice(len(files), min(a.limit, len(files)), replace=False).tolist())
        files = [p for i, p in enumerate(files) if i in keep]
    log(f"{len(files)} Guitar Pro files in {len(folders)} folders")
    songs, skips, notes, fams = [], Counter(), Counter(), Counter()
    lens, feats = [], {n: [] for n in ("hash", "pc", "on", "low", "gn", "sec")}
    jobs = [(str(p), str(src)) for p in files]
    with open(out / "tokens.bin", "wb") as fh, mp.Pool(a.workers) as pool:
        bar0 = 0
        for n, (status, r) in enumerate(pool.imap_unordered(_prepare_one, jobs, chunksize=8), 1):
            if n % 2000 == 0:
                log(f"  {n}/{len(files)}: {len(songs)} songs, skipped {dict(skips)}")
            if status == "skip":
                skips[r] += 1
                continue
            # The same tab in several folders is kept, once per folder: copying a file into an album folder is
            # how a tab is assigned, and the folders are the truth. So versions are counted per top-level folder
            # - otherwise --max-versions, meant to stop one popular song flooding a genre, would drop the copy
            # that carries the album label. A copy is marked (dup) rather than dropped: it trains, since being in
            # two folders is a deliberate reweighting, but it supplies no validation or test gaps, where it would
            # only score the same bars twice.
            parts = r["file"].split("/")
            if fams[parts[0], r["family"]] >= a.max_versions:
                skips["more versions of one song"] += 1
                continue
            fams[parts[0], r["family"]] += 1
            fh.write(r["tokens"].astype(np.uint16).tobytes())
            lens.append(r["lens"])
            for k in feats:
                feats[k].append(r["feats"][k])
            nb = len(r["lens"])
            songs.append({k: r[k] for k in ("file", "family", "labels", "tempo")} |
                         {"bar0": bar0, "nb": nb, "tokens": int(r["lens"].sum()), "holdout": r["holdout"],
                          "_digest": r["digest"]})
            bar0 += nb
    # Which copy of a tab counts as the original is settled here and not in the loop above: the workers finish in
    # whatever order they finish in, so marking copies as they arrived would hand the fixed validation gaps to a
    # different copy on every run. The order is fixed and prefers the plain genre copy, so a tab assigned to an
    # album is scored as corpus material and the album sets stay out of the evaluation entirely.
    seen, copies = {}, Counter()
    for s in sorted(songs, key=lambda s: (s["file"].startswith("tabs - own - "), s["file"])):
        d = s.pop("_digest")
        copies[d] += 1
        if d in seen:
            s["dup"] = True
        seen[d] = True
    if n := sum(v - 1 for v in copies.values()):
        notes["copies of a tab in another folder"] = n
    if not songs:
        # Every file was skipped. Without this the next line dies on an empty list with "need at least one
        # array to concatenate", which says nothing about why; the skip counts are the answer and they are
        # right here. It happens when a folder holds only tabs the encoder cannot use - drum, bass or piano
        # transcriptions with no guitar track, say.
        sys.exit(f"no song survived: {len(files)} file(s) read, all skipped\n"
                 + "\n".join(f"  {n:6d}  {why}" for why, n in skips.most_common()))
    lens = np.concatenate(lens)
    np.save(out / "bar_off.npy", np.concatenate([[0], np.cumsum(lens, dtype=np.int64)]))
    for k, v in feats.items():
        np.save(out / f"bar_{k}.npy", np.concatenate(v))
    for s in songs:                                     # split by song family (README §10), own songs by holdout/
        h = int(hashlib.md5(s["family"].encode()).hexdigest()[:8], 16) % 1000
        s["split"] = "val" if s.pop("holdout") else "val" if h < 15 else "test" if h < 30 else "train"
    gaps = fixed_gaps(songs, {k: np.concatenate(v) for k, v in feats.items()}, a.seed)
    labels = sorted({f"{k}:{v}" for s in songs for k, v in s["labels"].items()})
    train_tok = sum(s["tokens"] for s in songs if s["split"] == "train")
    stats = {"files": len(files), "songs": len(songs), "skipped": dict(skips), "notes": dict(notes),
             "copied_most": max(copies.values(), default=0), "bars": int(len(lens)),
             "tokens": int(lens.sum()), "train_tokens": int(train_tok), "tokens_per_bar": float(lens.mean()),
             "tokens_per_bar_p90": float(np.percentile(lens, 90)), "bars_per_song": float(len(lens) / max(len(songs), 1)),
             "splits": dict(Counter(s["split"] for s in songs)),
             "labels": dict(Counter(f"{k}:{v}" for s in songs for k, v in s["labels"].items()))}
    (out / "meta.json").write_text(json.dumps({"tokens": make_tokens(), "labels": labels, "stats": stats,
                                               "songs": songs, "gaps": gaps}), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log(f"{len(songs)} songs, {stats['tokens'] / 1e6:.1f}M tokens ({stats['tokens_per_bar']:.0f}/bar), "
        f"{len(gaps)} fixed val/test gaps -> {out}\nskipped: {dict(skips)}"
        + (f"\nnote: {dict(notes)}" if notes else ""))


def fixed_gaps(songs, feats, seed, per_split=3000) -> list[dict]:
    """Validation and test gaps, chosen once (README §10), 4-16 bars with guitar notes and context on both sides."""
    out = Counter()
    gaps = []
    for i, s in enumerate(songs):
        if s["split"] == "train" or out[s["split"]] >= per_split or s["nb"] < 12 or s.get("dup"):
            continue                                    # a copy of a tab in another folder scores the same bars
        rng = rng_for(seed, i)
        gn = feats["gn"][s["bar0"]:s["bar0"] + s["nb"]]
        for _ in range(2):
            L = int(rng.choice((4, 8, 16), p=(0.3, 0.5, 0.2)))
            for _ in range(10):
                if s["nb"] - L - 1 < 1:
                    break
                st = int(rng.integers(1, s["nb"] - L))
                if (gn[st:st + L] > 0).mean() >= 0.5:
                    gaps.append({"song": i, "start": st, "len": L, "split": s["split"]})
                    out[s["split"]] += 1
                    break
    return gaps


def rng_for(*keys) -> np.random.Generator:
    return np.random.default_rng([int(k) & 0xFFFFFFFF for k in keys])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class SongView:
    """One song's bars and features, from the corpus or from a file being filled."""

    def __init__(self, tokens, off, feats, labels, tempo):
        self.tok, self.off, self.f, self.labels, self.tempo = tokens, off, feats, labels, tempo
        self.nb = len(off) - 1
        self.lens = np.diff(off)

    def bars(self, lo: int, hi: int) -> list[np.ndarray]:
        return [np.asarray(self.tok[self.off[i]:self.off[i + 1]]) for i in range(lo, hi)]

    @classmethod
    def from_encoded(cls, enc: dict, labels: dict):
        lens = [len(b) for b in enc["bars"]]
        return cls(np.concatenate(enc["bars"]), np.concatenate([[0], np.cumsum(lens)]), enc["feats"], labels, enc["tempo"])


class Corpus:
    """data_encoded/encoded: tokens.bin + per-bar arrays (memory-mapped) + meta.json."""

    def __init__(self, path):
        self.dir = Path(path)
        meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.vocab, self.labels, self.stats = Vocab(meta["tokens"]), meta["labels"], meta["stats"]
        self.songs, self.gaps = meta["songs"], meta["gaps"]
        self._mm = None
        self.cache = {}                                  # derived sets that cost more than one evaluation

    def _open(self):
        if self._mm is None:
            off = np.load(self.dir / "bar_off.npy", mmap_mode="r")
            tok = np.memmap(self.dir / "tokens.bin", np.uint16, "r") if off[-1] else np.zeros(0, np.uint16)
            feats = {k: np.load(self.dir / f"bar_{k}.npy", mmap_mode="r") for k in ("hash", "pc", "on", "low", "gn", "sec")}
            self._mm = tok, off, feats
        return self._mm

    def view(self, sid: int) -> SongView:
        tok, off, feats = self._open()
        s = self.songs[sid]
        b0, nb = s["bar0"], s["nb"]
        o = np.asarray(off[b0:b0 + nb + 1])
        return SongView(tok[o[0]:o[-1]], o - o[0], {k: np.asarray(v[b0:b0 + nb]) for k, v in feats.items()},
                        s["labels"], s["tempo"])

    def __getstate__(self):
        return self.__dict__ | {"_mm": None, "cache": {}}


# ============================================================================ training examples

POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], np.uint8)
GAP_LEN, GAP_P = (1, 2, 4, 8, 16), (0.05, 0.1, 0.25, 0.35, 0.25)


def jaccard(a, b):
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    u = POP16[a | b]
    return np.where(u > 0, POP16[a & b] / np.maximum(u, 1), 1.0)


def far_repeats(f, lo, hi, dist=32) -> list[int]:
    same = {}
    for i in range(lo, hi):
        if f["gn"][i]:
            same.setdefault(int(f["hash"][i]), []).append(i)
    return [i for ids in same.values() if len(ids) > 1 for i in ids if any(abs(i - j) >= dist for j in ids)]


def pick_gaps(f, rng, lo, hi, n, frac, avoid=()) -> list[tuple[int, int]]:
    """Up to n gaps (start, length) in [lo, hi), until about ``frac`` of the bars are hidden. Every gap has
    guitar notes; most start on a phrase or section border, 25 % on a bar that repeats >= 32 bars away."""
    size, gaps, hidden = hi - lo, [], 0
    border = [i for i in range(lo, hi) if f["sec"][i] >= 0 or (i - lo) % 4 == 0]
    far = far_repeats(f, lo, hi)
    taken = list(avoid)
    for _ in range(n * 10):
        if len(gaps) == n or (gaps and hidden >= frac * size):
            break
        L = min(int(rng.choice(GAP_LEN, p=GAP_P)), max(1, int(size * 0.5) - hidden))
        r = rng.random()
        s = (int(rng.choice(far)) - int(rng.integers(0, L)) if far and r < 0.25 else
             int(rng.choice(border)) if border and r < 0.85 else int(rng.integers(lo, hi)))
        s = max(lo, min(s, hi - L))
        if L >= size or f["gn"][s:s + L].sum() == 0 or any(s <= b + l and b <= s + L for b, l in taken):
            continue
        gaps.append((s, L))
        taken.append((s, L))
        hidden += L
    return gaps


def relation(f, s, L) -> str:
    """README §9: how the gap relates to the riff before it."""
    seg = slice(s, s + L)
    busy = f["gn"][seg] > 0
    if busy.any() and ((POP16[f["pc"][seg].astype(np.int64)] <= 2) & (f["low"][seg] >= 70))[busy].mean() >= 0.75:
        return "chug"
    n = min(L, s)
    if n == 0:
        return "new"
    cur, prev = slice(s, s + n), slice(s - n, s)
    if (f["hash"][cur] == f["hash"][prev]).all():
        return "copy"
    sp, so = jaccard(f["pc"][cur], f["pc"][prev]).mean(), jaccard(f["on"][cur], f["on"][prev]).mean()
    return "variation" if (sp + so) / 2 >= 0.6 else "answer" if so >= 0.6 else "new"


def example_slots(V, view, rel, rng, train) -> list[int]:
    labels = dict(view.labels) | ({"rel": rel} if rel else {})
    if train:                                            # README §8: every label is dropped 15 % of the time
        labels = {k: v for k, v in labels.items() if rng.random() >= 0.15}
    return V.slots(labels)


def augment(bars, V, rng, guitar_drop=0.2, info=None) -> list[np.ndarray]:
    """README §2, §3, §8, §10, for the whole example at once: drop drums / bass (15 % each), drop one of
    several guitars (20 %), shift the song 1-3 semitones within Drop A (30 %), drop section markers (30 %).
    ``info`` receives the shift."""
    lens = np.array([len(b) for b in bars])
    cat = np.concatenate(bars).astype(np.int64)
    ids, flags = cat & MASK, cat & FLAG
    bar_of = np.repeat(np.arange(len(bars)), lens)
    head = (V.kind[ids] == K_TRK) | (V.kind[ids] == K_TS) | (ids == V.eob)
    seg = np.cumsum(head) - 1
    seg_head = ids[head]
    drop = np.zeros(len(seg_head), bool)
    for trk in (V.trk_dr, V.trk_bs):
        if rng.random() < 0.15:
            drop |= seg_head == trk
    present = [g for g in range(len(V.trk_g)) if (seg_head == V.trk_g[g]).any()]
    if len(present) >= 2 and rng.random() < guitar_drop:
        g = int(rng.choice(present))
        drop |= seg_head == V.trk_g[g]
        for h in range(g + 1, len(V.trk_g)):
            ids[ids == V.trk_g[h]] = V.trk_g[h - 1]
    keep = ~drop[seg]
    if rng.random() < 0.3:
        keep &= V.kind[ids] != K_SEC
    if rng.random() < 0.3:
        for sh in rng.permutation([-3, -2, -1, 1, 2, 3]):
            new = V.shift[int(sh)][ids]
            if (new[keep] >= 0).all():
                ids = np.where(new >= 0, new, ids)
                if info is not None:
                    info["shift"] = int(sh)
                break
    out = (ids | flags)[keep].astype(np.uint16)
    return np.split(out, np.cumsum(np.bincount(bar_of[keep], minlength=len(bars)))[:-1])


def choose_context(f, lens, s, L, holes, budget) -> list[int]:
    """README §7: bars for a window that cannot hold the whole song: the nearest bars (50 %), then related
    bars (to 85 %): repeats of the bars next to the gap and what followed them, intro and outro."""
    nb = len(lens)
    cand = [i for i in range(nb) if not s <= i < s + L and i not in holes]
    if lens[cand].sum() <= budget:
        return cand
    ok, chosen, used = set(cand), set(), 0

    def take(i):
        nonlocal used
        if i in ok and i not in chosen and used + lens[i] <= budget:
            chosen.add(i)
            used += int(lens[i])

    near = [x for pair in itertools.zip_longest(range(s - 1, -1, -1), range(s + L, nb)) for x in pair if x is not None]
    for i in near:
        if used >= budget * 0.5:
            break
        take(i)
    same = {}
    for i in range(nb):
        if f["gn"][i]:
            same.setdefault(int(f["hash"][i]), []).append(i)
    related = []
    for a in list(range(s - 1, max(-1, s - 5), -1)) + list(range(s + L, min(nb, s + L + 4))):
        for r in same.get(int(f["hash"][a]), []):
            if r != a:
                related += [r] + (list(range(r + 1, min(nb, r + 1 + L))) if a < s else list(range(max(0, r - L), r)))
    related += list(range(min(8, nb))) + list(range(max(0, nb - 8), nb))
    for i in related:
        if used >= budget * 0.85:
            break
        take(i)
    for i in near:
        take(i)
    return sorted(chosen)


def flat_example(V, view, rng, ctx, target=None, holes=(), train=True, slots=None, ctx_weight=0.5):
    """Baseline example (README §6): [labels tempo len <pre> before <suf> after <mid> gap bars]. Skipped
    bars become gap:N. Returns (ids, weights, prompt length) or None."""
    nb, f = view.nb, view.f
    holes = set(holes)
    if target is None:
        g = pick_gaps(f, rng, 0, nb, 1, 0.0)
        if not g:
            return None
        target = g[0]
        if rng.random() < 0.2:
            for s2, L2 in pick_gaps(f, rng, 0, nb, 1, 0.0, avoid=[target]):
                holes |= set(range(s2, s2 + L2))
    s, L = target
    while L > 1 and view.lens[s:s + L].sum() > ctx // 2:
        L -= 1
    budget = int(0.95 * ctx) - 16 - int(view.lens[s:s + L].sum())
    keep = choose_context(f, view.lens, s, L, holes, budget)
    order = keep + list(range(s, s + L))
    loaded = [np.asarray(view.tok[view.off[i]:view.off[i + 1]]) for i in order]
    if train:
        loaded = augment(loaded, V, rng)
    bar = dict(zip(order, loaded))
    if slots is None:
        slots = example_slots(V, view, relation(f, s, L), rng, train)
    ids, wts = [V.bos, *slots, V.tempo(view.tempo), V.id[f"len:{min(L, 64)}"], V.pre], [0.0] * 8

    def emit(lo, hi, cw):
        last = lo
        for i in [i for i in keep if lo <= i < hi] + [hi]:
            if i > last:
                ids.append(V.id[f"gap:{min(i - last, 64)}"])
                wts.append(0.0)
            if i < hi:
                b = bar[i]
                ids.extend((b & MASK).tolist())
                wts.extend(((b & FLAG) > 0) * cw)
            last = i + 1

    emit(0, s, ctx_weight if train else 0.0)
    ids.append(V.suf)
    wts.append(0.0)
    emit(s + L, nb, ctx_weight if train else 0.0)
    ids.append(V.mid)
    wts.append(0.0)
    n_prompt = len(ids)
    for i in range(s, s + L):
        ids.extend((bar[i] & MASK).tolist())
        wts.extend(((bar[i] & FLAG) > 0) * 1.0)
    if len(ids) > ctx:
        return None
    return np.array(ids, np.int64), np.array(wts, np.float32), n_prompt


def phrase_pos(sec) -> np.ndarray:
    """Bar k (0-15) of its phrase: counted from the last section marker (README §6)."""
    out, start = np.zeros(len(sec), np.int64), 0
    for i, x in enumerate(sec):
        if x >= 0:
            start = i
        out[i] = (i - start) % 16
    return out


def hier_example(V, view, rng, max_bars, targets=None, holes=(), train=True, slots=None):
    """Big/huge example (README §6): a window of whole bars with 1-4 target gaps (10-50 % of the bars),
    sometimes an unknown placeholder gap, and the labels."""
    nb, f = view.nb, view.f
    holes = set(holes)
    if targets is None:
        lo = int(rng.integers(0, nb - max_bars + 1)) if nb > max_bars else 0
        hi = min(nb, lo + max_bars)
        targets = pick_gaps(f, rng, lo, hi, 4, rng.uniform(0.1, 0.5))
        if not targets:
            return None
        if rng.random() < 0.3:
            for s2, L2 in pick_gaps(f, rng, lo, hi, 1, 0.0, avoid=targets):
                holes |= set(range(s2, s2 + L2))
        if rng.random() < 0.1:                           # hide the nearest bars: the model must look far
            s, L = targets[0]
            k = int(rng.integers(4, 9))
            holes |= (set(range(max(lo, s - k), s)) | set(range(s + L, min(hi, s + L + k)))) - \
                {b for s2, L2 in targets for b in range(s2, s2 + L2)}
    else:
        s, L = targets[0]
        lo = max(0, min(s + L // 2 - max_bars // 2, nb - max_bars))
        hi = min(nb, lo + max_bars)
    roles = np.zeros(hi - lo, np.int64)
    for b in holes:
        if lo <= b < hi:
            roles[b - lo] = 1
    for s, L in targets:
        roles[s - lo:s + L - lo] = 2
    bars = view.bars(lo, hi)
    if train:
        bars = augment(bars, V, rng)
    if slots is None:
        slots = example_slots(V, view, relation(f, *targets[0]) if len(targets) == 1 else None, rng, train)
    sec = f["sec"][lo:hi]
    eff = np.maximum.accumulate(np.where(sec >= 0, np.arange(len(sec)), -1))
    return {"bars": bars, "roles": roles, "targets": [(s - lo, L) for s, L in targets], "slots": slots,
            "sec": np.where(eff >= 0, sec[np.maximum(eff, 0)], -1), "phr": phrase_pos(sec),
            "hash": f["hash"][lo:hi], "pc": f["pc"][lo:hi], "on": f["on"][lo:hi], "gn": f["gn"][lo:hi],
            "ntok": int(sum(len(b) for b in bars))}


def length_chunks(lengths, budget) -> list[list[int]]:
    """Indices grouped by similar length, each group at most ``budget`` padded tokens (less padding waste)."""
    chunks, cur, mx = [], [], 0
    for i in np.argsort(lengths, kind="stable"):
        L = int(lengths[i])
        if cur and max(mx, L) * (len(cur) + 1) > budget:
            chunks.append(cur)
            cur, mx = [], 0
        cur.append(int(i))
        mx = max(mx, L)
    return chunks + [cur] if cur else chunks


def pad_rows(rows, fill, dtype=torch.long) -> torch.Tensor:
    out = torch.full((len(rows), max(len(r) for r in rows)), fill, dtype=dtype)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.as_tensor(np.asarray(r))
    return out


def collate_flat(exs, V):
    ids = pad_rows([e[0] for e in exs], V.pad)
    w = pad_rows([e[1][1:] for e in exs], 0.0, torch.float32)
    return {"ids": ids, "w": w, "ntok": int(sum(len(e[0]) for e in exs))}


def enc_chunks(bars, V, budget=16384):
    """Bars for the bar encoder: <bos> (read out as the bar vector) + the bar's tokens, chunked by length."""
    rows = [np.concatenate([[V.bos], b & MASK]) for b in bars]
    return [(torch.tensor(c), pad_rows([rows[i] for i in c], V.pad)) for c in length_chunks([len(r) for r in rows], budget)]


def collate_hier(exs, V, cfg):
    """Global sequence per example: 4 label slots, the known bars in order (unknown ones as <unk_bar>), then
    every target gap: <gap_start> and the gap's bars shifted by one (README §6: fill-in-the-middle order).
    Positions are the bars' true indices, so distances stay musical."""
    huge, mem_len = bool(cfg.get("pointer")), cfg.get("mem_len", 160)
    Vn = V.size
    enc, g_idx, g_pos, g_seg, g_phr, g_sec, nctx = [], [], [], [], [], [], []
    cond, loc_rows, loc_tgt, loc_w, ptr_cand, ptr_sim, plan_pc, plan_on = [], [], [], [], [], [], [], []
    for e_i, e in enumerate(exs):
        idx, pos, seg, phr, sec = list(e["slots"]), [0] * 4, [0] * 4, [0] * 4, [-1] * 4
        ctx_enc, ctx_bars = [], []
        for i, r in enumerate(e["roles"]):
            if r == 2:
                continue
            if r == 0:
                ctx_enc.append(len(enc))
                ctx_bars.append(i)
                idx.append(Vn + len(enc))
                enc.append(e["bars"][i])
            else:
                idx.append(V.unk_bar)
            pos.append(i + 1)
            seg.append(1 if r == 0 else 2)
            phr.append(int(e["phr"][i]))
            sec.append(int(e["sec"][i]) if r == 0 else -1)
        nctx.append(len(idx))
        for s, L in e["targets"]:
            for j in range(L):
                b = s + j
                if j == 0:
                    idx.append(V.gap_start)
                else:
                    idx.append(Vn + len(enc))
                    enc.append(e["bars"][b - 1])
                pos.append(b + 1)
                seg.append(3)
                phr.append(int(e["phr"][b]))
                sec.append(-1)
                cond.append(e_i * 100000 + len(idx) - 1)
                bar = e["bars"][b]
                loc_rows.append(np.concatenate([[V.pad], bar[:-1] & MASK]))      # position 0 carries the plan
                loc_tgt.append(bar & MASK)
                loc_w.append(((bar & FLAG) > 0).astype(np.float32))
                if huge:
                    ptr_cand.append(ctx_enc)
                    h = e["hash"][ctx_bars]
                    sim = 0.5 * (jaccard(e["pc"][ctx_bars], e["pc"][b]) + jaccard(e["on"][ctx_bars], e["on"][b]))
                    ptr_sim.append(np.where((h == e["hash"][b]) & (e["gn"][ctx_bars] > 0), 1.0, sim))
                    plan_pc.append([(int(e["pc"][b]) >> k) & 1 for k in range(12)])
                    plan_on.append([(int(e["on"][b]) >> k) & 1 for k in range(16)])
        g_idx.append(idx)
        g_pos.append(pos)
        g_seg.append(seg)
        g_phr.append(phr)
        g_sec.append(sec)
    T = max(len(x) for x in g_idx)
    cond = [(c // 100000) * T + c % 100000 for c in cond]
    lens, nctx = torch.tensor([len(x) for x in g_idx]), torch.tensor(nctx)
    ar = torch.arange(T)
    mid = (ar[None, :, None] >= nctx[:, None, None]) & (ar[None, None, :] <= ar[None, :, None])
    mask = (ar[None, None, :] < lens[:, None, None]) & ((ar[None, None, :] < nctx[:, None, None]) | mid)
    b = {"enc": enc_chunks(enc, V), "g_idx": pad_rows(g_idx, V.pad), "g_pos": pad_rows(g_pos, 0),
         "g_seg": pad_rows(g_seg, 0), "g_phr": pad_rows(g_phr, 0), "g_mask": mask[:, None],
         "cond": torch.tensor(cond), "ntok": int(sum(e["ntok"] for e in exs)), "loc": []}
    for c in length_chunks([len(r) for r in loc_rows], 16384):
        b["loc"].append((torch.tensor(c), pad_rows([loc_rows[i] for i in c], V.pad),
                         pad_rows([loc_tgt[i] for i in c], V.pad), pad_rows([loc_w[i] for i in c], 0.0, torch.float32)))
    if huge:
        b["sec"] = pad_rows(g_sec, -1)
        b["mem_bank"] = pad_rows([(x[(x & FLAG) > 0] & MASK)[:mem_len] if ((x & FLAG) > 0).any() else [V.pad]
                                  for x in enc], V.pad)[:, :mem_len]
        C = max(1, max(len(x) for x in ptr_cand))
        b["ptr_cand"] = pad_rows([x or [0] for x in ptr_cand], 0)[:, :C]
        b["ptr_mask"] = pad_rows([[True] * len(x) or [False] for x in ptr_cand], False, torch.bool)[:, :C]
        b["ptr_sim"] = pad_rows([x if len(x) else [0.0] for x in ptr_sim], 0.0, torch.float32)[:, :C]
        b["plan_pc"] = torch.tensor(plan_pc, dtype=torch.float32)
        b["plan_on"] = torch.tensor(plan_on, dtype=torch.float32)
    return b


# ============================================================================ models

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.w


def rope_tables(pos: torch.Tensor, dh: int, base: float = 10000.0):
    inv = 1.0 / base ** (torch.arange(0, dh, 2, device=pos.device, dtype=torch.float32) / dh)
    ang = pos.float()[..., None] * inv
    if ang.dim() == 2:
        ang = ang[None]
    return ang.cos()[:, None], ang.sin()[:, None]          # (B|1, 1, T, dh/2)


def apply_rope(x, cos, sin):
    a, b = x.float().chunk(2, -1)
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1).type_as(x)


class Attention(nn.Module):
    """Multi-head attention with rotary positions and normalised queries/keys (stable at higher learning rates)."""

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q, self.kv, self.o = nn.Linear(d, d, bias=False), nn.Linear(d, 2 * d, bias=False), nn.Linear(d, d, bias=False)
        self.qn, self.kn = RMSNorm(self.dh), RMSNorm(self.dh)

    def qkv(self, x, src, rope_q, rope_k):
        B, T, _ = x.shape
        q = self.qn(self.q(x).view(B, T, self.h, self.dh)).transpose(1, 2)
        k, v = self.kv(src).view(B, src.shape[1], 2, self.h, self.dh).unbind(2)
        k, v = self.kn(k).transpose(1, 2), v.transpose(1, 2)
        if rope_q is not None:
            q = apply_rope(q, *rope_q)
        if rope_k is not None:
            k = apply_rope(k, *rope_k)
        return q, k, v

    def forward(self, x, rope=None, mask=None, causal=False, mem=None, kv_out=None):
        q, k, v = self.qkv(x, x if mem is None else mem, rope, rope if mem is None else None)
        if kv_out is not None:
            kv_out.append((k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None)
        return self.o(y.transpose(1, 2).flatten(2))

    def step(self, x, rope, shared, own):
        """One new token per row. Rows are candidates x label rows; the prompt's keys are stored once per
        label row and shared by every candidate (README §17)."""
        q, k, v = self.qkv(x, x, rope, rope)
        own["k"] = k if own["k"] is None else torch.cat([own["k"], k], 2)
        own["v"] = v if own["v"] is None else torch.cat([own["v"], v], 2)
        pk, pv = shared
        N, R, P = q.shape[0], pk.shape[0], pk.shape[2]
        q1 = q[:, :, 0]
        s1 = torch.einsum("crhd,rhpd->crhp", q1.view(N // R, R, self.h, self.dh), pk).reshape(N, self.h, P)
        s2 = torch.einsum("nhd,nhtd->nht", q1, own["k"])
        p = torch.softmax(torch.cat([s1, s2], -1).float() * self.dh ** -0.5, -1).to(q.dtype)
        y = (torch.einsum("crhp,rhpd->crhd", p[..., :P].reshape(N // R, R, self.h, P), pv).reshape(N, self.h, self.dh)
             + torch.einsum("nht,nhtd->nhd", p[..., P:], own["v"]))
        return self.o(y.reshape(N, 1, -1))


class SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        h = int(8 * d / 3 + 63) // 64 * 64
        self.w1, self.w3, self.w2 = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d, heads, drop=0.0, cross=False):
        super().__init__()
        self.n1, self.att, self.n2, self.mlp = RMSNorm(d), Attention(d, heads), RMSNorm(d), SwiGLU(d)
        self.drop, self.cross = nn.Dropout(drop), cross
        if cross:                                            # starts closed: training opens it only if it helps
            self.nx, self.xatt, self.gate = RMSNorm(d), Attention(d, heads), nn.Parameter(torch.zeros(1))

    def forward(self, x, rope, mask=None, causal=False, mem=None, mem_mask=None, kv_out=None):
        x = x + self.drop(self.att(self.n1(x), rope, mask, causal, kv_out=kv_out))
        if self.cross and mem is not None:
            x = x + torch.tanh(self.gate) * self.drop(self.xatt(self.nx(x), None, mem_mask, mem=mem))
        return x + self.drop(self.mlp(self.n2(x)))

    def step(self, x, rope, shared, own):
        x = x + self.att.step(self.n1(x), rope, shared, own)
        return x + self.mlp(self.n2(x))


class Stack(nn.Module):
    def __init__(self, d, layers, drop=0.0, cross_every=0, ckpt=False):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, d // 64, drop, cross=bool(cross_every) and i % cross_every == cross_every - 1)
                                    for i in range(layers))
        self.norm, self.ckpt = RMSNorm(d), ckpt

    def forward(self, x, pos, mask=None, causal=False, mem=None, mem_mask=None, kv_out=None):
        rope = rope_tables(pos, 64)
        for b in self.blocks:
            if self.ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(b, x, rope, mask, causal, mem, mem_mask, use_reentrant=False)
            else:
                x = b(x, rope, mask, causal, mem, mem_mask, kv_out)
        return self.norm(x)


def token_loss(logits, tgt, w, kind, smooth=0.0, focal=0.0):
    """Weighted cross-entropy sum and per-class sums (README §19: rhythm usually gets easier than pitch).
    ``smooth`` label-smooths and ``focal`` rescales each token by (1-p)^focal, which spends the step on the
    tokens the model still gets wrong - the opposite of smoothing, for a corpus this predictable (README §12).
    Both change what is returned for the backward pass only: the reported "loss" stays the plain cross-entropy,
    so runs with different smoothing or focusing remain comparable (the grid ranks on it). The modulating
    factor is detached, so focusing reweights the plain gradient instead of adding a term of its own."""
    lg = logits.float().flatten(0, 1)
    nll = F.cross_entropy(lg, tgt.flatten(), reduction="none").view_as(tgt)
    back = (nll if not smooth else
            F.cross_entropy(lg, tgt.flatten(), reduction="none", label_smoothing=smooth).view_as(tgt))
    if focal:
        back = back * (1.0 - torch.exp(-nll.detach())).pow(focal)
    k = kind[tgt]
    st = {"loss": (nll * w).sum(), "w": w.sum(), "correct": ((logits.argmax(-1) == tgt).float() * w).sum()}
    for name, sel in (("rhythm", k == K_DUR), ("pitch", k == K_NOTE), ("technique", (k == K_FX) | (k == K_TREM))):
        st[f"nll_{name}"], st[f"w_{name}"] = (nll * w * sel).sum(), (w * sel).sum()
    return (back * w).sum(), {n: v.detach() for n, v in st.items()}


class FlatModel(nn.Module):
    """baseline: one decoder over the fill-in-the-middle window."""

    def __init__(self, cfg, V: Vocab):
        super().__init__()
        self.cfg, self.pad = cfg, V.pad
        self.emb = nn.Embedding(V.size, cfg["d"])
        self.body = Stack(cfg["d"], cfg["layers"], cfg.get("drop", 0.1), ckpt=cfg.get("ckpt", False))
        self.register_buffer("kind", torch.as_tensor(V.kind), persistent=False)

    def forward(self, ids):
        h = self.body(self.emb(ids), torch.arange(ids.shape[1], device=ids.device), causal=True)
        return h @ self.emb.weight.T

    def loss(self, b):
        return token_loss(self(b["ids"][:, :-1]), b["ids"][:, 1:], b["w"], self.kind,
                          self.cfg.get("smooth", 0.0), self.cfg.get("focal", 0.0))

    def prefill(self, ids):
        kv = []
        h = self.body(self.emb(ids), torch.arange(ids.shape[1], device=ids.device), causal=True, kv_out=kv)
        return h[:, -1] @ self.emb.weight.T, kv

    def step(self, ids, pos, shared, owns):
        x = self.emb(ids)
        rope = rope_tables(torch.tensor([pos], device=ids.device), 64)
        for blk, sh, own in zip(self.body.blocks, shared, owns):
            x = blk.step(x, rope, sh, own)
        return self.body.norm(x)[:, -1] @ self.emb.weight.T


class HierModel(nn.Module):
    """big / huge: bar encoder -> global model over bars -> local decoder (README §6)."""

    def __init__(self, cfg, V: Vocab):
        super().__init__()
        self.cfg, self.pad = cfg, V.pad
        de, dg, dl, drop, ck = cfg["d_enc"], cfg["d_glob"], cfg["d_loc"], cfg.get("drop", 0.1), cfg.get("ckpt", False)
        self.tok_e = nn.Embedding(V.size, de)
        self.enc = Stack(de, cfg["enc_layers"], drop, ckpt=ck)
        self.enc_out = nn.Linear(de, dg)
        self.g_tok, self.g_seg = nn.Embedding(V.size, dg), nn.Embedding(4, dg)
        self.g_phr = nn.Embedding(16, dg) if cfg.get("phrase") else None
        self.glob = Stack(dg, cfg["glob_layers"], drop, ckpt=ck)
        self.to_loc = nn.Linear(dg, dl)
        self.loc_e = nn.Embedding(V.size, dl)
        self.K = cfg.get("pointer", 0)
        self.loc = Stack(dl, cfg["loc_layers"], drop, cross_every=2 if self.K else 0, ckpt=ck)
        self.register_buffer("kind", torch.as_tensor(V.kind), persistent=False)
        if self.K:                                           # huge: pointer memory + planning heads
            M = cfg.get("mem_len", 160)
            self.ptr_q, self.ptr_k = nn.Linear(dg, dg), nn.Linear(dg, dg)
            self.mem_pos, self.mem_slot = nn.Embedding(M, dl), nn.Embedding(self.K, dl)
            self.mem_null, self.mem_norm = nn.Parameter(torch.zeros(1, 1, dl)), RMSNorm(dl)
            self.pc_head, self.on_head = nn.Linear(dg, 12), nn.Linear(dg, 16)
            self.sec_head = nn.Linear(dg, len(SECTIONS))

    def encode(self, chunks):
        outs, order = [], []
        for idx, ids in chunks:
            h = self.enc(self.tok_e(ids), torch.arange(ids.shape[1], device=ids.device), mask=(ids != self.pad)[:, None, None])
            outs.append(self.enc_out(h[:, 0]))
            order.append(idx)
        return torch.cat(outs)[torch.argsort(torch.cat(order))]

    def glob_states(self, idx, pos, seg, phr, mask, vecs):
        bank = torch.cat([self.g_tok.weight, vecs.to(self.g_tok.weight.dtype)], 0)
        x = bank[idx] + self.g_seg(seg)
        if self.g_phr is not None:
            x = x + self.g_phr(phr)
        return self.glob(x, pos, mask=mask)

    def memory(self, ids, valid):
        """Pointer memory: the guitar tokens of the K related bars, so far riffs can be copied note by note."""
        n, K, M = ids.shape
        x = self.loc_e(ids) + self.mem_pos.weight[:M][None, None] + self.mem_slot.weight[:K][None, :, None]
        m = ((ids != self.pad) & valid[..., None]).flatten(1)
        x = torch.cat([self.mem_null.expand(n, 1, -1).to(x.dtype), x.flatten(1, 2)], 1)
        m = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=m.device), m], 1)
        return self.mem_norm(x), m[:, None, None]

    def pointer(self, cond, vecs, cand, cmask, sim=None):
        q, k = self.ptr_q(cond).float(), self.ptr_k(vecs).float()
        sc = (q @ k.T / math.sqrt(q.shape[-1])).gather(1, cand).masked_fill(~cmask, -1e4)
        K = min(self.K, cand.shape[1])
        pick = sc.topk(K, -1).indices
        if sim is not None and self.training:              # half the time the true related bars (teacher forcing)
            true = sim.masked_fill(~cmask, -1).topk(K, -1).indices
            pick = torch.where(torch.rand(len(pick), 1, device=pick.device) < 0.5, true, pick)
        return sc, cand.gather(1, pick), cmask.gather(1, pick)

    def local(self, cond, inp, mem=None, mem_mask=None):
        x = torch.cat([self.to_loc(cond)[:, None].to(self.loc_e.weight.dtype), self.loc_e(inp[:, 1:])], 1)
        h = self.loc(x, torch.arange(inp.shape[1], device=inp.device), causal=True, mem=mem, mem_mask=mem_mask)
        return h @ self.loc_e.weight.T

    def loss(self, b):
        vecs = self.encode(b["enc"])
        g = self.glob_states(b["g_idx"], b["g_pos"], b["g_seg"], b["g_phr"], b["g_mask"], vecs)
        cond = g.flatten(0, 1)[b["cond"]]
        aux = {}
        if self.K:
            sc, sel, valid = self.pointer(cond, vecs, b["ptr_cand"], b["ptr_mask"], b["ptr_sim"])
            target = torch.softmax(b["ptr_sim"].masked_fill(~b["ptr_mask"], -1e4) / 0.1, -1)
            has = b["ptr_mask"].any(-1)
            aux["ptr"] = -(target * torch.log_softmax(sc, -1)).sum(-1)[has].mean() if has.any() else sc.sum() * 0
            aux["pc"] = F.binary_cross_entropy_with_logits(self.pc_head(cond).float(), b["plan_pc"])
            aux["onset"] = F.binary_cross_entropy_with_logits(self.on_head(cond).float(), b["plan_on"])
            secl = self.sec_head(g).float().flatten(0, 1)
            sec = b["sec"].flatten()
            aux["section"] = F.cross_entropy(secl, sec, ignore_index=-1) if (sec >= 0).any() else secl.sum() * 0
            mem_ids = b["mem_bank"][sel]
        total, stats = 0.0, {}
        for ti, inp, tgt, w in b["loc"]:
            mem = self.memory(mem_ids[ti], valid[ti]) if self.K else (None, None)
            l, st = token_loss(self.local(cond[ti], inp, *mem), tgt, w, self.kind,
                               self.cfg.get("smooth", 0.0), self.cfg.get("focal", 0.0))
            total = total + l
            for k, v in st.items():
                stats[k] = stats.get(k, 0) + v
        if aux:
            a = sum(aux.values())
            total = total + self.cfg.get("aux", 0.1) * a * stats["w"]
            stats.update({f"aux_{k}": v.detach() * stats["w"] for k, v in aux.items()})
        return total, stats


MODELS = {
    # flat decoder over a 4,096-token fill-in-the-middle window (~60M)
    "baseline": {"kind": "flat", "d": 640, "layers": 12, "ctx": 4096, "epochs": 4, "micro_tokens": 32768},
    # bar encoder 4x512 -> global 12x768 over up to 512 (train) / 1,024 (generate) bars -> local 6x512 (~120M)
    "big": {"kind": "hier", "d_enc": 512, "enc_layers": 4, "d_glob": 768, "glob_layers": 12, "d_loc": 512,
            "loc_layers": 6, "max_bars": 512, "epochs": 4, "micro_tokens": 65536, "ckpt": True},
    # big, wider, + pointer memory (3 related bars), planning heads, phrase positions (~280M)
    "huge": {"kind": "hier", "d_enc": 640, "enc_layers": 6, "d_glob": 1024, "glob_layers": 16, "d_loc": 640,
             "loc_layers": 8, "max_bars": 512, "pointer": 3, "mem_len": 160, "phrase": True, "epochs": 3,
             "micro_tokens": 49152, "ckpt": True},
}
TINY = {"d": 64, "layers": 2, "ctx": 1024, "d_enc": 64, "enc_layers": 1, "d_glob": 64, "glob_layers": 2,
        "d_loc": 64, "loc_layers": 2, "max_bars": 128, "micro_tokens": 8192, "ckpt": False}


def model_cfg(name: str, tiny: bool = False) -> dict:
    cfg = dict(MODELS[name]) | {"name": name}
    return cfg | TINY if tiny else cfg


def build_model(cfg, V) -> nn.Module:
    m = FlatModel(cfg, V) if cfg["kind"] == "flat" else HierModel(cfg, V)
    for n, p in m.named_parameters():
        if p.dim() >= 2:
            nn.init.normal_(p, std=0.02)
    return m


def n_params(m: nn.Module, embeddings: bool = False) -> int:
    return sum(p.numel() for n, p in m.named_parameters()
               if embeddings or not re.search(r"(^|\.)(emb|tok_e|g_tok|loc_e|g_seg|g_phr|mem_pos|mem_slot)\b", n))


# ============================================================================ 4) generation

class BarWriter:
    """The tokens allowed next in a bar the model writes (README §5, §16): the forced head (time signature,
    section, drums, bass of the gap bar), then every guitar track, each empty or filled to exactly the bar
    length, notes low string first with one note per string, then the end of the bar."""

    def __init__(self, V: Vocab, head, n_guitars, bar_len):
        self.V, self.queue, self.ng, self.L = V, list(head), n_guitars, bar_len
        self.g, self.rem, self.beat, self.beats, self.last_s, self.fx, self.trem_ok = -1, bar_len, False, 0, 8, None, False
        self.done, self.tup_open = False, 1

    def allowed(self) -> list[int]:
        V = self.V
        if self.done:
            return [V.pad]
        if self.queue:
            return [self.queue[0]]
        if self.g < 0:
            return [V.trk_g[0]] if self.ng else [V.eob]
        opts = []
        if self.rem >= V.min_dur:
            durs = [d for d in V.durs_asc[:bisect.bisect_right(V.ticks_asc, self.rem)] if not V.dotted_tup[d]
                    and (self.rem == V.dur[d] or self.rem - V.dur[d] >= V.min_dur)]   # never an unfillable rest
            if (self.L - self.rem) % (TICKS // 128):           # inside a tuplet group: finish it with its kind
                durs = [d for d in durs if V.tup[d] == self.tup_open]
            opts += durs
        if self.beat:
            if self.trem_ok:
                opts.append(V.trem)
            opts += V.notes[:self.last_s - 1].ravel().tolist()
            if self.fx is not None:
                opts += [V.fx[e] for e in EFFECTS if e not in self.fx]
        if self.rem < V.min_dur or self.beats == 0 or not opts:   # (no duration fits: end, never get stuck)
            opts.append(V.trk_g[self.g + 1] if self.g + 1 < self.ng else V.eob)
        return opts

    def push(self, t: int) -> None:
        V = self.V
        if self.queue:
            self.queue.pop(0)
            return
        k = V.kind[t]
        if t == V.eob:
            self.done = True
        elif k == K_TRK:
            self.g, self.rem, self.beat, self.beats, self.fx = self.g + 1, self.L, False, 0, None
        elif k == K_DUR:
            if (self.L - self.rem) % (TICKS // 128) == 0:
                self.tup_open = int(V.tup[t])
            self.rem -= int(V.dur[t])
            self.beat, self.beats, self.last_s, self.fx, self.trem_ok = True, self.beats + 1, 8, None, True
        elif t == V.trem:
            self.trem_ok = False
        elif k == K_NOTE:
            self.last_s, self.fx, self.trem_ok = int(V.n_sf[t, 0]), set(), False
        elif k == K_FX:
            self.fx.add(V.tokens[t][3:])


def cfg_combine(logits, w):
    """README §8: L_uncond + sum_i w_i (L_i - L_uncond); row 0 is unconditional. logits (cand, rows, V)."""
    if logits.shape[1] == 1:
        return logits[:, 0]
    base = logits[:, :1]
    return logits[:, 0] + ((logits[:, 1:] - base) * w[1:, None]).sum(1)


def choose(logits, writers, w, temp, top_p, gen, lp, nlp):
    """Sample one token per candidate under the bar rules; returns the tokens (cand,)."""
    comb = cfg_combine(logits.float().view(len(writers), -1, logits.shape[-1]), w)
    mask = torch.zeros_like(comb, dtype=torch.bool)
    for c, wr in enumerate(writers):
        mask[c, wr.allowed()] = True
    lg = torch.log_softmax((comb / max(temp, 1e-4)).masked_fill(~mask, -1e9), -1)
    probs = lg.exp()
    if top_p < 1:
        sp, si = probs.sort(-1, descending=True)
        drop = sp.cumsum(-1) - sp > top_p
        probs = probs.scatter(-1, si, sp.masked_fill(drop, 0))
    tok = torch.multinomial(probs.cpu() + 1e-12 * mask.cpu(), 1, generator=gen)[:, 0]
    for c, wr in enumerate(writers):
        if not wr.done and not wr.queue and len(wr.allowed()) > 1:
            lp[c] += float(lg[c, tok[c]])
            nlp[c] += 1
    return tok


@torch.no_grad()
def gen_flat(model, V, prompt, rows, w, plans, n_cand, temp, top_p, seed):
    dev = next(model.parameters()).device
    R = len(rows)
    P = torch.tensor(prompt, device=dev).repeat(R, 1)
    P[:, 1:5] = torch.tensor(rows, device=dev)
    last, shared = model.prefill(P)
    owns = [{"k": None, "v": None} for _ in shared]
    logits = last[None].expand(n_cand, R, -1).reshape(n_cand * R, -1)
    gen = torch.Generator().manual_seed(seed)
    writers = [BarWriter(V, *plans[0]) for _ in range(n_cand)]
    bar_no, out, cur = [0] * n_cand, [[] for _ in range(n_cand)], [[] for _ in range(n_cand)]
    lp, nlp, pos = [0.0] * n_cand, [0] * n_cand, len(prompt)
    for _ in range(len(plans) * (MAX_BAR + 4)):
        tok = choose(logits, writers, w, temp, top_p, gen, lp, nlp)
        for c, wr in enumerate(writers):
            if bar_no[c] >= len(plans):
                continue
            wr.push(int(tok[c]))
            cur[c].append(int(tok[c]))
            if wr.done:
                out[c].append(cur[c])
                cur[c], bar_no[c] = [], bar_no[c] + 1
                writers[c] = BarWriter(V, *plans[bar_no[c]]) if bar_no[c] < len(plans) else wr
        if all(b >= len(plans) for b in bar_no):
            break
        logits = model.step(tok.to(dev).repeat_interleave(R)[:, None], pos, shared, owns)
        pos += 1
    return [(out[c], lp[c] / max(nlp[c], 1)) for c in range(n_cand)]


@torch.no_grad()
def gen_hier(model, V, ex, rows, w, plans, n_cand, temp, top_p, seed):
    """Bars are written one by one; the global model runs again after each, so bar j sees bars 1..j-1."""
    dev = next(model.parameters()).device
    R, N = len(rows), n_cand * len(rows)
    roles, bars, (s, L) = ex["roles"], ex["bars"], ex["targets"][0]
    ctx = [i for i, r in enumerate(roles) if r == 0]
    vecs = model.encode([(i.to(dev), x.to(dev)) for i, x in enc_chunks([bars[i] for i in ctx], V)])
    Vn = V.size
    head_idx, head_pos, head_seg, head_phr, k = [], [], [], [], 0
    for i, r in enumerate(roles):
        if r == 2:
            continue
        head_idx.append(Vn + k if r == 0 else V.unk_bar)
        k += r == 0
        head_pos.append(i + 1)
        head_seg.append(1 if r == 0 else 2)
        head_phr.append(int(ex["phr"][i]))
    nctx = 4 + len(head_idx)
    mem_bank = None
    if model.K and ctx:
        mem_bank = pad_rows([(bars[i][(bars[i] & FLAG) > 0] & MASK)[:model.cfg.get("mem_len", 160)] for i in ctx],
                            V.pad).to(dev)
    gen = torch.Generator().manual_seed(seed)
    rows_t = torch.tensor(rows, device=dev)
    out, lp, nlp, gen_vecs = [[] for _ in range(n_cand)], [0.0] * n_cand, [0] * n_cand, []
    for j in range(L):
        mid_idx = [V.gap_start] + [Vn + len(ctx) + jj * n_cand for jj in range(j)]
        idx = torch.tensor(head_idx + mid_idx, device=dev)[None].repeat(N, 1)
        for jj in range(j):                                   # each candidate reads its own earlier bars
            idx[:, len(head_idx) + 1 + jj] += torch.arange(n_cand, device=dev).repeat_interleave(R)
        idx = torch.cat([rows_t.repeat(n_cand, 1), idx], 1)
        T = idx.shape[1]
        pos = torch.tensor([0] * 4 + head_pos + [s + jj + 1 for jj in range(j + 1)], device=dev)
        seg = torch.tensor([0] * 4 + head_seg + [3] * (j + 1), device=dev)
        phr = torch.tensor([0] * 4 + head_phr + [int(ex["phr"][s + jj]) for jj in range(j + 1)], device=dev)
        ar = torch.arange(T, device=dev)
        mask = (ar[None, :] < nctx) | ((ar[:, None] >= nctx) & (ar[None, :] <= ar[:, None]))
        allv = torch.cat([vecs] + gen_vecs) if gen_vecs else vecs
        g = model.glob_states(idx, pos[None].expand(N, T), seg[None].expand(N, T), phr[None].expand(N, T),
                              mask[None, None], allv)[:, -1]
        mem = (None, None)
        if mem_bank is not None:
            cand = torch.arange(len(ctx), device=dev)[None].expand(N, -1)
            _, sel, valid = model.pointer(g, vecs, cand, torch.ones_like(cand, dtype=torch.bool))
            mem = model.memory(mem_bank[sel], valid)
        writers = [BarWriter(V, *plans[j]) for _ in range(n_cand)]
        seq = torch.full((N, 1), V.pad, device=dev)
        toks = [[] for _ in range(n_cand)]
        for _ in range(MAX_BAR + 4):
            logits = model.local(g, seq, *mem)[:, -1]
            tok = choose(logits, writers, w, temp, top_p, gen, lp, nlp)
            for c, wr in enumerate(writers):
                if not wr.done:
                    wr.push(int(tok[c]))
                    toks[c].append(int(tok[c]))
            if all(wr.done for wr in writers):
                break
            seq = torch.cat([seq, tok.to(dev).repeat_interleave(R)[:, None]], 1)
        new = [flag_bar(t, V) for t in toks]
        for c in range(n_cand):
            out[c].append(toks[c])
        gen_vecs.append(model.encode([(i.to(dev), x.to(dev)) for i, x in enc_chunks(new, V)]))
    return [(out[c], lp[c] / max(nlp[c], 1)) for c in range(n_cand)]


def flag_bar(toks, V) -> np.ndarray:
    """Generated ids -> a stored bar (flags on the guitar part, which starts at the first guitar track)."""
    a = np.array(toks, np.uint16)
    first = next((i for i, t in enumerate(toks) if t == V.trk_g[0] or t == V.eob), len(toks))
    a[first:] |= FLAG
    return a


def gap_plans(V, bars, section=None):
    """Per gap bar: the forced head (everything before the guitar part), the number of guitars, the length."""
    plans = []
    for j, bar in enumerate(bars):
        ids = (bar & MASK).tolist()
        n_head = int(np.argmax((bar & FLAG) > 0)) if ((bar & FLAG) > 0).any() else len(ids)
        head = ids[:n_head]
        if section and j == 0:
            head = [head[0], V.id[f"sec:{section}"]] + [t for t in head[1:] if V.kind[t] != K_SEC]
        plans.append((head, sum(t in V.trk_g for t in ids), int(V.ts_len[ids[0]])))
    return plans


def fill_gap(model, V, cfg, view, target, holes, rows, w, n_cand=8, temp=1.0, top_p=0.95, seed=0, section=None):
    """Candidates for one gap of a song: [(bars, mean log-probability)]."""
    s, L = target
    plans = gap_plans(V, view.bars(s, s + L), section)
    rng = rng_for(seed)
    dev = next(model.parameters()).device
    wt = torch.tensor(w, dtype=torch.float32, device=dev)
    if cfg["kind"] == "flat":
        ids, _, n_prompt = flat_example(V, view, rng, cfg["ctx"], target, holes, train=False, slots=rows[0])
        res = gen_flat(model, V, ids[:n_prompt].tolist(), rows, wt, plans, n_cand, temp, top_p, seed)
    else:
        ex = hier_example(V, view, rng, cfg.get("max_bars_infer", 2 * cfg["max_bars"]), [target], holes,
                          train=False, slots=rows[0])
        res = gen_hier(model, V, ex, rows, wt, plans, n_cand, temp, top_p, seed)
    return [([flag_bar(b, V) for b in bars], score) for bars, score in res]


# ============================================================================ 2) training

def song_weight(labels: dict, focus: dict) -> float:
    w = 1.0
    for k, v in labels.items():
        w *= focus.get(f"{k}:{v}", 1.0)
    return w


def epoch_order(c: Corpus, cfg, rng, focus) -> list[int]:
    """One epoch: every training song as often as it fills windows (README §11-12: --focus reweights labels)."""
    ids = []
    for i, s in enumerate(c.songs):
        if s["split"] != "train":
            continue
        n = s["tokens"] / cfg["ctx"] if cfg["kind"] == "flat" else s["nb"] / cfg["max_bars"]
        r = max(n, 1.0) * song_weight(s["labels"], focus)
        ids += [i] * (int(r) + (rng.random() < r - int(r)))
    return rng.permutation(ids).tolist()


def epoch_tokens(c: Corpus, focus) -> float:
    return sum(s["tokens"] * song_weight(s["labels"], focus) for s in c.songs if s["split"] == "train")


def make_example(c: Corpus, sid, cfg, rng):
    view = c.view(sid)
    if cfg["kind"] == "flat":
        return flat_example(c.vocab, view, rng, cfg["ctx"])
    return hier_example(c.vocab, view, rng, cfg["max_bars"])


def collate(exs, V, cfg):
    return collate_flat(exs, V) if cfg["kind"] == "flat" else collate_hier(exs, V, cfg)


def ex_tokens(ex) -> int:
    return len(ex[0]) if isinstance(ex, tuple) else ex["ntok"]


class Stream(torch.utils.data.IterableDataset):
    """Micro-batches built on the fly in the loader workers (README §10, §18): new gaps and augmentations every
    epoch, every example from its own seed (seed, epoch, position), so the data does not depend on the workers."""

    def __init__(self, data, cfg, micro, seed, epoch, focus):
        self.data, self.cfg, self.micro, self.seed, self.epoch, self.focus = data, cfg, micro, seed, epoch, focus

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        w, nw = (info.id, info.num_workers) if info else (0, 1)
        c, epoch = Corpus(self.data), self.epoch
        while True:
            order = epoch_order(c, self.cfg, rng_for(self.seed, epoch), self.focus)
            buf, n = [], 0
            for k in range(w, len(order), nw):
                ex = make_example(c, order[k], self.cfg, rng_for(self.seed, epoch, k))
                if ex is None:
                    continue
                buf.append(ex)
                n += ex_tokens(ex)
                if n >= self.micro:
                    yield {"batch": collate(buf, c.vocab, self.cfg), "epoch": epoch}
                    buf, n = [], 0
            epoch += 1


def to_dev(x, dev):
    if torch.is_tensor(x):
        return x.to(dev, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_dev(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_dev(v, dev) for v in x)
    return x


def autocast(dev):
    if dev.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(dev.type, enabled=False)


class LoRA(nn.Module):
    """README §13: a frozen layer plus a small trainable update B·A (B starts at zero)."""

    def __init__(self, base: nn.Linear, r: int):
        super().__init__()
        self.base, self.scale = base, 2.0
        self.a = nn.Parameter(torch.randn(r, base.in_features) / r)
        self.b = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.a.T.to(x.dtype)) @ self.b.T.to(x.dtype) * self.scale


def apply_lora(model: nn.Module, r: int) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and name in ("q", "kv", "o", "w1", "w2", "w3"):
                setattr(mod, name, LoRA(child, r))
    for n, p in model.named_parameters():                    # label (album) embeddings learn too
        if re.search(r"(^|\.)(emb|tok_e|g_tok|loc_e)\.weight$", n):
            p.requires_grad_(True)


def lr_at(frac, peak, warm=0.02, floor=0.1):
    if frac < warm:
        return peak * frac / warm
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, (frac - warm) / (1 - warm)))))


def save_ckpt(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    V = Vocab(ck["tokens"])
    model = build_model(ck["cfg"], V)
    if ck.get("lora"):
        apply_lora(model, ck["lora"])
    model.load_state_dict(ck["model"])
    return model.to(dev).eval(), V, ck["cfg"], ck


def train_run(cfg, data, run, lr, batch_tokens, epochs, eval_every=0.5, budget=None, resume=False, init=None,
              lora=0, focus=None, workers=4, seed=1234, eval_gaps=400, gen_gaps=8, dev=None, keep_ckpt=True) -> dict:
    """Train one model; evaluate every ``eval_every`` epochs (and at the end); keep best.pt and last.pt.
    ``keep_ckpt=False`` only reports the losses (the grid search needs no weights, and writing last.pt with the
    optimiser state at every evaluation would cost more than the short run itself)."""
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    c, focus = Corpus(data), focus or {}
    V = c.vocab
    torch.manual_seed(seed)
    model = build_model(cfg, V)
    start = {"seen": 0, "step": 0, "best": float("inf"), "epoch": 0}
    if init:
        ck = torch.load(init, map_location="cpu", weights_only=False)
        if ck["tokens"] != V.tokens:
            sys.exit("--init checkpoint was trained on another vocabulary (preprocess and pretrain again)")
        model.load_state_dict(ck["model"])
    if lora:
        apply_lora(model, lora)
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.1},
              {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=dev.type == "cuda")
    if resume and (run / "last.pt").is_file():
        ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["state"]
        log(f"resumed at {start['seen'] / 1e6:.1f}M tokens")
    ep_tok = epoch_tokens(c, focus)
    total = budget or epochs * ep_tok
    every = eval_every * ep_tok if eval_every else total
    # the accumulation loop runs micro-batches until it reaches batch_tokens, so a micro-batch bigger than the step
    # would silently train at the micro size instead: split the step into equal micro-batches that fit the model's
    micro = max(4096, -(-batch_tokens // max(1, math.ceil(batch_tokens / cfg["micro_tokens"]))))
    info = {"cfg": cfg, "lr": lr, "batch_tokens": batch_tokens, "epochs": epochs, "budget": budget, "seed": seed,
            "params": n_params(model), "params_total": n_params(model, True), "epoch_tokens": ep_tok, "lora": lora,
            "focus": focus, "data": str(data), "micro_tokens": micro}
    (run / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log(f"{cfg['name']}: {info['params'] / 1e6:.1f}M parameters ({info['params_total'] / 1e6:.1f}M with embeddings), "
        f"lr {lr:.2e}, {batch_tokens} tokens/step ({-(-batch_tokens // micro)} x {micro}), "
        f"{total / 1e6:.0f}M tokens on {dev}")
    loader = torch.utils.data.DataLoader(
        Stream(data, cfg, micro, seed, start["epoch"], focus), batch_size=None, num_workers=workers,
        persistent_workers=workers > 0, prefetch_factor=4 if workers else None, pin_memory=dev.type == "cuda")
    it = iter(loader)
    st, logf = dict(start), open(run / "metrics.jsonl", "a", encoding="utf-8")
    next_eval = (st["seen"] // every + 1) * every
    acc, t0, last = {}, time.time(), {}
    model.train()
    while st["seen"] < total:
        got, wsum = 0, 0.0
        while got < batch_tokens:
            item = next(it)
            st["epoch"] = item["epoch"]
            b = to_dev(item["batch"], dev)
            with autocast(dev):
                loss, stats = model.loss(b)
            loss.backward()
            got += b["ntok"]
            wsum += float(stats["w"])
            for k, v in stats.items():
                acc[k] = acc.get(k, 0.0) + float(v)
        for p in params:
            if p.grad is not None:
                p.grad.div_(max(wsum, 1.0))                # README pitfalls: normalised by the step's target tokens
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        cur = lr_at(st["seen"] / total, lr)
        for g in opt.param_groups:
            g["lr"] = cur
        if math.isfinite(gn):
            opt.step()
        opt.zero_grad(set_to_none=True)
        st["seen"] += got
        st["step"] += 1
        if st["step"] % 20 == 0 or st["seen"] >= total:
            row = {"step": st["step"], "tokens": st["seen"], "epoch": st["seen"] / ep_tok, "lr": cur, "grad_norm": gn,
                   "tok_s": got * 20 / max(time.time() - t0, 1e-6)} | summarise(acc)
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            log(f"step {st['step']} ep {row['epoch']:.2f} loss {row.get('loss', 0):.3f} acc {row.get('acc', 0):.3f} "
                f"lr {cur:.2e} {row['tok_s'] / 1e3:.0f}k tok/s")
            acc, t0 = {}, time.time()
        if st["seen"] >= next_eval or st["seen"] >= total:
            next_eval += every
            last = evaluate(model, c, cfg, "val", eval_gaps, gen_gaps, out=run / f"samples_{st['step']:06d}.txt")
            logf.write(json.dumps({"step": st["step"], "tokens": st["seen"], "eval": last}) + "\n")
            logf.flush()
            log("eval " + " ".join(f"{k} {v:.4f}" for k, v in last.items()))
            best = last["loss"] < st["best"]
            st["best"] = min(st["best"], last["loss"])
            if best:
                st["best_eval"] = last                   # the whole evaluation at the best step, not just its loss
            if keep_ckpt:
                ck = {"cfg": cfg, "tokens": V.tokens, "model": model.state_dict(), "lora": lora, "state": st}
                if best:
                    save_ckpt(run / "best.pt", ck)
                save_ckpt(run / "last.pt", ck | {"opt": opt.state_dict()})
            model.train()
    logf.close()
    return {"best": st["best"], "last": last, "best_eval": st.get("best_eval", last), "tokens": st["seen"],
            "tokens_per_step": st["seen"] / max(st["step"], 1), "micro_tokens": micro}


def summarise(acc: dict) -> dict:
    w = max(acc.get("w", 0.0), 1e-9)
    out = {"loss": acc.get("loss", 0.0) / w, "acc": acc.get("correct", 0.0) / w}
    for name in ("rhythm", "pitch", "technique"):
        if acc.get(f"w_{name}"):
            out[f"nll_{name}"] = acc[f"nll_{name}"] / acc[f"w_{name}"]
    for k, v in acc.items():
        if k.startswith("aux_"):
            out[k] = v / w
    return out


# ============================================================================ 3) evaluation

def bar_feats(bar, V):
    """Pitch classes, onset grid and note count of a bar's guitar part."""
    ids, L = bar & MASK, int(V.ts_len[int(bar[0]) & MASK])
    pc = on = n = pos = start = 0
    guitar = False
    for t, fl in zip(ids.tolist(), ((bar & FLAG) > 0).tolist()):
        k = V.kind[t]
        if k == K_TRK:
            guitar, pos = fl, 0
        elif k == K_DUR:
            start, pos = pos, pos + int(V.dur[t])
        elif k == K_NOTE and guitar:
            s, f = V.n_sf[t]
            pc |= 1 << ((DROP_A[s - 1] + f) % 12)
            on |= 1 << min(15, start * 16 // max(L, 1))
            n += 1
    return pc, on, n


CMP_CTX = 4096                  # the flat window: the comparable subset is defined against it and never against
                                # the model being scored, so every architecture is scored on the same tokens


def gap_repeats(view: SongView, s: int, L: int) -> bool:
    """Does any bar of the gap occur again outside it? A model that finds such a bar in its context has not
    written anything, and the flat window carries those bars as plain tokens while the hierarchical ones see
    only one vector per bar, so these gaps measure the architecture's reach rather than its writing."""
    outside = {int(h) for i, h in enumerate(view.f["hash"]) if view.f["gn"][i] and not s <= i < s + L}
    return any(int(view.f["hash"][i]) in outside for i in range(s, s + L) if view.f["gn"][i])


def comparable_gaps(c: Corpus, split="val") -> list[tuple[int, dict]]:
    """The fixed gaps every architecture can be ranked on (README §19): no bar of the gap occurs anywhere else
    in the song, so none of them can be copied from the context, and the gap is short enough that
    ``flat_example`` does not have to shorten it - on these the flat and the hierarchical builders weight an
    identical set of target tokens. Cached on the corpus: the set is a property of the corpus, not of a run."""
    key = f"cmp_{split}"
    if key not in c.cache:
        out = []
        for j, g in enumerate(g for g in c.gaps if g["split"] == split):
            view, s, L = c.view(g["song"]), g["start"], g["len"]
            if view.lens[s:s + L].sum() <= CMP_CTX // 2 and not gap_repeats(view, s, L):
                out.append((j, g))
        c.cache[key] = out
    return c.cache[key]


@torch.no_grad()
def gap_loss(model, c: Corpus, cfg, gaps, dev) -> dict:
    """Summed loss statistics over the given fixed gaps. The context has weight 0 at evaluation, so these
    cover the gaps' guitar tokens alone. ``gaps`` is a list of (index, gap); the index only labels the gap,
    because with the target given and ``train=False`` neither example builder draws from its generator."""
    V, acc, buf, n = c.vocab, {}, [], 0
    for item in list(gaps) + [None]:
        if item is not None:
            j, g = item
            view, rng = c.view(g["song"]), rng_for(0, j)
            tgt = (g["start"], g["len"])
            ex = (flat_example(V, view, rng, cfg["ctx"], tgt, train=False) if cfg["kind"] == "flat" else
                  hier_example(V, view, rng, cfg.get("max_bars_infer", 2 * cfg["max_bars"]), [tgt], train=False))
            if ex is not None:
                buf.append(ex)
                n += ex_tokens(ex)
        if buf and (n >= cfg["micro_tokens"] or item is None):
            with autocast(dev):
                _, st = model.loss(to_dev(collate(buf, V, cfg), dev))
            for k, v in st.items():
                acc[k] = acc.get(k, 0.0) + float(v)
            buf, n = [], 0
    return acc


class NgramModel(nn.Module):
    """The model-free reference for the gap loss: an add-alpha bigram over the corpus's own bar tokens, scored
    through the same example builder, the same gaps and the same weights as a trained model, so "nats below
    the bigram" is one scale every architecture can be placed on, whatever it is allowed to see."""

    def __init__(self, logp: np.ndarray, V: Vocab, micro_tokens=16384):
        super().__init__()
        self.cfg = {"kind": "flat", "ctx": CMP_CTX, "micro_tokens": micro_tokens, "name": "bigram"}
        self.register_buffer("logp", torch.as_tensor(logp), persistent=False)
        self.register_buffer("kind", torch.as_tensor(V.kind), persistent=False)

    def loss(self, b):
        return token_loss(self.logp[b["ids"][:, :-1]], b["ids"][:, 1:], b["w"], self.kind)


def fit_bigram(c: Corpus, alpha=0.5, chunk=2000) -> np.ndarray:
    """Bigram counts over the training songs' bar tokens, add-alpha smoothed, as log-probabilities. Pairs are
    taken within one song at a time, so no count crosses a song boundary."""
    n = c.vocab.size
    cnt, buf = np.zeros(n * n, np.int64), []
    for sid, s in enumerate(c.songs):
        if s["split"] == "train":
            t = (np.asarray(c.view(sid).tok) & MASK).astype(np.int64)
            if len(t) > 1:
                buf.append(t[:-1] * n + t[1:])
        if len(buf) >= chunk or (sid == len(c.songs) - 1 and buf):
            cnt += np.bincount(np.concatenate(buf), minlength=n * n)
            buf = []
    p = cnt.reshape(n, n).astype(np.float64) + alpha
    return np.log(p / p.sum(1, keepdims=True)).astype(np.float32)


def ngram_ref(c: Corpus, split="val", build=False) -> dict | None:
    """The bigram's losses on the same gaps, computed once per corpus and cached beside it. Returns None when
    it has not been built yet (``riff.py ngram``), so a run without it simply reports no lift."""
    path = c.dir / f"ngram_{split}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if not build:
        return None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"fitting the bigram on the {c.stats['train_tokens'] / 1e6:.0f}M training tokens")
    model = NgramModel(fit_bigram(c), c.vocab).to(dev).eval()
    gaps = [g for g in c.gaps if g["split"] == split]
    cmp = comparable_gaps(c, split)
    ref = {"loss": summarise(gap_loss(model, c, model.cfg, list(enumerate(gaps)), dev))["loss"],
           "loss_cmp": summarise(gap_loss(model, c, model.cfg, cmp, dev))["loss"],
           "gaps": len(gaps), "gaps_cmp": len(cmp), "alpha": 0.5}
    path.write_text(json.dumps(ref, indent=1), encoding="utf-8")
    return ref


@torch.no_grad()
def evaluate(model, c: Corpus, cfg, split="val", max_gaps=400, gen_gaps=8, out=None) -> dict:
    """README §19: loss on the fixed gaps (all, rhythm, pitch, technique), the same loss on the gaps that
    nothing in the song repeats (``loss_cmp``, comparable across architectures) and how far that is below the
    corpus's own bigram (``lift_cmp``), and for a few gaps, written candidates: pitch-class and rhythm
    similarity to the real bars, copies of a context bar, note density."""
    model.eval()
    dev = next(model.parameters()).device
    V, gaps = c.vocab, [g for g in c.gaps if g["split"] == split][:max_gaps]
    res = summarise(gap_loss(model, c, cfg, list(enumerate(gaps)), dev))
    cmp = comparable_gaps(c, split)
    if cmp and not isinstance(model, NgramModel):
        cs = summarise(gap_loss(model, c, cfg, cmp, dev))
        res |= {"loss_cmp": cs["loss"], "acc_cmp": cs["acc"]}
        ref = ngram_ref(c, split)
        if ref:
            res["lift_cmp"] = ref["loss_cmp"] - cs["loss"]
    sims, lines = [], []
    for g in gaps[:gen_gaps]:
        view = c.view(g["song"])
        s, L = g["start"], g["len"]
        truth = view.bars(s, s + L)
        ctx_h = {int(h) for i, h in enumerate(view.f["hash"]) if view.f["gn"][i] and not s <= i < s + L}
        with autocast(dev):
            cands = fill_gap(model, V, cfg, view, (s, L), (), [V.slots(view.labels)], [1.0], n_cand=2, seed=g["song"])
        for bars, score in cands:
            tf, gf = [bar_feats(b, V) for b in truth], [bar_feats(b, V) for b in bars]
            sims.append({"pitch_sim": float(np.mean([jaccard(a[0], b[0]) for a, b in zip(tf, gf)])),
                         "rhythm_sim": float(np.mean([jaccard(a[1], b[1]) for a, b in zip(tf, gf)])),
                         "copy": float(any(ghash(b) in ctx_h and f[2] for b, f in zip(bars, gf))),
                         "density": sum(f[2] for f in gf) / max(sum(f[2] for f in tf), 1), "logp": score})
            lines.append(f"# {c.songs[g['song']]['file']} bars {s + 1}-{s + L} logp {score:.3f}\n" +
                         "\n".join(" ".join(V.tokens[t] for t in (b & MASK).tolist()) for b in bars))
    if sims:
        res |= {k: float(np.mean([x[k] for x in sims])) for k in sims[0]}
    if out and lines:
        Path(out).write_text("\n\n".join(lines), encoding="utf-8")
    model.train()
    return res


# ============================================================================ batch sizes and learning rates

def flops_per_token(cfg, stats) -> float:
    """Non-embedding training FLOPs per token, counted like DeepSeek LLM (2024): 72·layers·d² + 6·layers·d·context."""
    Lb = stats["tokens_per_bar"]
    if cfg["kind"] == "flat":
        return 72 * cfg["layers"] * cfg["d"] ** 2 + 6 * cfg["layers"] * cfg["d"] * cfg["ctx"]
    nb = min(stats["bars_per_song"], cfg["max_bars"])
    enc = 72 * cfg["enc_layers"] * cfg["d_enc"] ** 2 + 6 * cfg["enc_layers"] * cfg["d_enc"] * Lb
    loc = 72 * cfg["loc_layers"] * cfg["d_loc"] ** 2 + 6 * cfg["loc_layers"] * cfg["d_loc"] * Lb
    if cfg.get("pointer"):
        loc += cfg["loc_layers"] // 2 * (24 * cfg["d_loc"] ** 2 + 6 * cfg["d_loc"] * cfg["pointer"] * cfg["mem_len"])
    glob = (72 * cfg["glob_layers"] * cfg["d_glob"] ** 2 + 6 * cfg["glob_layers"] * cfg["d_glob"] * nb) / Lb
    return enc + 0.3 * loc + glob                           # the local decoder writes ~30 % of the bars


def suggest(cfg, stats, epochs=None) -> dict:
    """Starting values from the DeepSeek LLM fit: lr = 0.3118·C^-0.125, batch = 0.2920·C^0.3271 tokens."""
    M = flops_per_token(cfg, stats)
    D = (epochs or cfg["epochs"]) * stats["train_tokens"]
    C = M * D
    lr = min(max(0.3118 * C ** -0.125, 1e-4), 3e-3)
    B = int(min(max(0.2920 * C ** 0.3271, 16384), 2 ** 21)) // 4096 * 4096
    return {"flops_per_token": M, "tokens": D, "compute": C, "lr": lr, "batch_tokens": B,
            "hours_4090": C / 6.6e13 / 3600}                  # ~40 % of the 4090's 165 bf16 TFLOPS


def grid_points(h, bms=(0.5, 1, 2), lms=(2 ** -0.5, 1, 2 ** 0.5)) -> list[tuple[int, float]]:
    """Batch sizes (multiples of the fit) × learning rates (multiples of the fit, itself scaled with √batch as
    Adam needs). DeepSeek's near-optimal band is wide; this brackets it without bf16 blow-ups."""
    out = []
    for bm in bms:
        B = max(int(h["batch_tokens"] * bm) // 4096 * 4096, 4096)
        for lm in lms:
            out.append((B, h["lr"] * math.sqrt(bm) * lm))
    return out


def grid_variants(cfg, aux: str, smooth: str, focal: str = "") -> list[dict]:
    """The loss axis of this model: the aux-head weight where there are aux heads, otherwise the focusing
    exponent if one is given, otherwise label smoothing."""
    key, pick = ("aux", aux) if cfg.get("pointer") else ("focal", focal) if focal.strip() else ("smooth", smooth)
    return [{key: float(x)} for x in pick.split(",") if x.strip()] or [{}]


# ============================================================================ 6) segmentation (sections)

SEG_NF = 12 + 16 + 12
TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)          # border thresholds tried on the validation songs


def seg_features(f) -> np.ndarray:
    """Per-bar inputs besides the tokens: pitch classes, onset grid, density, low-string share, position in
    the song, and repetition: how often the bar recurs, whether it is its first time, where it first came,
    whether it equals the bar 1/4/8 back, and its pitch/rhythm similarity to the previous bar."""
    pc, on, gn, low, h = (np.asarray(f[k]).astype(np.int64) for k in ("pc", "on", "gn", "low", "hash"))
    nb, idx = len(pc), np.arange(len(pc))
    _, inv, counts = np.unique(h, return_inverse=True, return_counts=True)
    first = np.full(len(counts), nb)
    np.minimum.at(first, inv, idx)
    busy = gn > 0

    def same(k):
        return np.concatenate([np.zeros(min(k, nb)), (h[k:] == h[:-k]) & busy[k:]]) if nb > k else np.zeros(nb)

    prev_pc, prev_on = np.concatenate([[0], pc[:-1]]), np.concatenate([[0], on[:-1]])
    sc = np.stack([np.log1p(gn) / 3, low / 100, idx / max(nb - 1, 1), np.full(nb, math.log(nb) / 7),
                   np.log(np.where(busy, counts[inv], 1)) / 3, (first[inv] == idx) & busy, first[inv] / max(nb - 1, 1),
                   same(1), same(4), same(8), jaccard(pc, prev_pc), jaccard(on, prev_on)], 1)
    return np.concatenate([(pc[:, None] >> np.arange(12)) & 1, (on[:, None] >> np.arange(16)) & 1, sc], 1).astype(np.float32)


def seg_targets(sec) -> tuple[np.ndarray, np.ndarray]:
    """The section in effect per bar (-1 before the first marker) and where a marker starts a section."""
    sec = np.asarray(sec, np.int64)
    eff = np.maximum.accumulate(np.where(sec >= 0, np.arange(len(sec)), -1))
    return np.where(eff >= 0, sec[np.maximum(eff, 0)], -1), (sec >= 0).astype(np.float32)


OTHER = SECTIONS.index("other")


def seg_ssm(f) -> np.ndarray:
    """Bar self-similarity: the same guitar part = 1, else the mean of pitch-class and onset Jaccard."""
    pc, on, h = (np.asarray(f[k]).astype(np.int64) for k in ("pc", "on", "hash"))
    S = 0.5 * (jaccard(pc[:, None], pc[None, :]) + jaccard(on[:, None], on[None, :]))
    return np.where(h[:, None] == h[None, :], 1.0, S).astype(np.float32)


def seg_item(V: Vocab, view: SongView, lo=0, hi=None, rng=None, named=False, pseudo=None) -> dict:
    """One song (window) for the segmenter. With ``rng``: drums/bass dropped, transposed (pitch classes rotated
    with it). ``named``: 'other' bars are unlabeled (their markers still count as borders)."""
    hi = view.nb if hi is None else hi
    bars, shift = view.bars(lo, hi), 0
    if rng is not None:
        info = {}
        bars = augment(bars, V, rng, guitar_drop=0.0, info=info)
        shift = info.get("shift", 0) % 12
    tok = np.concatenate(bars).astype(np.int64)
    ids, g = tok & MASK, (tok & FLAG) > 0
    bar_of = np.repeat(np.arange(hi - lo), [len(b) for b in bars])
    keep = V.kind[ids] != K_SEC                              # the markers are the answer: never an input
    f = {k: np.asarray(v)[lo:hi] for k, v in view.f.items()}
    if shift:
        pc = f["pc"].astype(np.int64)
        f["pc"] = ((pc << shift) | (pc >> (12 - shift))) & 0xFFF
    sec, bnd = seg_targets(f["sec"]) if pseudo is None else (pseudo[0][lo:hi], pseudo[1][lo:hi])
    if named:
        sec = np.where(sec == OTHER, -1, sec)
    return {"tok": (ids + V.size * g)[keep], "cnt": np.bincount(bar_of[keep], minlength=hi - lo),
            "feats": seg_features(f), "sec": sec, "bnd": bnd, "ssm": seg_ssm(f)}


def seg_collate(items) -> dict:
    B, T = len(items), max(len(x["sec"]) for x in items)
    where = np.concatenate([b * T + np.arange(len(x["sec"])) for b, x in enumerate(items)])
    cnt = np.concatenate([x["cnt"] for x in items])
    out = {"tok": torch.as_tensor(np.concatenate([x["tok"] for x in items])),
           "off": torch.as_tensor(np.concatenate([[0], np.cumsum(cnt)[:-1]])), "where": torch.as_tensor(where),
           "feats": torch.zeros(B, T, SEG_NF), "mask": torch.zeros(B, T, dtype=torch.bool),
           "sec": torch.full((B, T), -1, dtype=torch.long), "bnd": torch.zeros(B, T), "ssm": torch.zeros(B, T, T)}
    for b, x in enumerate(items):
        n = len(x["sec"])
        out["feats"][b, :n] = torch.as_tensor(x["feats"])
        out["mask"][b, :n] = True
        out["sec"][b, :n] = torch.as_tensor(x["sec"])
        out["bnd"][b, :n] = torch.as_tensor(x["bnd"])
        out["ssm"][b, :n, :n] = torch.as_tensor(x["ssm"])
    return out


class SegModel(nn.Module):
    """Bars -> section of every bar + where sections start. A bar is a bag of its tokens (guitar tokens and
    context tokens kept apart) plus its features; a bidirectional transformer over the bars (rotary positions =
    bar index) sees the whole song, so repetition and position can tell verse from chorus. With ``ssm`` the
    bar self-similarity is added to every attention score (a learned weight per head): a chorus bar looks at
    the other chorus bars from the first step."""

    def __init__(self, V: Vocab, d=256, layers=6, drop=0.1, ssm=False, named=False, active=None):
        super().__init__()
        self.cfg = {"d": d, "layers": layers, "ssm": ssm, "named": named,
                    "active": list(active) if active is not None else list(range(len(SECTIONS)))}
        self.bag = nn.EmbeddingBag(2 * V.size, d, mode="mean")
        self.fin = nn.Sequential(nn.Linear(SEG_NF, d), nn.SiLU(), nn.Linear(d, d))
        self.body = Stack(d, layers, drop)
        self.sec, self.bnd = nn.Linear(d, len(SECTIONS)), nn.Linear(d, 1)
        self.ssm_w = nn.Parameter(torch.ones(d // 64)) if ssm else None

    def forward(self, b):
        B, T = b["mask"].shape
        e = self.bag(b["tok"], b["off"])
        x = torch.zeros(B * T, e.shape[1], device=e.device, dtype=e.dtype).index_copy(0, b["where"], e).view(B, T, -1)
        mask = b["mask"][:, None, None]
        if self.ssm_w is not None:
            mask = (self.ssm_w.view(1, -1, 1, 1) * b["ssm"][:, None]).masked_fill(~mask, float("-inf"))
        h = self.body(x + self.fin(b["feats"]), torch.arange(T, device=e.device), mask=mask)
        ls = self.sec(h)
        off = torch.ones(len(SECTIONS), dtype=torch.bool, device=ls.device)   # sections the labels never use
        off[self.cfg.get("active", list(range(len(SECTIONS))))] = False
        if self.cfg.get("named"):
            off[OTHER] = True
        ls = ls.masked_fill(off, -1e4)
        return ls, self.bnd(h)[..., 0]


def load_seg(path, V, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    model = SegModel(V, c["d"], c["layers"], ssm=c.get("ssm", False), named=c.get("named", False),
                     active=c.get("active")).to(dev)
    model.load_state_dict(ck["model"])
    return model, ck


class SegSet:
    """Songs with >= 2 section markers, split by song family (75 % train, 5 % val, 20 % test), and the songs
    without markers (the ones to fill)."""

    def __init__(self, data):
        self.c = Corpus(data)
        sec = self.c._open()[2]["sec"]
        self.ids, self.unmarked, self.pseudo, self.named = {"train": [], "val": [], "test": []}, [], {}, False
        for i, s in enumerate(self.c.songs):
            n = int((np.asarray(sec[s["bar0"]:s["bar0"] + s["nb"]]) >= 0).sum())
            if n < 2:
                self.unmarked.append(i)
            elif s["nb"] >= 8:
                h = int(hashlib.md5(s["family"].encode()).hexdigest()[:8], 16) % 100
                self.ids["test" if h < 20 else "val" if h < 25 else "train"].append(i)

    def item(self, sid, rng=None, max_bars=512):
        view = self.c.view(sid)
        lo = int(rng.integers(0, view.nb - max_bars + 1)) if rng is not None and view.nb > max_bars else 0
        return seg_item(self.c.vocab, view, lo, min(view.nb, lo + max_bars) if rng is not None else None,
                        rng=rng, named=self.named, pseudo=self.pseudo.get(sid))


def seg_batches(ds: SegSet, ids, budget, rng=None):
    """Micro-batches of whole songs grouped by length, at most ``budget`` padded bars each."""
    nbs = np.array([min(ds.c.songs[i]["nb"], 512 if rng is not None else 4096) for i in ids])
    order = np.argsort(nbs, kind="stable")
    chunks, cur = [], []
    for k in order:
        if cur and nbs[k] * (len(cur) + 1) > budget:
            chunks.append(cur)
            cur = []
        cur.append(ids[k])
    chunks += [cur] if cur else []
    if rng is not None:
        chunks = [chunks[i] for i in rng.permutation(len(chunks))]
    for ch in chunks:
        yield seg_collate([ds.item(i, rng) for i in ch]), sum(min(ds.c.songs[i]["nb"], 512) for i in ch), ch


def seg_loss(model, b, cls_w):
    ls, lb = model(b)
    keep = [i for i in model.cfg["active"] if not (model.cfg.get("named") and i == OTHER)]   # sections in use
    remap = torch.full((len(SECTIONS),), -1, dtype=torch.long, device=ls.device)
    remap[keep] = torch.arange(len(keep), device=ls.device)
    tgt = torch.where(b["sec"] >= 0, remap[b["sec"].clamp(min=0)], -1)
    ok = tgt >= 0
    sec_l = (F.cross_entropy(ls[ok][:, keep], tgt[ok], weight=cls_w[keep], label_smoothing=0.05) if ok.any()
             else ls.sum() * 0)
    m = b["mask"].float()
    pos = (b["bnd"] * m).sum().clamp(min=1)
    bnd_l = (F.binary_cross_entropy_with_logits(lb, b["bnd"], reduction="none",
                                                pos_weight=torch.tensor(8.0, device=lb.device)) * m).sum() / m.sum()
    return sec_l + bnd_l, float(pos)


def novelty_fast(S: np.ndarray, w: int) -> np.ndarray:
    """Foote novelty with a box checkerboard kernel from an integral image: O(n) per window after the
    O(n²) cumulative sum (riffgen/analysis/riffs.py loops in Python, O(n·w²))."""
    n = len(S)
    nov = np.zeros(n, np.float32)
    if n < 2 * w + 1:
        return nov
    I = np.zeros((n + 1, n + 1))
    I[1:, 1:] = S.astype(np.float64).cumsum(0).cumsum(1)
    i = np.arange(w, n - w + 1)

    def box(r0, r1, c0, c1):
        return (I[r1, c1] - I[r0, c1] - I[r1, c0] + I[r0, c0]) / (w * w)

    nov[i] = (box(i - w, i, i - w, i) + box(i, i + w, i, i + w)) / 2 - box(i - w, i, i, i + w)
    return nov


def rule_borders(f, threshold=0.22) -> set[int]:
    """The old rule-based borders (riffs.py: novelty peaks of the bar self-similarity at 2 and 4 bars, plus an
    8-bar grid in long stretches), on the stored bar features, as a baseline."""
    S = seg_ssm(f)
    n, out = len(S), {0}
    for w in (2, 4):
        nov = novelty_fast(S, w)
        for i in range(w, n - w + 1):
            if nov[i] >= threshold and nov[i] >= nov[max(0, i - 2):min(n, i + 3)].max():
                out.add(i)
    edges = sorted(out) + [n]
    for a, b in zip(edges[:-1], edges[1:]):
        out |= set(range(a + 8, b, 8)) if b - a > 8 else set()
    return out


def peak_borders(p: np.ndarray, tau: float) -> set[int]:
    left, right = np.concatenate([[-1], p[:-1]]), np.concatenate([p[1:], [-1]])
    return set(np.where((p >= tau) & (p >= left) & (p >= right))[0].tolist()) | {0}


def border_prf(pred: set, true: set, tol: int) -> tuple[int, int, int]:
    """Matched borders (each true border at most once, within ``tol`` bars), predicted, true; bar 0 excluded."""
    pred, true = sorted(pred - {0}), sorted(true - {0})
    used, hit = set(), 0
    for t in true:
        cands = [p for p in pred if abs(p - t) <= tol and p not in used]
        if cands:
            used.add(min(cands, key=lambda p: abs(p - t)))
            hit += 1
    return hit, len(pred), len(true)


@torch.no_grad()
def seg_predict(model, batches, dev) -> dict:
    """{song: (section probabilities, border probabilities)}."""
    model.eval()
    out = {}
    for b, _, ids in batches:
        b = to_dev(b, dev)
        ls, lb = model(b)
        ps, pb = torch.softmax(ls.float(), -1).cpu().numpy(), torch.sigmoid(lb.float()).cpu().numpy()
        for k, n in enumerate(b["mask"].sum(1).tolist()):
            out[ids[k]] = (ps[k, :n], pb[k, :n])
    return out


def seg_transitions(ds: SegSet) -> tuple[np.ndarray, np.ndarray]:
    """From the training songs' markers: which named section a song starts with, and which follows which
    (the same one again too: Verse 1 -> Verse 2)."""
    K = OTHER
    trans, start = np.ones((K, K)), np.ones(K)
    for i in ds.ids["train"]:
        sec = np.asarray(ds.c.view(i).f["sec"])
        named = sec[(sec >= 0) & (sec != OTHER)]
        if len(named):
            start[named[0]] += 1
            np.add.at(trans, (named[:-1], named[1:]), 1)
    return trans / trans.sum(1, keepdims=True), start / start.sum()


INTRO = SECTIONS.index("intro")


def seg_viterbi(ps, pb, trans, start, min_len=1, allowed=None) -> tuple[np.ndarray, set]:
    """Sections as segments: the most likely sequence of (section, bars so far) states. A new section starts
    with probability p(start) of that bar, follows the learned transitions, and may only start once the
    current section has lasted ``min_len`` bars (the intro may be shorter; so may nothing else, the last
    section included). Sections start at bar starts by construction. ``allowed`` (K x K bools, from -> to)
    forbids transitions outright. Returns (labels, section starts)."""
    K, n, X = OTHER, len(ps), max(min_len, 1)
    R = max(X, 2)                                            # run-length states 1..R (R = "R or more")
    NEG = -1e18
    e = np.log(ps[:, :K] / ps[:, :K].sum(1, keepdims=True) + 1e-9)
    stay, sw, lt = np.log(np.clip(1 - pb, 1e-6, 1)), np.log(np.clip(pb, 1e-6, 1)), np.log(trans + 1e-9)
    if allowed is not None:
        lt = np.where(allowed, lt, NEG)
    done = np.zeros((K, R), bool)                            # states a section may end in
    done[:, X - 1:] = True
    done[INTRO, :] = True
    dp = np.full((K, R), NEG)
    dp[:, 0] = np.log(start + 1e-9) + e[0]
    bl, bk, bs = (np.zeros((n, K), np.int64) for _ in range(3))
    for i in range(1, n):
        leave = np.where(done, dp, NEG)
        kbest, vbest = leave.argmax(1), leave.max(1)
        cand = vbest[:, None] + lt
        bl[i] = cand.argmax(0)
        bk[i] = kbest[bl[i]]
        new = np.full((K, R), NEG)
        new[:, 0] = cand.max(0) + sw[i]
        new[:, 1:R - 1] = dp[:, 0:R - 2] + stay[i]
        last = np.stack([dp[:, R - 2], dp[:, R - 1]], 1)
        bs[i] = last.argmax(1)
        new[:, R - 1] = last.max(1) + stay[i]
        dp = new + e[i][:, None]
    end = np.where(done, dp, NEG)
    if end.max() <= NEG / 2:                                  # a song shorter than one section
        end = dp
    lab, r = np.unravel_index(int(end.argmax()), end.shape)
    labels, starts = np.empty(n, np.int64), {0}
    for i in range(n - 1, -1, -1):
        labels[i] = lab
        if i == 0:
            break
        if r == 0:
            starts.add(i)
            lab, r = bl[i, lab], bk[i, lab]
        elif r < R - 1:
            r -= 1
        else:
            r = R - 2 + bs[i, lab]
    return labels, starts


def min_len_borders(ps, pb, tau, min_len) -> set:
    """Predicted section starts, the most confident first, keeping only those that leave every section at
    least ``min_len`` bars long (a first section the model calls intro may be shorter)."""
    n, acc = len(pb), [0]
    for c in sorted(peak_borders(pb, tau) - {0}, key=lambda i: -pb[i]):
        left = max(a for a in acc if a < c)
        right = min([a for a in acc if a > c] + [n])
        intro_ok = left == 0 and ps[0:c, :OTHER].mean(0).argmax() == INTRO
        if (c - left >= min_len or intro_ok) and right - c >= min_len:
            acc.append(c)
    return set(acc)


def seg_rule_scores(ds: SegSet, ids, preds, tau, hmm, min_len) -> dict:
    """Named-section bar accuracy, macro F1 and section-start F1 (exact, ±1 bar) under a minimum length."""
    conf = {k: np.zeros((len(SECTIONS), len(SECTIONS)), np.int64) for k in ("borders", "viterbi")}
    hits = {f"{k}_{t}": np.zeros(3, np.int64) for k in conf for t in (0, 1)}
    for sid in ids:
        ps, pb = preds[sid]
        sec, bnd = seg_targets(ds.c.view(sid).f["sec"])
        true_b = set(np.where(bnd > 0)[0].tolist())
        starts = min_len_borders(ps, pb, tau, min_len)
        edges = sorted(starts) + [len(pb)]
        lab_b = np.empty(len(pb), np.int64)
        for a, c in zip(edges[:-1], edges[1:]):
            lab_b[a:c] = ps[a:c, :OTHER].mean(0).argmax()
        lab_v, starts_v = seg_viterbi(ps, pb, *hmm, min_len)
        ok = (sec >= 0) & (sec != OTHER)
        for k, (lab, st) in {"borders": (lab_b, starts), "viterbi": (lab_v, starts_v)}.items():
            np.add.at(conf[k], (sec[ok], lab[ok]), 1)
            for t in (0, 1):
                hits[f"{k}_{t}"] += border_prf(st, true_b, t)
    out = {}
    for k, c in conf.items():
        tp, fp, fn = np.diag(c), c.sum(0) - np.diag(c), c.sum(1) - np.diag(c)
        f1 = [2 * tp[i] / max(2 * tp[i] + fp[i] + fn[i], 1) for i in range(OTHER) if c[i].sum() > 0]
        out[k] = {"bar_acc_named": float(np.trace(c) / max(c.sum(), 1)), "macro_f1_named": float(np.mean(f1))}
        for t in (0, 1):
            h, p, n = hits[f"{k}_{t}"]
            P, R = h / max(p, 1), h / max(n, 1)
            out[k][f"start_F1_{t}"] = 2 * P * R / max(P + R, 1e-9)
    return out


def collapse_preds(preds) -> dict:
    """Section probabilities moved onto the sections of the current configs/sections.json (a model trained with
    another mapping: interlude -> bridge, prechorus -> other); 'other' gets none (only named sections are scored)."""
    M = np.zeros((len(SECTIONS), len(SECTIONS)))
    for i, s in enumerate(SECTIONS):
        M[i, section_of(s)] = 1
    out = {}
    for sid, (ps, pb) in preds.items():
        q = ps @ M
        q[:, OTHER] = 0
        out[sid] = (q / np.clip(q.sum(1, keepdims=True), 1e-9, None), pb)
    return out


def tune_tau(ds: SegSet, ids, preds, tol=1) -> float:
    """The section-start threshold with the best start F1 (within ``tol`` bars) on these songs."""
    true = {sid: set(np.where(seg_targets(ds.c.view(sid).f["sec"])[1] > 0)[0].tolist()) for sid in ids}

    def f1(t):
        h = sum((np.array(border_prf(peak_borders(preds[sid][1], t), true[sid], tol)) for sid in ids), np.zeros(3))
        P, R = h[0] / max(h[1], 1), h[0] / max(h[2], 1)
        return 2 * P * R / max(P + R, 1e-9)
    return max(TAUS, key=f1)


def section_lengths(ds: SegSet, ids) -> np.ndarray:
    """Lengths in bars of the real sections (marker to marker), the first one (often an intro) excluded."""
    out = []
    for sid in ids:
        sec = np.asarray(ds.c.view(sid).f["sec"])
        m = list(np.where(sec >= 0)[0]) + [len(sec)]
        out += [b - a for a, b in zip(m[1:-1], m[2:])]
    return np.array(out)


def seg_score(ds: SegSet, ids, preds, tau, prior=None, hmm=None) -> dict:
    """Bar accuracy (all, and on named sections) and F1 per section for three decodings (each bar on its own;
    one label per predicted segment; Viterbi), borders P/R/F within 0 and 1 bars, and the baselines."""
    conf = {k: np.zeros((len(SECTIONS), len(SECTIONS)), np.int64) for k in ("raw", "smoothed", "viterbi")}
    hits = {k: np.zeros(3, np.int64) for k in ("model_0", "model_1", "rule_0", "rule_1", "grid8_0", "grid8_1")}
    base = Counter()
    for sid in ids:
        ps, pb = preds[sid]
        v = ds.c.view(sid)
        f = {k: np.asarray(x) for k, x in v.f.items()}
        sec, bnd = seg_targets(f["sec"])
        pred = {"raw": ps.argmax(-1), "smoothed": np.empty(len(ps), np.int64)}
        borders = sorted(peak_borders(pb, tau)) + [len(pb)]
        for a, c in zip(borders[:-1], borders[1:]):
            pred["smoothed"][a:c] = ps[a:c].mean(0).argmax()
        if hmm is not None:
            pred["viterbi"] = seg_viterbi(ps, pb, *hmm)[0]
        ok = sec >= 0
        for k, p in pred.items():
            np.add.at(conf[k], (sec[ok], p[ok]), 1)
        true_b = set(np.where(bnd > 0)[0].tolist())
        for name, pb_set in (("model", peak_borders(pb, tau)), ("rule", rule_borders(f)),
                             ("grid8", set(range(0, len(pb), 8)))):
            for tol in (0, 1):
                hits[f"{name}_{tol}"] += border_prf(pb_set, true_b, tol)
        if prior is not None:
            pos = np.minimum((np.arange(len(sec)) * 10) // max(len(sec), 1), 9)
            base["prior_ok"] += int((prior[pos] == sec)[ok].sum())
            base["major_ok"] += int((sec[ok] == prior[10]).sum())
            base["n"] += int(ok.sum())

    def acc(c, named=False):
        rows = [i for i in range(len(SECTIONS)) if not named or i != OTHER]
        return float(np.trace(c[rows][:, rows]) / max(c[rows].sum(), 1))

    def f1s(c):
        tp, fp, fn = np.diag(c), c.sum(0) - np.diag(c), c.sum(1) - np.diag(c)
        return {s: float(2 * tp[i] / max(2 * tp[i] + fp[i] + fn[i], 1)) for i, s in enumerate(SECTIONS)}

    res = {"songs": len(ids), "bars": int(conf["raw"].sum())}
    for k, c in conf.items():
        if c.sum():
            f1 = f1s(c)                                  # (averaged over the sections the mapping has)
            res[k] = {"bar_acc": acc(c), "bar_acc_named": acc(c, True), "f1": f1, "macro_f1_named": float(np.mean(
                [f1[s] for i, s in enumerate(SECTIONS) if i != OTHER and c[i].sum() > 0]))}
    best = max((k for k in conf if conf[k].sum()), key=lambda k: res[k]["macro_f1_named"])
    res["macro_f1_named"], res["bar_acc_named"] = res[best]["macro_f1_named"], res[best]["bar_acc_named"]
    res["bar_acc_smoothed"], res["bar_acc_named_smoothed"] = res["smoothed"]["bar_acc"], res["smoothed"]["bar_acc_named"]
    res["best_decoding"] = best
    for k, (h, p, t) in hits.items():
        P, R = h / max(p, 1), h / max(t, 1)
        res[f"border_{k}"] = {"P": P, "R": R, "F": 2 * P * R / max(P + R, 1e-9)}
    if base["n"]:
        res["baseline_majority_acc"] = base["major_ok"] / base["n"]
        res["baseline_position_prior_acc"] = base["prior_ok"] / base["n"]
    res["confusion"] = conf[best].tolist()
    return res


def cmd_segment(a) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    ds = SegSet(a.data)
    V = ds.c.vocab
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    n_bars = {k: sum(ds.c.songs[i]["nb"] for i in v) for k, v in ds.ids.items()}
    log(f"songs with sections: train {len(ds.ids['train'])} ({n_bars['train']} bars), val {len(ds.ids['val'])}, "
        f"test {len(ds.ids['test'])} ({n_bars['test']} bars); without sections: {len(ds.unmarked)}")
    if a.action == "train":
        torch.manual_seed(a.seed)
        ds.named = a.named
        used = np.zeros(len(SECTIONS), bool)                 # the sections this mapping produces
        for i in ds.ids["train"]:
            s = np.asarray(ds.c.view(i).f["sec"])
            used[s[s >= 0]] = True
        model = SegModel(V, a.d, a.layers, ssm=a.ssm, named=a.named, active=np.where(used)[0].tolist()).to(dev)
        log(f"sections in use: {[SECTIONS[i] for i in model.cfg['active']]}")
        train_ids = list(ds.ids["train"])
        if a.teacher:                                        # noisy student: the teacher labels the unmarked tabs
            teacher, tck = load_seg(a.teacher, V, dev)
            kept = 0
            for sid, (ps, pb) in collapse_preds(seg_predict(teacher, seg_batches(ds, ds.unmarked, a.micro), dev)).items():
                lab = np.where(ps.max(-1) >= a.pl_conf, ps.argmax(-1), -1)
                bnd = np.zeros(len(pb), np.float32)
                bnd[sorted(peak_borders(pb, tck["tau"]))] = 1.0
                ds.pseudo[sid] = (lab, bnd)
                kept += int((lab >= 0).sum())
            train_ids += list(ds.pseudo)
            log(f"pseudo-labels for {len(ds.pseudo)} unmarked songs: {kept} bars above confidence {a.pl_conf}")
            del teacher
        n_bars["train"] = sum(ds.c.songs[i]["nb"] for i in train_ids)
        counts = np.zeros(len(SECTIONS))
        for i in ds.ids["train"][::10]:
            s, _ = seg_targets(ds.c.view(i).f["sec"])
            counts += np.bincount(s[s >= 0], minlength=len(SECTIONS))
        cls_w = torch.tensor((counts.sum() / np.maximum(counts, 1) / len(SECTIONS)) ** 0.5, dtype=torch.float32, device=dev)
        M = 72 * a.layers * a.d ** 2 + 6 * a.layers * a.d * 110          # FLOPs per bar, as in hparams
        C = M * a.epochs * n_bars["train"]
        lr = a.lr or min(0.3118 * C ** -0.125, 3e-3)
        batch = a.batch or int(0.2920 * C ** 0.3271)
        log(f"{n_params(model) / 1e6:.1f}M parameters; lr {lr:.2e}, {batch} bars per step (DeepSeek fit, C={C:.2e})")
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
        total, seen, best, t_start = a.epochs * n_bars["train"], 0, -1.0, time.time()
        for ep in range(a.epochs):
            model.train()
            rng, acc_bars, t0 = rng_for(a.seed, ep), 0, time.time()
            for b, nb, _ in seg_batches(ds, train_ids, a.micro, rng):
                loss, _ = seg_loss(model, to_dev(b, dev), cls_w)
                (loss * nb / batch).backward()
                acc_bars += nb
                if acc_bars >= batch:
                    seen += acc_bars
                    for g in opt.param_groups:
                        g["lr"] = lr_at(seen / total, lr)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    acc_bars = 0
            preds = seg_predict(model, seg_batches(ds, ds.ids["val"], a.micro), dev)
            best_tau = max((seg_score(ds, ds.ids["val"], preds, t)["border_model_1"]["F"], t) for t in TAUS)
            r = seg_score(ds, ds.ids["val"], preds, best_tau[1])
            log(f"epoch {ep + 1}: {time.time() - t0:.0f}s, loss {float(loss):.3f}, val named bar acc "
                f"{r['raw']['bar_acc_named']:.3f}, macro F1 {r['raw']['macro_f1_named']:.3f}, "
                f"border F1 +-1 {r['border_model_1']['F']:.3f} (tau {best_tau[1]})")
            if r["raw"]["macro_f1_named"] > best:
                best = r["raw"]["macro_f1_named"]
                save_ckpt(run / "seg.pt", {"model": model.state_dict(), "cfg": model.cfg, "tokens": V.tokens,
                                           "tau": best_tau[1], "epoch": ep + 1})
        log(f"trained in {(time.time() - t_start) / 60:.1f} min -> {run / 'seg.pt'}")
    elif a.action == "eval":
        model, ck = load_seg(run / "seg.pt", V, dev)
        prior = np.zeros(11, np.int64)                     # baselines: class per tenth of the song, majority
        cnt = np.zeros((10, len(SECTIONS)), np.int64)
        for i in ds.ids["train"]:
            s, _ = seg_targets(ds.c.view(i).f["sec"])
            pos = np.minimum((np.arange(len(s)) * 10) // len(s), 9)
            np.add.at(cnt, (pos[s >= 0], s[s >= 0]), 1)
        prior[:10], prior[10] = cnt.argmax(1), cnt.sum(0).argmax()
        val = seg_predict(model, seg_batches(ds, ds.ids["val"], a.micro), dev)
        t0 = time.time()
        preds = seg_predict(model, seg_batches(ds, ds.ids["test"], a.micro), dev)
        t_pred = time.time() - t0
        if a.collapse:
            val, preds = collapse_preds(val), collapse_preds(preds)
        tau = tune_tau(ds, ds.ids["val"], val)
        res = seg_score(ds, ds.ids["test"], preds, tau, prior, seg_transitions(ds)) | {
            "predict_seconds": t_pred, "tau": tau, "epoch": ck["epoch"], "cfg": ck["cfg"]}
        (run / "seg_eval_test.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: ({s: v[s] for s in v if s != "f1"} if isinstance(v, dict) and "f1" in v else v)
                          for k, v in res.items() if k != "confusion"}, indent=1))
    elif a.action == "fuse":                                # the old rule-based borders blended into the model's
        model, ck = load_seg(run / "seg.pt", V, dev)        # section starts: p = (1 - w) p_model + w [rule border]
        hmm = seg_transitions(ds)
        preds = {s: seg_predict(model, seg_batches(ds, ds.ids[s], a.micro), dev) for s in ("val", "test")}
        if a.collapse:
            preds = {s: collapse_preds(p) for s, p in preds.items()}
        rule = {}
        for s in ("val", "test"):
            for sid in ds.ids[s]:
                r = np.zeros(ds.c.songs[sid]["nb"], np.float32)
                r[sorted(rule_borders({k: np.asarray(v) for k, v in ds.c.view(sid).f.items()}))] = 1.0
                rule[sid] = r
        res = {"min_len": a.min_len, "w": {}}
        for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
            fused = {s: {sid: (ps, (1 - w) * pb + w * rule[sid]) for sid, (ps, pb) in p.items()} for s, p in preds.items()}
            tau = tune_tau(ds, ds.ids["val"], fused["val"])
            v = seg_rule_scores(ds, ds.ids["val"], fused["val"], tau, hmm, a.min_len)
            t = seg_rule_scores(ds, ds.ids["test"], fused["test"], tau, hmm, a.min_len)
            res["w"][w] = {"tau": tau, "val": v, "test": t}
            log(f"rule weight {w}: tau {tau} | " + " | ".join(
                f"{k}: named acc {t[k]['bar_acc_named']:.3f} F1 {t[k]['macro_f1_named']:.3f} "
                f"starts {t[k]['start_F1_0']:.3f}/{t[k]['start_F1_1']:.3f}" for k in t))
        bw = max(res["w"], key=lambda w: res["w"][w]["val"]["viterbi"]["start_F1_1"])
        log(f"best rule weight on validation: {bw}")
        (run / f"seg_fuse{'_collapsed' if a.collapse else ''}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    elif a.action == "rules":                               # minimum section length X, chosen on validation
        model, ck = load_seg(run / "seg.pt", V, dev)
        hmm = seg_transitions(ds)
        val = seg_predict(model, seg_batches(ds, ds.ids["val"], a.micro), dev)
        tau = max((seg_score(ds, ds.ids["val"], val, t)["border_model_1"]["F"], t) for t in TAUS)[1]
        test = seg_predict(model, seg_batches(ds, ds.ids["test"], a.micro), dev)
        lens = section_lengths(ds, ds.ids["test"])
        res = {"tau": tau, "cfg": ck["cfg"], "epoch": ck["epoch"], "X": {}}
        for X in range(1, 9):
            v, t = seg_rule_scores(ds, ds.ids["val"], val, tau, hmm, X), seg_rule_scores(ds, ds.ids["test"], test, tau, hmm, X)
            res["X"][X] = {"val": v, "test": t, "true_sections_shorter": float((lens < X).mean())}
            log(f"X={X}: real sections shorter {100 * (lens < X).mean():4.1f}% | " + " | ".join(
                f"{k}: named acc {t[k]['bar_acc_named']:.3f} F1 {t[k]['macro_f1_named']:.3f} "
                f"starts {t[k]['start_F1_0']:.3f}/{t[k]['start_F1_1']:.3f}" for k in t))
        for k in ("borders", "viterbi"):
            bx = max(res["X"], key=lambda X: res["X"][X]["val"][k]["macro_f1_named"])
            res[f"best_X_{k}"] = bx
            log(f"best X for {k} (on validation): {bx} -> test {res['X'][bx]['test'][k]}")
        (run / "seg_rules.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    elif a.action == "label":
        model, ck = load_seg(run / "seg.pt", V, dev)
        t0 = time.time()
        preds = seg_predict(model, seg_batches(ds, ds.unmarked, a.micro), dev)
        t_pred = time.time() - t0
        bars = sum(len(p[0]) for p in preds.values())
        log(f"predicted sections for {len(preds)} songs without markers ({bars} bars) in {t_pred:.1f}s on {dev}")
        out = {"songs": len(preds), "bars": bars, "predict_seconds": t_pred,
               "labels": dict(Counter(SECTIONS[int(x)] for ps, _ in preds.values() for x in ps.argmax(-1)))}
        if a.write:                                          # a sample written as GP5 copies, to time the write step
            out |= seg_write_sample(ds, model, seg_transitions(ds), a.min_len, dev, a.write, run / "labelled")
        (run / "seg_label.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1))


def seg_write_sample(ds, model, hmm, min_len, dev, n, out_dir) -> dict:
    """Parse, encode, predict (Viterbi, sections of at least ``min_len`` bars) and write section markers into
    copies of ``n`` tabs without sections (never the originals); returns the seconds per file of each step."""
    import guitarpro as gp
    out_dir.mkdir(parents=True, exist_ok=True)
    t = Counter()
    for sid in np.random.default_rng(0).choice(ds.unmarked, min(n, len(ds.unmarked)), replace=False).tolist():
        path = ROOT / "data" / ds.c.songs[sid]["file"]
        t0 = time.time()
        song = gp.parse(str(path))
        enc = encode_song(song, ds.c.vocab)
        t1 = time.time()
        view = SongView.from_encoded(enc, {})
        ps, pb = seg_predict(model, [(seg_collate([seg_item(ds.c.vocab, view)]), 0, [sid])], dev)[sid]
        t2 = time.time()
        labels, starts = seg_viterbi(ps, pb, *hmm, min_len)
        for a in sorted(starts):
            m = enc["orig"][a]
            if song.measureHeaders[m].marker is None:
                song.measureHeaders[m].marker = gp.Marker(title=SECTIONS[int(labels[a])].capitalize())
        gp.write(song, str(out_dir / Path(path).name), version=(5, 1, 0))
        t3 = time.time()
        t["parse_encode"] += t1 - t0
        t["predict"] += t2 - t1
        t["write"] += t3 - t2
    return {f"sec_per_file_{k}": v / max(n, 1) for k, v in t.items()}


# ============================================================================ commands

def corpus_stats(data) -> dict:
    p = Path(data) / "meta.json"
    if p.is_file():
        return Corpus(data).stats
    log(f"{p} not found: using estimates from a sample of the 61k tabs (77 tokens/bar, 100 bars/song)")
    return {"tokens_per_bar": 77.0, "bars_per_song": 100.0, "train_tokens": 4.3e8}


def cmd_hparams(a) -> None:
    stats = corpus_stats(a.data)
    V = Corpus(a.data).vocab if (Path(a.data) / "meta.json").is_file() else Vocab(make_tokens())
    print(f"{'model':9} {'params':>8} {'FLOPs/tok':>10} {'tokens':>8} {'compute':>9} {'lr':>8} {'batch':>9} {'4090 h':>7}")
    for name in a.models.split(","):
        cfg = model_cfg(name)
        h = suggest(cfg, stats)
        n = n_params(build_model(cfg, V))
        print(f"{name:9} {n / 1e6:7.0f}M {h['flops_per_token'] / 1e6:9.0f}M {h['tokens'] / 1e9:7.2f}B "
              f"{h['compute']:9.2e} {h['lr']:8.2e} {h['batch_tokens']:9d} {h['hours_4090']:7.1f}")
        print("          grid: " + ", ".join(f"{B // 1024}k@{lr:.1e}" for B, lr in grid_points(h)))


def parse_focus(s) -> dict:
    return {k: float(v) for k, v in (x.split("=") for x in s.split(",") if x)} if s else {}


def cmd_train(a) -> None:
    cfg = model_cfg(a.model, a.tiny)
    if a.micro_tokens:
        cfg = cfg | {"micro_tokens": a.micro_tokens}
    if a.focal:
        cfg = cfg | {"focal": a.focal}
    if a.smooth:
        cfg = cfg | {"smooth": a.smooth}
    if a.ckpt:
        cfg = cfg | {"ckpt": True}
    h = suggest(cfg, corpus_stats(a.data), a.epochs)
    train_run(cfg, a.data, a.run or ROOT / "runs" / cfg["name"], a.lr or h["lr"], a.batch_tokens or h["batch_tokens"],
              a.epochs or cfg["epochs"], a.eval_every, a.budget, a.resume, a.init, a.lora, parse_focus(a.focus),
              a.workers, a.seed, a.eval_gaps, a.gen_gaps)


def cmd_eval(a) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, V, cfg, _ = load_ckpt(a.ckpt, dev)
    c = Corpus(a.data)
    if c.vocab.tokens != V.tokens:
        sys.exit("the checkpoint was trained on another vocabulary")
    res = evaluate(model, c, cfg, a.split, a.gaps, a.gen_gaps, out=Path(a.ckpt).with_name(f"samples_{a.split}.txt"))
    print(json.dumps(res, indent=1))
    Path(a.ckpt).with_name(f"eval_{a.split}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def cmd_ngram(a) -> None:
    """Build the model-free reference the lift is measured against. Once per corpus: the result only depends
    on the corpus, so every run and every architecture is compared with the same number."""
    c = Corpus(a.data)
    ref = ngram_ref(c, a.split, build=True)
    print(json.dumps(ref, indent=1))
    log(f"bigram on the {a.split} gaps: loss {ref['loss']:.4f} over {ref['gaps']} gaps, "
        f"{ref['loss_cmp']:.4f} over the {ref['gaps_cmp']} comparable ones")


def cmd_grid(a) -> None:
    """README §12: every model x 3 batch sizes x 3 learning rates, each a short run of --budget tokens; the
    lowest validation loss wins. Finished runs are skipped, so a stopped grid continues."""
    stats, out = corpus_stats(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {r["run"]: r for r in map(json.loads, res_path.read_text().splitlines())} if res_path.is_file() else {}
    for name in a.models.split(","):
        cfg = model_cfg(name, a.tiny)
        if a.micro_tokens:
            cfg = cfg | {"micro_tokens": a.micro_tokens}
        if a.ckpt:
            cfg = cfg | {"ckpt": True}
        bms = [float(x) for x in a.batches.split(",") if x.strip()]
        lms = [float(x) for x in a.lrs.split(",") if x.strip()]
        # centre the grid on the run length actually used (--budget), not on the model's default epochs
        eps = a.budget / stats["train_tokens"] if a.budget else None
        for B, lr in grid_points(suggest(cfg, stats, eps), bms, lms):
            for var in grid_variants(cfg, a.aux, a.smooth, a.focal):
                c = cfg | var
                run = f"{name}_b{B // 1024}k_lr{lr:.1e}" + "".join(f"_{k}{v:g}" for k, v in var.items())
                if run in done:
                    continue
                r = train_run(c, a.data, out / run, lr, B, c["epochs"], a.eval_every, a.budget, workers=a.workers,
                              seed=a.seed, eval_gaps=a.eval_gaps, gen_gaps=0, keep_ckpt=False)
                loss = r["best"] if math.isfinite(r["best"]) else 99.0  # a diverged run sorts last
                be = r["best_eval"]
                done[run] = {"run": run, "model": name, "batch_tokens": B, "lr": lr, "val_loss": loss,
                             "real_tokens_per_step": round(r["tokens_per_step"]), "micro_tokens": r["micro_tokens"],
                             "tokens": r["tokens"]} | var | {
                    k: be[k] for k in ("acc", "loss_cmp", "acc_cmp", "lift_cmp") if k in be}
                with open(res_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(done[run]) + "\n")
    for name in a.models.split(","):
        rows = sorted((r for r in done.values() if r["model"] == name), key=lambda r: r["val_loss"])
        for r in rows:
            extra = "".join(f"  {k} {r[k]:.4f}" for k in ("loss_cmp", "lift_cmp") if k in r)
            print(f"{r['run']:40} val loss {r['val_loss']:.4f}{extra}")
        if rows:
            extra = " ".join(f"{k} {rows[0][k]:g}" for k in ("aux", "smooth") if rows[0].get(k) is not None)
            print(f"best {name}: --batch-tokens {rows[0]['batch_tokens']} --lr {rows[0]['lr']:.2e}"
                  + (f" ({extra})" if extra else "") + "\n")


def find_gaps(song, guitars, min_len) -> list[tuple[int, int]]:
    """Runs of >= min_len measures in which every guitar track is empty (README §1), 0-based inclusive."""
    empty = [all(not any(b.notes for v in song.tracks[t].measures[m].voices for b in v.beats) for t in guitars)
             for m in range(len(song.measureHeaders))]
    runs, start = [], None
    for m, e in enumerate(empty + [False]):
        if e and start is None:
            start = m
        if not e and start is not None:
            if m - start >= min_len:
                runs.append((start, m - 1))
            start = None
    return runs


def write_bars(path_in, enc, positions, bars, V, out_path) -> None:
    """Splice generated bars into a fresh copy of the song (README §2: everything else stays as written)."""
    import guitarpro as gp
    song = gp.parse(str(path_in))
    for p, bar in zip(positions, bars):
        m = enc["orig"][p]
        parts, slot, beat = {}, -1, None
        for t in (bar & MASK).tolist():
            k = V.kind[t]
            if k == K_TRK:
                slot = V.trk_g.index(t) if t in V.trk_g else -1
                parts.setdefault(slot, [])
            elif slot >= 0 and k == K_DUR:
                beat = [DUR_TABLE[V.tokens[t]], False, []]
                parts[slot].append(beat)
            elif slot >= 0 and t == V.trem:
                beat[1] = True
            elif slot >= 0 and k == K_NOTE:
                beat[2].append([int(V.n_sf[t, 0]), int(V.n_sf[t, 1]), set()])
            elif slot >= 0 and k == K_FX:
                beat[2][-1][2].add(V.tokens[t][3:])
        for slot, ti in enumerate(enc["guitars"]):
            if slot not in parts:          # a bar too dense to hold every guitar carries fewer slots than the
                continue                   # song has: that track keeps the bar as it was written
            measure = song.tracks[ti].measures[m]
            for v in measure.voices:
                v.beats = []
            voice, start = measure.voices[0], measure.start
            for (dv, dot, u), trem, notes in parts.get(slot, []):
                d = gp.Duration(value=dv, isDotted=dot, tuplet=gp.Tuplet(u, TUPLETS[u]))
                b = gp.Beat(voice, duration=d, start=start, status=gp.BeatStatus.normal if notes else gp.BeatStatus.rest)
                for s, f, fx in notes:
                    n = gp.Note(b, value=f, string=s, type=gp.NoteType.tie if "tie" in fx else
                                gp.NoteType.dead if "dead" in fx else gp.NoteType.normal)
                    e = n.effect
                    e.ghostNote, e.palmMute, e.hammer, e.vibrato = "ghost" in fx, "pm" in fx, "ho" in fx, "vib" in fx
                    e.letRing, e.accentuatedNote, e.staccato = "ring" in fx, "acc" in fx, "stacc" in fx
                    if "slide" in fx:
                        e.slides = [gp.SlideType.shiftSlideTo]
                    if "bend" in fx:
                        e.bend = gp.BendEffect(type=gp.BendType.bend, value=100, points=[
                            gp.BendPoint(0, 0), gp.BendPoint(6, 4), gp.BendPoint(12, 4)])
                    if "harm" in fx:
                        e.harmonic = gp.NaturalHarmonic()
                    if trem:
                        e.tremoloPicking = gp.TremoloPickingEffect(duration=gp.Duration(value=16))
                    b.notes.append(n)
                voice.beats.append(b)
                start += d.time
    gp.write(song, str(out_path), version=(5, 1, 0))


def cmd_infer(a) -> None:
    import guitarpro as gp
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    model, V, cfg, _ = load_ckpt(a.ckpt, dev)
    song = gp.parse(a.song)
    enc = encode_song(song, V, max_bars=4096)
    if any(open_strings(song.tracks[t]) != list(DROP_A) for t in enc["guitars"]):
        sys.exit("every guitar track of the song must be 7-string Drop A (A E A D G B E), capo 0 (README §2)")
    view = SongView.from_encoded(enc, {})
    gaps = ([tuple(int(x) - 1 for x in a.gap.split("-"))] if a.gap else find_gaps(song, enc["guitars"], a.min_gap))
    if not gaps:
        sys.exit(f"no gap: no run of {a.min_gap}+ measures with empty guitars (or give --gap 33-40)")
    pos_of = {}
    for p, m in enumerate(enc["orig"]):
        pos_of.setdefault(m, []).append(p)
    rows, w = [[V.none] * 4], [0.0]                         # row 0: unconditional (README §8)
    for spec, weight in [(x, a.cfg) for x in a.labels.split(",") if x] + [(x, -a.neg_cfg) for x in a.neg.split(",") if x]:
        if f"lbl:{spec}" not in V.id:
            sys.exit(f"unknown label {spec}; known: {', '.join(t[4:] for t in V.tokens if t.startswith('lbl:'))}")
        r = [V.none] * 4
        r[SLOTS.index(spec.split(":")[0])] = V.id[f"lbl:{spec}"]
        rows.append(r)
        w.append(weight)
    if len(rows) == 1:
        w = [1.0]
    out_root = Path(a.out or ROOT / "outputs")
    all_gap_pos = {p for g0, g1 in gaps for m in range(g0, g1 + 1) for p in pos_of.get(m, [])}
    ctx_h = {int(h) for i, h in enumerate(enc["feats"]["hash"]) if enc["feats"]["gn"][i]}
    runs = []                                               # a gap across a repeat sign is played in pieces:
    for g0, g1 in gaps:                                     # each piece is filled on its first pass
        first = sorted(pos_of[m][0] for m in range(g0, g1 + 1) if m in pos_of)
        for _, grp in itertools.groupby(enumerate(first), lambda x: x[1] - x[0]):
            ps = [p for _, p in grp]
            runs.append((enc["orig"][ps[0]], enc["orig"][ps[-1]], ps))
    for g0, g1, first in runs:
        target, holes = (first[0], len(first)), all_gap_pos - set(first)
        with autocast(dev):
            cands = fill_gap(model, V, cfg, view, target, holes, rows, w, a.candidates, a.temperature, a.top_p,
                             a.seed, a.section)
        scored = []
        for k, (bars, score) in enumerate(cands):
            copy = any(ghash(b) in ctx_h and bar_feats(b, V)[2] for b in bars)
            if copy and not a.allow_copy:
                continue
            scored.append((score, k, bars))
        scored.sort(key=lambda x: -x[0])
        folder = out_root / f"{Path(a.song).stem}_gap_{g0 + 1}-{g1 + 1}"
        folder.mkdir(parents=True, exist_ok=True)
        report = []
        for rank, (score, k, bars) in enumerate(scored[:a.keep], 1):
            path = folder / f"cand_{rank:02d}.gp5"
            write_bars(a.song, enc, first, bars, V, path)
            report.append({"file": path.name, "candidate": k, "seed": a.seed, "logp": score})
        (folder / "report.json").write_text(json.dumps({"labels": a.labels, "neg": a.neg, "cfg": a.cfg,
                                                        "candidates": report}, indent=1), encoding="utf-8")
        log(f"measures {g0 + 1}-{g1 + 1}: {len(report)} of {len(cands)} candidates kept -> {folder}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    data = str(ROOT / "data_encoded" / "encoded")
    p = sub.add_parser("prepare", help="1) split, curate and encode the corpus")
    p.add_argument("--src", default=str(ROOT / "data"))
    p.add_argument("--out", default=data)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--limit", type=int, default=0, help="a random subset of N files (a quick look)")
    p.add_argument("--exclude", default="", help="drop files by label, e.g. album:*,src:own")
    p.add_argument("--frac", type=float, default=1.0,
                   help="keep this fraction of the files of each folder, before the train/test split")
    p.add_argument("--max-versions", type=int, default=3, help="versions of one song kept")
    p.add_argument("--seed", type=int, default=1234)
    p = sub.add_parser("hparams", help="the calculated batch sizes and learning rates")
    p.add_argument("--data", default=data)
    p.add_argument("--models", default="baseline,big,huge")
    p = sub.add_parser("train", help="2) train one model, 3) evaluating every --eval-every epochs")
    p.add_argument("--model", choices=list(MODELS), default="big")
    p.add_argument("--focal", type=float, default=0.0, help="focusing exponent (README §12); 0 is off")
    p.add_argument("--smooth", type=float, default=0.0, help="label smoothing; the Nov 5 grid found it hurts")
    p.add_argument("--ckpt", action="store_true",
                   help="recompute the activations instead of storing them: same arithmetic, ~a third slower")
    p.add_argument("--data", default=data)
    p.add_argument("--run", default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--batch-tokens", type=int, default=None, help="default: calculated (python riff.py hparams)")
    p.add_argument("--micro-tokens", type=int, default=0,
                   help="tokens per micro-batch (memory only; the optimiser step stays --batch-tokens)")
    p.add_argument("--lr", type=float, default=None, help="default: calculated")
    p.add_argument("--eval-every", type=float, default=0.5, help="epochs between evaluations")
    p.add_argument("--budget", type=float, default=None, help="stop after N tokens")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init", default=None, help="start from a checkpoint (style stage, album fine-tuning)")
    p.add_argument("--lora", type=int, default=0, help="LoRA rank (album fine-tuning, README §13)")
    p.add_argument("--focus", default="", help="label sampling weights, e.g. album:cs=20,src:early=0.3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eval-gaps", type=int, default=400)
    p.add_argument("--gen-gaps", type=int, default=8)
    p.add_argument("--tiny", action="store_true", help="a tiny version of the model (smoke tests)")
    p = sub.add_parser("eval", help="3) evaluate a checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default=data)
    p.add_argument("--split", default="test")
    p.add_argument("--gaps", type=int, default=2000)
    p.add_argument("--gen-gaps", type=int, default=32)
    p = sub.add_parser("infer", help="4) fill the gaps of a song")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--song", required=True)
    p.add_argument("--gap", default=None, help="measures to fill, e.g. 33-40 (default: every gap)")
    p.add_argument("--min-gap", type=int, default=5)
    p.add_argument("--labels", default="", help="e.g. genre:prog,genre:doom_stoner_sludge,album:cs,rel:answer")
    p.add_argument("--neg", default="src:early,rel:copy,rel:chug", help="labels to steer away from")
    p.add_argument("--cfg", type=float, default=1.5, help="guidance weight of every label")
    p.add_argument("--neg-cfg", type=float, default=0.5)
    p.add_argument("--section", choices=SECTIONS, default=None)
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--keep", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--allow-copy", action="store_true", help="keep candidates that repeat a bar of the song")
    p.add_argument("--out", default=None)
    p.add_argument("--cpu", action="store_true")
    p = sub.add_parser("grid", help="5) grid search: models x 3 batch sizes x 3 learning rates (x --aux weights)")
    p.add_argument("--data", default=data)
    p.add_argument("--models", default="baseline,huge",
                   help="big is left out: all 36 huge runs of the Nov 5 grid beat big's best (README §12)")
    p.add_argument("--budget", type=float, default=1e8, help="tokens per run")
    p.add_argument("--out", default=str(ROOT / "runs" / "grid"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eval-gaps", type=int, default=400)
    p.add_argument("--eval-every", type=float, default=0.125, help="epochs between evaluations (0: only at the end)")
    p.add_argument("--aux", default="", help="aux-loss weights to try on the pointer model, e.g. 0,0.1,0.2")
    p.add_argument("--smooth", default="", help="label smoothings to try on the models without aux heads, e.g. 0,0.05")
    p.add_argument("--focal", default="", help="focusing exponents to try instead of label smoothing, e.g. 0,1,2")
    p.add_argument("--ckpt", action="store_true",
                   help="recompute the activations instead of storing them: exactly the same arithmetic in "
                        "about a third more time, for a micro-batch that does not fit otherwise")
    p.add_argument("--batches", default="0.5,1,2", help="batch sizes as multiples of the calculated fit")
    p.add_argument("--lrs", default="0.707,1,1.414", help="learning rates as multiples of the calculated fit")
    p.add_argument("--micro-tokens", type=int, default=0,
                   help="tokens per micro-batch (memory only; the optimiser step stays the grid's batch size)")
    p.add_argument("--tiny", action="store_true")
    p = sub.add_parser("ngram", help="the bigram reference the lift is measured against (once per corpus)")
    p.add_argument("--data", default=data)
    p.add_argument("--split", default="val")
    p = sub.add_parser("segment", help="6) sections: train / eval / label the tabs without markers")
    p.add_argument("action", choices=("train", "eval", "rules", "fuse", "label"))
    p.add_argument("--data", default=data)
    p.add_argument("--run", default=str(ROOT / "runs" / "segment"))
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=10, help="validation peaks around epoch 9, then it memorises")
    p.add_argument("--lr", type=float, default=None, help="default: calculated")
    p.add_argument("--batch", type=int, default=None, help="bars per step (default: calculated)")
    p.add_argument("--micro", type=int, default=8192, help="padded bars per micro-batch (memory)")
    p.add_argument("--write", type=int, default=0, help="label: write N labelled copies to time the whole job")
    p.add_argument("--ssm", action="store_true", help="bar self-similarity as attention bias")
    p.add_argument("--named", action="store_true", help="'other' bars are unlabeled; only named sections are predicted")
    p.add_argument("--teacher", default=None, help="a seg.pt that pseudo-labels the unmarked tabs (noisy student)")
    p.add_argument("--pl-conf", type=float, default=0.6, help="keep pseudo-labels at least this confident")
    p.add_argument("--min-len", type=int, default=4, help="label, fuse: sections at least this many bars (intro excepted)")
    p.add_argument("--collapse", action="store_true",
                   help="eval/rules/fuse: a model trained with another section mapping, scored under the current one")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    {"prepare": cmd_prepare, "hparams": cmd_hparams, "train": cmd_train, "eval": cmd_eval, "infer": cmd_infer,
     "grid": cmd_grid, "ngram": cmd_ngram, "segment": cmd_segment}[a.cmd](a)


if __name__ == "__main__":
    mp.freeze_support()
    main()
