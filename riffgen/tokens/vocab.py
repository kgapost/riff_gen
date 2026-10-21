"""Vocabulary: the DadaGP tokens first (their IDs never change), our tokens on top (D21).

The DadaGP part is the token list in DadaGP's ``_DadaGP_song_tokens_frequency.json``
(vendored in third_party/dadagp), in file order. We add:

* DadaGP-format tokens that are valid but missing from that list: every guitar note
  on strings 1-7 and frets 0-29 for all five guitar slots (7-string notes are 0.1 % of
  DadaGP, so many Drop A notes would otherwise be unknown), tempo values, rests.
* our tokens: specials, ``task:``, ``track:``, ``tuning:``, ``ts:``, ``section:``,
  ``bars:``, ``gap:``.

The vocabulary is built deterministically, saved next to the processed data, and copied
into every checkpoint. A store and a model must use the same vocabulary (checked by hash).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from pathlib import Path

import numpy as np

from riffgen import THIRD_PARTY_DIR
from riffgen.tokens import sections as sections_mod
from riffgen.tokens import tunings as tunings_mod

GUITARS = ("distorted0", "distorted1", "distorted2", "clean0", "clean1")
GUITAR_SLOT = {name: i for i, name in enumerate(GUITARS)}
SLOT_KIND = ("distorted", "distorted", "distorted", "clean", "clean")
OTHER_INSTRUMENTS = ("bass", "leads", "pads", "drums")
MAX_STRINGS = 7
MAX_FRET = 29
MAX_GAP = 64
TS_DENOMINATORS = (1, 2, 4, 8, 16, 32)
TEMPO_VALUES = tuple(range(10, 410, 10))

SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>", "<fim_prefix>", "<fim_suffix>", "<fim_middle>")


class K(IntEnum):
    """Token kinds."""
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    FIM_PREFIX = 4
    FIM_SUFFIX = 5
    FIM_MIDDLE = 6
    ARTIST = 7
    DOWNTUNE = 8
    TEMPO = 9
    START = 10
    END = 11
    TASK = 12
    TRACK = 13
    TUNING = 14
    BARS = 15
    NEW_MEASURE = 16
    MEASURE = 17
    TS = 18
    SECTION = 19
    GAP = 20
    NOTE = 21
    REST = 22
    NFX = 23
    PARAM = 24
    BFX = 25
    WAIT = 26
    OTHER_INST = 27


# ---------------------------------------------------------------------------
# DadaGP wait splitting (port of token_splitter.split_wait_token, MIT, Dadabots)

COMMON_WAITS = frozenset([1920, 16, 4096, 240, 3840, 512, 2048, 960, 64, 2880, 320, 256, 32, 16384, 480, 8,
                          1024, 640, 1, 360, 192, 128, 120, 40, 60, 2, 8192, 4, 1440, 80, 180, 720, 160])


def split_wait(n: int) -> list[int]:
    """Split a wait of n ticks the way DadaGP v1.1 does."""
    if n <= 0:
        return []
    if n in COMMON_WAITS:
        return [n]
    out = []
    for i in range(1, 20):
        p = 1 << i
        r = n % p
        n -= r
        if r > 0:
            out.append(r)
    return out[::-1]


# ---------------------------------------------------------------------------

_NOTE_RE = re.compile(r"^([a-z]+\d?):note:s(\d+):f(-?\d+)$")
_DRUM_RE = re.compile(r"^drums:note:(\d+)$")


@dataclass
class TokenInfo:
    kind: int
    inst: int = -1          # guitar slot 0-4 for NOTE/REST
    string: int = 0
    fret: int = 0
    wait: int = 0
    num: int = 0            # ts numerator
    den: int = 0            # ts denominator
    n: int = 0              # gap:N / bars:N
    tempo: int = 0          # tempo:N or bfx:tempo_change:N
    tempo_bfx: bool = False
    fx_params: bool = False  # nfx/bfx that may be followed by param: tokens
    section: int = -1
    tuning: int = -1


def parse_token(tok: str) -> TokenInfo:
    if tok in SPECIALS:
        return TokenInfo(kind=[K.PAD, K.BOS, K.EOS, K.UNK, K.FIM_PREFIX, K.FIM_SUFFIX, K.FIM_MIDDLE][SPECIALS.index(tok)])
    if tok == "new_measure":
        return TokenInfo(K.NEW_MEASURE)
    if tok == "start":
        return TokenInfo(K.START)
    if tok == "end":
        return TokenInfo(K.END)
    head, _, rest = tok.partition(":")
    if head == "wait":
        return TokenInfo(K.WAIT, wait=int(rest))
    if head == "nfx":
        return TokenInfo(K.NFX, fx_params=rest.split(":")[0] in ("bend", "grace", "trill"))
    if head == "bfx":
        name = rest.split(":")[0]
        info = TokenInfo(K.BFX, fx_params=(name == "tremolo_bar"))
        if name in ("tempo_change", "tempo_interpolation"):
            info.tempo_bfx = True
            if name == "tempo_change":
                info.tempo = int(rest.split(":")[1])
        return info
    if head == "param":
        return TokenInfo(K.PARAM)
    if head == "measure":
        return TokenInfo(K.MEASURE)
    if head == "artist":
        return TokenInfo(K.ARTIST)
    if head == "downtune":
        return TokenInfo(K.DOWNTUNE)
    if head == "tempo":
        return TokenInfo(K.TEMPO, tempo=int(rest))
    if head == "task":
        return TokenInfo(K.TASK)
    if head == "track":
        return TokenInfo(K.TRACK, inst=GUITAR_SLOT[rest])
    if head == "tuning":
        return TokenInfo(K.TUNING)
    if head == "bars":
        return TokenInfo(K.BARS, n=int(rest))
    if head == "gap":
        return TokenInfo(K.GAP, n=int(rest))
    if head == "ts":
        num, den = rest.split("/")
        return TokenInfo(K.TS, num=int(num), den=int(den))
    if head == "section":
        return TokenInfo(K.SECTION, section=sections_mod.labels().index(rest))
    if head in GUITARS:
        if rest == "rest":
            return TokenInfo(K.REST, inst=GUITAR_SLOT[head])
        m = _NOTE_RE.match(tok)
        if m:
            return TokenInfo(K.NOTE, inst=GUITAR_SLOT[head], string=int(m.group(2)), fret=int(m.group(3)))
    if head in OTHER_INSTRUMENTS:
        return TokenInfo(K.OTHER_INST)
    return TokenInfo(K.UNK)


def note_token(slot: int, string: int, fret: int) -> str:
    return f"{GUITARS[slot]}:note:s{string}:f{fret}"


def rest_token(slot: int) -> str:
    return f"{GUITARS[slot]}:rest"


def ts_token(num: int, den: int) -> str:
    return f"ts:{num}/{den}"


def dadagp_token_list() -> list[str]:
    path = THIRD_PARTY_DIR / "dadagp" / "_DadaGP_song_tokens_frequency.json"
    with open(path, encoding="utf-8") as f:
        return list(json.load(f).keys())


def build_token_list() -> list[str]:
    toks: list[str] = []
    seen: set[str] = set()

    def add(t: str) -> None:
        if t not in seen:
            seen.add(t)
            toks.append(t)

    for t in dadagp_token_list():
        add(t)
    # valid DadaGP-format tokens missing from the frequency list
    for slot in range(len(GUITARS)):
        add(rest_token(slot))
        for s in range(1, MAX_STRINGS + 1):
            for f in range(0, MAX_FRET + 1):
                add(note_token(slot, s, f))
    for v in TEMPO_VALUES:
        add(f"tempo:{v}")
        add(f"bfx:tempo_change:{v}")
    for w in range(0, 15):
        add(f"wait:{1 << w}")
    for t in ("start", "end", "new_measure"):
        add(t)
    # our tokens
    for t in SPECIALS:
        add(t)
    add("task:song")
    add("task:infill")
    for g in GUITARS:
        add(f"track:{g}")
    for name in tunings_mod.all_names():
        add(f"tuning:{name}")
    for den in TS_DENOMINATORS:
        for num in range(1, 33):
            add(ts_token(num, den))
    for lab in sections_mod.labels():
        add(f"section:{lab}")
    for n in range(1, MAX_GAP + 1):
        add(f"bars:{n}")
    for n in range(1, MAX_GAP + 1):
        add(f"gap:{n}")
    return toks


class Vocab:
    def __init__(self, tokens: list[str]):
        self.tokens = list(tokens)
        self.index = {t: i for i, t in enumerate(self.tokens)}
        if len(self.index) != len(self.tokens):
            raise ValueError("duplicate tokens in vocabulary")
        self.pad_id = self.index["<pad>"]
        self.bos_id = self.index["<bos>"]
        self.eos_id = self.index["<eos>"]
        self.unk_id = self.index["<unk>"]
        self.fim_prefix_id = self.index["<fim_prefix>"]
        self.fim_suffix_id = self.index["<fim_suffix>"]
        self.fim_middle_id = self.index["<fim_middle>"]
        self.new_measure_id = self.index["new_measure"]
        self.start_id = self.index["start"]
        self.end_id = self.index["end"]
        self.task_song_id = self.index["task:song"]
        self.task_infill_id = self.index["task:infill"]
        self._build_table()

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls) -> "Vocab":
        return cls(build_token_list())

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["tokens"] if isinstance(data, dict) else data)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"hash": self.hash, "size": len(self), "tokens": self.tokens}, f, indent=0)

    @property
    def hash(self) -> str:
        return hashlib.sha1("\n".join(self.tokens).encode("utf-8")).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.tokens)

    # -- lookups -----------------------------------------------------------
    def id(self, tok: str) -> int:
        return self.index.get(tok, self.unk_id)

    def ids(self, toks) -> list[int]:
        get = self.index.get
        unk = self.unk_id
        return [get(t, unk) for t in toks]

    def decode(self, ids) -> list[str]:
        return [self.tokens[int(i)] for i in ids]

    def _build_table(self) -> None:
        V = len(self.tokens)
        self.kind = np.zeros(V, np.int8)
        self.inst = np.full(V, -1, np.int8)
        self.string = np.zeros(V, np.int8)
        self.fret = np.zeros(V, np.int16)
        self.wait = np.zeros(V, np.int32)
        self.ts_num = np.zeros(V, np.int16)
        self.ts_den = np.zeros(V, np.int16)
        self.n = np.zeros(V, np.int16)
        self.tempo = np.zeros(V, np.int16)
        self.tempo_bfx = np.zeros(V, bool)
        self.fx_params = np.zeros(V, bool)
        self.section = np.full(V, -1, np.int8)
        self.note_id = np.full((len(GUITARS), MAX_STRINGS + 1, MAX_FRET + 1), -1, np.int32)
        self.rest_id = np.full(len(GUITARS), -1, np.int32)
        self.track_id = np.full(len(GUITARS), -1, np.int32)
        self.gap_ids: dict[int, int] = {}
        self.bars_ids: dict[int, int] = {}
        self.ts_ids: dict[tuple[int, int], int] = {}
        self.section_ids: dict[str, int] = {}
        self.tuning_ids: dict[str, int] = {}
        self.tempo_ids: dict[int, int] = {}
        for i, tok in enumerate(self.tokens):
            info = parse_token(tok)
            self.kind[i] = info.kind
            self.inst[i] = info.inst
            self.string[i] = info.string
            self.fret[i] = info.fret
            self.wait[i] = info.wait
            self.ts_num[i] = info.num
            self.ts_den[i] = info.den
            self.n[i] = info.n
            self.tempo[i] = info.tempo
            self.tempo_bfx[i] = info.tempo_bfx
            self.fx_params[i] = info.fx_params
            self.section[i] = info.section
            if info.kind == K.NOTE and 0 <= info.fret <= MAX_FRET and info.string <= MAX_STRINGS:
                self.note_id[info.inst, info.string, info.fret] = i
            elif info.kind == K.REST:
                self.rest_id[info.inst] = i
            elif info.kind == K.TRACK:
                self.track_id[info.inst] = i
            elif info.kind == K.GAP:
                self.gap_ids[info.n] = i
            elif info.kind == K.BARS:
                self.bars_ids[info.n] = i
            elif info.kind == K.TS:
                self.ts_ids[(info.num, info.den)] = i
            elif info.kind == K.SECTION:
                self.section_ids[tok.split(":", 1)[1]] = i
            elif info.kind == K.TUNING:
                self.tuning_ids[tok.split(":", 1)[1]] = i
            elif info.kind == K.TEMPO:
                self.tempo_ids[info.tempo] = i
        self.wait_values = sorted({int(w) for w in self.wait[self.kind == K.WAIT]})
        self.wait_ids = {int(self.wait[i]): i for i in np.nonzero(self.kind == K.WAIT)[0]}

    # -- helpers used by the serializer and the generator --------------------
    def wait_tokens(self, ticks: int) -> list[int]:
        return [self.wait_ids[w] for w in split_wait(ticks)]

    def tempo_token(self, bpm: float) -> int:
        v = int(min(max(round(bpm / 10) * 10, TEMPO_VALUES[0]), TEMPO_VALUES[-1]))
        return self.tempo_ids[v]

    def ts_token(self, num: int, den: int) -> int:
        return self.ts_ids.get((int(num), int(den)), self.unk_id)

    def gap_tokens(self, n_bars: int) -> list[int]:
        out = []
        while n_bars > 0:
            k = min(n_bars, MAX_GAP)
            out.append(self.gap_ids[k])
            n_bars -= k
        return out

    def tuning_token(self, name: str) -> int:
        return self.tuning_ids.get(name, self.tuning_ids["other"])


@lru_cache(maxsize=4)
def default_vocab() -> Vocab:
    return Vocab.build()


def swap_table(vocab: Vocab, slot_map: dict[int, int]) -> np.ndarray:
    """Token-id remap for renaming guitar slots (``distorted0:note:...`` -> ``distorted1:note:...``)."""
    table = np.arange(len(vocab), dtype=np.int32)
    for old, new in slot_map.items():
        if old == new:
            continue
        sel = np.nonzero(vocab.note_id[old] >= 0)
        for s, f in zip(*sel):
            dst = vocab.note_id[new, s, f]
            table[vocab.note_id[old, s, f]] = dst if dst >= 0 else vocab.unk_id
        table[vocab.rest_id[old]] = vocab.rest_id[new]
    return table
