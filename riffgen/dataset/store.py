"""Processed song store: a few flat, memory-mapped numpy arrays plus one JSON line per song.

    <store>/meta.json      version, vocab hash, counts
    <store>/vocab.json     the vocabulary the token ids refer to
    <store>/songs.jsonl    per song: id, family, split, tracks/tunings, offsets, borders, segments
    <store>/bars.npy       one row per bar (length, time signature, tempo, section, flags, events)
    <store>/events.npy     one row per (onset, guitar): tick, slot, token span
    <store>/tokens.npy     uint16 token ids of all events

This is the "index per song" of §3.6: the sampler only slices these arrays, nothing is
parsed during training, and worker processes share the pages (mmap) on Linux and Windows.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from riffgen.dataset.song import SongData, TrackInfo
from riffgen.tokens import sections as sections_mod
from riffgen.tokens.vocab import Vocab

STORE_VERSION = 2
N_PARTNERS = 4

BAR_DTYPE = np.dtype([
    ("ticks", "i4"), ("ts_num", "i2"), ("ts_den", "i2"), ("tempo", "f4"), ("section", "i1"),
    ("triplet", "i1"), ("gap", "?"), ("ctx", "?"), ("drop", "?"), ("orig", "i4"),
    ("ev_start", "i4"), ("ev_count", "i4"), ("partners", "i4", (N_PARTNERS,)),
])
EVENT_DTYPE = np.dtype([("tick", "i4"), ("track", "i1"), ("tok_start", "i4"), ("tok_len", "i2")])


@dataclass
class SongView:
    """Everything the sampler needs about one song (store slice or in-memory song)."""
    meta: dict
    bars: np.ndarray
    events: np.ndarray
    tokens: np.ndarray

    @property
    def n_bars(self) -> int:
        return len(self.bars)

    @property
    def tracks(self) -> dict[int, TrackInfo]:
        return {int(k): TrackInfo.from_dict(v) for k, v in self.meta["tracks"].items()}

    def bar_events(self, b: int) -> np.ndarray:
        bar = self.bars[b]
        return self.events[bar["ev_start"]: bar["ev_start"] + bar["ev_count"]]

    def event_tokens(self, ev) -> np.ndarray:
        return self.tokens[ev["tok_start"]: ev["tok_start"] + ev["tok_len"]]

    def bar_has_notes(self, b: int, vocab: Vocab) -> bool:
        from riffgen.tokens.vocab import K
        for ev in self.bar_events(b):
            if ev["track"] >= 0 and np.any(vocab.kind[self.event_tokens(ev)] == K.NOTE):
                return True
        return False


def song_to_arrays(song: SongData, vocab: Vocab) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    labels = sections_mod.labels()
    bars = np.zeros(len(song.bars), BAR_DTYPE)
    events, tokens = [], []
    partners = song.analysis.get("partners")
    unknown = 0
    for i, bar in enumerate(song.bars):
        row = bars[i]
        row["ticks"] = bar.ticks
        ts = bar.ts or (4, 4)
        row["ts_num"], row["ts_den"] = ts
        row["tempo"] = bar.tempo
        row["section"] = labels.index(bar.section) if bar.section in labels else -1
        row["triplet"] = bar.triplet_feel
        row["gap"] = bar.gap
        row["ctx"] = bar.ctx
        row["drop"] = bar.drop
        row["orig"] = bar.orig
        row["ev_start"] = len(events)
        row["ev_count"] = len(bar.events)
        row["partners"] = partners[i] if partners is not None and len(partners) == len(song.bars) else -1
        for ev in bar.events:
            ids = vocab.ids(ev.tokens)
            unknown += sum(1 for x in ids if x == vocab.unk_id)
            events.append((ev.tick, ev.track, len(tokens), len(ids)))
            tokens.extend(ids)
    ev_arr = np.array(events, EVENT_DTYPE) if events else np.zeros(0, EVENT_DTYPE)
    tok_arr = np.array(tokens, np.uint16)
    meta = {
        "id": song.song_id, "family": song.family, "artist": song.artist, "title": song.title,
        "source": song.source, "kind": song.kind, "tempo": song.tempo,
        "tracks": {str(s): t.to_dict() for s, t in song.tracks.items()},
        "borders": song.analysis.get("borders", [0]),
        "segments": [{k: s[k] for k in ("start", "end", "kind", "period", "repeats", "section", "action")}
                     for s in song.analysis.get("segments", [])],
        "dup_tracks": [list(p) for p in song.analysis.get("dup_tracks", [])],
        "stats": {**{k: v for k, v in song.stats.items() if k != "gap_runs"}, "unknown_tokens": unknown},
        "warnings": song.warnings[:20],
    }
    return bars, ev_arr, tok_arr, meta


def view_from_song(song: SongData, vocab: Vocab) -> SongView:
    bars, events, tokens, meta = song_to_arrays(song, vocab)
    return SongView(meta=meta, bars=bars, events=events, tokens=tokens)


def song_from_view(view: SongView, vocab: Vocab) -> SongData:
    """Rebuild an in-memory song (token strings) from a store view."""
    from riffgen.dataset.song import Bar, Event
    labels = sections_mod.labels()
    bars = []
    for b in range(view.n_bars):
        row = view.bars[b]
        evs = [Event(int(ev["tick"]), int(ev["track"]), vocab.decode(view.event_tokens(ev))) for ev in view.bar_events(b)]
        sec = int(row["section"])
        bars.append(Bar(ticks=int(row["ticks"]), events=evs, ts=(int(row["ts_num"]), int(row["ts_den"])),
                        tempo=float(row["tempo"]), section=labels[sec] if sec >= 0 else None,
                        marker=labels[sec].capitalize() if sec >= 0 else None, orig=int(row["orig"]),
                        gap=bool(row["gap"]), ctx=bool(row["ctx"]), drop=bool(row["drop"])))
    m = view.meta
    return SongData(song_id=m["id"], tracks=view.tracks, bars=bars, tempo=float(m.get("tempo", 120)),
                    artist=m.get("artist", ""), title=m.get("title", ""), family=m.get("family", ""),
                    source=m.get("source", ""), kind=m.get("kind", ""))


class StoreWriter:
    def __init__(self, path: str | Path, vocab: Vocab, info: dict | None = None):
        self.path = Path(path)
        self.vocab = vocab
        self.info = info or {}
        self._bars, self._events, self._tokens, self._meta = [], [], [], []
        self._n_bars = self._n_events = self._n_tokens = 0

    def add(self, song: SongData, split: str = "train") -> None:
        bars, events, tokens, meta = song_to_arrays(song, self.vocab)
        meta.update(split=split, n_bars=len(bars), bar_off=self._n_bars, ev_off=self._n_events,
                    tok_off=self._n_tokens, n_events=len(events), n_tokens=len(tokens))
        self._bars.append(bars)
        self._events.append(events)
        self._tokens.append(tokens)
        self._meta.append(meta)
        self._n_bars += len(bars)
        self._n_events += len(events)
        self._n_tokens += len(tokens)
        if self._n_tokens >= 2**31 - 1:
            raise OverflowError("store too large for int32 offsets inside a song")

    def __len__(self) -> int:
        return len(self._meta)

    def close(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / "bars.npy", np.concatenate(self._bars) if self._bars else np.zeros(0, BAR_DTYPE))
        np.save(self.path / "events.npy", np.concatenate(self._events) if self._events else np.zeros(0, EVENT_DTYPE))
        np.save(self.path / "tokens.npy", np.concatenate(self._tokens) if self._tokens else np.zeros(0, np.uint16))
        with open(self.path / "songs.jsonl", "w", encoding="utf-8") as f:
            for m in self._meta:
                f.write(json.dumps(m, default=_json_default) + "\n")
        self.vocab.save(self.path / "vocab.json")
        meta = {"version": STORE_VERSION, "vocab_hash": self.vocab.hash, "songs": len(self._meta),
                "bars": self._n_bars, "events": self._n_events, "tokens": self._n_tokens,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), **self.info}
        with open(self.path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Store:
    def __init__(self, path: str | Path, mmap: bool = True):
        self.path = Path(path)
        if not (self.path / "meta.json").is_file():
            raise FileNotFoundError(f"no processed store at {self.path} (run riffgen.dataset.preprocess first)")
        with open(self.path / "meta.json", encoding="utf-8") as f:
            self.info = json.load(f)
        mode = "r" if mmap else None
        self.bars = np.load(self.path / "bars.npy", mmap_mode=mode)
        self.events = np.load(self.path / "events.npy", mmap_mode=mode)
        self.tokens = np.load(self.path / "tokens.npy", mmap_mode=mode)
        with open(self.path / "songs.jsonl", encoding="utf-8") as f:
            self.songs = [json.loads(line) for line in f if line.strip()]
        self.by_id = {s["id"]: i for i, s in enumerate(self.songs)}

    @property
    def vocab_hash(self) -> str:
        return self.info["vocab_hash"]

    def load_vocab(self) -> Vocab:
        return Vocab.load(self.path / "vocab.json")

    def __len__(self) -> int:
        return len(self.songs)

    def view(self, idx: int) -> SongView:
        m = self.songs[idx]
        bars = self.bars[m["bar_off"]: m["bar_off"] + m["n_bars"]]
        events = self.events[m["ev_off"]: m["ev_off"] + m["n_events"]]
        tokens = self.tokens[m["tok_off"]: m["tok_off"] + m["n_tokens"]]
        return SongView(meta=m, bars=bars, events=events, tokens=tokens)

    def indices(self, split: str | None = None, ids: set[str] | None = None) -> list[int]:
        return [i for i, s in enumerate(self.songs)
                if (split is None or s.get("split") == split) and (ids is None or s["id"] in ids)]
