"""Plagiarism check (README §7.6, D38-D40, D48).

Every guitar part becomes note events (onset, length, string/fret, pitches). Pieces --
n-grams of 8 events, single bars and pairs of bars -- get fingerprints on three levels:

    L1 exact       string + fret + rhythm                  direct copies
    L2 transposed  intervals + chord shape + rhythm        same riff in another key/position
    L3 rhythm      onsets only                             info only, never a reason to reject

Fingerprints are looked up in an SQLite index of the album's training set (own songs,
favourites with their copies, influences, and the DadaGP songs mixed into fine-tuning).
A piece found in >= 3 song families and >= 1 % of the families is a common idiom and is
ignored (D39); copies of one song are one family.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from riffgen import CONFIG_DIR
from riffgen.analysis.notes import bar_onsets
from riffgen.dataset.song import SongData
from riffgen.utils.config import load_config

KIND_NGRAM, KIND_BAR1, KIND_BAR2 = 0, 1, 2
LEVELS = (1, 2, 3)


@dataclass
class NoteEvent:
    tick: int          # absolute onset (ticks from song start)
    bar: int
    in_bar: int
    dur: int
    tab: tuple         # ((string, fret), ...)
    pitches: tuple


def track_events(song: SongData, slot: int, bars: range | None = None) -> list[NoteEvent]:
    tr = song.tracks[slot]
    out = []
    start = 0
    for b, bar in enumerate(song.bars):
        if bars is None or b in bars:
            for on in bar_onsets(bar, slot, tr.tuning):
                if not on.notes or not on.is_attack:
                    continue
                notes = [n for n in on.notes if not n.dead]
                if not notes:
                    continue
                out.append(NoteEvent(start + on.tick, b, on.tick, on.dur,
                                     tuple(sorted((n.string, n.fret) for n in notes)),
                                     tuple(sorted(n.pitch for n in notes))))
        start += bar.ticks
    return out


def _h(key) -> int:
    return int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "little", signed=True)


def _q(t: int, quantum: int) -> int:
    return int(round(t / quantum))


def _shape(ev: NoteEvent) -> tuple:
    lo = ev.pitches[0]
    return tuple(p - lo for p in ev.pitches)


def piece_keys(evs: list[NoteEvent], level: int, quantum: int, bar_mode: bool) -> tuple:
    """Bar pieces keep the position inside the bar; n-grams are relative to their first event."""
    if not evs:
        return ()
    first = evs[0]
    if bar_mode:
        rel = [_q(e.in_bar, quantum) + 100000 * (e.bar - first.bar) for e in evs]
    else:
        rel = [_q(e.tick - first.tick, quantum) for e in evs]
    if level == 1:
        return tuple((r, _q(e.dur, quantum), e.tab) for r, e in zip(rel, evs))
    if level == 2:
        lo0 = first.pitches[0]
        return tuple((r, _q(e.dur, quantum), e.pitches[0] - lo0, _shape(e)) for r, e in zip(rel, evs))
    return tuple(rel)


def fingerprints(evs: list[NoteEvent], n: int = 8, quantum: int = 40) -> list[tuple[int, int, int, int, int]]:
    """(level, kind, hash, position, bar) for every piece of one guitar part."""
    out = []
    for i in range(0, max(0, len(evs) - n + 1)):
        win = evs[i:i + n]
        for lv in LEVELS:
            out.append((lv, KIND_NGRAM, _h((lv, piece_keys(win, lv, quantum, False))), i, win[0].bar))
    by_bar = defaultdict(list)
    for e in evs:
        by_bar[e.bar].append(e)
    bars = sorted(by_bar)
    for b in bars:
        if len(by_bar[b]) < 2:
            continue
        for lv in LEVELS:
            out.append((lv, KIND_BAR1, _h((lv, "b1", piece_keys(by_bar[b], lv, quantum, True))), -1, b))
        if b + 1 in by_bar:
            pair = by_bar[b] + by_bar[b + 1]
            for lv in LEVELS:
                out.append((lv, KIND_BAR2, _h((lv, "b2", piece_keys(pair, lv, quantum, True))), -1, b))
    return out


class Index:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute("CREATE TABLE IF NOT EXISTS fp (level INTEGER, kind INTEGER, h INTEGER, family TEXT, "
                        "song TEXT, track INTEGER, pos INTEGER, bar INTEGER)")
        self.db.execute("CREATE TABLE IF NOT EXISTS families (family TEXT PRIMARY KEY, own INTEGER, name TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS idioms (level INTEGER, kind INTEGER, h INTEGER)")
        self.db.execute("CREATE TABLE IF NOT EXISTS songs (song TEXT PRIMARY KEY, orig_bars TEXT)")

    def add_song(self, song: SongData, own: bool, cfg) -> int:
        n = int(cfg.get("ngram", 8))
        quantum = int(cfg.get("onset_quantum", 40))
        rows = []
        for slot in song.tracks:
            for lv, kind, h, pos, bar in fingerprints(track_events(song, slot), n, quantum):
                rows.append((lv, kind, h, song.family, song.song_id, slot, pos, bar))
        self.db.executemany("INSERT INTO fp VALUES (?,?,?,?,?,?,?,?)", rows)
        name = song.source or song.song_id
        self.db.execute("INSERT OR REPLACE INTO families VALUES (?,?,?)", (song.family, int(own), name))
        self.db.execute("INSERT OR REPLACE INTO songs VALUES (?,?)",
                        (song.song_id, ",".join(str(b.orig + 1) for b in song.bars)))
        return len(rows)

    def finalize(self, cfg) -> None:
        idiom = cfg.get("idiom", {}) or {}
        n_fam = self.db.execute("SELECT COUNT(*) FROM families").fetchone()[0]
        min_f = max(int(idiom.get("min_families", 3)), int(round(float(idiom.get("min_share", 0.01)) * n_fam)))
        self.db.execute("DELETE FROM idioms")
        self.db.execute("INSERT INTO idioms SELECT level, kind, h FROM fp GROUP BY level, kind, h "
                        "HAVING COUNT(DISTINCT family) >= ?", (min_f,))
        self.db.execute("CREATE INDEX IF NOT EXISTS fp_lookup ON fp(level, kind, h)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idiom_lookup ON idioms(level, kind, h)")
        self.db.commit()

    def lookup(self, level: int, kind: int, hashes: list[int]) -> list[tuple]:
        out = []
        for i in range(0, len(hashes), 500):
            chunk = hashes[i:i + 500]
            q = ",".join("?" * len(chunk))
            out += self.db.execute(
                f"SELECT h, family, song, track, pos, bar FROM fp WHERE level=? AND kind=? AND h IN ({q}) "
                f"AND h NOT IN (SELECT h FROM idioms WHERE level=? AND kind=?)",
                (level, kind, *chunk, level, kind)).fetchall()
        return out

    def family_info(self) -> dict[str, tuple[bool, str]]:
        return {f: (bool(o), n) for f, o, n in self.db.execute("SELECT family, own, name FROM families")}

    def orig_bar(self, song: str, bar: int) -> int:
        row = self.db.execute("SELECT orig_bars FROM songs WHERE song=?", (song,)).fetchone()
        if not row:
            return bar + 1
        bars = row[0].split(",")
        return int(bars[bar]) if 0 <= bar < len(bars) else bar + 1

    def close(self) -> None:
        self.db.close()


@dataclass
class Match:
    verdict: str = "GREEN"
    self_quote: bool = False
    level: str = ""
    source: str = ""
    source_bars: str = ""
    cand_bars: str = ""
    run_bars: int = 0
    coverage: float = 0.0
    rhythm_source: str = ""
    details: list = field(default_factory=list)

    def label(self) -> str:
        if self.verdict == "GREEN":
            return "GREEN"
        return f"SELF-QUOTE ({self.verdict})" if self.self_quote else self.verdict


def check_song(index: Index, song: SongData, bars: range | None, cfg=None) -> Match:
    """Check the guitar parts of ``song`` (only ``bars`` if given) against the index."""
    cfg = cfg or load_config(CONFIG_DIR / "plagiarism.yaml")
    n = int(cfg.get("ngram", 8))
    quantum = int(cfg.get("onset_quantum", 40))
    fam_info = index.family_info()
    best = Match()
    ticks = [b.ticks for b in song.bars]
    bar_len = (sum(ticks[b] for b in bars) / max(1, len(bars))) if bars else (sum(ticks) / max(1, len(ticks)))
    # per family and level: matched n-gram starts per guitar, and matched bar pieces
    ngram_hits: dict[tuple[str, int], dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    hits: dict[str, list[tuple[int, float, str, int, int]]] = defaultdict(list)  # fam -> (cand bar, bars, src song, src bar, level)
    rhythm_fams: dict[str, int] = defaultdict(int)
    events: dict[int, list[NoteEvent]] = {}
    total_events = 0
    for slot in song.tracks:
        evs = track_events(song, slot, bars)
        events[slot] = evs
        total_events += len(evs)
        fps = fingerprints(evs, n, quantum)
        for lv in LEVELS:
            for kind in (KIND_NGRAM, KIND_BAR1, KIND_BAR2):
                by_h = defaultdict(list)
                for (l, k, h, pos, bar) in fps:
                    if l == lv and k == kind:
                        by_h[h].append((pos, bar))
                if not by_h:
                    continue
                for h, fam, src_song, _track, _src_pos, src_bar in index.lookup(lv, kind, list(by_h)):
                    for pos, bar in by_h[h]:
                        if lv == 3:
                            rhythm_fams[fam] += 1
                        elif kind == KIND_NGRAM:
                            ngram_hits[(fam, lv)][slot].add(pos)
                            hits[fam].append((bar, 0.0, src_song, src_bar, lv))
                        else:
                            hits[fam].append((bar, 2.0 if kind == KIND_BAR2 else 1.0, src_song, src_bar, lv))
    # n-gram runs: consecutive matched starts -> a stretch of music; its length in bars
    runs: dict[str, float] = defaultdict(float)
    coverage: dict[str, float] = defaultdict(float)
    for (fam, lv), per_slot in ngram_hits.items():
        covered = 0
        for slot, starts in per_slot.items():
            evs = events[slot]
            idx = set()
            for p in sorted(starts):
                idx.update(range(p, min(p + n, len(evs))))
            covered += len(idx)
            run_start = prev = None
            for p in sorted(starts) + [None]:
                if p is not None and prev is not None and p == prev + 1:
                    prev = p
                    continue
                if run_start is not None:
                    last = evs[min(prev + n - 1, len(evs) - 1)]
                    span = last.tick + last.dur - evs[run_start].tick
                    runs[fam] = max(runs[fam], span / max(bar_len, 1))
                run_start = prev = p
        coverage[fam] = max(coverage[fam], covered / max(1, total_events))
    red, yellow = cfg.get("red", {}), cfg.get("yellow", {})
    for fam, fam_hits in hits.items():
        run = max([runs.get(fam, 0.0)] + [b for _, b, *_ in fam_hits])
        cov = coverage.get(fam, 0.0)
        lvl = min(lv for *_, lv in fam_hits)
        if run >= int(red.get("run_bars", 2)) or cov >= float(red.get("coverage", 0.5)):
            verdict = "RED"
        elif run >= int(yellow.get("run_bars", 1)) or cov >= float(yellow.get("coverage", 0.25)):
            verdict = "YELLOW"
        else:
            continue
        rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}
        if rank[verdict] > rank[best.verdict] or (verdict == best.verdict and run > best.run_bars):
            first = min(fam_hits, key=lambda x: x[0])
            width = max(1, int(round(run)))
            own, name = fam_info.get(fam, (False, fam))
            src_b = index.orig_bar(first[2], first[3])
            best = Match(verdict=verdict, self_quote=own, level=f"L{lvl}", source=name,
                         source_bars=f"{src_b}-{src_b + width - 1}",
                         cand_bars=f"{first[0] + 1}-{first[0] + width}", run_bars=round(run, 2), coverage=round(cov, 3),
                         details=sorted({(h[0] + 1, h[2]) for h in fam_hits})[:10])
    if rhythm_fams:
        fam = max(rhythm_fams, key=rhythm_fams.get)
        best.rhythm_source = fam_info.get(fam, (False, fam))[1]
    return best


def build_album_index(acfg, out: str | Path, extra_folders: list[str] | None = None) -> str:
    """D40: the album's training set + the DadaGP songs mixed into fine-tuning (D48)."""
    import json

    from riffgen.dataset.store import Store, song_from_view
    from riffgen.train.common import repo_path
    cfg = load_config(CONFIG_DIR / "plagiarism.yaml")
    out = Path(out)
    if out.exists():
        out.unlink()
    idx = Index(out)
    store = Store(repo_path(acfg.get_path("data.store", f"data/processed/albums/{acfg.album_id}")))
    vocab = store.load_vocab()
    n_songs = n_fp = 0
    for i, meta in enumerate(store.songs):
        song = song_from_view(store.view(i), vocab)
        n_fp += idx.add_song(song, own=meta.get("set") == "own", cfg=cfg)
        n_songs += 1
    sel_path = out.parent / "dadagp_mix.json"
    if sel_path.is_file():
        sel = json.loads(sel_path.read_text())
        dpath = repo_path((acfg.get("dadagp_mix") or {}).get("store", "data/processed/dadagp"))
        if (dpath / "meta.json").is_file():
            dstore = Store(dpath)
            dv = dstore.load_vocab()
            for sid in sel["ids"]:
                if sid in dstore.by_id:
                    n_fp += idx.add_song(song_from_view(dstore.view(dstore.by_id[sid]), dv), own=False, cfg=cfg)
                    n_songs += 1
    for folder in extra_folders or []:
        n_songs += add_folder(idx, folder, cfg)
    idx.finalize(cfg)
    idx.close()
    return f"{out} ({n_songs} songs, {n_fp} fingerprints)"


def add_folder(idx: Index, folder: str | Path, cfg) -> int:
    """--extra-reference: any folder of GP files (any tuning)."""
    from riffgen.dataset import pipeline as pl
    n = 0
    folder = Path(folder)
    for p in sorted(x for ext in ("*.gp3", "*.gp4", "*.gp5") for x in folder.rglob(ext)):
        try:
            s = pl.process_gp_file(p, folder, pl.PipelineConfig(analyze=False), kind="reference", family_group="ref")
            s.family = pl.family_key(s.artist, p.stem)
            idx.add_song(s, own=False, cfg=cfg)
            n += 1
        except Exception:
            continue
    return n
