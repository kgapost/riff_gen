"""The preprocessing pipeline of §3.6, for one song.

    GP3/4/5 file or DadaGP token file
      1. load (PyGuitarPro 0.6)                        D26
      2. tuning: own sets must be Drop A; DadaGP keeps its tuning   D10, D11
      3. guitars only (max 3 distorted + 2 clean)      D13
      4. unroll repeats                                D14
      5. section markers -> section labels             D25
      6. >= 4 empty guitar bars -> gap placeholder     D15
      7. riffs/sections, simple-riff filter            §3.7
      (8. normalized GP5 export and 9./10. tokens happen in preprocess.py / store.py)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from riffgen.analysis import filters as filters_mod
from riffgen.analysis import riffs as riffs_mod
from riffgen.analysis.notes import song_onsets
from riffgen.dataset.song import (SongData, assign_tempi, mark_gaps, parse_dadagp, raw_to_song, read_token_file,
                                  ts_from_ticks, ts_ticks)
from riffgen.dataset.unroll import unroll
from riffgen.gp import reader as gp_reader
from riffgen.tokens import sections as sections_mod
from riffgen.tokens import tunings as tunings_mod

GP_EXTS = (".gp5", ".gp4", ".gp3")


@dataclass
class PipelineConfig:
    min_gap_bars: int = 4
    require_tuning: str | None = None           # "drop_a7" for album sets and input songs
    use_gp_files: bool = True                   # DadaGP: read the GP file next to the tokens
    filters: filters_mod.FilterConfig = field(default_factory=filters_mod.FilterConfig)
    analyze: bool = True
    overrides: dict | None = None

    @classmethod
    def from_cfg(cls, cfg: dict | None, **kw) -> "PipelineConfig":
        cfg = cfg or {}
        out = cls(min_gap_bars=int(cfg.get("min_gap_bars", 4)), filters=filters_mod.FilterConfig.from_cfg(cfg.get("filters")),
                  overrides=cfg.get("filters"))
        for k, v in kw.items():
            setattr(out, k, v)
        return out


# ---------------------------------------------------------------------------
# song identity

_COPY_RE = re.compile(r"__v\d+$")


def normalize_title(title: str) -> str:
    t = title.lower()
    for _ in range(3):
        t = re.sub(r"\.(gp[345x]?|tokens\.txt|txt|pygp)$", "", t)
    t = _COPY_RE.sub("", t)
    t = re.sub(r"\s*\(\d+\)$", "", t)
    t = re.sub(r"\((live|acoustic|demo|remaster\w*|cover|tab|guitar pro|gp\d?|ver\w*|version)[^)]*\)", "", t)
    t = re.sub(r"\b(ver|version|v)\s*\d+\b", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return t


def family_key(artist: str, title: str) -> str:
    a = re.sub(r"[^a-z0-9]+", " ", (artist or "").lower()).strip()
    return f"{a}|{normalize_title(title)}"


def find_gp_for_tokens(token_path: Path) -> Path | None:
    name = token_path.name
    if name.endswith(".tokens.txt"):
        base = token_path.with_name(name[: -len(".tokens.txt")])
        if base.suffix.lower() in GP_EXTS and base.is_file():
            return base
        for ext in GP_EXTS:
            cand = base.with_suffix(ext)
            if cand.is_file():
                return cand
    return None


# ---------------------------------------------------------------------------

def apply_meta(song: SongData, meta: gp_reader.GPMeta | None) -> None:
    if meta is None:
        return
    if meta.n_measures != len(song.bars):
        song.warn(f"GP file has {meta.n_measures} measures, tokens have {len(song.bars)}; GP metadata ignored")
        return
    for bar, ts, marker in zip(song.bars, meta.ts, meta.markers):
        bar.ts = ts
        if marker and marker.strip():
            bar.marker = marker.strip()
            bar.section = sections_mod.normalize(marker)
    song.tempo = meta.tempo or song.tempo
    for slot, name in meta.slot_names.items():
        if slot in song.tracks:
            song.tracks[slot].name = name
            song.tracks[slot].program = meta.slot_programs.get(slot, -1)
            song.tracks[slot].orig_index = meta.slot_track_index.get(slot, -1)
    if meta.title and not song.title:
        song.title = meta.title


def fix_bar_lengths(song: SongData) -> None:
    """Bar lengths come from the waits; the last bar only holds its last event's length."""
    bars = song.bars
    for i, bar in enumerate(bars):
        last = i == len(bars) - 1
        if bar.ts is not None:
            nominal = ts_ticks(bar.ts)
            if last or bar.ticks <= 0:
                bar.ticks = nominal
            elif bar.ticks != nominal:
                # overfull/underfull header vs content: trust the timeline, fix the signature
                bar.ts = ts_from_ticks(bar.ticks)
        elif last and i > 0:
            max_tick = max((e.tick for e in bar.events), default=0)
            bar.ticks = max(bars[i - 1].ticks, max_tick + 1) if bar.ticks < bars[i - 1].ticks else bar.ticks
        if bar.ts is None:
            bar.ts = ts_from_ticks(bar.ticks)
    keep = [b for b in bars if b.ticks > 0]
    if len(keep) != len(bars):
        song.warn(f"{len(bars) - len(keep)} zero-length bars removed")
    song.bars = keep


def finalize(song: SongData, meta: gp_reader.GPMeta | None, cfg: PipelineConfig) -> SongData:
    apply_meta(song, meta)
    fix_bar_lengths(song)
    song.stats["written_bars"] = len(song.bars)
    song.bars = unroll(song.bars, song.warnings)
    song.stats["bars"] = len(song.bars)
    assign_tempi(song)
    gaps = mark_gaps(song, cfg.min_gap_bars)
    song.stats["gap_runs"] = gaps
    if cfg.analyze:
        analyze(song, cfg)
    return song


def analyze(song: SongData, cfg: PipelineConfig, rel_path: str | None = None) -> None:
    if not song.bars:
        return
    onsets = song_onsets(song)
    S, _ = riffs_mod.song_ssm(song, onsets)
    borders = riffs_mod.find_borders(song, S)
    segments = riffs_mod.make_segments(song, S, borders)
    filters_mod.apply_filters(song, segments, onsets, cfg.filters)
    notes = filters_mod.apply_overrides(song, segments, rel_path or song.source, cfg.overrides)
    if notes:
        song.warnings.extend(notes)
    for seg in segments:
        for i in range(seg.start, seg.end):
            bar = song.bars[i]
            bar.ctx = seg.action == "context"
            bar.drop = seg.action == "drop"
    song.analysis = {
        "borders": [s.start for s in segments if s.kind == "music"],
        "segments": [s.to_dict() for s in segments],
        "partners": riffs_mod.bar_partners(S),
        "dup_tracks": riffs_mod.duplicate_tracks(song),
    }


# ---------------------------------------------------------------------------
# entry points

def process_dadagp_file(token_path: str | Path, root: str | Path, cfg: PipelineConfig) -> SongData:
    token_path = Path(token_path)
    rel = token_path.relative_to(root).as_posix() if root else token_path.name
    tokens = read_token_file(token_path)
    raw = parse_dadagp(tokens)
    meta = None
    gp_path = find_gp_for_tokens(token_path) if cfg.use_gp_files else None
    if gp_path is not None:
        try:
            meta = gp_reader.read_gp_meta(gp_path)
        except gp_reader.SkipSong:
            meta = None
    slot_tunings = meta.slot_tunings if meta else None
    song_id = rel[: -len(".tokens.txt")] if rel.endswith(".tokens.txt") else rel
    song = raw_to_song(raw, song_id=song_id, slot_tunings=slot_tunings, dadagp_normalised=True, kind="dadagp")
    parts = Path(song_id).parts
    artist = parts[-2] if len(parts) >= 2 else raw.artist.replace("artist:", "")
    song.artist = artist
    song.title = Path(song_id).name
    song.family = family_key(artist, song.title)
    song.source = str(token_path)
    if not song.tracks:
        raise gp_reader.SkipSong("no guitar track")
    if cfg.require_tuning:
        for tr in song.tracks.values():
            if tr.tuning_name != cfg.require_tuning:
                raise gp_reader.SkipSong(f"tuning {tr.tuning_name} != {cfg.require_tuning}")
    return finalize(song, meta, cfg)


def own_family(rel_path: str, group: str = "own") -> str:
    stem = Path(rel_path).name
    for ext in GP_EXTS:
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    stem = _COPY_RE.sub("", stem)
    return f"{group}|{normalize_title(stem)}"


def process_gp_file(gp_path: str | Path, root: str | Path | None, cfg: PipelineConfig, kind: str = "own",
                    family_group: str = "own") -> SongData:
    gp_path = Path(gp_path)
    rel = gp_path.relative_to(root).as_posix() if root else gp_path.name
    enc = gp_reader.encode_own_gp(gp_path, require_tuning=cfg.require_tuning)
    raw = parse_dadagp(enc.tokens)
    song = raw_to_song(raw, song_id=rel, slot_tunings=enc.slot_tunings, dadagp_normalised=False, kind=kind)
    song.warnings.extend(enc.warnings)
    if enc.removed_tracks:
        song.stats["removed_tracks"] = enc.removed_tracks
    song.title = enc.meta.title or gp_path.stem
    song.artist = enc.meta.artist or "unknown"
    song.family = own_family(rel, family_group)
    song.source = str(gp_path)
    if not song.tracks:
        raise gp_reader.SkipSong("no guitar notes")
    return finalize(song, enc.meta, cfg)


def song_tokens_per_bar(song: SongData) -> list[int]:
    """Rough token count per bar (for D27 statistics), before serialization extras."""
    out = []
    for b in song.bars:
        n = 1 + sum(len(e.tokens) for e in b.events)
        ticks = sorted({e.tick for e in b.events})
        n += max(1, len(ticks))  # waits
        out.append(n)
    return out


def describe_song(song: SongData) -> dict:
    tpb = song_tokens_per_bar(song)
    return {"id": song.song_id, "bars": len(song.bars), "tracks": {s: (t.tuning_name, t.name) for s, t in song.tracks.items()},
            "tokens_per_bar_mean": float(np.mean(tpb)) if tpb else 0.0, "warnings": song.warnings,
            "tunings": [tunings_mod.describe(t.tuning) for t in song.tracks.values()]}
