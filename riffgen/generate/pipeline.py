"""From a song with gaps to ranked candidate files (README §7)."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from riffgen import REPO_DIR
from riffgen.analysis import filters as filters_mod
from riffgen.analysis import riffs as riffs_mod
from riffgen.analysis.notes import bar_onsets, song_onsets
from riffgen.dataset import pipeline as pl
from riffgen.dataset.sampler import ExampleBuilder, SamplerConfig
from riffgen.dataset.serialize import SEG_MIDDLE, SEG_PREFIX, SEG_SUFFIX, Featurizer, TrackPlan
from riffgen.dataset.song import Bar, SongData, parse_dadagp, raw_to_song, ts_ticks
from riffgen.dataset.store import SongView, view_from_song
from riffgen.generate.decode import Candidate
from riffgen.tokens import sections as sections_mod
from riffgen.tokens.vocab import Vocab
from riffgen.utils.log import get_logger

log = get_logger("riffgen.generate")


@dataclass
class Gap:
    start: int                 # unrolled bar index
    length: int
    orig_measures: list[int]   # 0-based GP measure indices to write into

    @property
    def label(self) -> str:
        return f"{self.orig_measures[0] + 1}-{self.orig_measures[-1] + 1}"


def load_input_song(path: str | Path, require_tuning: str = "drop_a7", min_gap_bars: int = 4) -> SongData:
    cfg = pl.PipelineConfig(require_tuning=require_tuning, min_gap_bars=min_gap_bars, analyze=True)
    return pl.process_gp_file(path, Path(path).parent, cfg, kind="input", family_group="input")


def find_gaps(song: SongData, gap_arg: str | None, min_gap_bars: int = 4) -> list[Gap]:
    """D34/D35: runs of >= min_gap_bars bars without guitar notes, or bars given by hand (GP numbers)."""
    gaps = []
    if gap_arg:
        a, b = filters_mod.parse_bar_range(gap_arg)
        wanted = list(range(a, b))
        idx = []
        for m in wanted:
            occ = [i for i, bar in enumerate(song.bars) if bar.orig == m]
            if not occ:
                raise SystemExit(f"bar {m + 1} does not exist in the song")
            idx.append(occ[0])
        if idx != list(range(idx[0], idx[0] + len(idx))):
            raise SystemExit("--gap must be consecutive bars (inside a repeat, give the first pass)")
        return [Gap(idx[0], len(idx), wanted)]
    seen = set()
    i = 0
    while i < len(song.bars):
        if song.bars[i].gap:
            j = i
            while j < len(song.bars) and song.bars[j].gap:
                j += 1
            orig = [song.bars[k].orig for k in range(i, j)]
            if tuple(orig) not in seen:        # a gap inside a repeat appears twice: fill the first pass
                seen.add(tuple(orig))
                gaps.append(Gap(i, j - i, orig))
            i = j
        else:
            i += 1
    return gaps


def prepare_view(song: SongData, gap: Gap, vocab: Vocab, use_gap_signatures: bool) -> tuple[SongView, list[int], list]:
    """Song view with the gap bars set to their forced length and time signature (D33)."""
    view = view_from_song(song, vocab)
    bars = view.bars.copy()
    before = gap.start - 1 if gap.start > 0 else gap.start + gap.length
    ref_ts = (int(bars[before]["ts_num"]), int(bars[before]["ts_den"])) if 0 <= before < len(bars) else (4, 4)
    ticks, ts = [], []
    for k in range(gap.start, gap.start + gap.length):
        t = (int(bars[k]["ts_num"]), int(bars[k]["ts_den"])) if use_gap_signatures else ref_ts
        bars[k]["ts_num"], bars[k]["ts_den"] = t
        bars[k]["ticks"] = ts_ticks(t)
        ticks.append(ts_ticks(t))
        ts.append(t)
    view.bars = bars
    return view, ticks, ts


def last_notes_before(song: SongData, gap: Gap) -> dict[int, set]:
    out = {}
    if gap.start == 0:
        return out
    bar = song.bars[gap.start - 1]
    for slot, tr in song.tracks.items():
        ons = [o for o in bar_onsets(bar, slot, tr.tuning) if o.notes]
        if ons:
            out[slot] = {(n.string, n.fret) for n in ons[-1].notes}
    return out


def build_prompt(view: SongView, vocab: Vocab, sc: SamplerConfig, gap: Gap, section: str | None,
                 reserve_per_bar: int = 110) -> tuple[dict, Featurizer]:
    builder = ExampleBuilder(vocab, sc)
    sec = sections_mod.labels().index(section) if section else None
    spec = {"task": "infill", "start": gap.start, "len": gap.length,
            "reserve": min(int(sc.max_len * sc.max_middle_frac), reserve_per_bar * gap.length)}
    if sec is not None:
        spec["section"] = sec
    ex = builder.build_infill(view, np.random.default_rng(0), spec=spec, augment=False)
    if ex is None:
        raise RuntimeError("could not build a prompt for this gap")
    ids = ex["ids"]
    cut = int(np.flatnonzero(ids == vocab.fim_middle_id)[0]) + 1
    prompt = {k: (v[:cut] if isinstance(v, np.ndarray) and v.shape[:1] == ids.shape[:1] else v) for k, v in ex.items()}
    if "song" in ex:
        prompt["song"] = dict(ex["song"])
        prompt["song"]["tok_bar"] = ex["song"]["tok_bar"][:cut]
    # featurizer positioned right after <fim_middle> (same set-up as in build_infill)
    feat = Featurizer(vocab, view.bars["ticks"], np.stack([view.bars["ts_num"], view.bars["ts_den"]], 1),
                      TrackPlan.identity(view).tunings,
                      {SEG_PREFIX: int(ex["meta"]["first_pre"]), SEG_SUFFIX: gap.start + gap.length,
                       SEG_MIDDLE: gap.start},
                      ref_bar=gap.start, bar_phrase=builder._phrase(view, gap.start, gap.length))
    for t in ids[:cut]:
        feat.step(int(t))
    return prompt, feat


def candidate_bars(cand: Candidate, vocab: Vocab, song: SongData, ticks: list[int], ts: list) -> list[Bar]:
    toks = [t for t in vocab.decode(cand.tokens) if not t.startswith("ts:") and t not in ("<eos>",)]
    raw = parse_dadagp(["start"] + toks)
    s = raw_to_song(raw, "cand", {slot: tr.tuning for slot, tr in song.tracks.items()}, dadagp_normalised=False)
    bars = s.bars[: len(ticks)]
    while len(bars) < len(ticks):
        bars.append(Bar(ticks=ticks[len(bars)]))
    for b, t, sig in zip(bars, ticks, ts):
        b.ticks = t
        b.ts = sig
    return bars


def song_with_bars(song: SongData, gap: Gap, bars: list[Bar]) -> SongData:
    s = copy.copy(song)
    s.bars = list(song.bars)
    for k, b in enumerate(bars):
        nb = copy.copy(b)
        nb.orig = song.bars[gap.start + k].orig
        nb.tempo = song.bars[gap.start + k].tempo
        nb.gap = False
        s.bars[gap.start + k] = nb
    return s


def neighbour_copy_score(song: SongData, gap: Gap) -> float:
    """D36: mean over candidate bars of the best similarity to the L bars before/after the gap."""
    a, b = gap.start, gap.start + gap.length
    lo, hi = max(0, a - gap.length), min(len(song.bars), b + gap.length)
    sub = copy.copy(song)
    sub.bars = song.bars[lo:hi]
    S, _ = riffs_mod.song_ssm(sub, song_onsets(sub))
    cand = range(a - lo, b - lo)
    neigh = [i for i in range(len(sub.bars)) if i not in cand and not sub.bars[i].gap]
    if not neigh:
        return 0.0
    return float(np.mean([max(S[i, j] for j in neigh) for i in cand]))


def grid_flag(bars: list[Bar], grids=(20, 24)) -> bool:
    for b in bars:
        for e in b.events:
            if not any(e.tick % g == 0 for g in grids):
                return True
    return False


def default_out_dir(album_id: str, song_path: Path, gap: Gap) -> Path:
    return REPO_DIR / "outputs" / album_id / f"{time.strftime('%Y-%m-%d')}_{song_path.stem}_gap_{gap.label}"
