"""Riffs and sections (§5.2): bar similarity, borders and segments.

A riff is a group of 1-8 bars that repeats (exactly or almost). We compute a bar
self-similarity matrix S (0..1), mostly from the rhythm guitar, and find borders from:
section markers, gap edges, novelty peaks in S (Foote's checkerboard kernel), and a
4/8-bar phrase grid inside long stretches without other borders. Segments between
borders are what the filters judge and where training gaps start (D29).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from riffgen.analysis.notes import Onset, bar_vectors
from riffgen.dataset.song import SongData

W_RHYTHM, W_PITCH, W_HARM = 0.4, 0.4, 0.2


@dataclass
class Segment:
    start: int
    end: int                      # exclusive
    kind: str = "music"           # music | gap
    period: int = 0               # riff unit length in bars (0 = no clear repetition)
    repeats: int = 0              # similar segments elsewhere in the song
    section: str | None = None    # section label in effect
    action: str = "keep"          # keep | context | drop
    scores: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "kind": self.kind, "period": self.period,
                "repeats": self.repeats, "section": self.section, "action": self.action,
                "scores": self.scores, "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(**d)


def _norm_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-8)


def track_ssm(bars_onsets: list[list[Onset]], ticks: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Similarity matrix of one guitar and the per-bar 'has notes' mask."""
    R, P, H, has = [], [], [], []
    for ons, t in zip(bars_onsets, ticks):
        r, p, h, ok = bar_vectors(ons, t)
        R.append(r)
        P.append(p)
        H.append(h)
        has.append(ok)
    R, P, H = _norm_rows(np.stack(R)), _norm_rows(np.stack(P)), _norm_rows(np.stack(H))
    S = W_RHYTHM * (R @ R.T) + W_PITCH * (P @ P.T) + W_HARM * (H @ H.T)
    return np.clip(S, 0.0, 1.0).astype(np.float32), np.array(has, bool)


def song_ssm(song: SongData, onsets: dict[int, list[list[Onset]]]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Combined similarity matrix; the rhythm guitar (lowest mean pitch) weighs double."""
    n = len(song.bars)
    ticks = [b.ticks for b in song.bars]
    if n == 0:
        return np.zeros((0, 0), np.float32), {}
    per_track: dict[int, np.ndarray] = {}
    weights: dict[int, float] = {}
    has_map: dict[int, np.ndarray] = {}
    mean_pitch: dict[int, float] = {}
    for slot, bars in onsets.items():
        S, has = track_ssm(bars, ticks)
        per_track[slot] = S
        has_map[slot] = has
        ps = [on.low_pitch for b in bars for on in b if on.low_pitch is not None]
        mean_pitch[slot] = float(np.mean(ps)) if ps else 1e9
    rhythm = min(mean_pitch, key=mean_pitch.get) if mean_pitch else None
    num = np.zeros((n, n), np.float32)
    den = np.zeros((n, n), np.float32)
    for slot, S in per_track.items():
        w = 2.0 if slot == rhythm else 1.0
        weights[slot] = w
        h = has_map[slot]
        either = h[:, None] | h[None, :]
        both = h[:, None] & h[None, :]
        num += w * np.where(both, S, 0.0) * either
        den += w * either
    out = np.where(den > 0, num / np.maximum(den, 1e-8), 1.0).astype(np.float32)
    np.fill_diagonal(out, 1.0)
    return out, per_track


def novelty(S: np.ndarray, w: int) -> np.ndarray:
    n = len(S)
    nov = np.zeros(n, np.float32)
    for i in range(w, n - w + 1):
        a = S[i - w:i, i - w:i].mean()
        b = S[i:i + w, i:i + w].mean()
        c = S[i - w:i, i:i + w].mean()
        nov[i] = (a + b) / 2 - c
    return nov


def find_borders(song: SongData, S: np.ndarray, novelty_threshold: float = 0.22) -> list[int]:
    n = len(song.bars)
    borders = {0}
    for i, b in enumerate(song.bars):
        if b.marker:
            borders.add(i)
        if i > 0 and b.gap != song.bars[i - 1].gap:
            borders.add(i)
    for w in (2, 4):
        if n < 2 * w + 1:
            continue
        nov = novelty(S, w)
        for i in range(w, n - w + 1):
            lo, hi = max(0, i - 2), min(n, i + 3)
            if nov[i] >= novelty_threshold and nov[i] >= nov[lo:hi].max():
                borders.add(i)
    # phrase grid inside long stretches
    ordered = sorted(borders) + [n]
    extra = set()
    for a, b in zip(ordered[:-1], ordered[1:]):
        if b - a <= 8 or song.bars[a].gap:
            continue
        step = 4 if a + 4 < b and np.mean([S[i, i + 4] for i in range(a, min(b - 4, a + 8))]) >= 0.8 else 8
        for k in range(a + step, b, step):
            extra.add(k)
    borders |= extra
    return sorted(x for x in borders if 0 <= x < n)


def segment_period(S: np.ndarray, a: int, b: int, threshold: float = 0.85) -> int:
    for L in (1, 2, 4, 8):
        if b - a < 2 * L:
            break
        vals = [S[i, i + L] for i in range(a, b - L)]
        if vals and float(np.mean(vals)) >= threshold:
            return L
    return 0


def block_similarity(S: np.ndarray, a0: int, a1: int, b0: int, b1: int) -> float:
    L = min(a1 - a0, b1 - b0)
    if L <= 0:
        return 0.0
    return float(np.mean([S[a0 + d, b0 + d] for d in range(L)]))


def make_segments(song: SongData, S: np.ndarray, borders: list[int], min_len: int = 2,
                  max_len: int = 16) -> list[Segment]:
    n = len(song.bars)
    edges = sorted(set(borders) | {n})
    segs: list[Segment] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        kind = "gap" if song.bars[a].gap else "music"
        segs.append(Segment(a, b, kind))
    # merge short music segments into the more similar neighbour
    changed = True
    while changed:
        changed = False
        for k, seg in enumerate(segs):
            if seg.kind != "music" or seg.end - seg.start >= min_len or len(segs) == 1:
                continue
            cands = []
            for j in (k - 1, k + 1):
                if 0 <= j < len(segs) and segs[j].kind == "music":
                    o = segs[j]
                    cands.append((block_similarity(S, seg.start, seg.end, o.start, o.end), j))
            if not cands:
                continue
            _, j = max(cands)
            o = segs[j]
            merged = Segment(min(o.start, seg.start), max(o.end, seg.end), "music")
            lo, hi = min(j, k), max(j, k)
            segs[lo:hi + 1] = [merged]
            changed = True
            break
    # split very long segments
    out: list[Segment] = []
    for seg in segs:
        if seg.kind == "music" and seg.end - seg.start > max_len:
            for a in range(seg.start, seg.end, 8):
                out.append(Segment(a, min(seg.end, a + 8), "music"))
            if out[-1].end - out[-1].start < min_len and len(out) >= 2 and out[-2].kind == "music":
                last = out.pop()
                out[-1].end = last.end
        else:
            out.append(seg)
    # section label in effect, period, repeats
    section = None
    marks = {i: b.section for i, b in enumerate(song.bars) if b.section}
    for seg in out:
        for i in range(seg.start, seg.end):
            if i in marks:
                section = marks[i]
                break
        seg.section = section
        if seg.kind == "music":
            seg.period = segment_period(S, seg.start, seg.end)
    for k, seg in enumerate(out):
        if seg.kind != "music":
            continue
        L = seg.end - seg.start
        seg.repeats = sum(1 for j, o in enumerate(out) if j != k and o.kind == "music"
                          and abs((o.end - o.start) - L) <= 1
                          and block_similarity(S, seg.start, seg.end, o.start, o.end) >= 0.8)
    return out


def bar_partners(S: np.ndarray, threshold: float = 0.9, k: int = 4) -> np.ndarray:
    """For each bar up to k other bars with similarity >= threshold (-1 = none)."""
    n = len(S)
    out = np.full((n, k), -1, np.int32)
    if n == 0:
        return out
    M = S.copy()
    np.fill_diagonal(M, -1.0)
    for i in range(n):
        idx = np.argsort(-M[i])[:k]
        good = [j for j in idx if M[i, j] >= threshold]
        out[i, :len(good)] = good
    return out


def duplicate_tracks(song: SongData, threshold: float = 0.9) -> list[tuple[int, int]]:
    """Pairs of guitars that play (almost) the same thing (double-tracked rhythm)."""
    slots = sorted(song.tracks)
    sig = {s: [] for s in slots}
    for bar in song.bars:
        per = {s: [] for s in slots}
        for e in bar.events:
            if e.track in per:
                per[e.track].append((e.tick, tuple(t.split(":", 1)[1] for t in e.tokens)))
        for s in slots:
            sig[s].append(tuple(per[s]))
    pairs = []
    for i, a in enumerate(slots):
        for b in slots[i + 1:]:
            played = [k for k in range(len(song.bars)) if sig[a][k] or sig[b][k]]
            if not played:
                continue
            same = sum(1 for k in played if sig[a][k] == sig[b][k])
            if same / len(played) >= threshold:
                pairs.append((a, b))
    return pairs
