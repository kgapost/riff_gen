"""Training filter (§3.7): simplicity of riffs (D17).

A heuristic. Its job is to be roughly right and *auditable* (D18): every decision goes to
a CSV with its scores, and album configs can override single songs and bars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riffgen.analysis.notes import Onset
from riffgen.analysis.riffs import Segment


@dataclass
class FilterConfig:
    simple_threshold: float = 0.35
    simple_riff_action: str = "context"      # keep | context | drop

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "FilterConfig":
        cfg = cfg or {}
        return cls(**{k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg})


def complexity(onsets_by_track: dict[int, list[list[Onset]]], a: int, b: int, ticks: list[int]) -> dict:
    """Pitch and rhythm complexity of bars [a, b) over all guitars (0 = trivial, 1 = rich).

    "Simple" needs BOTH simple pitch and simple rhythm, so one-note polyrhythms (Meshuggah)
    and tremolo-picked chord changes (Mayhem, Thorns) are kept.
    """
    pitch_sets, lows, iois, patterns = set(), [], set(), set()
    pm_var = 0.0
    for slot, bars in onsets_by_track.items():
        pm_flags, acc_flags = [], []        # per guitar: a muted rhythm + an open double is no "pattern"
        for i in range(a, b):
            ons = [on for on in bars[i] if on.notes and on.is_attack]
            pat = []
            for k, on in enumerate(ons):
                pitch_sets.add(on.pitches)
                lows.append((i, on.tick, slot, on.low_pitch))
                pm_flags.append(on.palm_mute)
                acc_flags.append(on.accent)
                pat.append((on.tick * 48 // max(ticks[i], 1), on.palm_mute, on.accent))
                if k + 1 < len(ons):
                    iois.add(ons[k + 1].tick - on.tick)
            patterns.add((slot, tuple(pat)))
        if (any(pm_flags) and not all(pm_flags)) or (any(acc_flags) and not all(acc_flags)):
            pm_var = 1.0
    if not lows:
        return {"pitch": 0.0, "rhythm": 0.0, "complexity": 0.0, "simple": True}
    lows.sort()
    seq = [p for *_, p in lows if p is not None]
    changes = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    change_rate = changes / max(1, len(seq) - 1)
    pitch_c = max(float(np.clip((len(pitch_sets) - 2) / 3.0, 0, 1)), float(np.clip(change_rate / 0.4, 0, 1)))
    ioi_var = float(np.clip((len(iois) - 1) / 3.0, 0, 1))
    n_slots = max(1, len({s for s, _ in patterns}))
    pat_var = float(np.clip((len(patterns) / n_slots - 1) / 3.0, 0, 1))
    rhythm_c = float(np.clip(0.4 * ioi_var + 0.4 * pm_var + 0.3 * pat_var, 0, 1))
    return {"pitch": round(pitch_c, 3), "rhythm": round(rhythm_c, 3),
            "complexity": round(0.5 * pitch_c + 0.5 * rhythm_c, 3), "simple": None}


def apply_filters(song, segments: list[Segment], onsets: dict[int, list[list[Onset]]], fcfg: FilterConfig) -> None:
    ticks = [b.ticks for b in song.bars]
    for seg in segments:
        if seg.kind != "music":
            continue
        c = complexity(onsets, seg.start, seg.end, ticks)
        simple = c["pitch"] < fcfg.simple_threshold and c["rhythm"] < fcfg.simple_threshold
        c["simple"] = simple
        seg.scores["complexity"] = c
        if simple and fcfg.simple_riff_action != "keep":
            seg.action = fcfg.simple_riff_action
            seg.reason = "simple riff"


def parse_bar_range(text: str) -> tuple[int, int]:
    """'49-56' (1-based, inclusive, as Guitar Pro shows them) -> (48, 56) half-open, 0-based."""
    text = str(text)
    if "-" in text:
        a, b = text.split("-", 1)
        return int(a) - 1, int(b)
    return int(text) - 1, int(text)


def apply_overrides(song, segments: list[Segment], rel_path: str, overrides: dict | None) -> list[str]:
    """force_keep / force_drop from the album YAML. Bars are GP bar numbers (before unrolling)."""
    notes = []
    if not overrides:
        return notes
    for key, action in (("force_keep", "keep"), ("force_drop", "drop")):
        for item in overrides.get(key, []) or []:
            if str(item.get("file", "")).replace("\\", "/") not in rel_path.replace("\\", "/"):
                continue
            a, b = parse_bar_range(item.get("bars", "1-100000"))
            hit = [i for i, bar in enumerate(song.bars) if a <= bar.orig < b]
            if not hit:
                continue
            for seg in segments:
                if seg.kind != "music" or not any(seg.start <= i < seg.end for i in hit):
                    continue
                seg.action = action
                seg.reason = key
            notes.append(f"{key} {item.get('bars')} applied")
    return notes
