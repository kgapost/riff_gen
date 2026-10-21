"""D14: unroll repeats and alternate endings.

DadaGP keeps the repeat signs as measure tokens:

* ``measure:repeat_open``
* ``measure:repeat_close:N``  -- N = number of *extra* passes (PyGuitarPro 0.6 normalises
  GP5's stored value; "play twice" is N=1, the most common value in DadaGP)
* ``measure:repeat_alternative:M`` -- M is a *bitmask* of the passes that play this
  ending (3 = endings 1 and 2), not an ending number.
* ``measure:direction:*`` / ``measure:from_direction:*`` (segno, coda, D.C., ...) are not
  unrolled; songs that use them are played straight through and get a warning.

An alternate ending covers the bars from the bar with the flag up to the next repeat close;
an ending without a close (usually the last one) covers just its first bar, after which
playback continues linearly, which is what a musician would do.
"""

from __future__ import annotations

import copy

from riffgen.dataset.song import Bar

MAX_EXPANSION = 8      # unrolled song may be at most 8x the written length
MAX_PASSES = 16


def _info(bar: Bar) -> tuple[bool, int, int, bool]:
    is_open, close, alt, direction = False, 0, 0, False
    for t in bar.measure_tokens:
        if t == "measure:repeat_open":
            is_open = True
        elif t.startswith("measure:repeat_close:"):
            close = max(0, int(t.rsplit(":", 1)[1]))
        elif t.startswith("measure:repeat_alternative:"):
            alt = int(t.rsplit(":", 1)[1])
        elif t.startswith("measure:direction:") or t.startswith("measure:from_direction:"):
            direction = True
    return is_open, close, alt, direction


def ending_regions(bars: list[Bar]) -> dict[int, int]:
    """Map bar index -> alternative bitmask for every bar inside an alternate ending."""
    infos = [_info(b) for b in bars]
    region: dict[int, int] = {}
    n = len(bars)
    i = 0
    while i < n:
        alt = infos[i][2]
        if alt <= 0:
            i += 1
            continue
        j = i
        end = i  # default: ending without close covers one bar
        while j < n:
            if j > i and (infos[j][2] > 0 or infos[j][0]):
                break
            if infos[j][1] > 0:
                end = j
                break
            j += 1
        for k in range(i, end + 1):
            region[k] = alt
        i = end + 1
    return region


def unroll(bars: list[Bar], warnings: list[str] | None = None) -> list[Bar]:
    """Return the bars in playing order. Bars are shallow copies; ``orig`` is preserved."""
    n = len(bars)
    if n == 0:
        return []
    infos = [_info(b) for b in bars]
    if not any(inf[0] or inf[1] or inf[2] for inf in infos):
        out = bars
    else:
        regions = ending_regions(bars)
        out_idx: list[int] = []
        repeat_start = 0
        pass_no = 1              # current pass through the current repeat section
        done_closes: dict[int, int] = {}
        i = 0
        limit = MAX_EXPANSION * n + 16
        while i < n and len(out_idx) < limit:
            is_open, close, _, _ = infos[i]
            if is_open and (i != repeat_start or pass_no == 1):
                if i != repeat_start:
                    repeat_start = i
                    pass_no = 1
                    done_closes = {}
            alt = regions.get(i, 0)
            if alt and not (alt >> (pass_no - 1)) & 1:
                i += 1
                continue
            out_idx.append(i)
            if close > 0:
                times = min(close, MAX_PASSES)
                done = done_closes.get(i, 0)
                if done < times:
                    done_closes[i] = done + 1
                    pass_no += 1
                    i = repeat_start
                    continue
                # section finished: following bars start a new (implicit) section
                repeat_start = i + 1
                pass_no = 1
                done_closes = {}
            i += 1
        if len(out_idx) >= limit and warnings is not None:
            warnings.append("repeat unrolling hit the expansion limit; repeats may be wrong")
        out = [bars[k] for k in out_idx]

    if warnings is not None and any(inf[3] for inf in infos):
        warnings.append("song uses segno/coda/D.C. directions; they are not unrolled")

    result = []
    for b in out:
        nb = copy.copy(b)
        nb.measure_tokens = [t for t in b.measure_tokens
                             if not t.startswith(("measure:repeat", "measure:direction", "measure:from_direction"))]
        result.append(nb)
    return result
