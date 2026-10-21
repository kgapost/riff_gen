"""Guitar tunings: names, classification and the DadaGP fret normalisation.

Strings are listed from string 1 (highest) to string N (lowest), as MIDI numbers.

DadaGP only accepts four guitar tuning *shapes* (any uniform downtune allowed):

    std6     E A D G B E          (-5,-4,-5,-5,-5)
    std7     B E A D G B E        (-5,-4,-5,-5,-5,-5)
    drop6    D A D G B E          (-5,-4,-5,-5,-7)
    dropdbl7 A D A D G B E        (-5,-4,-5,-5,-7,-5)   DadaGP calls this "g7_drop"

Our target, 7-string Drop A (A E A D G B E, shape (-5,-4,-5,-5,-5,-7)), is NOT one of
them: dadagp.py raises "unsupported guitar tuning" for it. See
riffgen/gp/dadagp_bridge.py for how we get around that.

DadaGP also *pitch-normalises* drop strings: it subtracts 2 from the fret on the
dropped strings, so the open low D of drop D is written ``s6:f-2``. We convert
everything back to physical frets (what the player actually plays) and keep the
real tuning in a ``tuning:`` token.
"""

from __future__ import annotations

NOTE_NAMES = ("c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b")

STD6 = (64, 59, 55, 50, 45, 40)
STD7 = STD6 + (35,)
DROP_A7 = (64, 59, 55, 50, 45, 40, 33)
TARGET_TUNING_NAME = "drop_a7"

SHAPES: dict[tuple[str, int], tuple[int, ...]] = {
    ("std", 6): (-5, -4, -5, -5, -5),
    ("std", 7): (-5, -4, -5, -5, -5, -5),
    ("drop", 6): (-5, -4, -5, -5, -7),
    ("drop", 7): (-5, -4, -5, -5, -5, -7),
    ("dropdbl", 7): (-5, -4, -5, -5, -7, -5),
}

# Downtunes for which names are generated. Within this range names are unique
# (the name keeps the pitch class of the lowest string, not its octave).
DOWNTUNE_RANGE = range(-11, 1)


def _diffs(strings) -> tuple[int, ...]:
    return tuple(int(strings[i + 1]) - int(strings[i]) for i in range(len(strings) - 1))


def classify(strings) -> tuple[str, int, int] | None:
    """Return (shape, n_strings, downtune) or None if the tuning is not supported."""
    n = len(strings)
    d = _diffs(strings)
    for (shape, sn), sd in SHAPES.items():
        if sn == n and sd == d:
            return shape, n, int(strings[0]) - 64
    return None


def make(shape: str, n: int, downtune: int = 0) -> tuple[int, ...]:
    diffs = SHAPES[(shape, n)]
    out = [64 + downtune]
    for step in diffs:
        out.append(out[-1] + step)
    return tuple(out)


def name_of(strings) -> str:
    c = classify(strings)
    if c is None:
        return "other"
    shape, n, downtune = c
    if downtune not in DOWNTUNE_RANGE:
        return "other"
    low = NOTE_NAMES[int(strings[-1]) % 12]
    if shape == "std":
        return f"{low}_std{n}"
    if shape == "drop":
        return f"drop_{low}{n}"
    return f"dropdbl_{low}{n}"


def all_names() -> list[str]:
    names = []
    for (shape, n) in SHAPES:
        for dt in DOWNTUNE_RANGE:
            names.append(name_of(make(shape, n, dt)))
    names.append("other")
    return names


_BY_NAME: dict[str, tuple[int, ...]] = {}


def by_name(name: str) -> tuple[int, ...]:
    if not _BY_NAME:
        for (shape, n) in SHAPES:
            for dt in DOWNTUNE_RANGE:
                t = make(shape, n, dt)
                _BY_NAME[name_of(t)] = t
    if name not in _BY_NAME:
        raise KeyError(f"unknown tuning name {name!r}")
    return _BY_NAME[name]


def dadagp_drop_strings(strings) -> frozenset[int]:
    """Strings whose frets DadaGP shifted by -2 (its drop normalisation)."""
    c = classify(strings)
    if c is None:
        return frozenset()
    shape, n, _ = c
    if shape == "drop" and n == 6:
        return frozenset({6})
    if shape == "dropdbl" and n == 7:
        return frozenset({6, 7})
    return frozenset()


def infer_dadagp_tuning(downtune: int, n_strings: int, has_drop: bool) -> tuple[int, ...]:
    """Best guess of a DadaGP track's tuning when no GP file is available.

    Mirrors the DadaGP decoder: 7 strings if string 7 is used, drop if a fret -1/-2
    appears on a drop string. A drop-tuned song that never plays frets 0-1 on the drop
    string cannot be told apart; its frets then stay pitch-correct for standard tuning.
    """
    if n_strings == 7:
        return make("dropdbl" if has_drop else "std", 7, downtune)
    return make("drop" if has_drop else "std", 6, downtune)


def midi_to_name(m: int) -> str:
    return f"{NOTE_NAMES[m % 12].replace('s', '#').upper()}{m // 12 - 1}"


def describe(strings) -> str:
    """Low to high, e.g. 'A1 E2 A2 D3 G3 B3 E4'."""
    return " ".join(midi_to_name(int(s)) for s in reversed(list(strings)))
