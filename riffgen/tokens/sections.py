"""Section markers ("Chorus 2", "REFRAIN", ...) -> section labels (D25)."""

from __future__ import annotations

import json
import re
from functools import lru_cache

from riffgen import CONFIG_DIR


@lru_cache(maxsize=1)
def _load() -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    with open(CONFIG_DIR / "sections.json", encoding="utf-8") as f:
        data = json.load(f)
    return tuple(data["labels"]), tuple((p.lower(), lab) for p, lab in data["patterns"])


def labels() -> tuple[str, ...]:
    return _load()[0]


def normalize(marker: str | None) -> str | None:
    if not marker:
        return None
    text = re.sub(r"[^a-z\- ]+", " ", marker.lower())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    for pattern, label in _load()[1]:
        if pattern in text:
            return label
    return "other"
