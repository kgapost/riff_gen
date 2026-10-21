"""D46: one seed controls every random operation.

There is no shared global random state. Each random decision builds its own
generator from the main seed plus a fixed key that describes the decision, e.g.
``derive(seed, "gap", epoch, song_id, example_no)``. Text parts of the key are
hashed with zlib.crc32 (stable across runs, unlike Python's ``hash()``).
"""

from __future__ import annotations

import random
import zlib

import numpy as np


def _key_part(k) -> int:
    if isinstance(k, (bool, np.bool_)):
        return int(k)
    if isinstance(k, (int, np.integer)) and k >= 0:
        return int(k)
    return zlib.crc32(str(k).encode("utf-8"))


def derive(seed: int, *keys) -> np.random.Generator:
    ss = np.random.SeedSequence(int(seed), spawn_key=tuple(_key_part(k) for k in keys))
    return np.random.default_rng(ss)


def derive_int(seed: int, *keys) -> int:
    return int(derive(seed, *keys).integers(0, 2**31 - 1))


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed torch (model init, dropout). Data randomness never uses these."""
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
