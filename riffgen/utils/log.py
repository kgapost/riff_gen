from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

_FMT = "%(asctime)s %(levelname).1s %(name)s: %(message)s"


def get_logger(name: str = "riffgen", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(log_file).resolve())
                   for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FMT))
            logger.addHandler(fh)
    return logger


class JsonlWriter:
    """Append-only JSON-lines log (metrics, sampler log, ratings)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        record = {"time": round(time.time(), 3), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
