"""YAML configs with inheritance and command-line overrides.

A config file may start with ``inherit: [other.yaml, ...]``; the listed files are
loaded first (paths relative to the file) and deep-merged, then the file itself.
Command-line overrides use dotted keys: ``train.lr=1e-4 data.augment.guitar_swap=0``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from riffgen import CONFIG_DIR, REPO_DIR


class Cfg(dict):
    """A dict with attribute access (``cfg.train.lr``). Nested dicts become Cfg."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            self[k] = _wrap(v)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _wrap(value)

    def __deepcopy__(self, memo):
        return Cfg(copy.deepcopy(dict(self), memo))

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = Cfg()
            node = node[part]
        node[parts[-1]] = _wrap(value)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self, default=str))


def _wrap(v: Any) -> Any:
    if isinstance(v, Cfg):
        return v
    if isinstance(v, dict):
        return Cfg(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_config_path(path: str | Path) -> Path:
    """Find a config file: as given, relative to the repo, or inside configs/."""
    p = Path(path)
    candidates = [p, REPO_DIR / p, CONFIG_DIR / p]
    if p.suffix == "":
        candidates += [c.with_suffix(".yaml") for c in candidates]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"config not found: {path}")


def _load_raw(path: Path, seen: set[Path]) -> dict:
    if path in seen:
        raise ValueError(f"config inheritance loop at {path}")
    seen = seen | {path}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parents = data.pop("inherit", []) or []
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for parent in parents:
        ppath = (path.parent / parent)
        if not ppath.is_file():
            ppath = resolve_config_path(parent)
        merged = deep_merge(merged, _load_raw(ppath.resolve(), seen))
    return deep_merge(merged, data)


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"override must look like key=value, got {text!r}")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Cfg:
    cfg = Cfg(_load_raw(resolve_config_path(path), set()))
    for ov in overrides:
        k, v = parse_override(ov)
        cfg.set_path(k, v)
    cfg["_source"] = str(resolve_config_path(path))
    return cfg


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)
