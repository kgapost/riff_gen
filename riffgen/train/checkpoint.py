"""Checkpoints.

    full checkpoint   {"kind": "full", model_cfg, state_dict, vocab, ...}
    LoRA album net    {"kind": "lora", base_model: <path>, lora: {...}, trainable_state: {...}}

A LoRA checkpoint only stores the adapters (and embeddings if they were trained) plus the
path of its base model, so five album networks cost a few MB each.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import torch

from riffgen import REPO_DIR
from riffgen.model.config import ModelConfig
from riffgen.model.lora import apply_lora, merge_lora, strip_lora
from riffgen.model.model import RiffGenModel
from riffgen.tokens.vocab import Vocab


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def save(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({**payload, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_commit()}, tmp)
    os.replace(tmp, path)   # atomic: a crash while saving never leaves a broken checkpoint


def full_payload(model: RiffGenModel, vocab: Vocab, **extra) -> dict:
    sd = {k: v for k, v in model.state_dict().items()}
    return {"kind": "full", "model_cfg": model.cfg.to_dict(), "state_dict": sd, "vocab": vocab.tokens,
            "vocab_hash": vocab.hash, **extra}


def lora_payload(model: RiffGenModel, vocab: Vocab, base_model: str, lora_cfg: dict, **extra) -> dict:
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    return {"kind": "lora", "base_model": str(base_model), "lora": lora_cfg, "trainable_state": trainable,
            "model_cfg": model.cfg.to_dict(), "vocab": vocab.tokens, "vocab_hash": vocab.hash, **extra}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_DIR / p
    return p


def load_model(path: str | Path, device: str | torch.device = "cpu", merge: bool = True,
               trainable_lora: bool = False) -> tuple[RiffGenModel, Vocab, dict]:
    """Load any checkpoint (full or LoRA). Returns (model, vocab, checkpoint dict)."""
    ckpt = torch.load(resolve(path), map_location="cpu", weights_only=False)
    if ckpt.get("kind") == "lora":
        model, vocab, _ = load_model(ckpt["base_model"], "cpu", merge=False)
        lc = ckpt["lora"]
        apply_lora(model, lc["rank"], lc["alpha"], lc.get("dropout", 0.0), tuple(lc["targets"]))
        missing = model.load_state_dict(ckpt["trainable_state"], strict=False)
        if missing.unexpected_keys:
            raise RuntimeError(f"unexpected keys in LoRA checkpoint: {missing.unexpected_keys[:5]}")
        if merge and not trainable_lora:
            merge_lora(model)
            strip_lora(model)
    else:
        cfg = ModelConfig.from_dict(ckpt["model_cfg"])
        model = RiffGenModel(cfg)
        model.load_state_dict(ckpt["state_dict"])
    vocab = Vocab(ckpt["vocab"])
    return model.to(device), vocab, ckpt
