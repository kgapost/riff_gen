"""Shared setup for pretraining and fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from riffgen import CONFIG_DIR, REPO_DIR
from riffgen.dataset.loader import build_fixed_examples
from riffgen.dataset.sampler import ExampleBuilder, SamplerConfig, make_val_specs
from riffgen.dataset.store import Store
from riffgen.model.config import ModelConfig
from riffgen.tokens.vocab import Vocab
from riffgen.utils.config import Cfg, deep_merge, load_config
from riffgen.utils.device import amp_dtype, pick_device, setup_backends
from riffgen.utils.seed import seed_everything


def repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_DIR / p


def load_model_yaml(name_or_path: str) -> Cfg:
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIG_DIR / "model" / f"{name_or_path}.yaml"
    return load_config(p)


def model_config(model_yaml: Cfg, vocab: Vocab) -> ModelConfig:
    d = {k: v for k, v in model_yaml.items() if k not in ("train", "_source")}
    d["vocab_size"] = ((len(vocab) + 63) // 64) * 64
    return ModelConfig.from_dict(d)


def stage_config(stage_cfg: Cfg, model_yaml: Cfg, stage: str) -> Cfg:
    """Model-size defaults (model yaml ``train.<stage>``) under the stage config."""
    over = (model_yaml.get("train") or {}).get(stage) or {}
    train_over = {k: v for k, v in over.items() if k not in ("method",)}
    merged = Cfg(deep_merge(stage_cfg, {"train": train_over}))
    if "method" in over and not stage_cfg.get_path("finetune.method_locked"):
        merged.set_path("finetune.method_default", over["method"])
    return merged


def setup_device(cfg: Cfg):
    seed_everything(int(cfg.get("seed", 1234)), bool(cfg.get("deterministic", False)))
    device = pick_device(cfg.get("device", "auto"))
    setup_backends(device)
    amp = amp_dtype(device, cfg.get("precision", "auto"))
    return device, amp


def sampler_config(cfg: Cfg, mcfg: ModelConfig) -> SamplerConfig:
    data = dict(cfg.get("data") or {})
    sc = SamplerConfig.from_cfg(data, max_len=mcfg.max_seq_len, hierarchical=mcfg.hierarchical)
    sc.max_song_bars = mcfg.max_song_bars
    sc.max_bar_tokens = mcfg.max_bar_tokens
    return sc


def check_vocab(store: Store, vocab: Vocab) -> None:
    if store.vocab_hash != vocab.hash:
        raise RuntimeError(
            f"store {store.path} was built with vocabulary {store.vocab_hash}, the model uses {vocab.hash}. "
            "Re-run preprocessing with the same code version as the base model.")


def fixed_val_examples(run_dir: Path, name: str, store: Store, vocab: Vocab, sc: SamplerConfig, seed: int,
                       n_infill: int, n_song: int, split: str = "val") -> list[dict]:
    """D43: validation examples are chosen once and saved, so every eval/run uses the same ones."""
    spec_path = run_dir / f"{name}_specs.json"
    if spec_path.is_file():
        specs = json.loads(spec_path.read_text())
    else:
        builder = ExampleBuilder(vocab, sc)
        views = [(i, store.view(i)) for i in store.indices(split)]
        specs = make_val_specs(views, builder, seed, n_infill, n_song)
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(specs, indent=1))
    return build_fixed_examples(specs, store, vocab, sc)


def count_params(model) -> str:
    n = model.n_params()
    parts = {"decoder": sum(p.numel() for n_, p in model.named_parameters() if n_.startswith("blocks")),
             "embeddings": sum(p.numel() for n_, p in model.named_parameters() if n_.startswith("emb"))}
    if hasattr(model, "song"):
        parts["song_memory"] = sum(p.numel() for p in model.song.parameters())
    return f"{n / 1e6:.1f}M (" + ", ".join(f"{k} {v / 1e6:.1f}M" for k, v in parts.items()) + ")"


def resolve_device_batch(train_cfg, device: torch.device) -> None:
    if device.type != "cuda":
        # CPU smoke runs: keep batches small
        train_cfg["micro_batch_tokens"] = min(int(train_cfg.get("micro_batch_tokens", 8192)), 8192)
