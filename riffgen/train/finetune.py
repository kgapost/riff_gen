"""Stage 2: fine-tune a base model on one album (D7, D8).

    python -m riffgen.train.finetune configs/albums/compound_structure.yaml
    python -m riffgen.train.finetune configs/albums/compound_structure.yaml --model huge
    python -m riffgen.train.finetune configs/albums/compound_structure.yaml finetune.method=full

Needs: the album store (python -m riffgen.dataset.preprocess album <config>) and a base
model (runs/pretrain_<model>/best.pt, or ``base_model`` in the album config).

Writes runs/<album_id>/<model>/{best.pt,last.pt,...}. ``best.pt`` is chosen by the loss on
fixed gaps of the held-out songs (D32) and training stops early when it stops improving.
Afterwards the album's plagiarism index is built (D40).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from riffgen import REPO_DIR
from riffgen.dataset.loader import SourceSpec, StreamConfig, TrainStream, build_fixed_examples, epoch_plan, _Sources
from riffgen.dataset.sampler import ExampleBuilder, holdout_specs, make_far_specs
from riffgen.dataset.store import Store
from riffgen.model.lora import apply_lora
from riffgen.train import checkpoint as ckpt_mod
from riffgen.train.common import (check_vocab, count_params, fixed_val_examples, load_model_yaml, repo_path,
                                  resolve_device_batch, sampler_config, setup_device, stage_config)
from riffgen.train.trainer import Trainer, TrainConfig
from riffgen.utils.config import load_config
from riffgen.utils.log import get_logger


def album_store_path(acfg) -> Path:
    return repo_path(acfg.get_path("data.store", f"data/processed/albums/{acfg.album_id}"))


def run_dir_for(acfg, model_name: str) -> Path:
    return REPO_DIR / "runs" / acfg.album_id / model_name


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("album", help="album config, e.g. configs/albums/album_01.yaml")
    ap.add_argument("--model", default=None, help="baseline | big | huge (default: album config 'model')")
    ap.add_argument("--base", default=None, help="base checkpoint (default runs/pretrain_<model>/best.pt)")
    ap.add_argument("--run", default=None)
    ap.add_argument("--no-index", action="store_true", help="skip building the plagiarism index")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_intermixed_args(argv)

    acfg0 = load_config(args.album, args.overrides)
    model_name = args.model or acfg0.get("model", "baseline")
    model_yaml = load_model_yaml(model_name)
    acfg = stage_config(acfg0, model_yaml, "finetune")
    for ov in args.overrides:
        k, v = ov.split("=", 1)
        acfg.set_path(k, yaml.safe_load(v))
    run_dir = repo_path(args.run) if args.run else run_dir_for(acfg, model_yaml.name)
    log = get_logger("riffgen.train", run_dir / "train.log")
    device, amp = setup_device(acfg)
    resolve_device_batch(acfg.train, device)

    base = args.base or acfg.get("base_model") or f"runs/pretrain_{model_yaml.name}/best.pt"
    base = repo_path(base)
    if not base.is_file():
        raise SystemExit(f"base model not found: {base}\nPretrain first: python -m riffgen.train.pretrain --model {model_name}")
    model, vocab, base_ckpt = ckpt_mod.load_model(base, "cpu")
    mcfg = model.cfg
    store = Store(album_store_path(acfg))
    check_vocab(store, vocab)

    ft = acfg.get("finetune") or {}
    method = ft.get("method") or ft.get("method_default") or "full"
    lora_cfg = None
    if method == "lora":
        lc = dict(ft.get("lora") or {})
        lora_cfg = {"rank": int(lc.get("rank", 16)), "alpha": float(lc.get("alpha", 32)),
                    "dropout": float(lc.get("dropout", 0.05)),
                    "targets": list(lc.get("targets", ["attn.qkv", "attn.proj", "mlp.w12", "mlp.w3",
                                                       "cross.q", "cross.kv", "cross.proj"]))}
        apply_lora(model, lora_cfg["rank"], lora_cfg["alpha"], lora_cfg["dropout"], tuple(lora_cfg["targets"]))
        if lc.get("train_embeddings", True):
            for name, p in model.named_parameters():
                if name.startswith("emb."):
                    p.requires_grad = True
        log.info("LoRA rank %d: %.2fM trainable parameters", lora_cfg["rank"], model.n_params(True) / 1e6)
    model.to(device)
    log.info("fine-tuning %s (%s) from %s on album %s", mcfg.name, method, base, acfg.album_id)
    log.info("model: %s", count_params(model))

    sc = sampler_config(acfg, mcfg)
    seed = int(acfg.seed)
    # --- training sources: album songs (+ similar DadaGP songs, D47)
    sources = [SourceSpec(path=str(store.path), split="train", bars_per_example=acfg.data.get("bars_per_example", 16))]
    mix = acfg.get("dadagp_mix") or {}
    if mix.get("enabled"):
        sel_path = run_dir.parent / "dadagp_mix.json"
        if not sel_path.is_file():
            from riffgen.dataset.similar import select_for_album
            select_for_album(acfg, out=sel_path)
        sel = json.loads(sel_path.read_text())
        dstore = repo_path(mix.get("store", "data/processed/dadagp"))
        sources.append(SourceSpec(path=str(dstore), split=None, share=float(mix.get("share", 0.15)),
                                  ids=sel["ids"], bars_per_example=acfg.data.get("bars_per_example", 16)))
        log.info("DadaGP mix: %d selected songs, share %.0f%%", len(sel["ids"]), 100 * float(mix.get("share", 0.15)))

    # --- validation: fixed gaps on the held-out songs (D32, D43)
    builder = ExampleBuilder(vocab, sc)
    hold_views = [(i, store.view(i)) for i in store.indices("holdout")]
    spec_path = run_dir / "val_gaps.json"
    if spec_path.is_file():
        specs = json.loads(spec_path.read_text())
    else:
        specs = holdout_specs(hold_views, builder, tuple(acfg.data.get("holdout_gap_lengths", [4, 8])))
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(specs, indent=1))
    val_sets = {}
    if specs:
        val_sets["val"] = build_fixed_examples(specs, store, vocab, sc)
        log.info("held-out: %d songs, %d fixed gaps", len(hold_views), len(val_sets["val"]))
        n_far = int(acfg.data.get("val_far", 0) or 0)
        far_path = run_dir / "val_far_specs.json"
        if n_far and not far_path.is_file():
            far_path.write_text(json.dumps(make_far_specs(hold_views, builder, seed, n_far), indent=1))
        if far_path.is_file() and json.loads(far_path.read_text()):
            val_sets["val_far"] = build_fixed_examples(json.loads(far_path.read_text()), store, vocab, sc)
    else:
        log.warning("no held-out songs (data.holdout): no early stopping, best.pt = last step")
    dv = int(acfg.data.get("dadagp_val", 0) or 0)
    dstore_path = repo_path(mix.get("store", "data/processed/dadagp"))
    if dv and (dstore_path / "meta.json").is_file():
        dstore = Store(dstore_path)
        val_sets["dadagp"] = fixed_val_examples(run_dir, "dadagp_val", dstore, vocab, sc, seed, dv // 2, dv // 2)

    # --- schedule in epochs (D43: an epoch = new gaps for every song)
    srcs = _Sources(sources)
    plan = epoch_plan(sources, srcs, seed, 0, sc.max_len, sc.task_mix)
    rng = np.random.default_rng(seed)
    probe = [builder.build(srcs.stores[k].view(i), t, rng) for k, i, t in plan[:48]]
    avg_targets = float(np.mean([e["loss"].sum() for e in probe if e is not None] or [100.0]))
    tcfg = TrainConfig.from_cfg(acfg.train)
    tcfg.lr = float(acfg.train.get("lr", tcfg.lr))
    steps_per_epoch = max(1, int(round(len(plan) * avg_targets / tcfg.tokens_per_step)))
    epochs = int(ft.get("max_epochs", 10))
    evals_per_epoch = int(ft.get("evals_per_epoch", 2))
    tcfg.max_steps = steps_per_epoch * epochs
    tcfg.eval_every = max(1, steps_per_epoch // evals_per_epoch)
    tcfg.save_every = tcfg.eval_every
    tcfg.early_stopping_patience = int(ft.get("early_stopping_patience", 2)) * evals_per_epoch if specs else 0
    tcfg.select_metric = "val/loss"
    tcfg.aux_weights = dict(mcfg.aux_weights) if mcfg.hierarchical else {}
    tcfg.warmup_steps = min(tcfg.warmup_steps, max(1, tcfg.max_steps // 10))
    log.info("%d examples/epoch, ~%.0f target tokens each -> %d steps/epoch, max %d epochs (%d steps)",
             len(plan), avg_targets, steps_per_epoch, epochs, tcfg.max_steps)

    stream = TrainStream(StreamConfig(sources=sources, seed=seed, sampler_log=str(run_dir / "sampler_log.jsonl")),
                         vocab, sc)
    train_files = [s["source"] for s in store.songs if s.get("split") == "train"]
    run_info = {"stage": "finetune", "album": acfg.album_id, "seed": seed, "method": method, "base_model": str(base),
                "config": acfg.to_dict(), "model": mcfg.to_dict(), "vocab_hash": vocab.hash,
                "git": ckpt_mod.git_commit(), "train_files": train_files,
                "holdout_files": [s["source"] for s in store.songs if s.get("split") == "holdout"]}

    def payload(m, **extra):
        if lora_cfg:
            return ckpt_mod.lora_payload(m, vocab, str(base), lora_cfg, album=acfg.album_id, stage="finetune", **extra)
        return ckpt_mod.full_payload(m, vocab, album=acfg.album_id, stage="finetune", base_model=str(base), **extra)

    trainer = Trainer(run_dir, model, vocab, tcfg, stream, val_sets, device, amp, payload, run_info)
    res = trainer.fit()
    log.info("done: step %d, best held-out loss %.4f -> %s", res["step"], res["best"], run_dir / "best.pt")
    if not args.no_index:
        from riffgen.check.plagiarism import build_album_index
        idx = build_album_index(acfg, run_dir.parent / "plagiarism_index.sqlite")
        log.info("plagiarism index: %s", idx)


if __name__ == "__main__":
    main()
