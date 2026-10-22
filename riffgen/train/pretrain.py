"""Stage 1: pretrain a base model on DadaGP.

    python -m riffgen.train.pretrain --model baseline
    python -m riffgen.train.pretrain --model big --resume
    python -m riffgen.train.pretrain --model huge train.max_steps=8000 train.micro_batch_tokens=12288

Writes runs/pretrain_<model>/{best.pt,last.pt,metrics.jsonl,train.log,run_info.json,val_specs.json}.
"""

from __future__ import annotations

import argparse


from riffgen.dataset.loader import SourceSpec, StreamConfig, TrainStream
from riffgen.dataset.store import Store
from riffgen.model.model import RiffGenModel
from riffgen.train import checkpoint as ckpt_mod
from riffgen.train.common import (check_vocab, count_params, fixed_far_examples, fixed_val_examples, load_model_yaml,
                                  model_config,
                                  repo_path, resolve_device_batch, sampler_config, setup_device, stage_config)
from riffgen.train.trainer import Trainer, TrainConfig
from riffgen.utils.config import load_config
from riffgen.utils.log import get_logger


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="baseline", help="baseline | big | huge | path to a model yaml")
    ap.add_argument("--config", default="train/pretrain.yaml")
    ap.add_argument("--run", default=None, help="run folder (default runs/pretrain_<model>)")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_intermixed_args(argv)

    model_yaml = load_model_yaml(args.model)
    cfg = stage_config(load_config(args.config, args.overrides), model_yaml, "pretrain")
    for ov in args.overrides:          # command line wins over model-size defaults
        k, v = ov.split("=", 1)
        import yaml
        cfg.set_path(k, yaml.safe_load(v))
    run_dir = repo_path(args.run or f"runs/pretrain_{model_yaml.name}")
    log = get_logger("riffgen.train", run_dir / "train.log")
    device, amp = setup_device(cfg)
    resolve_device_batch(cfg.train, device)

    stores = [Store(repo_path(s["path"])) for s in cfg.data.stores]
    vocab = stores[0].load_vocab()
    for st in stores:
        check_vocab(st, vocab)
    mcfg = model_config(model_yaml, vocab)

    resume = None
    if args.resume and (run_dir / "last.pt").is_file():
        model, vocab_ck, resume = ckpt_mod.load_model(run_dir / "last.pt", "cpu")
        if vocab_ck.hash != vocab.hash:
            raise RuntimeError("vocabulary changed since the checkpoint was written")
        log.info("resuming from %s at step %d", run_dir / "last.pt", resume["step"])
    else:
        model = RiffGenModel(mcfg)
    model.to(device)
    log.info("model %s: %s", mcfg.name, count_params(model))

    sc = sampler_config(cfg, mcfg)
    seed = int(cfg.seed)
    val_store = Store(repo_path(cfg.data.get("val_store", cfg.data.stores[0]["path"])))
    val = fixed_val_examples(run_dir, "val", val_store, vocab, sc, seed, int(cfg.data.get("val_infill", 256)),
                             int(cfg.data.get("val_song", 128)))
    val_sets = {"val": val}
    if val:
        val_sets["val_infill"] = [e for e in val if e["meta"]["task"] == "infill"]
    n_far = int(cfg.data.get("val_far", 0) or 0)
    if n_far:
        far = fixed_far_examples(run_dir, "val_far", val_store, vocab, sc, seed, n_far)
        if far:
            val_sets["val_far"] = far
    log.info("validation: %d fixed examples (%d infill, %d far)", len(val), len(val_sets.get("val_infill", [])),
             len(val_sets.get("val_far", [])))

    sources = [SourceSpec(path=str(repo_path(s["path"])), split=s.get("split", "train"),
                          weight=float(s.get("weight", 1.0)), share=s.get("share"),
                          bars_per_example=cfg.data.get("bars_per_example", "auto")) for s in cfg.data.stores]
    start_epoch = resume.get("epoch", 0) if resume else 0
    stream = TrainStream(StreamConfig(sources=sources, seed=seed, sampler_log=str(run_dir / "sampler_log.jsonl")),
                         vocab, sc, start_epoch=start_epoch)
    tcfg = TrainConfig.from_cfg(cfg.train)
    tcfg.aux_weights = dict(mcfg.aux_weights) if mcfg.hierarchical else {}
    tcfg.select_metric = "val_infill/loss" if val_sets.get("val_infill") else "val/loss"

    run_info = {"stage": "pretrain", "seed": seed, "config": cfg.to_dict(), "model": mcfg.to_dict(),
                "vocab_hash": vocab.hash, "git": ckpt_mod.git_commit(), "params": model.n_params(),
                "stores": [str(s.path) for s in stores],
                "train_songs": sum(len(st.indices("train")) for st in stores)}

    def payload(m, **extra):
        return ckpt_mod.full_payload(m, vocab, config=cfg.to_dict(), stage="pretrain", **extra)

    trainer = Trainer(run_dir, model, vocab, tcfg, stream, val_sets, device, amp, payload, run_info, resume)
    res = trainer.fit()
    log.info("done: step %d, best %s = %.4f -> %s", res["step"], tcfg.select_metric, res["best"], run_dir / "best.pt")


if __name__ == "__main__":
    main()
