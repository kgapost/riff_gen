#!/usr/bin/env python
"""Train the real model on the whole collection, with the configuration the grid search settled on.

    python train_n_test.py                          # prepare if needed, train, test the best checkpoint
    python train_n_test.py --epochs 2               # a shorter run
    python train_n_test.py --no-prepare             # the corpus is already encoded
    python train_n_test.py --test-only runs/final   # just score an existing run on the test split

Everything that was searched is already decided here, so the defaults ARE the answer and you should not
need a single flag. See grid_search.py for how they were found, and WHY nothing else is worth searching.

    baseline        a flat decoder over a 4096-token fill-in-the-middle window. It beat the hierarchical
                    models on loss_cmp - the metric whose gaps exclude every repeated bar, so it measures
                    writing rather than reach - by roughly 2x, 0.61 against 1.21.
    ctx 4096        an interior optimum, not a default: 2048 loses 0.024, and 16384 is worse on BOTH
                    metrics because 90 % of songs already fit in 4096 and the window sits half empty.
    batch 16384     with lr 3.74e-04. The whole 4-8e-4 band is within noise, so this is "a good point in
                    the band", not a knife edge; 5.29e-04 scored marginally better over three seeds.
    drop 0.0        adopted on priors rather than proof - it won by less than one seed sd, and two epochs
                    over a corpus this size should underfit.
    no smoothing    label smoothing hurt in every one of the 111 grid runs, and focal loss hurt
    no focal        monotonically (0.226 and 0.279 against 0.199 for the same cell unfocused).

THE SPLIT IS FIXED, NOT RE-ROLLED EACH EPOCH. riff.py assigns it by md5 of the song family, so every
version and every copy of one song lands on the same side (README §10). Re-drawing it per epoch would
put a take of a song in train and another take of the same song in test, and the reported loss would
then be partly a measure of memorisation. What happens each epoch is an EVALUATION on the fixed
validation split; the test split is touched once, at the end, by --test-only or the final step here.

Both checkpoints are kept, which is the other half of the name: best.pt (the lowest validation loss
seen) and last.pt (the final epoch, with the optimiser state, so --resume can continue it).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import riff  # noqa: E402

# --------------------------------------------------------------------------- the winning configuration
MODEL = "baseline"
LR = 3.74e-04          # grid2 cell baseline_b16k_lr3.7e-04_smooth0, val 0.19654
BATCH_TOKENS = 16384
MICRO_TOKENS = 16384   # memory only; activation checkpointing keeps this inside 24 GB
DROP = 0.0
SMOOTH = 0.0
EPOCHS = 4


def cfg_for(a) -> dict:
    cfg = riff.model_cfg(MODEL) | {"micro_tokens": a.micro_tokens, "drop": a.drop, "smooth": a.smooth}
    if a.ckpt:
        cfg = cfg | {"ckpt": True}
    if a.ctx:
        cfg = cfg | {"ctx": a.ctx}
    return cfg


def report(tag: str, res: dict) -> None:
    keys = ("loss", "acc", "loss_cmp", "acc_cmp", "lift_cmp", "nll_rhythm", "nll_pitch", "nll_technique")
    print(f"  {tag:<12}" + "  ".join(f"{k} {res[k]:.4f}" for k in keys if k in res))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(riff.ROOT / "data"), help="the tab folders")
    ap.add_argument("--data", default=str(riff.ROOT / "data_encoded" / "encoded"), help="the encoded corpus")
    ap.add_argument("--run", default=str(riff.ROOT / "runs" / "final"))
    ap.add_argument("--epochs", type=float, default=EPOCHS)
    ap.add_argument("--budget", type=int, default=None, help="stop after N tokens instead of --epochs")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batch-tokens", type=int, default=BATCH_TOKENS)
    ap.add_argument("--micro-tokens", type=int, default=MICRO_TOKENS)
    ap.add_argument("--ctx", type=int, default=0, help="override the model's 4096-token window")
    ap.add_argument("--drop", type=float, default=DROP)
    ap.add_argument("--smooth", type=float, default=SMOOTH)
    ap.add_argument("--eval-every", type=float, default=1.0, help="epochs between validations (default: 1)")
    ap.add_argument("--eval-gaps", type=int, default=400)
    ap.add_argument("--gen-gaps", type=int, default=8, help="gaps written out at each evaluation")
    ap.add_argument("--no-ckpt", dest="ckpt", action="store_false",
                    help="store activations instead of recomputing them: ~a third faster, much more memory")
    ap.add_argument("--no-prepare", action="store_true", help="the corpus at --data is ready")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action="store_true", help="continue from last.pt")
    ap.add_argument("--test-only", metavar="RUN", help="score RUN/best.pt on the test split and stop")
    a = ap.parse_args(argv)

    data, run = Path(a.data), Path(a.run)
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if a.test_only:
        run = Path(a.test_only)
        model, V, cfg, _ = riff.load_ckpt(run / "best.pt", dev)
        c = riff.Corpus(a.data)
        if c.vocab.tokens != V.tokens:
            sys.exit("the checkpoint was trained on another vocabulary; re-prepare or use its own corpus")
        print(f"=== {run.name}/best.pt on the held-out test split")
        res = riff.evaluate(model, c, cfg, "test", a.eval_gaps, a.gen_gaps, out=run / "samples_test.txt")
        report("test", res)
        (run / "eval_test.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 0

    # 1) encode. riff.py prepare is the only thing that writes the corpus, so the split, the fixed
    #    evaluation gaps and the label vocabulary all come from it and are identical for every run.
    if not a.no_prepare and not (data / "meta.json").is_file():
        print(f"=== encoding {a.src} -> {data}")
        riff.cmd_prepare(argparse.Namespace(src=a.src, out=str(data), workers=a.workers, limit=0,
                                            exclude="", frac=1.0, max_versions=3, seed=a.seed))
    elif not (data / "meta.json").is_file():
        sys.exit(f"no corpus at {data} (drop --no-prepare to build it)")

    c = riff.Corpus(a.data)
    st = json.loads((data / "stats.json").read_text(encoding="utf-8"))
    ep_tok = riff.epoch_tokens(c, {})
    print(f"=== corpus: {st['songs']} songs, {st['tokens'] / 1e6:.0f}M tokens, "
          f"{st['tokens_per_bar']:.0f}/bar, splits {st['splits']}")
    print(f"=== {ep_tok / 1e6:.0f}M tokens per epoch x {a.epochs} = "
          f"{(a.budget or a.epochs * ep_tok) / 1e6:.0f}M tokens to train on\n")

    # 2) train. train_run evaluates on the fixed validation split every --eval-every epochs and keeps
    #    best.pt (lowest validation loss) and last.pt (final, with the optimiser state).
    cfg = cfg_for(a)
    t0 = time.time()
    r = riff.train_run(cfg, a.data, run, a.lr, a.batch_tokens, a.epochs, a.eval_every, a.budget,
                       resume=a.resume, workers=a.workers, seed=a.seed, eval_gaps=a.eval_gaps,
                       gen_gaps=a.gen_gaps, keep_ckpt=True)
    hours = (time.time() - t0) / 3600
    print(f"\n=== trained {r['tokens'] / 1e6:.0f}M tokens in {hours:.2f} h "
          f"({r['tokens'] / max(time.time() - t0, 1) / 1e3:.0f}k tok/s)")
    report("best val", r["best_eval"])
    report("last val", r["last"])

    # 3) test, once, with the best checkpoint. Nothing above ever looked at this split.
    if (run / "best.pt").is_file():
        model, V, ck_cfg, _ = riff.load_ckpt(run / "best.pt", dev)
        print("\n=== best.pt on the held-out test split")
        res = riff.evaluate(model, c, ck_cfg, "test", a.eval_gaps, a.gen_gaps, out=run / "samples_test.txt")
        report("test", res)
        (run / "eval_test.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nweights: {run / 'best.pt'} (best) and {run / 'last.pt'} (final, resumable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
