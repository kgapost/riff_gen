#!/usr/bin/env python
"""The grid search, as 111 runs concluded it should be done - and as the conclusions say it mostly should NOT.

    python grid_search.py --list                    # what would run, and what is already settled
    python grid_search.py                           # the default sweep: lr x batch, 3 seeds each
    python grid_search.py --seeds 5 --lrs 0.707,1,1.414     # finer, with more seeds
    python grid_search.py --lrs 1 --drops 0,0.05,0.1        # settle the one axis still open

THE ONE RESULT THAT GOVERNS EVERY FUTURE SEARCH: on this task a single run tells you almost nothing.
The same cell (baseline, ctx 4096, batch 16384, lr 3.74e-04) run four times gave

    0.19654   0.19717   0.20015   0.19656        spread 0.0036, sd 0.0017

and two of those share a seed, differing by 0.00063 from the dataloader workers and non-deterministic
CUDA kernels alone. The 21-run baseline grid's top SEVEN cells span 0.0032 - all inside one spread - so
they were never distinguishable, and the nominal winner (lr 3.74e-04) was one lucky seed: over three
seeds lr 5.29e-04 has the marginally better mean. Hence this script runs every cell at --seeds seeds and
ranks on the MEAN, and prints a line saying which differences it refuses to call.

    Believe an effect only if it clears ~0.004 val CE, or if 3-seed means separate it cleanly.

ALREADY SETTLED - do not spend card time re-searching these:

    architecture  baseline (flat, 4096) beats the hierarchical big/huge by ~2x on loss_cmp (0.61 vs
                  1.21), the metric whose gaps exclude every repeated bar. huge's whole advantage is
                  reach, and this corpus does not need it: repeated bars sit a median of 2 bars apart
                  and choose_context already captures 97.5 % of them inside the flat window.
    context       4096 is an interior optimum. 2048 loses 0.024; 8192 wins val_loss by 0.0036 but loses
                  loss_cmp and costs 38 % more time; 16384 is worse on BOTH, because 90 % of songs
                  already fit in 4096 and the bigger window sits half empty.
    smoothing     hurts in all 111 runs. Dropped from the axes entirely.
    focal loss    hurts monotonically: 0.226 at focal 1 and 0.279 at focal 2, against 0.199 unfocused.
    batch         bigger is worse here. b132k's own best cell lost to every b16k/b32k/b64k cell in the
                  4-8e-4 band and degraded monotonically with lr above it.
    the lr fit    suggest()'s DeepSeek fit overestimates lr and batch by ~4x for this task, and its
                  sqrt(batch) scaling over-corrects: the best absolute lr sits in a narrow 3.7-7.5e-04
                  band at EVERY batch size from 16k to 132k.

What is genuinely open, and why each is here rather than in the list above:

    dropout       0.0 beat 0.1 by 0.0016, which is less than one seed sd, from a single run. Adopted on
                  priors (two epochs over this corpus should underfit), never actually measured.
    budget        deliberately never probed. encoded_grid held half the tokens, so 4x budget there was 8
                  epochs of a subsample and would not have transferred. It belongs on the real corpus.
    lr x batch    only worth re-running on the real corpus, and only with seeds, if the corpus changes
                  enough to move the optimum - which encoding every pitched part may well have done.

THE 121 RUNS BEHIND ALL OF THAT (this section replaces grid2.sh, grid2_all.sh, grid3.sh, probe.sh and
probe.py, which ran them and were deleted once their findings were recorded here). Every run was 410.2M
tokens = 2.00 epochs of data_encoded/encoded_grid, a halved corpus with the own-project tabs excluded,
at 400 evaluation gaps, cadence 0.125, micro-batch 16384 and activation checkpointing. The raw rows are
in runs/{grid,grid2,probe}/results.jsonl - which is NOT in git, because runs/ is gitignored, so those
121 rows live only on the machine that produced them.

  grid 1  84 runs, 2026-10-30..11-05, baseline x big x huge.
          Settled smoothing (hurts everywhere) and found the whole grid pegged against its own lower
          edge: batch and lr were monotone, best at the lowest point tried, so it did not bracket the
          optimum. baseline 0.198 vs big 0.973 vs huge 0.802, but on val CE, which is not comparable
          across architectures - the flat model reads its context as literal tokens while the
          hierarchical ones see one vector per bar.
  grid 2  27 runs, 11-05..11-10, shifted downward, and added loss_cmp to make the comparison fair.
          baseline beat huge 0.61 vs 1.21 on it, which is what closed the architecture question.
          huge-old (11 further huge cells) was cancelled once that was clear: ~29 h for nothing.
          Top of the table: b16k lr3.7e-04 0.19654, b16k lr5.3e-04 0.19769, b64k lr7.5e-04 0.19916,
          b32k lr7.5e-04 0.19920, b32k lr5.3e-04 0.19935. focal 1 -> 0.22578, focal 2 -> 0.27905.
  probe   10 runs, 11-10..11-11, the three axes the grid never touched.
          seeds    one cell four times: 0.19654 / 0.19717 / 0.20015 / 0.19656 (the result above).
          context  2048 -> 0.22081 | 4096 -> 0.19717 | 8192 -> 0.19354 | 16384 -> 0.21808. val_loss
                   and loss_cmp disagree in the middle (8192 wins one, loses the other) but agree at
                   both ends. At ctx 16384 the mean example is 5151 tokens in a ~15148-token window.
          dropout  0.0 -> 0.19601 | 0.1 -> 0.19761 (mean of 4) | 0.2 -> 0.20012. Monotone on both
                   metrics across six runs, but 0.0 beats 0.1 by less than one seed sd.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import riff  # noqa: E402

NOISE = 0.004          # the smallest val-CE difference worth believing; see the docstring
WINNER = {"lr": 3.74e-04, "batch_tokens": 16384, "ctx": 4096, "drop": 0.0, "smooth": 0.0}


def cells(a) -> list[dict]:
    """The (lr, batch, drop, ctx) points to run. lr and batch are multipliers on the winning
    configuration, so `--lrs 1` is the winner's own learning rate and `--lrs 0.707,1,1.414` is a
    half-step either side; drop and ctx are absolute."""
    lrs = [float(x) for x in a.lrs.split(",") if x.strip()]
    batches = [float(x) for x in a.batches.split(",") if x.strip()]
    drops = [float(x) for x in a.drops.split(",") if x.strip()]
    ctxs = [int(x) for x in a.ctxs.split(",") if x.strip()]
    out = []
    for ctx in ctxs:
        for b in batches:
            for lm in lrs:
                for dr in drops:
                    out.append({"lr": WINNER["lr"] * lm, "batch_tokens": int(WINNER["batch_tokens"] * b),
                                "drop": dr, "ctx": ctx})
    return out


def name_of(c: dict) -> str:
    name = f"b{c['batch_tokens'] // 1024}k_lr{c['lr']:.1e}_drop{c['drop']:g}"
    return name if c["ctx"] == WINNER["ctx"] else f"{name}_ctx{c['ctx']}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(riff.ROOT / "data_encoded" / "encoded_grid"))
    ap.add_argument("--out", default=str(riff.ROOT / "runs" / "grid_search"))
    ap.add_argument("--seeds", type=int, default=3, help="runs per cell; the mean is what is ranked")
    ap.add_argument("--seed0", type=int, default=1234)
    ap.add_argument("--lrs", default="0.707,1,1.414", help="multiples of the winning lr (3.74e-04)")
    ap.add_argument("--batches", default="1", help="multiples of the winning batch (16384)")
    ap.add_argument("--drops", default="0", help="dropout values to try")
    ap.add_argument("--ctxs", default=str(WINNER["ctx"]), help="context lengths to try (absolute)")
    ap.add_argument("--budget", type=int, default=0, help="tokens per run (0: 2 epochs of --data)")
    ap.add_argument("--micro-tokens", type=int, default=16384)
    ap.add_argument("--eval-every", type=float, default=0.125)
    ap.add_argument("--eval-gaps", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-ckpt", dest="ckpt", action="store_false")
    ap.add_argument("--list", action="store_true", help="print the plan and stop")
    a = ap.parse_args(argv)

    out = Path(a.out)
    res_path = out / "results.jsonl"
    rows = [json.loads(l) for l in res_path.read_text().splitlines()] if res_path.is_file() else []
    done = {r["run"] for r in rows}
    plan = [(c, a.seed0 + i) for c in cells(a) for i in range(a.seeds)]

    if a.list:
        print(f"{len(plan)} runs = {len(cells(a))} cells x {a.seeds} seeds "
              f"(ranked on the mean; differences under {NOISE} are not called)\n")
        for c, s in plan:
            run = f"{name_of(c)}_s{s}"
            print(f"  {'DONE' if run in done else 'todo'}  {run}")
        return 0

    if not (Path(a.data) / "meta.json").is_file():
        sys.exit(f"no corpus at {a.data}; prepare one first (a halved corpus keeps each run short)")
    out.mkdir(parents=True, exist_ok=True)
    stats = riff.corpus_stats(a.data)
    budget = a.budget or int(2 * stats["train_tokens"])
    base = riff.model_cfg("baseline") | {"micro_tokens": a.micro_tokens, "smooth": 0.0}
    if a.ckpt:
        base = base | {"ckpt": True}
    print(f"{len(plan)} runs, {budget / 1e6:.0f}M tokens each "
          f"({budget / stats['train_tokens']:.2f} epochs of {a.data})\n")

    for c, seed in plan:
        run = f"{name_of(c)}_s{seed}"
        if run in done:
            riff.log(f"skip {run}")
            continue
        cfg = base | {"drop": c["drop"], "ctx": c["ctx"]}
        riff.log(f"=== {run}")
        t0 = time.time()
        try:
            r = riff.train_run(cfg, a.data, out / run, c["lr"], c["batch_tokens"], cfg["epochs"],
                               a.eval_every, budget, workers=a.workers, seed=seed,
                               eval_gaps=a.eval_gaps, gen_gaps=0, keep_ckpt=False)
        except Exception as e:                        # one bad cell must not abandon the sweep
            riff.log(f"!!! {run} failed after {(time.time() - t0) / 60:.0f} min: {type(e).__name__}: {e}")
            continue
        be = r["best_eval"]
        row = {"run": run, "cell": name_of(c), "seed": seed, **c,
               "val_loss": r["best"] if math.isfinite(r["best"]) else 99.0,
               "tokens": r["tokens"], "hours": round((time.time() - t0) / 3600, 3),
               **{k: be[k] for k in ("acc", "loss_cmp", "acc_cmp", "lift_cmp") if k in be}}
        with open(res_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        rows.append(row)
        riff.log(f"=== {run}: val {row['val_loss']:.5f}  ({row['hours']:.2f} h)")

    # ---- the table that matters: per cell, the MEAN over seeds, and what cannot be told apart
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cell"]].append(r["val_loss"])
    table = sorted(((c, statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else float("nan")), len(v))
                    for c, v in by.items()), key=lambda t: t[1])
    print(f"\n{len(rows)} runs over {len(table)} cells\n")
    print(f"  {'cell':28s} {'mean':>8} {'sd':>8} {'n':>3}")
    for cell, mean, sd, n in table:
        print(f"  {cell:28s} {mean:8.5f} {sd:8.5f} {n:3d}")
    if table:
        best = table[0][1]
        tied = [c for c, m, _, _ in table if m - best < NOISE]
        print(f"\n  best: {table[0][0]} at {best:.5f}")
        if len(tied) > 1:
            print(f"  NOT DISTINGUISHABLE from it ({NOISE} is the smallest difference worth believing):")
            for c in tied[1:]:
                print(f"      {c}")
            print("  Pick among these on throughput or simplicity, not on the ranking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
