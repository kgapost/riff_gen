#!/usr/bin/env python
"""Fourth round (Nov 10): the three questions the 111-run grid search left open.

The grid search answered batch and lr and is finished (best: baseline, b16k, lr 3.74e-04, no smoothing,
val 0.19654). It never touched three things, and these nine runs do, all on the same corpus, budget,
cadence and micro-batch as runs/grid2 so every number joins that table:

  (a) SEEDS - the grid's top five cells span 0.00281 against a reproduction gap of 0.00125 that itself
      came from a single pair of runs, so the "winner" is not distinguishable from four other cells.
      Two extra seeds on each of the top two cells gives three seeds apiece and a real noise floor.

  (b) CONTEXT - baseline's 4096-token window is the one architectural lever never swept, and it is the
      lever that matters: the repeats this corpus contains sit a median of 2 bars away (p90 17), and
      choose_context already lands a matching bar in the window 97.5 % of the time, so 2048 may lose
      almost nothing and run much faster, while 8192 should gain almost nothing. Confirming that buys
      throughput on the real corpus; disproving it changes the final model.

  (c) DROPOUT - cfg["drop"] has defaulted to 0.1 for all 111 runs. Two epochs over 205M tokens with a
      60M-parameter model is very likely underfitting, so 0.0 may simply win; the full corpus has ~2x
      the tokens, which pushes the same way.

Not sweepable from the riff.py CLI (ctx and drop are baked into MODELS, and cmd_train always keeps
weights), so this calls train_run directly. It is the SAME function cmd_grid calls, with keep_ckpt=False
for the same reason the grid used it: 16 evaluations x ~700 MB of optimiser state per run would cost
more than the runs and would distort the throughput numbers that (b) is partly about.

Finished runs are skipped on their results row, not on their directory, so this is safe to stop,
restart, or lose to a reboot at any point - a partial run simply starts over.

    usage: python probe.py [--list]

RESULTS (all ten runs, 2026-11-10 08:41 to 11-11 07:50, runs/probe/results.jsonl)
--------------------------------------------------------------------------------
(a) SEEDS - the headline, and it ends the hyperparameter search. One cell (ctx 4096, drop 0.1,
    lr 3.74e-04) measured four times gives 0.19654 / 0.19717 / 0.20015 / 0.19656: a spread of 0.0036
    and sd 0.0017, against the 0.00125 that had been used as the noise floor. Two runs at the SAME seed
    differ by 0.00063 on their own (four dataloader workers and non-deterministic CUDA kernels).
    The grid's top SEVEN baseline cells span 0.0032 and are therefore mutually indistinguishable, and
    lr 5.29e-04 has the slightly LOWER mean of the two finalists (0.19726, n=3, against 0.19761, n=4;
    the difference is 0.3 standard errors). The grid's nominal winner was one lucky seed. No further
    single-seed cell comparison can rank anything here; only a change clearing ~0.004, or 3-seed means,
    can be believed.

(b) CONTEXT - the curve turns, and the two metrics disagree about where:
        ctx  2048   val 0.22081   cmp 0.63920   1.78 h
        ctx  4096   val 0.19717   cmp 0.61554   2.09 h   <- best on cmp
        ctx  8192   val 0.19354   cmp 0.63988   2.88 h   <- best on val
        ctx 16384   val 0.21808   cmp 0.67122   3.90 h
    val_loss covers 400 gaps of which 86 % contain a bar repeated elsewhere, so it rewards recall;
    loss_cmp's 114 gaps are selected for having no repeat at all, so it measures invention only, and on
    identical target tokens. More window therefore helps recall and hurts invention - but only up to a
    point: 16384 is worse on BOTH. The reason is that the window is already big enough. At ctx 16384 the
    mean example is 5151 tokens in a ~15148-token window, because 90 % of songs fit entirely, so the
    extra capacity is mostly unused while the longer position range and ~1.8x fewer songs per gradient
    step (3.2 against 5.8 at ctx 4096, a real confound) cost something. Keep ctx 4096: it wins the
    copy-proof metric, and 8192 buys 0.0036 of val_loss - two seed sd - for 38 % more card time.

(c) DROPOUT - directionally as predicted but not proven: 0.0 -> 0.19601 / cmp 0.61581, 0.1 -> 0.19761
    mean / cmp 0.61888 mean, 0.2 -> 0.20012 / cmp 0.62837. Monotone on both metrics across six runs, and
    0.2 is clearly worse (2.7 sd on cmp), but 0.0 beats 0.1 by 0.0016, just inside one seed sd from a
    single run. Adopt drop 0.0 anyway: it is free, it is simpler, and two epochs over 205M tokens with a
    60M-parameter model should be underfitting - the full corpus has ~2x the tokens and pushes the
    same way. It is a tie-break on priors, not a measured win.

RECOMMENDED FINAL CONFIGURATION: baseline, ctx 4096, batch_tokens 16384, lr 3.7-5.3e-04 (a coin flip
inside that band), drop 0.0, no label smoothing, no focal loss.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import riff                                                   # noqa: E402

# The corpus these ten runs used. It was removed from iti-1261 on 2026-11-11 along with data/encoded, when
# the tabs were packed into data/tabs_part*.zip, and it cannot be rebuilt: the genre and album folders have
# since been recurated and the label constants changed (21 -> 28 genres, new ALBUMS). So runs/probe and
# runs/grid2 are a closed table - re-running anything here produces numbers on a different corpus.
DATA = "data_encoded/encoded_grid"   # the path the rest of the repo now uses; nothing is there yet
OUT = Path("runs/probe")
BUDGET = 410_200_000                 # 2.00 epochs of this corpus, exactly as in runs/grid2
COMMON = dict(eval_every=0.125, eval_gaps=400, gen_gaps=0, workers=4, keep_ckpt=False)
LR_BEST, LR_2ND = 0.00037399248989315373, 0.0005288253807089194      # the grid's top two cells, verbatim
BATCH = 16384

# name                        lr        seed  cfg delta                    group
PLAN = [
    ("repro_ctx4096_s1234",   LR_BEST,  1234, {},                          "b"),
    ("ctx2048_s1234",         LR_BEST,  1234, {"ctx": 2048},               "b"),
    ("ctx8192_s1234",         LR_BEST,  1234, {"ctx": 8192},               "b"),
    # ctx 16384 is the no-truncation end of the sweep, not just one more point: the usable window (~15148
    # tokens after the 0.95 margin, the head and a median gap) holds every bar of 90 % of val songs, so
    # choose_context stops dropping bars and the model sees the whole piece. ~227 bars against 4096's ~52.
    ("ctx16384_s1234",        LR_BEST,  1234, {"ctx": 16384},              "b"),
    ("drop0.0_s1234",         LR_BEST,  1234, {"drop": 0.0},               "c"),
    ("drop0.2_s1234",         LR_BEST,  1234, {"drop": 0.2},               "c"),
    ("lr3.7e-04_s1235",       LR_BEST,  1235, {},                          "a"),
    ("lr3.7e-04_s1236",       LR_BEST,  1236, {},                          "a"),
    ("lr5.3e-04_s1235",       LR_2ND,   1235, {},                          "a"),
    ("lr5.3e-04_s1236",       LR_2ND,   1236, {},                          "a"),
]
# Order: the reproduction check first, so a harness that does not land on the grid's 0.19654 is caught in
# the first two hours rather than at the end. Then the two single runs that could change the final model
# (context, dropout), then the four-run seed block last: it is the most card time for the narrowest
# question, and each seed is still useful on its own if the batch is cut short.


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.jsonl"
    done = {json.loads(l)["run"] for l in res_path.read_text().splitlines()} if res_path.is_file() else set()

    if "--list" in sys.argv:
        for name, lr, seed, delta, grp in PLAN:
            mark = "DONE" if name in done else "todo"
            print(f"  {mark}  ({grp})  {name:22s} lr {lr:.3e}  seed {seed}  {delta or '{}'}")
        return

    # Baseline, plus the grid's two overrides: the micro-batch it actually ran at and activation
    # checkpointing. Both are memory-only and leave the arithmetic identical, but they are what runs/grid2
    # used, so they stay fixed here too.
    base = riff.model_cfg("baseline") | {"micro_tokens": 16384, "ckpt": True, "smooth": 0.0}

    for name, lr, seed, delta, grp in PLAN:
        if name in done:
            riff.log(f"skip {name} (already in results.jsonl)")
            continue
        cfg = base | delta
        riff.log(f"=== probe ({grp}) {name}: lr {lr:.3e} seed {seed} "
                 f"ctx {cfg['ctx']} drop {cfg.get('drop', 0.1)}")
        t0 = time.time()
        try:
            r = riff.train_run(cfg, DATA, OUT / name, lr, BATCH, cfg["epochs"], budget=BUDGET, seed=seed, **COMMON)
        except Exception as e:                   # one run that OOMs or diverges must not abandon the rest
            riff.log(f"!!! {name} failed after {(time.time() - t0) / 60:.0f} min: {type(e).__name__}: {e}")
            continue
        be = r["best_eval"]
        row = {"run": name, "group": grp, "model": "baseline", "lr": lr, "seed": seed,
               "batch_tokens": BATCH, "ctx": cfg["ctx"], "drop": cfg.get("drop", 0.1),
               "val_loss": r["best"] if riff.math.isfinite(r["best"]) else 99.0,
               "tokens": r["tokens"], "real_tokens_per_step": round(r["tokens_per_step"]),
               "micro_tokens": r["micro_tokens"], "hours": round((time.time() - t0) / 3600, 3),
               **{k: be[k] for k in ("acc", "loss_cmp", "acc_cmp", "lift_cmp") if k in be}}
        with open(res_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        riff.log(f"=== {name}: val {row['val_loss']:.5f}  cmp {row.get('loss_cmp', float('nan')):.5f}  "
                 f"{row['hours']:.2f} h")

    rows = [json.loads(l) for l in res_path.read_text().splitlines()]
    print(f"\n{len(rows)} probe runs\n")
    for grp, title in (("b", "context length"), ("c", "dropout"), ("a", "seed repeats")):
        rs = [r for r in rows if r["group"] == grp]
        if not rs:
            continue
        print(f"--- ({grp}) {title}")
        for r in sorted(rs, key=lambda r: r["val_loss"]):
            print(f"    {r['run']:22s} val {r['val_loss']:.5f}  cmp {r.get('loss_cmp', float('nan')):.5f}"
                  f"  ctx {r['ctx']:5d}  drop {r['drop']:.1f}  seed {r['seed']}  {r['hours']:.2f} h")
        print()


if __name__ == "__main__":
    main()
