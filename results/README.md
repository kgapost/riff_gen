# Grid search results

The 121 runs the hyperparameters rest on, one JSON object per line. They are
committed because `runs/` is gitignored, so these rows otherwise exist only on
the machine that produced them — and the corpus they were measured on
(`encoded_grid`) was deleted on 2026-11-11, so **no run here can be reproduced
or extended**. The table is closed; anything measured later is on a different
corpus and is not comparable to these numbers.

| file | runs | when | what it settled |
|---|---|---|---|
| `grid1_84runs.jsonl` | 84 | 2026-10-30 … 11-05 | label smoothing hurts; the DeepSeek fit overshoots ~4× |
| `grid2_27runs.jsonl` | 27 | 11-05 … 11-10 | `baseline` beats `huge` 0.61 vs 1.21 on `loss_cmp`; focal loss hurts |
| `probe_10runs.jsonl` | 10 | 11-10 … 11-11 | the seed noise floor, the context optimum, dropout |

Every run is 410.2 M tokens — 2.00 epochs of `encoded_grid`, a halved corpus
with the own-project tabs excluded — at 400 evaluation gaps, cadence 0.125,
micro-batch 16384 and activation checkpointing, so all 121 rows sit in one
table.

Fields: `run`, `model`, `lr`, `batch_tokens`, `val_loss` (the lowest validation
loss reached), `acc`, `loss_cmp` / `acc_cmp` / `lift_cmp` (the copy-proof
metric, comparable across architectures), `tokens`, `micro_tokens`,
`real_tokens_per_step`, plus the variant that cell carried (`smooth`, `focal`
or `aux`). `probe_10runs.jsonl` adds `seed`, `ctx`, `drop`, `group` and
`hours`.

**Read the noise floor before ranking anything with these.** One cell run four
times spans 0.0036 val CE, which is wider than the gap between the top seven
cells of the baseline grid. The narrative is in the root `README.md` under
"What the grid search found"; `grid_search.py` carries the same record and runs
further sweeps with seeds.
