# Training runbook and the problems we will run into

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
# PyTorch first, matching your GPU:
#   RTX 4090 (and other RTX 30/40):  pip install torch --index-url https://download.pytorch.org/whl/cu128
#   GTX 1650 / 1050 Ti (Turing/Pascal): pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"                                       # numpy, pyyaml, tqdm, PyGuitarPro 0.6, pytest
python -m riffgen.tools.smoke                                 # whole pipeline on synthetic data, ~1 min
python -m riffgen.tools.smoke --huge
python -m pytest -q                                           # 25 tests, ~1 min
```

PyGuitarPro 0.6 (D26) installs and runs on Python 3.10-3.12, so **one environment is
enough**. If pytest picks up foreign plugins (e.g. ROS), run it with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

## 2. Order of work

| Step | Command | Look at |
|---|---|---|
| 1. DadaGP arrives | `python -m riffgen.dataset.inspect --src data/dadaGP` | token files vs GP files, downtunes, 7-string share, bad frets |
| 2. Preprocess DadaGP | `python -m riffgen.dataset.preprocess dadagp --workers 12 --export-gp5 50` | `data/processed/dadagp/stats.json` (tokens per bar, bars per 4096/8192 tokens, skip reasons), `skipped.csv`, `filters.csv`; open some files in `normalized/` in Guitar Pro |
| 3. Preview examples | `python -m riffgen.dataset.preview --store data/processed/dadagp --n 20` | do the gaps and guitar swaps look right? |
| 4. Pretrain baseline | `python -m riffgen.train.pretrain --model baseline` | `runs/pretrain_baseline/train.log`, `metrics.jsonl` |
| 5. Evaluate it | `python -m riffgen.eval --checkpoint runs/pretrain_baseline/best.pt --store data/processed/dadagp --split val --listen reports/listen_base` | invalid mass, similarity to real bars; listen |
| 6. Pretrain big, then huge | `python -m riffgen.train.pretrain --model big` (add `--resume` after a crash) | same |
| 7. Album folders | put files in `data/own/<Album>/{own,favourites,influences,songs}`; set `data.holdout` in `configs/albums/<album>.yaml` | – |
| 8. Preprocess album | `python -m riffgen.dataset.preprocess album configs/albums/<album>.yaml` | `skipped.csv` (tuning!), `reports/filters_<album>.csv`, listen to `reports/audit_<album>/` |
| 9. Fine-tune | `python -m riffgen.train.finetune configs/albums/<album>.yaml --model big` | held-out loss, `dadagp/loss` (forgetting), `reports/dadagp_mix_<album>/` |
| 10. Generate | `python -m riffgen.generate --album <album> --song <song.gp5>` | `outputs/<album>/…/check_report.md`, open the files |
| 11. Rate | `python -m riffgen.tools.rate outputs/<album>/<folder>` | ratings go to `outputs/<album>/ratings.jsonl` |

Configs are layered: `configs/train/{pretrain,finetune}.yaml` < `configs/model/<size>.yaml`
(`train:` section) < album config < command line (`key=value`, e.g.
`train.micro_batch_tokens=16384`).

## 3. What "training works" looks like

| Metric (in `metrics.jsonl` / log) | Healthy |
|---|---|
| `val_infill/loss` (pretraining), `val/loss` (fine-tuning) | goes down, then flattens; `best.pt` is the minimum |
| `nll_bar`, `acc_bar` | bar lines and time signatures become almost free early (acc → ~1) |
| `nll_wait` vs `nll_note` | rhythm (wait) usually gets easier than pitch/shape (note); if not, look at the timing data |
| `invalid_mass` (eval, report.json) | the probability the model put on tokens the rules forbid; should fall below ~0.05 after pretraining. High values mean the constraints are hiding a model that has not learned the format |
| `aux_pc`, `aux_onset`, `aux_section`, `aux_contrastive` (huge) | go down; if flat, the structure part is not learning |
| `dadagp/loss` during fine-tuning | may rise slowly (forgetting). A fast rise = learning rate too high or too many epochs |
| tokens/s, `mem_gb` | for planning; lower `micro_batch_tokens` if memory is tight (OOM is caught automatically) |

## 4. Problems we will stumble upon (and what the code does)

### 4.1 Data

| # | Problem | What we do |
|---|---|---|
| 1 | **The DadaGP encoder rejects 7-string Drop A.** It only accepts four tuning shapes; A E A D G B E is not one of them (its "g7_drop" is A D A D G B E). Found by testing (D22). | Tuning bypass for your files: non-guitar tracks removed, guitars labelled standard for the encoder only, real tuning restored in our `tuning:` token. Tested: identical events to the DadaGP path, exact GP5 round trip. |
| 2 | DadaGP **pitch-normalises drop tunings**: open low D of drop D is written `s6:f-2`. | Converted to physical frets (what you play) using the tuning from the GP file, or from the negative frets when only tokens exist. Lossless. |
| 3 | A drop-tuned DadaGP song without GP file that never plays fret 0-1 on the drop string cannot be recognised as drop. | Treated as standard; its frets stay pitch-correct. Rare, harmless. |
| 4 | **7-string notes are 0.1 % of DadaGP's notes.** | Music features (big/huge) transfer from 6-string; fine-tuning teaches Drop A; the DadaGP mix prefers 7-string songs. |
| 5 | Garbage frets in DadaGP (up to 87). | Notes outside 0-29 removed and counted (`bad_fret` in stats). |
| 6 | DadaGP rounds tempo to 10 BPM and stores no time signatures (6/8 = 3/4). | Real time signatures from the GP files when present (D23). Generated files keep your song's exact tempo (we write into your file). |
| 7 | Repeat semantics: `repeat_close:N` = N *extra* passes; `repeat_alternative:M` is a **bit mask** (3 = endings 1 and 2), not an ending number. | Unrolled accordingly (tested). Segno/coda/D.C. are not unrolled; such songs get a warning. |
| 8 | **Duplicate songs** in DadaGP ("Song (2).gp5", several tabbers). | Song families by normalised artist+title: used for the validation split and the plagiarism idiom rule. |
| 9 | DadaGP decides "guitar or not" by MIDI program. A lead guitar with a synth sound becomes `leads` and is removed. | For your files guitars are found by tuning (7-string Drop A) or track name. Check `removed_tracks` in the store / log. |
| 10 | More than 3 distorted or 2 clean guitars → DadaGP refuses the song. | Identical (double-tracked) tracks are removed first, then extras (logged). |
| 11 | Tempo changes attached to the drum track would disappear with the drums. | Tempo changes are kept as global events. |
| 12 | Over/underfull bars in tabs. | The timeline is trusted, the time signature recomputed; the folder check reports bad bar lengths after manual edits. |
| 13 | Tuplets and tiny durations: GP5 cannot store notes shorter than a 64th; the DadaGP decoder rounds them (bars change length) and drops tremolo picking. | Our own GP5 writer: exact durations or a 20-tick grid; tremolo picking kept. Every written bar has the exact length (tested). |
| 14 | Ties across the gap border (a tie in the bar after the gap that pointed to the old empty bar). | Fixed when splicing (tie → normal note). |
| 15 | Many tabs have no section markers. | No `section:` token then; 30 % section dropout in training so the model works with and without `--section`. |
| 16 | The simple-riff filter will sometimes call a deliberately minimal riff "simple". | `filters.csv` + GP5 snippets; `force_keep`/`force_drop` in the album config (D18). |
| 17 | Changing `configs/sections.json` or the tuning list changes the vocabulary. | Stores and checkpoints carry a vocabulary hash; a mismatch stops with a clear error. Re-preprocess and re-pretrain after such changes. |

### 4.2 Training

| # | Problem | What we do |
|---|---|---|
| 18 | **Little data** (~70-90M tokens): models memorise. | Moderate sizes, dropout 0.1, `best.pt` by validation loss, LoRA for album fine-tuning, plagiarism rate as a metric. |
| 19 | Infill examples only train the hidden bars: much compute per useful token. | Pretraining also trains prefix and suffix tokens (`context_loss_weight: 1.0`, "FIM for free"). Fine-tuning keeps the README's middle-only loss. |
| 20 | Gradient accumulation with very different numbers of target tokens per micro-batch. | The loss is normalised by the total target tokens of the whole step. |
| 21 | Padding waste (short songs, infill examples). | Examples are sorted into micro-batches by length under a token budget. |
| 22 | Several data-loader workers producing the *same* examples (a classic bug), or different data with a different worker count. | Example *i* of epoch *e* is always built from `derive(seed, "example", e, i)`; worker w builds `i % W == w`. |
| 23 | Out of memory on long contexts. | Gradient checkpointing (big/huge); an OOM skips the step and lowers the micro-batch budget by 25 %. |
| 24 | NaN loss. | Step skipped; 10 in a row stop the run ("lower the learning rate"). |
| 25 | Crash after 10 hours. | `last.pt` every `save_every` steps, written atomically; `--resume`. |
| 26 | Windows: data-loader workers start slowly (spawn), `torch.compile` needs Triton (Linux only), PyTorch may not ship flash attention. | Works without compile; memory-efficient attention is used then. Train on Linux/WSL if possible. |
| 27 | Validation on 1-2 held-out songs is noisy. | All borders × 4 and 8 bars of the held-out songs (up to 200 fixed gaps, saved in `val_gaps.json`); plus a small DadaGP set to watch forgetting. |
| 28 | Fine-tuning learns the album too well (copies favourites). | LoRA, early stopping, max 2-4 copies per favourite (D9), plagiarism rate on candidates. |

### 4.3 Generation

| # | Problem | What we do |
|---|---|---|
| 29 | The rules (D33) can hide a model that writes nonsense. | `invalid_mass` per candidate and in `eval`. |
| 30 | Ranking by model probability prefers safe, repetitive riffs. | Complexity term in the score, "repeats the neighbour riff" filter, `--min-complexity`, temperature. |
| 31 | GTX 1650: no bf16; fp16 can overflow with weights trained in bf16. | fp16 with a check; automatic retry in fp32. |
| 32 | GTX 1050 Ti (Pascal): fp16 is very slow; new PyTorch builds may not support it. | fp32 is chosen automatically; use a cu126 PyTorch build. |
| 33 | 32 candidates × 8192-token prompts do not fit in 4 GB. | The prompt is computed once and its K/V cache is shared by all candidates (tested: identical to a full forward pass). |
| 34 | A gap inside a repeated section appears twice in the unrolled song. | The first pass is filled; the other stays a `gap:N` placeholder; the result is written into the measures of your original file. |
| 35 | D33 forces the time signature of the bar before the gap, but the empty bars in your file may have another one. | The output file gets the forced time signature in those measures; `--use-gap-signatures` keeps yours. |

### 4.4 Long context (stage 1 of LONG_CONTEXT.md)

| # | Problem | What we do |
|---|---|---|
| 36 | `gap:N` now also means "skipped, far away", not only "empty or unknown". | Intended (D63). The bars' positions on the musical-time axis keep the true distance; the model learns that a skipped run is unknown. |
| 37 | The related bars are chosen from repeats found by our riff analysis (similarity ≥ 0.9). A wrong analysis means wrong or missing far bars. | `far_in_context` per example (sampler log) shows how often the far repeat is in the window; `val_far/loss` and `python -m riffgen.eval --long-context N` show whether it helps. Listen to `reports/preview_*` (`python -m riffgen.dataset.preview`). |
| 38 | At generation the hidden bars are empty, so their repeats are unknown; only the repeats of the *neighbouring* bars, intro, outro and same section can be found. | Training builds the window the same way (it never looks up the hidden bars' own repeats), so there is no train/generation mismatch. Reprise gaps are still found because the repeat of the bar before the gap leads to what followed it. |
| 39 | The model could rely on far context too much and copy the intro. | `neighbour_copy` and the plagiarism check (SELF-QUOTE) flag it; the reprise share (25 %) is a config value (`augment.reprise_prob`). |
| 40 | Huge's song memory is 4× larger (512 bars × 192 tokens): more memory in the data loader and in the bar encoder (~1.2 s extra per step is plausible; not measured). | Gradient checkpointing is on; lower `max_song_bars` / `max_bar_tokens` in `configs/model/huge.yaml` if memory is tight. |
| 41 | Songs that fit the window are shown completely, so long-context tasks (reprise, bookend) matter mainly for long songs. | Preprocessing prints how many songs fit 4k/8k/16k/32k tokens and how many have far repeats; if few DadaGP songs do, the album fine-tuning set carries these tasks (the same config applies there). |

## 5. Rough time and memory budget

| | baseline | big | huge |
|---|---|---|---|
| Pretraining (default steps, RTX 4090, bf16) | ~4 h | ~12 h | ~16 h |
| Album fine-tuning (10 epochs max, early stop) | 10-30 min | 20-60 min | 30-90 min |
| GPU memory, pretraining | ~10 GB | ~12-16 GB | ~14-18 GB |
| Generating 32 candidates for an 8-bar gap, RTX 4090 | seconds | ~10-20 s | ~20-30 s |
| Same on GTX 1650 | ~1 min | ~3-5 min | ~5-8 min |

These are estimates from parameter counts and typical throughput. The logs report real
tokens/s and memory.
