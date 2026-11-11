# RiffGen

RiffGen fills the empty bars of a 50–90 % complete Guitar Pro 5 song with new
7-string Drop A guitar riffs. You choose the style with labels, for example
"genres prog + stoner, album cs". Each gap gets many candidates. Every
candidate is checked, then ranked. The output is a source of ideas: you keep
the few good candidates and edit them in Guitar Pro.

Everything is in one script, [riff.py](riff.py): preparing the corpus, three
models, training, evaluation, filling gaps and a grid search.

## Summary

| # | Item | Decision | In riff.py |
|---|------|----------|-----------|
| 1 | Task | Gap filling only: the model writes the missing bars (5+ empty bars, or `--gap`) | yes |
| 2 | Input / output | GP5 in, GP5 out. Output is guitar tracks only, 7-string Drop A | yes |
| 3 | Tunings | Every tuning is mapped onto Drop A, keeping riff shapes | yes |
| 4 | Tokens | Bar by bar: drums, bass, then each guitar as beats of string:fret notes and techniques | yes |
| 5 | Time signature & tempo | Taken from the song, never generated | yes |
| 6 | Architecture | Three models: flat baseline, hierarchical big, hierarchical huge with a pointer memory | yes |
| 7 | Long context | Structure-aware window (baseline); every bar of the song (big, huge) | yes |
| 8 | Labels | Genre, album, source, relation and section, combined with weighted guidance | yes |
| 9 | Relation labels | `copy` / `variation` / `answer` / `new` / `chug` per gap | yes |
| 10 | Gap sampler | New random gaps for every song in every epoch. Validation gaps are fixed | yes |
| 11 | Favourites | Sampling weights per label (`--focus`) | per label, not per song |
| 12 | Training stages | Pretrain on all data → style stage → per-album LoRA | yes |
| 13 | Album LoRA | `--lora 16` on top of a pretrained model | yes |
| 14 | Candidates | N candidates per gap → copy reject → rank by likelihood → top-k GP5 files | yes |
| 15 | Plagiarism | Three levels (exact / transposed / rhythm) | planned |
| 16 | Playability | Hard rules during decoding; ergonomic score | rules yes, score planned |
| 17 | Hardware | Train on an RTX 4090 in bf16; generate on a GTX 1650 in fp32 | yes |
| 18 | Reproducibility | One master seed; every example and candidate has its own seed | yes |
| 19 | Evaluation | Loss by token class and generation metrics, every `--eval-every` epochs | yes |
| 20 | Sections | A learned segmenter labels every bar and finds where sections start | yes |

## Discussion

**1. Task.** The model has one job: fill gaps. It sees the bars before and
after each gap, so its riffs fit what follows, such as a theme that returns
minutes later.

A gap is 5 or more consecutive measures that are empty in all guitar tracks;
`--gap 33-40` sets one by hand. Each gap is filled on its own. The song's
other gaps stay in the context as unknown placeholders. A gap that crosses a
repeat sign is filled in pieces, each on its first pass.

**2. Input / output.** GP5 is used because MIDI loses strings, frets,
techniques and tuning. Files are read and written with PyGuitarPro; GP3 and
GP4 are read too, and GP6/GP7 files are converted with `data_convert.py`
(see [Collecting tabs](#collecting-tabs)). The model reads drums and bass as
context. In training they are dropped 15 % of the time each, so the model
also works on guitar-only songs. Up to four guitar tracks are kept (`G_SLOTS`);
double-tracked copies are kept once. Each candidate is the whole song: the
new bars are spliced into your original file, so everything else stays
exactly as written. The song you fill must have its guitars in 7-string
Drop A, without capo.

**3. Tunings.** All songs are moved onto 7-string Drop A (A E A D G B E):

- One transposition per song, chosen so that most strings keep their frets.
  E standard keeps every fret on strings 1–6; Eb, D and C standard keep
  their shapes one to four semitones up.
- A 6-string drop tuning (Drop D, Drop C, …) lands on Drop A's six *lowest*
  strings, so one-finger drop chords stay drop chords on the low A.
- A string with a different interval (the B string of a drop tuning, the
  low B of B standard) keeps its pitch with a fret shift.
- A fret outside 0–24 moves, at the same pitch, to the nearest free string
  that can play it, or is dropped.

The bass is transposed by the same amount. For augmentation, 30 % of the
examples are shifted 1–3 semitones within Drop A when every fret allows it.

**4. Tokens.** Each bar is one token sequence, which every model uses:

```
ts:4/4 [sec:chorus] trk:dr <beats> trk:bs <beats> trk:g0 <beats> trk:g1 <beats> <eob>
beat (guitar) = d:8 [trem] n:7:0 fx:pm n:6:2 …   (duration, then notes low string first, each with its techniques)
beat (bass)   = d:8 bp:33                         (pitch after transposition)
beat (drums)  = d:16 dr:kick dr:hat
```

- Durations are Guitar Pro's own note values (whole to 64th, dotted,
  triplets, quintuplets, sextuplets, septuplets), so writing back is exact.
- Notes are `n:string:fret` on Drop A. Techniques follow their note: tie,
  dead, ghost, palm mute, hammer-on/pull-off, slide, bend, harmonic,
  vibrato, let ring, accent, staccato. Tremolo picking is a beat token.
- Every track fills its bar exactly, or is empty (a whole-bar rest).
  Overfull bars are cut and underfull ones end with rests.
- Repeats and alternative endings are unrolled (at most four passes).
- Everything the model *writes* (the guitar part and the end of bar)
  carries a flag in the stored data. The rest of the bar (time signature,
  section, drums, bass) is given to it at generation, so it is never trained
  as a target.

The vocabulary is about 700 tokens; labels are added at its end, so music
tokens keep their ids. Over the whole collection (61,123 tabs) 93.9 % are
kept (the rest have nothing pitched in them at all, or one bar too dense to
encode). A track does not have to be a guitar: a lead synth, a pad, a piano,
strings or a vocal line is written on a fretboard like any other Guitar Pro
track, so it is mapped onto Drop A at its own pitch and played. A real guitar
claims a slot first and is never displaced; the other parts fill what is left,
and a song with nothing but bass and drums has its bass line played. They
average 68 tokens per bar and 111 bars per song, which puts the corpus at
431 M tokens per epoch.

Why this encoding: bar-major order with the tracks one after another inside
each bar is what MMM and NotaGen's interleaved ABC use for multi-track
music. It keeps a bar together, so a bar can be a unit for the hierarchical
models. Guitar Pro's beat structure keeps rhythm exact and the tab's
fingering (string and fret) intact. DadaGP's token format was dropped
together with its encoder.

**5. Time signature & tempo.** These are never generated. Each gap bar keeps
the time signature written in that measure of your file, and the tempo is
the song's. `--section bridge` asks for a section on the gap's first bar.

**6. Architecture.** Data, not the GPU, is the limit: a few hundred million
parameters suit about 0.45 B tokens per epoch, and more would mostly
memorise other people's riffs. So the three models differ mainly in how
they see the song, not in width:

| | baseline | big | huge |
|---|---|---|---|
| Idea | flat fill-in-the-middle decoder | hierarchical bar model | big + pointer memory and planning |
| Parameters (non-embedding) | 59M | 118M | 282M |
| Layers × width | 12 × 640 | bar encoder 4 × 512, global 12 × 768, local 6 × 512 | 6 × 640, 16 × 1024, 8 × 640 |
| Context | 4,096 tokens (~50 bars) | every bar (512 in training, 1,024 when filling) | same as big |
| Training FLOPs per token | 543M | 118M | 280M |

*Baseline* (ProgGP, DadaGP-style GPT with fill-in-the-middle). The window is
`labels tempo len <pre> bars before <suf> bars after <mid> gap bars`; bars
that do not fit become `gap:N`. The context bars are trained too, at half
weight ("FIM for free", Bavarian et al.), which gives more learning per
token.

*Big* (MEGABYTE, NotaGen bar patches, BACH):

- A bar encoder turns every bar into one vector.
- A global model runs over the label slots, all known bars, then the gap's
  bars shifted by one, in fill-in-the-middle order. It attends both ways
  over the context and causally over the gap.
- Rotary positions are the bars' true indices, so "the same place 32 bars
  ago" is a fixed distance even though the gap comes last in the sequence.
- A local decoder writes each gap bar, token by token, from the global
  state of that bar.
- When filling, the global model runs again after every bar, so bar 3 of
  the gap knows bars 1 and 2.

The big model sees whole songs at a fifth of the baseline's compute per
token.

*Huge* adds three things to big:

- **Pointer memory.** A pointer head scores every context bar against the
  bar being written. The local decoder cross-attends to the guitar tokens of
  the 3 best bars, so a riff from 150 bars back can be quoted or varied note
  by note. A bar vector cannot copy exactly; token-level attention can
  ("Repeat After Me"). The pointer is trained towards the bars that really
  resemble the target (same riff, similar pitch classes and rhythm), with
  the true bars used half the time early on. This is Museformer's
  "fine attention to structure-related bars", with the related bars chosen
  by content.
- **Planning heads** on the global state. They predict each gap bar's pitch
  classes and 16-step onset grid before any note is written, and the
  section of every bar.
- **Phrase positions**: bar k (0–15) since the last section marker.

The cross-attention gates start closed, so training opens them only if they
help.

All three models use pre-norm RMSNorm, SwiGLU, rotary positions and
normalised queries and keys (stable at higher learning rates), with dropout
0.1. The big and huge models use gradient checkpointing.

**7. Long context.** Seeing the whole song needs three things:

1. The far bars must reach the model.
2. They must come at the right resolution: a 22-bar progression is ~2,000
   tab tokens but only 22 bars.
3. Training must reward using them, because almost every next token is
   predictable from the last few bars.

How the models handle them:

- **Baseline:** retrieval. When the song does not fit the window, the window
  is filled in order:
  1. the nearest bars, up to 50 % of the budget;
  2. related bars, up to 85 %: repeats of the four bars on each side of the
     gap and what followed (or led into) them, plus the first and last 8
     bars;
  3. more nearest bars for the rest.
- **Big and huge:** whole songs at bar resolution.
- **Huge:** also reaches back at token resolution through the pointer.
- **Training rewards far context** (item 10): about 25 % of the gaps sit on
  bars that repeat ≥ 32 bars away, and 10 % of the examples hide the bars
  around the gap.

Options for later: a beat-level harmony stream, a section-level form model,
and learned block-sparse attention (NSA, MoBA) for token-level whole-song
attention.

**8. Labels.** Each song gets labels from its folder:

- `genre:<folder>` for `tabs - <genre>` (for example `genre:prog`,
  `genre:doom_stoner_sludge`); a folder not listed in `GENRES` becomes
  `genre:other`;
- `src:early` for the early-years folders (`generic`, `exercises`,
  `metal generic`; edit `EARLY` in riff.py);
- `album:<id>` plus `src:own` for your own songs in
  `data/tabs - own - <album> - own/`, and `album:<id>` plus `src:insp` for the tabs you
  assign to that album as inspiration, in `data/tabs - own - <album> - insp/`.

The two sources of an album are a deliberate distinction: the model may copy
from `src:own` or come very close to it, but may only be informed by
`src:insp`, never reproduce it (§15). A file put
straight in `data/tabs - own - <album> - insp/ or - own/` with no subfolder is read as `insp`, the safe
side, and counted in `prepare`'s report.

The label vocabulary is built from the constants in riff.py (`ALBUMS`,
`GENRES`, `SOURCES`) and not from whatever labels a corpus happens to hold,
so every album is registered while its folder is still empty and preparing
the corpus again always yields the same token list. That is what keeps
checkpoints loadable as the album folders fill: `--init` refuses a
checkpoint trained on another vocabulary. Adding a name to those constants
renumbers the label ids and costs a fresh pretrain, so `prepare` stops if it
finds an album folder that is not in `ALBUMS`.

The model reads four label slots (genre, album, source, relation). In
training, each label is dropped 15 % of the time (MusicGen and Diff-Symbo
drop conditions 10–20 %), so the model also learns every unconditional
distribution.

When filling, composable classifier-free guidance combines the labels:
`L_uncond + Σ wᵢ·(L_labelᵢ − L_uncond)`, one row per label. Positive labels
use `--cfg` (1.5). `--neg` labels get a negative weight (`--neg-cfg`, 0.5)
and steer away. The default `--neg src:early,rel:copy,rel:chug` keeps the
early years, plain copies and chugs out. CFG was shown to work for
autoregressive models in "Stay on topic with CFG", and summing conditions
comes from composable diffusion. Several genres at once (`genre:prog,
genre:doom_stoner_sludge`) are just more rows. The prompt is computed once
per label row and shared by all candidates.

Section markers ("Chorus 2", "Refrain") become `sec:` tokens on the bar
where they start. They are dropped from 30 % of the training examples.

**9. Relation labels.** Each training gap is labelled by how it relates to
the bars before it, using exact bar identity and Jaccard similarity of
pitch classes and onsets:

- `copy`: the same bars again;
- `variation`: similar pitch and rhythm (≥ 0.6);
- `answer`: related rhythm, different pitches;
- `new`: neither;
- `chug`: mostly one or two pitch classes on the low strings.

When filling, `rel:answer` or `rel:variation` asks for one, and `rel:copy`
and `rel:chug` push away.

**10. Gap sampler.** New gaps for every example, from its own seed:

- lengths 1/2/4/8/16 bars (5/10/25/35/25 %);
- 60 % start on a section marker or every 4th bar, 25 % on far repeats, the
  rest anywhere;
- every gap has guitar notes.

The big and huge models hide 10–50 % of the window in up to four gaps, and
every hidden bar is a target. The baseline has one gap per window. 20–30 %
of the examples get an extra unknown placeholder gap. Validation and test
songs are split by song family (about 1.5 % each); their gaps are chosen
once at `prepare`.

**11. Favourites.** `--focus album:cs=20,src:early=0.3` changes how often
songs with a label are drawn. Every draw gets new gaps, so up-weighting
does not repeat examples. Per-song weights are planned.

**12. Training stages and hyperparameters.**

1. Pretrain on all ~61k tabs with labels.
2. Style stage: `--init runs/big/best.pt --focus genre:prog=4,src:early=0.2`.
3. Album: `--init … --lora 16 --focus album:cs=20`.

Loss is computed on what the model writes. It is normalised by the target
tokens of the whole step, and gradients are clipped at 1.0. The optimiser is
AdamW (β 0.9/0.95, weight decay 0.1) with 2 % warm-up and cosine decay to
10 %.

Batch size and learning rate are calculated, not guessed. DeepSeek LLM
(2024) fitted the optimum against compute C = M·D, with M the non-embedding
training FLOPs per token and D the training tokens:
`lr = 0.3118·C^−0.125`, `batch = 0.2920·C^0.3271` tokens. `riff.py hparams`
computes M for each model (attention included; for the hierarchical models,
the global model's cost spread over a bar's tokens and the local decoder on
~30 % of the bars) and D from the prepared corpus. With the estimate of
0.43 B training tokens per epoch:

| | epochs | tokens | compute | learning rate | batch (tokens) | 4090 hours* |
|---|---|---|---|---|---|---|
| baseline | 4 | 1.72 B | 9.3e17 | 1.8e-3 | 217k | 3.9 |
| big | 4 | 1.72 B | 2.0e17 | 2.1e-3 | 131k | 0.9 |
| huge | 3 | 1.29 B | 3.6e17 | 2.0e-3 | 160k | 1.5 |

\* At 40 % of the 4090's bf16 peak; the hierarchical models reach less in
plain PyTorch, so plan for 2–3× longer.

**Those calculated values are the prediction, and 121 runs disproved it.**
The fit overestimates both lr and batch by about 4× for this task, and the
√batch scaling of the learning rate over-corrects: the best absolute lr sits
in a narrow 3.7–7.5e-04 band at *every* batch size from 16k to 132k. What
the search actually settled on is

| | value | |
|---|---|---|
| model | `baseline` | flat decoder, 4096-token FIM window |
| context | 4096 | an interior optimum: 2048 and 16384 are both clearly worse |
| batch | 16384 tokens | the fit said 217k |
| learning rate | 3.74e-04 | the fit said 1.8e-3 |
| dropout | 0.0 | on priors, not proof |
| label smoothing | off | hurt in all 111 runs |
| focal loss | off | hurt monotonically |

`train_n_test.py` has these as its defaults, so it needs no flags. The full
account is in [What the grid search found](#what-the-grid-search-found);
read the noise floor there before trusting any ranking, including this one.
`grid_search.py` runs further sweeps, with seeds, and says which cells it
refuses to tell apart.

**13. Album LoRA.** About 200 tabs cannot safely move all the weights.
`--lora 16` freezes the model and adds rank-16 updates to every attention
and MLP projection; the token and label embeddings train too. Add replay
from the album's genres with `--focus`, and keep the checkpoint with the
lowest loss on the held-out songs (`data/tabs - own - <album> - own/holdout/` goes to
validation).

**14. Candidates.** Each candidate is sampled with temperature and top-p
(`--temperature`, `--top-p`) under the bar rules of item 16. Candidates that
repeat a bar of the song are dropped (`--allow-copy` keeps them). The rest
are ranked by mean log-probability. The top `--keep` go to
`outputs/<song>_gap_<a>-<b>/cand_NN.gp5` with a `report.json`.

**15. Plagiarism.** Planned for riff.py: winnowed 8–16-note fingerprints at
three levels (exact tab, transposed, rhythm), weighted by rarity, with
versions of one song counting once. The thresholds are per source, because
"not mine" is not one thing: `src:own` is only reported (copying your own
song is the point), `src:insp` is relaxed — a longer match is needed to
count, and transposed or rhythm-only matches are ignored, since a tab chosen
as an influence is meant to show through — and the rest of the corpus is
strict. Which tier applies is decided per album by the folders, so the same
tab is `insp` when filling the album it was assigned to and plain corpus
material anywhere else. `red` rejects a candidate, `yellow` keeps it and
lists it with its source. Until then, `--neg` and the copy reject keep plain
copies out, but quotes of other songs are not checked.

**16. Playability.** Enforced while writing:

- every guitar track fills its bar exactly or stays empty;
- tuplets come in complete groups of one kind;
- at most one note per string per beat, low string first;
- frets stay in 0–24, and each technique appears once per note.

An ergonomic score (fret span, shift speed at tempo, notes per second) is
planned.

**17. Hardware.** Training runs on the 4090 in bf16 (fp32 elsewhere).
Generation works on the GTX 1650 and on a CPU (`--cpu`). The baseline stores
the prompt's keys once per label row and shares them across candidates; the
big and huge models re-run the small global model per bar. On Windows,
`torch.compile` is unavailable and data workers start slowly; Linux or WSL
is faster.

**18. Reproducibility.**

- Every training example is built from `(seed, epoch, position)`, so the
  data does not depend on the number of loader workers.
- Every candidate comes from `--seed`.
- `run_info.json` records the configuration.
- Training is not bit-exact on GPUs.

**19. Evaluation** (every `--eval-every` epochs, and with `riff.py eval`):

- loss and accuracy on the fixed gaps, split into rhythm (durations), pitch
  (notes) and technique;
- for huge, the pointer, plan and section losses;
- for a few gaps, written candidates scored on pitch-class similarity and
  onset similarity to the real bars, copy rate and note density (1.0 = as
  dense as the real bars), saved as readable token text in
  `runs/<model>/samples_*.txt`.

`best.pt` is the checkpoint with the lowest validation loss; if validation
loss rises while training loss falls, the model is memorising. Planned: the
far-context test (loss on far-repeat gaps with the whole song vs. ±16
bars) and ratings of listened candidates.

**20. Sections.** 40 % of the tabs (22,684 songs, 3.0 M bars) have at least
two section markers; the other 33,532 have none. The section map knows
English, abbreviations ("pre ch", "vs", "ref") and the corpus's other
languages (estribillo, coro, verso, puente, pont, refrén, zwrotka, sloka,
przejście). Letters, "riff" and empty titles stay "other": they mark a
border but name no section.

`riff.py segment` learns sections from the marked tabs:

- **Inputs per bar:**
  - the bar's tokens, as a learned bag of embeddings (guitar and context
    tokens kept apart; section tokens removed, since they are the answer);
  - pitch classes, onset grid, density and low-string share;
  - position in the song;
  - repetition: how often the bar recurs, whether this is its first time,
    whether it equals the bar 1, 4 or 8 bars back.
- **Model:** a 6 × 256 bidirectional transformer over all bars (5.3M
  parameters), with rotary positions on the bar index.
- **Outputs:** the section of every bar, and whether a section starts there.

The old segmenter in `riffgen/analysis/riffs.py` found only borders, with
hand-set thresholds, and never names. It is kept here as a baseline, with
its novelty curve vectorised (2–14× faster).

What improved it (all options of `riff.py segment train`):

- `--ssm`: the bar self-similarity is added to every attention score, with a
  learned weight per head, so a chorus bar attends to the other chorus bars
  from the first step.
- `--named`: "other" bars are unlabeled (their markers still count as
  section starts), and the model only predicts named sections.
- Every epoch, drums or bass are dropped and the song is transposed 1–3
  semitones (pitch classes rotated with it).
- Viterbi decoding: one label per section; a new section may only start
  where the model expects one, and section transitions are learned from the
  training songs.
- `--teacher`: self-training ("noisy student"). The v2 model labels the
  33,532 unmarked tabs; the 1.35 M bars it is at least 60 % sure of train a
  fresh model together with the real labels.

Training ran on a GTX 1650 with 17,011 songs (2.26 M bars): v2 in
17 minutes, v3 in 23 minutes. Test: 4,506 other songs (585k labelled bars,
split by song family).

| | v1 (first model) | v2 (+ similarity bias, named, augmentation) | v3 (+ self-training) |
|---|---|---|---|
| bar accuracy, named sections, per bar | 55.5 % | 60.4 % | 61.0 % |
| bar accuracy, named sections, Viterbi | 58.9 % | 61.0 % | **61.5 %** |
| macro F1, 9 named sections | 0.41 | 0.42 | 0.42 |
| section starts F1, exact / ±1 bar | 0.62 / 0.66 | 0.64 / 0.68 | **0.65 / 0.68** |

Baselines: always the majority class 25.3 %; class by position in the song
33.1 %. The old rule-based borders reach start F1 0.33 / 0.42, and a start
every 8 bars 0.17 / 0.32. Per section (v1): intro 0.66, verse 0.58, chorus
0.57, solo 0.54, outro 0.50, prechorus 0.29, interlude 0.22, bridge 0.19,
breakdown 0.11. The weak ones are the sections tabbers name least
consistently.

Two hard rules were tried:

1. **A section starts at a bar start.** This holds by construction: markers
   sit on measures and the model predicts per bar.
2. **A section lasts at least X bars, except the intro** (`riff.py segment
   rules`, X = 1–8 on v2). It does not help: X ≤ 4 changes nothing (the
   model rarely predicts shorter sections), and X ≥ 5 costs accuracy,
   because 23 % of real sections are under 5 bars (7 % under 3). X = 3–4
   with Viterbi is a safe clean-up for written markers, with the same
   accuracy and slightly better section starts (±1 bar F1 0.669 vs 0.658).

`riff.py segment label` labels the 33,532 tabs without markers. The model
takes 41 s for all of them (3.25 M bars) on the GTX 1650. Writing markers
into the files costs about 0.08 s per file to read, encode and write, which
is about 1 hour on one core or about 10 minutes over the 12 cores (the
corpus encoding scaled 4.7× there). It writes labelled copies and never
changes the originals.

## What the grid search found

121 runs over twelve days (2026-10-30 to 11-11). Every one was 410.2 M tokens
— 2.00 epochs of `encoded_grid`, a halved corpus with the own-project tabs
excluded — at 400 evaluation gaps, cadence 0.125, micro-batch 16384 and
activation checkpointing, so all 121 numbers sit in one table. The raw rows
are committed in [`results/`](results/) — one JSON object per run — because
`runs/` is gitignored and they would otherwise exist only on the machine that
produced them. `grid_search.py` carries the same record in its docstring.
Note that **none of it can be reproduced or extended**: `encoded_grid` was
deleted on 2026-11-11, so anything measured later is on a different corpus.

**Round 1 — 84 runs, baseline × big × huge.** Settled label smoothing: it
hurt on both best and mean, on both architectures, without exception. It also
found the grid pegged against its own lower edge — batch and lr were monotone
and best at the lowest point tried — so it did not bracket the optimum, and
the DeepSeek fit was overshooting by about 4×. The raw losses (baseline 0.198,
big 0.973, huge 0.802) were *not* a like-for-like quality comparison: a flat
model reads its context as literal tokens and can copy a repeated bar, while
the hierarchical models see one pooled vector per bar.

**Round 2 — 27 runs, shifted downward, plus a fair metric.** `loss_cmp` scores
only the 114 validation gaps where **no bar of the gap occurs anywhere else in
the song**, so nothing can be copied and both architectures weight an identical
set of target tokens. On it, `baseline` beat `huge` 0.61 against 1.21 — every
baseline run ahead of every huge run. That closed the architecture question and
the remaining 11 `huge` cells were cancelled rather than spend ~29 h confirming
it. Focal loss was tried here and hurt monotonically: 0.226 at focal 1 and
0.279 at focal 2, against 0.199 for the same cell unfocused.

| top of the table | val CE | loss_cmp |
|---|---|---|
| `b16k lr 3.7e-04` | 0.19654 | 0.61456 |
| `b16k lr 5.3e-04` | 0.19769 | 0.61847 |
| `b64k lr 7.5e-04` | 0.19916 | 0.62126 |
| `b32k lr 7.5e-04` | 0.19920 | 0.62293 |
| `b32k lr 5.3e-04` | 0.19935 | 0.62278 |

**Round 3 — 10 runs, the three axes the grid never touched.**

*Seeds, and this is the result that governs everything else.* One cell
(ctx 4096, b16k, lr 3.74e-04) run four times gave **0.19654 / 0.19717 /
0.20015 / 0.19656** — a spread of **0.0036**, sd 0.0017. Two of those share a
seed and still differ by 0.00063, from the four dataloader workers and
non-deterministic CUDA kernels alone, so even an identical command does not
reproduce. The consequence is blunt: the table above spans 0.0028 across its
top five, and the top **seven** cells of the 21-run baseline grid span 0.0032
— all inside one seed spread. **They were never distinguishable**, and the
nominal winner was one lucky seed: over three seeds `lr 5.29e-04` has the
marginally better mean (0.19726 against 0.19761).

> Believe an effect only if it clears ~0.004 val CE, or if 3-seed means
> separate it cleanly. Single-run rankings on this task are noise.

*Context length has an interior optimum*, which is why it is 4096 and not a
default:

| ctx | val CE | loss_cmp | hours |
|---|---|---|---|
| 2048 | 0.22081 | 0.63920 | 1.78 |
| **4096** | 0.19717 | **0.61554** | 2.09 |
| 8192 | **0.19354** | 0.63988 | 2.88 |
| 16384 | 0.21808 | 0.67122 | 3.90 |

The two metrics disagree in the middle and agree at both ends. 8192 wins
val CE — which is dominated by the 86 % of gaps containing a repeated bar, so
it rewards *recall* — and loses `loss_cmp`, whose gaps are copy-proof, so it
measures *invention*. It also costs 38 % more time. 16384 is worse on both,
because the window is already big enough: 90 % of songs fit inside it entirely,
and at ctx 16384 the mean example is 5151 tokens in a ~15148-token window.
2048 losing 0.024 was a surprise — repeated bars sit a median of 2 bars apart
and `choose_context` already captures 97.5 % of them, so the window's value is
mostly plain local context, not retrieval.

*Dropout* came out 0.0 → 0.19601, 0.1 → 0.19761 (mean of four), 0.2 → 0.20012:
monotone on both metrics across six runs, but 0.0 beats 0.1 by less than one
seed sd. It is adopted on priors — two epochs over a corpus this size should
underfit — and is the one axis still genuinely open.

**Also settled, so do not pay for them again:** the architecture, the context
length, label smoothing, focal loss, large batches (`b132k`'s own best cell
lost to every b16k/b32k/b64k cell in the band and degraded monotonically
above it), and the shape of the lr/batch fit. **Still open:** dropout, budget
scaling (deliberately never probed — `encoded_grid` held half the tokens, so
4× budget there would have been 8 epochs of a subsample and would not have
transferred), and lr × batch if the corpus moves the optimum, which encoding
every pitched part may well have done.

## Pitfalls handled

- Repeats: `repeatClose` counts *extra* passes, and `repeatAlternative` is a
  bit mask (3 = endings 1 and 2).
- Duplicates: the same tab is kept once per folder, on purpose - copying a
  file into an album folder is how a tab is assigned, and the folders are
  the only truth. So at most 3 versions of one song are kept
  (`--max-versions`) *per top-level folder*, otherwise that limit would drop
  the very copy carrying the album label. Versions form a family that stays
  in one split. A copy still trains, since being in two folders is a
  deliberate reweighting, but it supplies no validation or test gaps, where
  it would only score the same bars twice; which copy counts as the original
  is settled by sorted path after encoding, so it does not depend on the
  order the workers finish in.
- Guitar or not: a 6–7 string track with a guitar program or a guitar-like
  name. A bass is found by program, name, 4–5 strings or a low tuning.
- A bar too dense for 384 tokens loses its drums, then its bass; a song
  with a bar still too dense is skipped.
- Tabs with an exotic time signature (denominator not 1–32) are skipped.
- Changing labels changes the vocabulary: a checkpoint used with another
  corpus stops with a clear error.
- A stopped training continues with `--resume`; `last.pt` is written
  atomically. A step with a non-finite gradient is skipped.
- Ranking by likelihood prefers safe, repetitive riffs: the copy reject, the
  `rel:copy` / `rel:chug` guidance and temperature counter that.

## Getting the tabs (Drive backup)

The tab collection (`data/tabs - */`, about 56.6k Guitar Pro files, 2.9 GB, 196 MB zipped) is **not in git**: the tabs are
other people's transcriptions. It is kept as three zip files plus `tabs_manifest.json` in a private Google Drive folder
(<https://drive.google.com/drive/folders/12636du_4adjeTp9BEyaGzSNEqL9PIedN>). On a new machine:

```
pip install gdown
python data_tabs.py fetch              # downloads the Drive folder to data_encoded/tabs_backup/ and unzips it into data/
python data_tabs.py fetch --from DIR   # the same from zips you downloaded by hand (DIR holds tabs_part*.zip)
python data_tabs.py verify             # file counts per folder against tabs_manifest.json
```

After the collection changed, `python data_tabs.py pack` rebuilds the zips and the manifest in `data_encoded/tabs_backup/`;
upload them to the Drive folder (replace the old files). The encoded corpora (`data_encoded/encoded*`) are not backed up:
`python riff.py prepare` rebuilds them from the tabs.

## Collecting tabs

The tab tools live next to `riff.py`, outside the package: `data_dl.py` collects
tabs and `data_convert.py` turns them into GP5.

`python data_dl.py list <artist URL>` adds the free community Guitar Pro tabs
rated above 4.4 from public artist pages to `data/list.csv`. It needs no
account, waits 7.5–15 s between requests and stops at the first refusal.

`python data_dl.py` drives a real Chromium through `data/list.csv` (one row
per tab id with its status) and presses each tab's Download button. Log in with
`UG_USER` / `UG_PASSWORD`; the profile in `data_encoded/profile` keeps the login.
Files land in `data/tabs - generic/` as `Artist - Song[__vN][__id].ext`.
File headers are checked, and HTML pages go to `data/errors/`. It pauses
2.5–5 s per tab and longer every 100 and 1000 tabs, stops when the site
reports a download limit, and resumes from `data/state.json`
(`--retry-failed`, `--only-first N`, `--report`).

`python data_dl.py get <tab URL | artist URL> ...` downloads one tab, or every
song of a band (the best-rated free community Guitar Pro version of each song
not on disk yet), the same way, and records them in `data/list.csv`.

`python data_curate.py <files or folders> --out <folder>` applies the same
curation rules to a collection of your own, leaving the originals alone
(`--genre-folders`, `--move`); it writes `curate_report.csv` in `--out`.

Afterwards `python data_dl.py sort --manifest data/state.json ...` renames
the downloads after the list and files them, and `python data_convert.py`
converts GP3/4/6/7/8 to GP5 in `data/gp5/`. Every output is read back and compared.
Unreadable files, GP1/2 files and 8-string songs are moved to `__erroneous`
or `__8string`.

Ultimate Guitar's robots.txt disallows `/tab/download`. The script uses your
own account, which is what is at risk, and a daily quota may end long runs.

## Before you start

- Put your own songs in `data/tabs - own - <album> - own/` and 1–2 of them per album in
  `data/tabs - own - <album> - own/holdout/`; put the tabs you want that album to be
  inspired by in `data/tabs - own - <album> - insp/`. The album folder must be named
  after an album in `ALBUMS`.
- Mark gaps with 5 or more empty measures in all guitar tracks, with the
  right time signatures. A marker such as "Bridge" helps.
- Add section markers to your own songs where you can.

## Code

Install PyTorch for your GPU first (`cu128` for RTX 30/40, `cu126` for the
GTX 1650), then `pip install -e ".[dev]"`.

```bash
python train_n_test.py                       # encode if needed, train the winning configuration, test best.pt
python train_n_test.py --test-only runs/final   # score an existing run on the held-out test split
python grid_search.py --list                 # what a further sweep would run, and what is already settled
python grid_search.py --lrs 1 --drops 0,0.05,0.1    # the one axis still open, 3 seeds a cell

python riff.py prepare                       # 1) split, curate, encode: data/tabs - */, data/tabs - own - */ -> data_encoded/encoded/
python riff.py hparams                       # the CALCULATED batch/lr - the grid disproved these, see §12
python riff.py grid                          # the original single-seed grid; grid_search.py supersedes it
python riff.py train --model big             # 2) train; 3) evaluates every --eval-every 0.5 epochs
python riff.py train --model big --init runs/big/best.pt --run runs/big_cs --lora 16 --focus album:cs=20
python riff.py eval --ckpt runs/big/best.pt  # 3) test-split loss and generation metrics
python riff.py infer --ckpt runs/big/best.pt --song song.gp5 \
    --labels genre:prog,album:cs,rel:answer --section bridge   # 4) fill every gap
python riff.py train --model huge --tiny --budget 4e6           # a tiny smoke test on CPU
python riff.py segment train --ssm --named --run runs/seg_v2                 # 6) sections (~17 min, GTX 1650)
python riff.py segment train --ssm --named --teacher runs/seg_v2/seg.pt --epochs 6   # self-training on the unmarked tabs
python riff.py segment eval                  # test set: bar accuracy, F1 per section, section-start F1, baselines
python riff.py segment rules                 # minimum section length X = 1..8
python riff.py segment label --write 300     # label the tabs without markers; --write times N labelled copies
```

The `riffgen/` package is the previous, DadaGP-based pipeline; riff.py
replaces it.

## References

- MEGABYTE: Yu et al., 2023 (global patch model + local byte decoder).
- NotaGen: arXiv 2502.18008, 2025 (bar-stream patching, interleaved ABC).
- BACH: [arXiv 2508.01394](https://arxiv.org/abs/2508.01394), 2025 (bar-level patches for long songs).
- MMM: Ens & Pasquier, 2020 (bar-major multi-track tokens).
- Museformer (NeurIPS 2022) and Theme Transformer (2021): attention to
  structure-related bars, conditioning on a returning theme.
- Repeat After Me (ICML 2024): transformers copy from context far better
  than state-space models; a returning riff is a copy task.
- Fill-in-the-middle: Bavarian et al., 2022. Variable-length infilling:
  [arXiv 2111.06046](https://arxiv.org/abs/2111.06046).
- Classifier-free guidance: MusicGen (Copet et al., 2023); "Stay on topic
  with CFG" (Sanchez et al., 2023); composable conditions (Liu et al.,
  2022); Diff-Symbo ([arXiv 2608.05222](https://arxiv.org/abs/2608.05222)).
- DeepSeek LLM: [arXiv 2401.02954](https://arxiv.org/abs/2401.02954), 2024
  (learning rate and batch size against compute).
- Malladi et al., 2022: square-root learning-rate scaling for Adam.
- Scaling Data-Constrained Language Models: Muennighoff et al., 2023.
- LoRA: Hu et al., 2021.
- ProgGP (Loth et al., 2023), GTR-CTRL (Sarmento et al., 2023), DadaGP
  (Sarmento et al., 2021); Fretting-Transformer
  ([arXiv 2506.14223](https://arxiv.org/abs/2506.14223), 2025) for string
  and fret assignment.

## Open decisions

- Maximum gap length (proposed: 16 bars).
- Whether a match against the user's own albums is rewarded in ranking or
  only reported.
