# RiffGen — a neural network that fills gaps in my songs with guitar riffs

RiffGen is a planned set of Python tools. You give it a Guitar Pro song with empty
bars (a "gap"). It writes guitar tab ideas to fill the gap. Every idea comes out as
a Guitar Pro file in 7-string Drop A.


## Table of contents

1. [The goal](#1-the-goal)
2. [Decision log](#2-decision-log)
3. [The data](#3-the-data)
4. [Tokens: how a tab becomes "text" for the model](#4-tokens-how-a-tab-becomes-text-for-the-model)
5. [Musical concepts](#5-musical-concepts)
6. [Training](#6-training)
7. [Filling gaps (generation)](#7-filling-gaps-generation)
8. [ToDo](#8-todo)
9. [References](#9-references)
10. [Code](#10-code)

---

## 1. The goal

- There are **5 albums** in progress. Each one still needs a number of **new riffs**.
  A bridge is also a riff, so the model has one job: write riffs.
- We train **one network per album**, so each network learns that album's mood.
- The style is mostly **metal and progressive music**. Influences: Gojira, Opeth,
  Meshuggah, Pain of Salvation, Periphery, The Beatles, Anathema, Pink Floyd,
  Frank Zappa, Arcturus, Thorns, Mayhem.
- The output is **guitars only**, in **7-string Drop A** (A E A D G B E).
  The musician writes the bass, drums and everything else.
- The input is always a **Guitar Pro file of a song that is 50–90 % complete**. 
  Where a riff is missing, the song has **4 or more empty bars** (a gap). The network writes a riff that fits
  between the bars before the gap and the bars after it.
- The network is a source of **ideas**, not finished songs. We generate many ideas
  for each gap, the musician keeps the few good ones and edits them in Guitar Pro.

---

## 2. Decision log

Each decision has an ID, so we can refer to it later ("let's change D14").
The last column links to the section with the details.

### 2.1 Scope

| ID | Decision | Why | More |
|---|---|---|---|
| D1 | The only job is **gap filling**. The input is a GP5 song that is 50–90 % complete; the model writes the missing bars. There is no "answer a theme" mode. | One job means one kind of training example to get right, and the song around the gap tells the model much more than a short theme. | [§5.4](#54-gaps) |
| D2 | The model only writes **riffs**. A bridge is a riff too. | That is what the albums need. | [§1](#1-the-goal) |
| D3 | Output is **guitars only, 7-string Drop A**. The number of guitar parts follows the song (max 3). | The musician writes everything else. | [§3.5](#35-tuning-7-string-drop-a) |
| D4 | Every idea is saved as a **GP5 file containing the whole song with the gap filled**. | You listen to the idea in context in Guitar Pro 7 and copy the good bars. | [§7.5](#75-output-files) |
| D5 | We generate **many candidates** per gap (e.g. 32), then filter and rank them. | Most ideas will be average. Choosing is part of the process. | [§7.4](#74-candidates-filters-and-ranking) |

### 2.2 Data

| ID | Decision | Why | More |
|---|---|---|---|
| D6 | Input format is **GP5** (also GP3/GP4). Guitar Pro 7 files are exported to GP5. Not MIDI. | MIDI loses strings, frets, palm mutes, bends, slides and the tuning. PyGuitarPro (and DadaGP) read and write GP3/4/5, but not GP6/GP7. | [§3.1](#31-file-format-gp5) |
| D7 | **Pretrain on DadaGP** (about 26,000 songs), then **fine-tune on ~200 tabs per album**. | 200 tabs are too few to learn guitar music from zero, but a good size to learn a style. Quality beats quantity. | [§3.3](#33-how-much-data) |
| D8 | **One base model, five album networks.** Each album has its own YAML config. | The musician's requirement. Pretraining happens once; fine-tuning is cheap. | [§6.1](#61-two-training-stages) |
| D9 | Favourite songs get more weight through **2–4 slightly changed copies made by hand**, not through a weight number. All versions of one song form a **song family** (`song__v2.gp5`, `song__v3.gp5`). | The musician's choice. Small changes also act as data augmentation. Families keep the held-out set and the plagiarism check honest. | [§3.4](#34-favourite-songs-copies-with-small-changes) |
| D10 | **All fine-tuning tabs are already in 7-string Drop A** (converted by hand by the musician). The pipeline only **checks** the tuning. A song with a guitar track in another tuning is skipped and logged, never converted automatically. | Automatic conversion can change fret shapes and make a riff harder or impossible to play. | [§3.5](#35-tuning-7-string-drop-a) |
| D11 | **DadaGP is used in its own tunings, without conversion.** Each song gets a `tuning:` header token. | Same reason as D10. The base model learns rhythm, techniques and shapes from all tunings; fine-tuning teaches Drop A; generation forces Drop A. | [§3.5](#35-tuning-7-string-drop-a) |
| D12 | **No pitch shifting.** Augmentation is limited to the methods in §3.8 plus the hand-made copies (D9). | Drop A riffs are built around the open low A string; moving them breaks that. | [§3.8](#38-data-augmentation) |
| D13 | **Guitar tracks only** (up to 3). Bass, drums, vocals and keys are removed, including DadaGP's `leads` and `pads` classes. | Guitar output only, and shorter sequences fit more bars into the model's memory. In DadaGP, `leads` holds piano, vocal melodies and rare instruments, not lead guitars. | [§3.6](#36-preprocessing-pipeline) |
| D14 | **Repeats are unrolled** (written out in full) before tokenizing. | The model sees the music as it is played. The DadaGP authors also report that their repeat token was hard for models to learn and caused too many repeated bars. | [§5.3](#53-structure-sections-and-repeats) |
| D15 | In training songs, runs of **4 or more empty guitar bars** are replaced by a **`gap:N` placeholder** token. The model reads it but never writes it. | The model must not learn "play nothing for 4 bars". The same token marks the other, unfilled gaps of a song at generation time (D35). | [§5.4](#54-gaps) |
| D17 | **Very simple riffs** are kept as **context only**: never trained to be written, never used as a hidden gap. "Simple" means simple pitch AND simple rhythm. | The model still needs them as context. Requiring both keeps one-note polyrhythm riffs and tremolo riffs. | [§3.7.1](#371-simple-riff-filter) |
| D18 | Every filter writes an **audit report** plus GP5 snippets of what it removed, and the album YAML can override it per song and bar. | Filters make mistakes, and we must be able to hear them. | [§3.7.2](#372-filter-audit-and-manual-overrides) |
| D20 | We accept DadaGP as **token files**: if the download has no original GP files, we decode the tokens to GP5 with the DadaGP decoder and run the same pipeline. | The DadaGP paper lists the token files, the encoder/decoder and statistics as the distributed content. Both cases work. | [§3.2](#32-where-the-data-comes-from) |

### 2.3 Tokens

| ID | Decision | Why | More |
|---|---|---|---|
| D21 | Tokenization **follows the DadaGP format**. Our extra tokens are **added on top**, never replacing DadaGP tokens. | Same format as the DadaGP dataset (the musician's requirement). | [§4](#4-tokens-how-a-tab-becomes-text-for-the-model) |
| D22 | **7-string guitars are supported** by the DadaGP encoder. We still start with a round-trip test (one own Drop A song → tokens → GP5) and listen to it. | The paper says 7-string guitars are supported, but it does not say how a dropped 7th string is written. **Tested: the encoder rejects 7-string Drop A** (A E A D G B E is not one of its four tuning shapes). Solved by D49; the round trip is exact. | [§4.1](#41-the-dadagp-token-format) |
| D23 | We add **`ts:` time-signature tokens**. | DadaGP does not store time signatures; it infers them from the `wait` tokens, which cannot tell 3/4 from 6/8. | [§4.2](#42-our-extra-tokens) |
| D24 | Tempo: DadaGP has a `tempo:` header token and `bfx:tempo_change` tokens. We keep both in training and **block tempo changes** inside generated bars. | The tempo of the gap is forced (D33). | [§7.2](#72-forced-rules) |
| D25 | **Section markers** ("Verse", "Chorus"…) are read by us with PyGuitarPro and become `section:` tokens. | The DadaGP token format does not include markers. | [§5.3](#53-structure-sections-and-repeats) |
| D26 | **PyGuitarPro is pinned to version 0.6** for the whole project. If 0.6 does not install on our Python version, the DadaGP encoder runs in its own virtual environment. | The DadaGP code only works with PyGuitarPro 0.6. **Tested: 0.6 works on Python 3.12, so one environment is enough.** | [§3.6](#36-preprocessing-pipeline) |
| D27 | **Context window: 4096 tokens.** When the song is longer, we keep the bars closest to the gap on both sides (changed by D63). We measure real tokens per bar on our data and revisit this if fewer than ~32 bars fit. | DadaGP averages ~4,400 tokens per song with all instruments. Guitars only is shorter; dense two-guitar metal is roughly 50–100 tokens per bar, so about 40–80 bars should fit. These are estimates. | [§5.1](#51-measure) |

### 2.4 Training

| ID | Decision | Why | More |
|---|---|---|---|
| D28 | Training uses **two kinds of examples**: song pieces (`task:song`) and fill-in-the-middle gaps (`task:infill`). | Song pieces teach general music; infill examples teach our one job. | [§4.3](#43-the-two-kinds-of-training-examples) |
| D29 | Hidden gaps in training (4–16 bars) are placed on **riff and section borders**, found by our riff detection. | Real gaps in the musician's songs sit where a riff is missing, not in the middle of one. | [§5.2](#52-riff) |
| D30 | Model: a small **decoder-only Transformer** (GPT style, ~85M parameters, 4096 tokens) with PyTorch + Hugging Face `transformers`. | Well known and well documented. Fits easily on both GPUs. This is now the **baseline**; two larger variants exist (D58), all in plain PyTorch (D59). | [§6.2](#62-model) |
| D31 | **Hardware:** training on the RTX 4090 (bf16), testing and generating on the GTX 1650 (the model needs well under 1 GB). | Both are enough for a model of this size. | [§6.3](#63-starting-training-settings) |
| D32 | **1–2 own songs per album are held out** (never trained on, never copied). Fine-tuning stops early when the loss on them stops improving. | It shows whether the model learned the style or only memorized. | [§6.4](#64-how-we-know-training-works) |

### 2.5 Generation

| ID | Decision | Why | More |
|---|---|---|---|
| D33 | **Tempo, time signature, tuning, bar length, length and "guitars only" are forced** by rules while the model writes (constrained decoding). The model cannot break them. Tempo and time signature come from the bar just before the gap. | The musician's requirement. | [§7.2](#72-forced-rules) |
| D34 | A gap is **4 or more empty bars in all guitar tracks** (`min_gap_bars: 4`). You can also give the bars by hand: `--gap 33-40`. | Answer to the old question "4 or more, or more than 4?" — 4 or more. It is one number in the config. | [§5.4](#54-gaps) |
| D35 | **One gap per run.** By default, every gap in the song is filled separately. The other gaps stay in the context as `gap:N` placeholders. | Each gap gets its own set of candidates, and you choose freely per gap. | [§7.3](#73-songs-with-several-gaps) |
| D36 | Candidates are removed when they **only repeat the riff before or after the gap** (similarity ≥ 0.95) or when they are **too simple**. | A gap needs a new riff, not a copy of its neighbour or a plain chug. | [§7.4](#74-candidates-filters-and-ranking) |
| D37 | Python is the main entry point. The `.sh` files are thin wrappers (Git Bash or WSL on Windows). | Everything also works from PowerShell. | [§7.1](#71-command) |

### 2.6 Checks

| ID | Decision | Why | More |
|---|---|---|---|
| D38 | Every candidate goes through a **plagiarism check** against the training songs. It compares on 3 levels: exact tab, transposed (intervals + rhythm) and rhythm only (info only). RED candidates are removed. | We must not put another band's riff on an album by accident. | [§7.6](#76-plagiarism-check) |
| D39 | Common idioms are ignored: a piece counts as an idiom only if it appears in **at least 3 different song families** and in at least 1 % of the families in the index. Copies of one song count once. | Otherwise the hand-made copies (D9) would turn a favourite riff into an "idiom" and the check would ignore it. | [§7.6](#76-plagiarism-check) |
| D40 | The default plagiarism index is the **album's training set** (including the DadaGP songs selected for it, D48) plus any extra folder you give. A DadaGP index is optional, built once, and used only on request. | A DadaGP-wide index is large (we measure it when we first build it); the album index is small and fast. | [§7.6](#76-plagiarism-check) |
| D41 | Every candidate goes through a **playability check** with limits from a **personal profile** (`configs/playability.yaml`). It can propose a better fingering. | "Playable" depends on the player. | [§7.7](#77-playability-check) |
| D42 | Both checks also run as a **standalone command on any folder** of GP5 files. | You edit ideas by hand, and the checks must still hold after that. | [§7.8](#78-running-the-checks-on-a-folder) |

### 2.7 Augmentation and extra fine-tuning data

| ID | Decision | Why | More |
|---|---|---|---|
| D43 | **New gaps every epoch.** The sampler picks new hidden parts (position and length) for every song in every epoch. Validation gaps are fixed. | ~200 songs with fixed gaps give a few hundred examples that the model memorizes quickly. New gaps turn the same songs into thousands of different examples. | [§3.8.1](#381-new-gaps-every-epoch) |
| D44 | **Guitar track variations:** swap guitars of the same kind and sometimes drop one guitar, always for the whole example. | Your songs have 1, 2 or 3 guitars, and the slot order means nothing. The model must handle all cases. | [§3.8.2](#382-guitar-track-variations) |
| D45 | All augmentation runs **on the fly in the data sampler**. Nothing augmented is saved as a file. | Every epoch is different, no extra files, config changes need no reprocessing, and song families and the held-out set stay intact. | [§3.8.3](#383-on-the-fly-with-one-seed) |
| D46 | **One seed controls every random operation**, through generators derived from the seed and a fixed key per decision. The seed is saved with every run and every candidate. | Runs can be repeated, comparisons are fair, and any candidate can be regenerated. | [§3.8.3](#383-on-the-fly-with-one-seed) |
| D47 | **Similar songs from DadaGP** are mixed into each album's fine-tuning (starting value: 15 % of examples), chosen by riff-level style similarity to the album set. | Real music close to the album's style is worth more than any transform, and it also keeps the model from forgetting general guitar music. | [§3.9](#39-similar-songs-from-dadagp) |
| D48 | The DadaGP songs selected for an album are **added to that album's plagiarism index**, and songs that also appear in the album folders or the held-out set are removed from the pool. | The model trains on them, so it can copy from them. Duplicates would leak held-out songs into training. | [§3.9](#39-similar-songs-from-dadagp) |

### 2.8 Implementation decisions (added with the code)

| ID | Decision | Why | More |
|---|---|---|---|
| D49 | **Tuning bypass** for our own files: non-guitar tracks are removed first, guitar tracks are labelled standard 6/7-string *for the encoder only*, the real tuning goes into our `tuning:` token. | The DadaGP encoder rejects Drop A (D22). It writes string + fret, not pitch, so the tab is unchanged. | [docs/TRAINING.md](docs/TRAINING.md) |
| D50 | **Physical frets everywhere.** DadaGP writes drop strings pitch-normalised (open low D = `s6:f-2`); we convert back to the fret you actually play, using the tuning. | Tabs are shapes (D10, D12). Playability and plagiarism L1 need real frets. Lossless. | [docs/TRAINING.md](docs/TRAINING.md) |
| D51 | **Canonical token order**: guitars in slot order inside an onset, notes by string, rests only where they end a note, tempo changes as global events (they survive removing the drum track). | One way to write the same music: less for the model to learn, and the generator can enforce it. | `riffgen/dataset/song.py` |
| D52 | Header: one `track:` + `tuning:` pair per guitar (instead of one `tuning:` token), then `tempo:`. DadaGP's `artist:` and `downtune:` are not given to the model. | Songs can mix 6- and 7-string guitars; the model sees which guitars exist even if one is silent around the gap. The artist would invite copying. | `riffgen/dataset/serialize.py` |
| D53 | Our own **GP5 writer** instead of the DadaGP decoder; candidates are **spliced into your original file**. | The DadaGP decoder rounds durations (bars change length) and drops tremolo picking. Splicing keeps bass, drums and everything else exactly as written. | `riffgen/gp/writer.py` |
| D54 | Pretraining also trains the prefix/suffix tokens of infill examples (`context_loss_weight: 1.0`); fine-tuning keeps middle-only loss (§4.3). | "FIM for free" (Bavarian et al.): ~3× more learning per token in pretraining. | `configs/train/*.yaml` |
| D55 | A segment with the filter action `drop` (`force_drop`, or `simple_riff_action: drop`) becomes a `gap:N` placeholder in context instead of splitting the song. | Same effect as D15/§3.7 (no false transitions), simpler. | `riffgen/dataset/sampler.py` |
| D56 | The forced rules (D33) also enforce the canonical order of D51, one note per string, effects only after their note, ties only onto a sounding note. | The model writes the format it was trained on; impossible tabs cannot appear. | `riffgen/generate/constraints.py` |
| D57 | Every candidate is sampled with its own generator (Gumbel-max), and all candidates share the prompt's attention cache. | D46 "regenerate candidate 7" holds with any batch size; 32 candidates fit on the GTX 1650. | `riffgen/generate/decode.py` |
| D58 | **Three model variants**: baseline (87M, tokens only), big (261M, musical embeddings + musical-time positions, 8192 tokens), huge (384M, big + song memory with planning heads). | Data, not GPU, is the limit; the variants add musical knowledge rather than width. | [docs/MODELS.md](docs/MODELS.md) |
| D59 | Models in **plain PyTorch**, not Hugging Face `transformers`. | Custom embeddings, multi-axis positions, the song memory and constrained decoding with a shared cache do not fit the stock models. | `riffgen/model/` |
| D60 | Album fine-tuning of big/huge uses **LoRA** (+ trained embeddings) by default; baseline fine-tunes fully. | 200 tabs cannot safely move 260-380M weights; LoRA regularises and an album network is a few MB. | `configs/model/*.yaml` |
| D61 | Album data lives in `data/own/<Album>/{own,favourites,influences,songs}`; one config per album in `configs/albums/` (`python -m riffgen.tools.new_album`). | Matches the folders you created. | §10 |
| D62 | The simple-riff filter judges all guitars of a segment together. | A simple lead line over a rich riff is not a "simple riff" (§3.7). | `riffgen/analysis/filters.py` |
| D63 | **Structure-aware context.** When a song is longer than the window, it is filled ~50 % with the bars closest to the gap, ~35 % with *related* bars, and the rest is skipped as `gap:N`. Related = for the 4 bars before/after the gap, the bars that repeat them elsewhere *and what followed there*, the first and last 8 bars (intro, outro), other occurrences of the gap's section. The hidden bars are never shown. Songs that fit are shown completely. | The surprises you want (an intro riff that returns, a verse/chorus cycle) are far from the gap, so "closest bars first" cuts them away first. On long synthetic songs the far repeat is in the window 88 % of the time instead of 7 %, at the same token count. | [docs/LONG_CONTEXT.md](docs/LONG_CONTEXT.md) |
| D64 | **Long-range training tasks.** 25 % of infill gaps are placed on bars that repeat a bar ≥ 32 bars away ("reprise"), 5 % in the last 16 bars where the intro returns ("bookend"), and in 10 % of examples the closest 4-8 bars on both sides are hidden. All with the same seed rules as the other augmentation (D45, D46). | Almost every next token is predictable from the last bars; without these tasks the model learns to ignore distant context. | `configs/train/*.yaml` |
| D65 | New token feature **bar k of its phrase** (1-16, from the riff/section borders; the gap edges count as borders, so training and generation compute the same value). Huge's song memory also gets it. | Metal and prog phrases follow 4/8-bar patterns; the model should not have to count bars. | `riffgen/dataset/serialize.py` |
| D66 | Huge's song memory holds up to **512 bars** with up to **192 tokens per bar** (was 256 and 96). | Long prog songs are seen completely, dense 3-guitar bars are not cut. | `configs/model/huge.yaml` |

---

## 3. The data

### 3.1 File format: GP5

**Decision D6.** We read `.gp5` (also `.gp4`, `.gp3`).

Why not MIDI? MIDI only keeps the pitch and timing of notes. It loses:

- which **string and fret** a note is played on (the shape under your hand),
- **palm mute**, slides, bends, hammer-ons, pull-offs, harmonics, vibrato,
- the **tuning**.

In metal, palm muting and fret shapes are a big part of a riff. The model can only
learn what it sees, so we keep them.

Why not the Guitar Pro 7 `.gp` file directly? **PyGuitarPro** reads and writes GP3,
GP4 and GP5, but not GP6 (`.gpx`) or GP7 (`.gp`). DadaGP is built on PyGuitarPro as
well. Exporting from Guitar Pro 7 to GP5 drops a few GP7-only features, which do not
matter for this project.

> **Later option:** a `.gp` (GP7) file is a zip archive with an XML file inside.
> If exporting hundreds of files one by one becomes painful, we can write a direct
> GP7 reader.

### 3.2 Where the data comes from

| Source | What | Used for |
|---|---|---|
| **DadaGP** | About 26,000 Guitar Pro songs in many genres (mostly rock and metal), as token files, with an encoder/decoder between GP files and tokens. Research use, on request (D20). | **Pretraining** the base model (once) |
| **Album folders** | For each album: the musician's own songs, favourite songs and songs from influence bands, ~200 tabs in total, all in 7-string Drop A | **Fine-tuning** one network per album |
| **Song files** | The musician's unfinished songs, with gaps | **Generating** ideas (not for training) |

### 3.3 How much data

| Data | Size | What it can do alone |
|---|---|---|
| Own songs of one album (~1 hour, maybe 12–15 songs) | ~1,500–2,000 bars | Not enough. The model would memorize or produce noise. |
| One album's fine-tune set (~200 tabs) | A few million tokens (we will measure this) | Good for learning a style, **if** the model already knows guitar music |
| DadaGP | About 116 million tokens with all instruments; less with guitars only (we will measure this) | Enough to teach "general guitar music" |

So **200 tabs per album is a good amount for fine-tuning**. More is fine, but
**quality beats quantity**. A tab with a wrong rhythm teaches the model a wrong
rhythm. 30 accurate tabs are worth more than 300 sloppy ones.

Rules for every album set:

- Hold out **1–2 of your own songs** as the validation set (D32).
- Prefer tabs that are **rhythmically correct** and have **section markers**.

### 3.4 Favourite songs: copies with small changes

**Decision D9.** To make the model learn more from the songs you like most, you put
**2–4 versions** of each of these songs in the album folder: the original plus
copies with **small musical changes** (another ending, a shifted accent, a different
palm-mute pattern, an extra note in a chord…).

This works. It has the same effect as showing the song more often, and the small
changes are a bonus: the model sees the *idea* in a few forms, so it learns the idea
more than the exact notes. Four rules make it safe:

1. **Name the copies after the original:** `song_a.gp5`, `song_a__v2.gp5`,
   `song_a__v3.gp5`. The tools treat all versions as **one song family**.
2. **2–4 versions per song in total.** More than that and the model starts to copy
   those songs.
3. **Keep the copies clean:** still 7-string Drop A, correct bar lengths, playable.
   Run the folder check (§7.8) on them once.
4. **Never copy a held-out song.**

Why the song families matter:

- **Held-out set:** if a copy of a held-out song were in training, the held-out loss
  would look better than it is. All versions of a family are always on the same side.
- **Plagiarism check:** a piece that appears in many *different* songs is treated as
  a common idiom and ignored. Without families, four copies of a favourite riff would
  look like four songs, and the check would stop protecting you from exactly that
  riff. Families count once (D39), and matches are reported with the original name.

Signs that a song has too many copies:

- the loss on the held-out songs goes **up** while the training loss goes down,
- the plagiarism check finds long matches with that song (RED, or SELF-QUOTE for
  your own songs).

Example album config:

```yaml
# configs/albums/album_01.yaml  (python -m riffgen.tools.new_album writes one per folder, D61)
inherit: train/finetune.yaml
album_id: album_01
title: "Working title of album 1"
model: big                        # which pretrained base: baseline | big | huge (D58)
# base_model: runs/pretrain_big/best.pt

data:
  root: data/own/Album 1
  own: own                        # my songs (used for SELF-QUOTE)
  others: [favourites, influences]  # songs I envy (+ their __vN copies), similar bands
  holdout:                        # never trained on, never copied
    - own/song_07.gp5
  songs: songs                    # unfinished songs with gaps (generation only)

finetune:
  # method: lora                  # default per model size (D60)
  max_epochs: 10
  early_stopping_patience: 2      # epochs; stop when held-out loss stops improving
```

### 3.5 Tuning: 7-string Drop A

7-string Drop A, string by string (string 1 = highest):

| String | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|
| Note | A1 | E2 | A2 | D3 | G3 | B3 | E4 |
| MIDI number | 33 | 40 | 45 | 50 | 55 | 59 | 64 |

- **Album sets (D10):** the musician converts every tab to Drop A by hand in Guitar
  Pro before exporting, so the fret shapes stay playable. The pipeline only checks
  that every guitar track has these 7 strings with these notes. A song with a guitar
  track in another tuning is skipped and listed in the log.
- **DadaGP (D11):** songs keep their own tuning. The pipeline reads it and writes a
  header token such as `tuning:e_std6`, `tuning:drop_d6`, `tuning:b_std7` or
  `tuning:drop_a7` (names defined in `configs/tunings.json`; anything unknown is
  `tuning:other`). DadaGP itself already dropped songs in rare tunings.
- **Generation:** always starts with `tuning:drop_a7`, and only strings 1–7 and
  frets 0–24 are allowed (D33).
- **The song you give the model** must be in Drop A as well. Otherwise the tool stops
  with an error.

### 3.6 Preprocessing pipeline

Every GP file goes through the same steps, both for DadaGP and for the album folders:

```
GP3/4/5 file (or DadaGP tokens → decoded to GP5 first, D20)
  │ 1. load with PyGuitarPro 0.6                                         (D26)
  │ 2. read the tuning: album sets must be Drop A; DadaGP → tuning token (D10, D11)
  │ 3. keep guitar tracks only (max 3); remove bass, drums, vocals, keys  (D13)
  │ 4. unroll repeats and alternate endings                              (D14)
  │ 5. read section markers ("Verse", "Chorus"...) → section labels       (D25)
  │ 6. replace runs of ≥4 empty guitar bars with a gap:N placeholder      (D15)
  │ 7. detect riffs and sections; run the simple-riff filter             (§3.7)
  │ 8. save as a "normalized" GP5 file  ← you can open and check it in GP7!
  │ 9. DadaGP encoder → tokens
  │10. add our extra tokens (tuning, time signature, sections, gaps)     (§4.2)
  ▼
token file (.txt) + statistics (tokens per bar, tunings, time signatures...)
+ filter audit report (what was removed and why)
```

Step 8 matters for debugging: before the model sees anything, a human can open the
normalized files in Guitar Pro 7 and hear whether they are correct.

The pipeline also saves a small **index per song**: the token span of every bar of
every guitar track, the riff and section borders, and the filter action of every
segment. Data augmentation (§3.8) works on this index during training, so no GP file
has to be parsed again.

Songs are **skipped** (and listed in a log file) if:
- they have no guitar track,
- they are in an album set and a guitar track is not in 7-string Drop A,
- the file cannot be read.

### 3.7 Training filter: very simple riffs

We want the model to write **riffs**. Very simple riffs (for example 4 bars of straight
chugs on the open A) don't help with that: they are not worth generating.

The filter judges **segments**: the riffs and sections found in §5.2 and §5.3, with
at least 2 bars each. It never judges single bars (one bar is too short), and it
never throws away whole songs.

The filter chooses one of three actions (set in the album YAML, `filters:`):

| Action | What happens to the segment |
|---|---|
| `keep` | Normal training. |
| `context` | The model **reads** it but is never trained to **write** it (loss masking). It is never used as a hidden gap. |
| `drop` | Removed. The song is split at that point. The bars before and after are **not** joined together, because that would teach the model transitions that never existed. |

#### 3.7.1 Simple-riff filter

For each segment we measure:

| Measure | "Simple" when… |
|---|---|
| Pitch variety | Only 1–2 different notes or chords |
| Pitch movement | The pitch rarely changes |
| Rhythm variety | One rhythm pattern repeats (e.g. straight 8th notes) |
| Repetition length | The pattern already repeats after one beat or half a bar |
| Accents and techniques | No accents, no palm-mute/open changes, no slides etc. |

**A riff is "simple" only if BOTH its pitch and its rhythm are simple.** This
matters for our influences:

| Example | Pitch | Rhythm | Result |
|---|---|---|---|
| Straight 8th chugs on the open A for 4 bars | simple | simple | **simple → `context`** |
| Whole-note power chords that never change | simple | simple | **simple → `context`** |
| One-note riff with polyrhythmic accents (Meshuggah style) | simple | complex | kept |
| Tremolo picking with chord changes (Mayhem, Thorns) | complex | simple | kept |
| Slow chord progression (Pink Floyd, Anathema) | complex (chords change) | simple | kept |

Palm-mute patterns count as rhythm. A riff that alternates muted and open chugs has
a rhythm idea, so it is not simple.

- Default action: **`context`**. Simple parts are normal in real songs, and the
  model must know them to understand the music around them. It just never learns
  to *write* them.
- The same simplicity score is used at **generation time** to remove candidates that
  are too simple (`--min-complexity`, §7.4).

#### 3.7.2 Filter audit and manual overrides

The filter makes mistakes, so we must be able to **listen** to what it did (D18):

- `reports/filters_album_01.csv`: every filtered segment, its scores, the action and the reason.
- `reports/audit_album_01/`: a short GP5 file for each filtered segment. Open them in
  Guitar Pro 7 and listen. Hearing 20 random ones is usually enough to see whether
  the thresholds are right.
- Manual overrides in the album YAML:

```yaml
filters:
  simple_riff_action: context      # keep | context | drop
  force_keep:
    - file: own/song_03.gp5
      bars: 49-56                  # looks simple, but it is the main theme
  force_drop:
    - file: influences/song_x.gp5
      bars: 1-16
```

### 3.8 Data augmentation

The fine-tuning set is small (~200 tabs per album). Data augmentation means making
**more and more varied training examples from the same songs**. One rule guides
every choice:

> **Change the context freely, but keep the target real music.**
> The target is the part the model learns to write (the hidden gap). If we transform
> the target, the model learns to write the transformation.

Version 1 uses two methods, described below: **new gaps every epoch** and **guitar
track variations**. Both run in pretraining and fine-tuning, but they matter most in
fine-tuning. The hand-made song copies (§3.4) stay, and there is no pitch shifting
(D12). Real extra data comes from similar DadaGP songs (§3.9).

#### 3.8.1 New gaps every epoch

**Decision D43.** If we chose the hidden parts once, ~200 songs would give a few
hundred infill examples, the model would see exactly the same ones every epoch, and
it would memorize them fast. Instead, the sampler picks **new gaps in every epoch**:

| Choice | How it is picked (starting values) |
|---|---|
| How many gaps per song | About one infill example per 16 bars of the song, per epoch. Every version of a song family counts as a song. |
| Gap length | 4–16 bars: 4 bars 35 %, 8 bars 35 %, any other length 30 %. 4 and 8 are the most common riff lengths. |
| Gap start | 80 %: exactly on a riff or section border (D29). 20 %: shifted 1–2 bars from a border, because borders found automatically can be off by a bar, and your own gaps may not sit exactly on one. |
| What the gap may cover | Only `keep` segments. A gap never covers a `context` or `drop` part (§3.7), so the model never learns to write very simple riffs. |
| Context | If the song fits into the window: all of it. Otherwise ~50 % the closest bars, ~35 % related bars (D63), the rest skipped as `gap:N`. |
| Extra placeholders | In 20 % of examples, another part of the context becomes `gap:N` (§4.3). |

Example: a 120-bar song gives about 7 infill examples per epoch. Over 10 epochs that
is about 70 different gaps from one song, with different lengths, positions and
contexts.

**Validation gaps are fixed.** For the held-out songs we pick the gaps **once** (with
the seed, §3.8.3) and save them in `runs/album_01/val_gaps.json`. Every epoch and
every run is measured on exactly the same gaps, so held-out loss can be compared.

#### 3.8.2 Guitar track variations

**Decision D44.** Your unfinished songs will have 1, 2 or 3 guitars, and which guitar
is "guitar 1" means nothing. Most training songs have 2 guitars, often one rhythm
and one lead. Without variation, the model may learn habits like "guitar 1 always
plays the rhythm" or "there are always two guitars".

| Variation | What happens | Probability (start) |
|---|---|---|
| **Swap guitars** | Guitars of the same kind change places: `distorted0` ↔ `distorted1`, `clean0` ↔ `clean1`. Never distorted ↔ clean, because that changes the sound and the role of the part. | 0.5 |
| **Drop a guitar** | One guitar track is removed. Only for songs with 2–3 guitars, never the last guitar. If two tracks are (almost) identical, as in double-tracked rhythm guitars, the duplicate is dropped first, because that loses no information. | 0.2 |

Rules:
- **The same variation applies to the whole example**: context before the gap, the gap
  itself and the context after it. Otherwise the model would learn that guitars appear
  or disappear at the gap borders, which never happens in your songs.
- At generation the number of guitars follows your song (D33), so the model must be
  good at 1, 2 and 3 guitars.

How it works on tokens:
- **Swap:** rename the instrument part of each note token (`distorted0:note:…` ↔
  `distorted1:note:…`).
- **Drop:** remove that track's note and effect tokens, then merge `wait` tokens that
  are now next to each other. In the DadaGP format, `wait` tokens are shared by all
  tracks, so the timing of the other guitars stays exactly the same.

#### 3.8.3 On the fly, with one seed

**Decision D45: all augmentation happens on the fly**, inside the data sampler, while
training runs. Nothing augmented is ever saved as a file. Why:

- **Every epoch is different.** Saved files would freeze one set of variations.
- **No extra files.** Thousands of augmented copies would fill the disk and get out of
  date every time a filter or the tokenizer changes.
- **Changing a setting is instant.** A different probability in the YAML needs no
  reprocessing.
- **Song families and the held-out set stay intact.** Augmented examples are never
  files, so they can never end up in the wrong folder, the wrong split or the
  plagiarism index.
- **The normalized GP5 files (§3.6) stay the single source** of all training data.

To keep this fast, the sampler only slices and edits token lists, using the index per
song from §3.6. No GP file is parsed during training, and the sampling runs in CPU
worker processes while the GPU trains.

Because augmented examples are not files, we need ways to **see and hear** them:
- `python -m riffgen.dataset.preview --album album_01 --n 20` decodes 20 sampled
  examples (with their augmentations) into GP5 files. Open them in Guitar Pro 7 and
  check that the gaps and variations make sense.
- For a small share of examples (e.g. 1 %), the sampler writes one line to
  `runs/album_01/sampler_log.jsonl`: song, gap bars, variations used. If the model
  shows a strange habit, we can trace it back.

**Decision D46: one seed controls every random operation.** One number in the config
(`seed`) decides all random choices: gap positions and lengths, track variations,
the example mix, the validation gaps, the order of examples, the model's starting
weights, dropout, and the sampling of candidates at generation time.

How it works:
- There is **no shared global random state**. Each random decision creates its own
  random generator from the main seed plus a fixed key that describes the decision,
  for example `derive(seed, "gap", epoch, song_id, example_no)` or
  `derive(seed, "guitar_drop", epoch, song_id, example_no)`.
- In Python this is `numpy.random.SeedSequence(seed, spawn_key=(...))`. Text parts of
  the key are turned into numbers with a stable hash (e.g. `zlib.crc32`), **not**
  Python's `hash()`, which changes every time Python starts.
- PyTorch gets the same seed for model initialization and dropout
  (`torch.manual_seed`), and each data-loader worker gets its own derived seed.

What this gives us:
- **Repeatable data.** The same seed gives exactly the same training examples, no
  matter how many worker processes run or in which order they finish.
- **Fair comparisons.** Each kind of decision has its own key, so switching one thing
  on or off does not reshuffle the rest. Example: turning guitar variations off does
  not change which gaps are chosen, so the two runs differ only in the variations.
- **Regenerating a candidate.** Each candidate gets its own seed,
  `derive(seed, gap, candidate_no)`, saved in `report.json`. With the same model and
  settings, "candidate 7" can be created again exactly.
- **A record of every run.** The seed, the full config, the git commit and the list
  of training files are saved in `runs/album_01/run_info.json`.

Limits:
- Training on a GPU is **not bit-for-bit repeatable**, even with the same seed, because
  some CUDA operations add numbers in a varying order. The data stream is identical,
  the results are very close but not identical. An option `deterministic: true` forces
  repeatable GPU math for debugging; it is slower.
- **One seed can be lucky or unlucky.** Important comparisons (e.g. with or without
  the DadaGP mix) should run with 2–3 different seeds. A small difference between two
  single runs may only be the seed.

Example config:

```yaml
seed: 1234

augment:
  gaps_per_bars: 16              # ~1 infill example per 16 bars of a song, per epoch
  gap_length:
    4: 0.35
    8: 0.35
    other: 0.30                  # any length from 5 to 16
  border_shift:
    prob: 0.2                    # gap start moved away from a border
    max_bars: 2
  extra_gap_placeholder: 0.2     # another part of the context becomes gap:N
  guitar_swap: 0.5
  guitar_drop: 0.2

deterministic: false             # true = repeatable GPU math (slower), for debugging
```

### 3.9 Similar songs from DadaGP

**Decisions D47, D48.** DadaGP has about 26,000 songs, and many of them are close to
the style of an album: the same bands, the same genres, similar riffs. Mixing a
**selected subset** of them into fine-tuning gives the model more real music in the
album's style. This is worth more than any transform, because it is real music, and
it also keeps the model from forgetting general guitar music while it learns the
album.

The selection is made **per album**, in six steps.

**Step 1: the pool.** All preprocessed DadaGP songs (guitars only, filters applied).
We remove:
- songs that are **also in the album folders or the held-out set**, found by matching
  artist and title and by the same fingerprints the plagiarism check uses (§7.6).
  A held-out song in the pool would make the held-out loss meaningless.
- songs whose guitar parts are mostly `drop` or `context` after filtering.

**Step 2: describe every riff with a style vector.** Two options:

| Option | What it uses | When |
|---|---|---|
| **A. Musical features** | Numbers computed from the tokens: tempo, share of odd time signatures, notes per beat, palm-mute share, chords vs single notes, techniques used, which strings are used (how much on the low strings), interval and rhythm (note-start) patterns, simplicity score (§3.7.1). | Available from day one. Easy to read and debug. |
| **B. Model embeddings** | The average internal state of the base model over the riff's tokens. | Only after pretraining. Captures more, but is harder to explain. |

We start with option A. When the base model exists, we try option B and compare the
two selections by listening.

The features are computed **once for all of DadaGP** and saved
(`data/dadagp_riff_features.parquet`). Selecting songs for an album then takes
seconds.

**Step 3: score every DadaGP song.**
- For every DadaGP riff, find the **5 most similar riffs in the album set**
  (own songs, favourites and influences, without the held-out songs) and average
  their similarity.
- A song's score is the average of its **3 best riffs**.

We compare **riffs, not whole songs**, because in progressive music one song can hold
very different parts. A long Opeth song with three riffs in your style should count,
even if the rest of it is acoustic.

**Step 4: select.**
- Take the best songs until `max_songs` (e.g. 1,000).
- Limit each artist to `max_per_artist` songs (e.g. 20), so one band cannot dominate.
- Artists in `always_include_artists` (your influences) are always in the pool,
  matched by DadaGP's artist label. `exclude_artists` removes artists completely.
- Tuning: songs in **7-string drop tunings** (Drop A, Drop G#, Drop G…) are preferred,
  because their fret shapes are the same as Drop A. Other tunings are allowed and
  keep their own `tuning:` token (D11).

**Step 5: use them in fine-tuning.** The sampler takes `share` of the examples
(start: 15 %) from the selected songs. They get the same example types and the same
augmentations as the album songs (§3.8). The share is set per example, so it does not
depend on how many songs were selected.

**Step 6: audit and protect.**
- `reports/dadagp_mix_album_01.csv` lists every selected song with its artist, score
  and best riffs, and `reports/dadagp_mix_album_01/` holds GP5 snippets of 20 random
  selected riffs. Listen to them, like the filter audit (D18).
- The selected songs are training data now, so the model can copy from them. They are
  **added to the album's plagiarism index automatically** (D48). They count as other
  people's songs, never as SELF-QUOTE.

**Does it help?** We check with an experiment: fine-tune album 1 with and without the
DadaGP mix, with the same seeds (2–3 each, §3.8.3), and compare held-out loss,
plagiarism rate and blind listening. We keep the mix only if it helps, and we try
`share` values between 10 % and 20 %.

Example config:

```yaml
# configs/albums/album_01.yaml
dadagp_mix:
  enabled: true
  share: 0.15                    # share of fine-tuning examples from DadaGP
  method: features               # features | embeddings
  k_nearest: 5                   # album riffs compared per DadaGP riff
  top_riffs_per_song: 3          # song score = average of its best riffs
  max_songs: 1000
  max_per_artist: 20
  prefer_tunings: [drop_a7, drop_gs7, drop_g7]
  always_include_artists: [Gojira, Opeth, Meshuggah, Periphery]
  exclude_artists: []
```

---

## 4. Tokens: how a tab becomes "text" for the model

### 4.1 The DadaGP token format

**Decision D21.** We use the DadaGP encoder, and new tokens are only added on top.
DadaGP writes a song as a list of tokens, one musical event per token. Every song
starts with `artist`, `downtune`, `tempo` and `start`. It looks roughly like this
(illustrative; the exact spelling comes from the DadaGP code):

```
artist:unknown_artist
downtune:0
tempo:140
start
new_measure
distorted0:note:s7:f0      ← guitar 1, string 7, fret 0
nfx:palm_mute              ← note effect: palm mute
wait:240                   ← move forward a 16th note (240 ticks)
distorted0:note:s7:f0
nfx:palm_mute
wait:240
distorted0:note:s5:f2
distorted0:note:s4:f2      ← two notes at the same time = a chord
wait:480
...
new_measure
...
end
```

Important ideas:
- **`new_measure`** marks the start of every bar.
- **`wait:N`** moves time forward by N ticks. A quarter note is 960 ticks, so an
  eighth note is 480 and a 16th note is 240. Notes between two `wait` tokens sound
  together.
- Guitar classes: up to 3 distorted guitars (`distorted0–2`) and 2 clean or acoustic
  guitars (`clean0–1`). `bass`, `drums`, `leads` and `pads` are removed by us (D13).
- `nfx:` = note effect (palm mute, slide, bend…). `bfx:` = beat effect, including
  `bfx:tempo_change`.
- 7-string guitars are supported (D22).

### 4.2 Our extra tokens

These are added to the end of the DadaGP vocabulary, so the DadaGP tokens keep their
IDs. They are removed again before the DadaGP decoder turns tokens back into GP5.

| Token | Meaning | Where |
|---|---|---|
| `tuning:drop_a7`, `tuning:e_std6`, … | The tuning of the song (D11) | Header |
| `ts:7/8`, `ts:4/4`, … | Time signature of the bar (D23) | After `new_measure` when it changes |
| `section:verse`, `section:chorus`, `section:bridge`, `section:solo`, `section:breakdown`, `section:intro`, `section:outro`, `section:interlude`, `section:other` | Section label read from the tab's markers | Start of a section |
| `task:song`, `task:infill` | Which kind of example this is | Header |
| `bars:N` | How many bars to write (the length of the gap) | Header (infill) |
| `gap:N` | N bars here are unknown: another gap, a part without guitars, or bars skipped because the song is longer than the window (D63). Read only, never written. (D15) | Instead of those bars |
| `<fim_prefix>`, `<fim_suffix>`, `<fim_middle>` | Before the gap / after the gap / the gap itself | Infill |

The section names come from a mapping file, `configs/sections.json`. For example,
"Chorus 2", "Refrain" and "CHORUS" all become `section:chorus`.

### 4.3 The two kinds of training examples

**Decision D28.**

**A. Song piece** (`task:song`): a random window of a song. It teaches general music.

```
tuning:drop_a7 tempo:140 task:song start new_measure ... (up to 4096 tokens)
```

**B. Fill in the middle** (`task:infill`): we hide 4–16 bars of a real song, on a
riff or section border (D29). This is called FIM, from "fill in the middle".

```
tuning:drop_a7 tempo:140 task:infill bars:8 section:bridge
<fim_prefix> ... bars before the hidden part ...
<fim_suffix> ... bars after the hidden part ...
<fim_middle> ... the hidden 8 bars (what the model learns to write) ... end
```

- The model is trained to write only the middle part (loss masking). The prefix and
  suffix are context.
- In some examples (starting value: 20 %) we also replace **another** 4–16-bar part
  of the prefix or suffix with `gap:N`. This way the model practises the real
  situation: a song that is only 50–90 % complete, with more than one gap.

The mix is set in the config (starting point: 70 % A / 30 % B for pretraining,
30 % A / 70 % B for fine-tuning).

---

## 5. Musical concepts

### 5.1 Measure

- Every bar starts with `new_measure`, so the model always knows where bars begin.
- Every bar has a time signature (`ts:` token, D23).
- **During generation we count ticks.** A 4/4 bar is 3840 ticks and a 7/8 bar is
  3360 ticks. The model cannot write a `wait` that goes past the end of the bar.
  When the bar is full, we force `new_measure`. So every generated bar has the
  correct length, also in odd meters.
- A riff spans several bars. The model sees many bars at once (D27), so it can
  continue patterns across bar lines.

### 5.2 Riff

A neural network has no built-in concept of "riff". It learns patterns. For our
tools, a riff is defined like this:

> **A riff is a group of 1–8 bars on a rhythm guitar that repeats at least twice
> in the song (exactly, or with small changes).**

Our analysis code finds riffs by comparing bars and groups of bars inside each song,
with a **similarity score** from 0 (nothing in common) to 1 (identical), made of:
- **rhythm similarity:** where notes start in each bar (on a fine grid), and
- **pitch similarity:** the notes used (pitch classes) and the melodic movement (intervals).

The detected riffs are used to:
1. place the hidden parts of infill training examples on riff and section borders (D29),
2. let the filters judge whole riffs instead of single bars (§3.7),
3. remove candidates that only repeat the riff next to the gap (D36).

So the model never needs the word "riff". It is trained on gaps that start and end
at riff borders, and it learns from them what a riff is.

### 5.3 Structure: sections and repeats

- **Repeats** (repeat signs, alternate endings) are **unrolled** (D14). The model
  sees the real order of bars, including how often a riff really repeats.
- **Section markers** in the tab ("Verse", "Chorus", "Bridge"…) become `section:`
  tokens (D25). Many tabs have markers, but not all. If a tab has none, that's fine:
  no section token.
- **At generation time** you can say what the gap should be: `--section bridge`.
  If you don't, the model reads the marker at the gap (if any), or guesses from the
  context.

### 5.4 Gaps

- **How a gap is found (D34):** 4 or more bars in a row where **all guitar tracks
  are empty** (rests). The number `min_gap_bars: 4` is in the config. You can also
  give the bars by hand: `--gap 33-40`.
- The model gets the bars **before** the gap (`<fim_prefix>`) **and** the bars
  **after** the gap (`<fim_suffix>`), then writes exactly as many bars as the gap
  has (`bars:N`). So the riff leads somewhere: the model knows where the music has
  to arrive.
- When the song does not fit into the window, we keep the bars closest to the gap and
  the bars **related** to it (repeats of the bars next to the gap and what followed them,
  intro, outro, same section); everything else becomes `gap:N` (D63).
- Training hides 4–16 bars of real songs, so the model practises exactly this job.

---

## 6. Training

### 6.1 Two training stages

200 tabs are not enough to learn guitar music from zero. So:

```
Stage 1: PRETRAIN (once)            Stage 2: FINE-TUNE (per album, ×5)    Then: FILL GAPS
DadaGP, ~26,000 songs ──► base ──►  album_01 set (~200 Drop A tabs) ────► album_01 network ──► ideas
                          model  ─► album_02 set ──────────────────────► album_02 network ──► ideas
                                 ─► ... album_05
```

- **Stage 1** teaches general guitar music in all common tunings.
- **Stage 2** teaches the style of one album, in Drop A, with a share of similar DadaGP
  songs mixed in (§3.9). It is short (minutes to an hour).
- Then the album network fills the gaps in that album's songs.

### 6.2 Model

**Decision D30.** Starting point (in `configs/model/small.yaml`):

| Setting | Value |
|---|---|
| Type | Decoder-only Transformer (GPT style, rotary positions) |
| Layers / width / heads | 12 / 768 / 12 |
| Parameters | ~85 million |
| Context window | 4096 tokens |
| Vocabulary | DadaGP tokens + our tokens (§4.2) |

We pretrain our own base model instead of reusing an existing DadaGP model, because
our data is guitars only and has extra tokens. If a public DadaGP checkpoint exists,
we can compare against it.

### 6.3 Starting training settings

| | Pretraining | Album fine-tuning |
|---|---|---|
| GPU | RTX 4090 | RTX 4090 |
| Learning rate | 3e-4, cosine decay | 5e-5 |
| Precision | bf16 | bf16 |
| Stop rule | fixed number of steps | early stopping on held-out own songs |
| Example mix (song piece / infill) | 70 / 30 | 30 / 70 |
| Similar DadaGP songs (§3.9) | — | 15 % of examples |
| Augmentation (§3.8) | new gaps every epoch, guitar variations | same |
| Seed | `seed` in config (§3.8.3) | same |

These are only first values. They live in YAML files and we will change them after
experiments.

### 6.4 How we know training works

1. **Loss on held-out songs:** it should go down. If it goes up while the training
   loss still goes down, stop: the model is starting to memorize.
2. **Plagiarism rate:** the share of candidates with a RED or YELLOW verdict (§7.6).
   It should stay low. If it rises during fine-tuning, a song has too many copies
   (§3.4) or training ran too long.
3. **Listening:** the most important test. You rate candidates (1–5), and the ratings
   are saved to `ratings.jsonl`. Later these ratings could help us rank candidates.

---

## 7. Filling gaps (generation)

### 7.1 Command

```bash
python -m riffgen.generate \
  --album album_01 \
  --song songs/song_03.gp5 \
  --gap 33-40 \
  --section bridge \
  --candidates 32
```

- `--gap` is optional. Without it, every gap in the song is filled (D35).
- `--section` is optional (§5.3).
- `--temperature` (default 1.0) and `--top-p` (default 0.95) set how adventurous the
  model is. Lower = safer and more predictable, higher = wilder.

### 7.2 Forced rules

**Decision D33.** These are not suggestions to the model. They are enforced while it
writes each token:

| Rule | How it is enforced |
|---|---|
| **Tempo = tempo of the bar before the gap** | Written in the header (the model knows if it's slow or fast) and in the output file. Tempo-change tokens are blocked. If the gap is at the very start of the song, the first bar after the gap is used. |
| **Time signature = time signature of the bar before the gap** | Every new bar gets this time signature. Time-signature-change tokens are blocked. Bars are filled to exactly the right length (§5.1). Option `--use-gap-signatures` uses the time signatures already written in the empty bars instead. |
| **7-string Drop A** | The header is `tuning:drop_a7`. Only strings 1–7 and frets 0–24 are allowed. |
| **Guitars only** | Only guitar instrument tokens are allowed, and only for the guitar tracks the song already has (max 3). |
| **Length** | Exactly the number of bars in the gap. Generation stops there. |

### 7.3 Songs with several gaps

A song that is 50–90 % complete usually has more than one gap. **Decision D35:**

- Each run fills **one** gap. By default the tool runs once per gap, so every gap gets
  its own set of candidates.
- The other gaps stay in the context as `gap:N` placeholders. The model has seen this
  in training (§4.3), so it knows that part of the song is still unknown.
- Gaps are filled independently: a candidate for gap 1 is not used as context for
  gap 2. When you have chosen and pasted a riff, run the tool again for the next gap
  and it will see your choice.

### 7.4 Candidates, filters and ranking

**Decisions D5, D36, D38, D41.**

```
generate N candidates for the gap (e.g. 32)
  → remove: candidates that only repeat the riff before or after the gap (similarity ≥ 0.95)
  → remove: candidates that are too simple (simplicity score, §3.7.1, --min-complexity)
  → remove/flag: plagiarism check (§7.6)
  → remove/flag: playability check, with automatic re-fingering (§7.7)
  → rank: by model confidence + playability score
  → save the best K (e.g. 10) + check report
```

### 7.5 Output files

**Decision D4.**

```
outputs/album_01/2026-10-18_song_03_gap_33-40/
  cand_01.gp5        ← the whole song with the gap filled, Drop A, correct tempo/time signature
  cand_02.gp5
  ...
  report.json        ← scores and settings used
  check_report.md    ← plagiarism + playability results (§7.6–7.8)
  check_report.csv
```

With several gaps, each gap gets its own folder. Open the files in Guitar Pro 7,
listen with RSE, and copy the good bars into your song.

### 7.6 Plagiarism check

**Goal:** be sure that no riff on the album is somebody else's riff (D38). The check
also tells you when the model only repeats **your own** songs. That is not a legal
problem, but it is not a new idea either.

**What it compares against: the reference index (D40).**
- By default: all songs in the album's training set (own, favourites with their
  copies, influences, and the similar DadaGP songs selected for this album, §3.9).
- Optional: any other folder, e.g. your whole tab collection, with `--extra-reference PATH`.
- Optional: all of DadaGP with `--reference dadagp` (big and slower, built once).
- The index is built once after fine-tuning and saved as
  `runs/album_01/plagiarism_index.sqlite`.

**How it works, step by step:**

1. Every guitar part becomes a list of **note events** (start time, length, pitch or chord).
2. The list is cut into short overlapping pieces (**n-grams**): groups of 8 note
   events, plus 1-bar and 2-bar pieces.
3. Each piece gets a **fingerprint** (a short code that is always the same for the
   same music) on three levels:

   | Level | What the fingerprint keeps | What it catches |
   |---|---|---|
   | **L1 exact** | string, fret, rhythm | Direct copies |
   | **L2 transposed** | the steps between notes (intervals) + rhythm, without the absolute pitch or string | The same riff moved to another key or another place on the neck |
   | **L3 rhythm** | only where the notes start | Rhythm copies. Info only, never a reason to reject: rhythms are shared by everybody. |

4. The fingerprints are looked up in the index to find matches.
5. **Common idioms are ignored (D39).** A piece that appears in at least 3 different
   **song families** and in at least 1 % of the families in the index is a shared
   idiom, like an open-string chug or a plain scale run. The hand-made copies of a
   song (§3.4) are one family, so they never turn a favourite riff into an "idiom".
6. For every candidate we compute:
   - the **longest matching run** (in bars and in note events),
   - the **coverage**: the share of the candidate's notes that sit inside matching pieces,
   - the **source**: which song (original name) and which bars.

**Verdicts** (starting values in `configs/plagiarism.yaml`):

| Verdict | Rule |
|---|---|
| **RED: copy** | An L1 or L2 run of **≥ 2 bars**, or **≥ 50 %** coverage from one song family |
| **YELLOW: close** | A run of **≥ 1 bar**, or **25–50 %** coverage |
| **GREEN: OK** | Below that |
| **SELF-QUOTE** | Same rules as red/yellow, but the source is one of **your own** songs. Reported separately. |

During generation, RED candidates are removed and YELLOW ones are kept but flagged.

**Output:**
- Lines in `check_report.md/.csv` with candidate, verdict, level, candidate bars,
  source song and source bars.
- With `--mark`: a copy of the GP5 file with **markers on the matching bars**, e.g.
  `COPY? influences/song_a.gp5 bars 12-13 (L2)`, so you can see it in Guitar Pro.

**Limits:**
- It only finds songs that are **in the index**. A bigger index (DadaGP, your tab
  collection) means better protection.
- It is a technical check, **not a legal opinion**.
- It only checks guitars.

### 7.7 Playability check

Generation already forces strings 1–7, frets 0–24 and correct bar lengths (§7.2).
But a tab can still be **impossible or very hard for a real hand**, and once you
edit a file by hand, nothing forces the rules any more. The playability check looks
at the **physical** side (D41):

| Check | What it looks for | Starting value | Severity |
|---|---|---|---|
| **Hand span** | Distance between the lowest and highest fretted note of a chord, or of notes played close together in time. Open strings don't count. | ≤ 4 frets below fret 5, ≤ 5 frets up to fret 12, ≤ 6 frets above (frets get narrower) | error / warning |
| **Fingers** | Number of different fretted positions in a chord. A barre (same fret on neighbouring strings) counts as one finger. | ≤ 4 | error |
| **One note per string** | Two notes at the same time on the same string | — | error |
| **Position jumps** | Fret jump compared with the time available, at the song's tempo | > 7 frets in less than a 16th note → warning | warning |
| **Picking speed** | Notes per second for the picking hand | > 16 notes/s → warning (limit set in your profile) | warning |
| **String skipping at speed** | Jumps over 2 or more strings at high speed | profile value | warning |
| **Techniques** | Do they make sense? No bend on an open string; hammer-on to a *higher* fret on the same string; pull-off to a *lower* fret; slide changes the fret; natural harmonics only at frets 3, 4, 5, 7, 9, 12, … | — | error |
| **Let-ring conflicts** | A ringing note on a string while another note is played on that same string | — | warning |
| **Bar length** | Each bar is exactly as long as its time signature (important after manual edits) | — | error |
| **Tuning** | All guitar tracks are 7-string Drop A | — | error |

Time in seconds is computed from the tempo: `seconds = ticks / 960 × 60 / BPM`.
The same fret jump can be easy at 80 BPM and impossible at 220 BPM.

**Result per file:**
- a list of **errors** (impossible) and **warnings** (hard), each with bar and beat,
- a **playability score** from 0 to 100.

**Automatic fix (`--fix`).** Many problems are only a bad fingering. The checker can
re-fret the notes. It keeps **the same pitches and the same rhythm** and only changes
string and fret. It uses a small fingering algorithm (dynamic programming) that:
- allows only frets 0–24 and one note per string at a time,
- keeps chord shapes within your hand span (open strings excluded),
- avoids big jumps between neighbouring notes,
- prefers low strings for low notes, and changes as few notes as possible.

The result is saved as a new file, e.g. `cand_03.fixed.gp5`. The original file is
never overwritten. If no playable fingering exists, the problem stays in the report.

**Your personal profile.** "Playable" depends on the player, so all limits live in
`configs/playability.yaml`. You can keep two profiles, e.g. a strict `live` one and a
looser `studio` one:

```yaml
# configs/playability.yaml
profile: live
max_span:            # frets, by position on the neck
  below_5: 4
  below_12: 5
  above_12: 6
max_fingers: 4
max_notes_per_second: 16
max_jump_frets_per_16th: 7
allow_tapping: true
```

### 7.8 Running the checks on a folder

Both checks also run as a **standalone command on any folder of GP5 files** (D42):
generated candidates, candidates you edited, your hand-made song copies (§3.4), or
your own new songs.

```bash
python -m riffgen.check \
  --album album_01 \
  --folder outputs/album_01/2026-10-18_song_03_gap_33-40 \
  --plagiarism --playability \
  --fix --mark
```

It writes `check_report.md` and `check_report.csv` into that folder. Example summary:

```
file           plagiarism                                   playability  errors  warnings
cand_01.gp5    GREEN                                        96           0       1
cand_02.gp5    RED    influences/song_a.gp5 bars 5-7 (L2)   88           0       2
cand_03.gp5    SELF-QUOTE  own/song_02.gp5 bars 17-18       71           1       3   → cand_03.fixed.gp5
```

Tips:
- **Run the check again after you edit an idea.** Your changes can bring it closer
  to a known song, or make it unplayable.
- Guitar Pro 7 saves as `.gp`. Before checking an edited file, **export it to GP5**
  again (or wait for the direct GP7 reader, §3.1).

---

## 8. ToDo

1. **Request DadaGP access** from the DadaGP authors (Dadabots / Pedro Sarmento),
   for research use. Read the license terms.
2. **Convert all fine-tuning tabs to 7-string Drop A** in Guitar Pro 7, keeping the
   fret shapes playable. Listen to them once.
3. **Export your songs to GP5** from Guitar Pro 7 (File → Export → GP5).
   First export **one** own song, so we can run the round-trip test (D22).
4. **Sort the tabs into folders**, one set per album: `own/`, `favourites/`,
   `influences/`.
5. **Make 2–4 versions of the songs you like most**, with small changes, named
   `song__v2.gp5`, `song__v3.gp5`… (§3.4).
6. **Choose 1–2 own songs per album to hold out.** Don't make copies of them.
7. **Mark the gaps** in your unfinished songs: leave 4 or more empty bars in all
   guitar tracks. Keep the right tempo and time signature in the song. A marker such
   as "Bridge" at the gap helps.
8. If possible, **add section markers** to your own songs (Verse, Chorus, Bridge…).
9. **Set your playability profile** in `configs/playability.yaml` (hand span,
   fastest picking you are comfortable with).
10. After editing a generated idea in Guitar Pro 7, **export it to GP5 again** before
    running the folder check.

---

## 9. References

- **DadaGP:** Sarmento et al., *"DadaGP: A Dataset of Tokenized GuitarPro Songs for
  Sequence Models"*, ISMIR 2021. Code: `github.com/dada-bots/dadaGP`.
- **ProgGP:** Loth et al., *"ProgGP: From GuitarPro Tablature Neural Generation to
  Progressive Metal Production"*, 2023. Fine-tunes a DadaGP-pretrained model on
  progressive metal, very close to this project.
- **GTR-CTRL:** Sarmento et al., 2023. Controls guitar tab generation by genre and instrument.
- **PyGuitarPro:** Python library to read/write GP3/GP4/GP5 files.
- **Fill-in-the-middle:** Bavarian et al., *"Efficient Training of Language Models to
  Fill in the Middle"*, 2022 (the idea behind our infill examples).
- **LoRA:** Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"*, 2021.
- **Repeated data:** Muennighoff et al., *"Scaling Data-Constrained Language Models"*, 2023
  (why a few passes over DadaGP are fine, and why much bigger models would memorize).

---

## 10. Code

The three model variants are explained in [docs/MODELS.md](docs/MODELS.md); the step-by-step
runbook and the list of problems we will run into are in [docs/TRAINING.md](docs/TRAINING.md);
how the model can look at the whole song is researched in [docs/LONG_CONTEXT.md](docs/LONG_CONTEXT.md).

```bash
pip install -e ".[dev]"                       # after installing PyTorch for your GPU, see docs/TRAINING.md
python -m riffgen.tools.smoke                 # whole pipeline on synthetic data (also: --huge)

# Stage 1 (once per model size)
python -m riffgen.dataset.inspect --src data/dadaGP
python -m riffgen.dataset.preprocess dadagp --workers 12
python -m riffgen.train.pretrain --model baseline          # big | huge; --resume after a crash

# Collecting tabs: a shortlist of free Guitar Pro tabs rated above 4.4 from Ultimate Guitar artist pages,
# then filing what you downloaded by hand (the site's robots.txt disallows automated downloads)
python -m riffgen.tools.ultimate_guitar list https://www.ultimate-guitar.com/artist/meshuggah_3907
python -m riffgen.tools.ultimate_guitar sort --downloads ~/Downloads --dest "data/own/Compound Structure/influences"

# Stage 2 (per album)
python -m riffgen.tools.new_album --all                    # configs/albums/<album>.yaml per folder in data/own
python -m riffgen.dataset.preprocess album configs/albums/compound_structure.yaml
python -m riffgen.dataset.preview --album compound_structure --n 20
python -m riffgen.train.finetune configs/albums/compound_structure.yaml --model big

# Ideas
python -m riffgen.generate --album compound_structure --song unfinished.gp5 --gap 33-40 --section bridge
python -m riffgen.check --album compound_structure --folder outputs/compound_structure/<folder> --fix --mark
python -m riffgen.tools.rate outputs/compound_structure/<folder>
python -m riffgen.eval --checkpoint runs/compound_structure/big/best.pt \
    --store data/processed/albums/compound_structure --split holdout --listen reports/listen
```

| Folder | Content |
|---|---|
| `riffgen/tokens/` | vocabulary (DadaGP tokens first, D21), tunings, section names |
| `riffgen/gp/` | Guitar Pro reading (tuning bypass, D49), DadaGP encoder bridge, GP5 writer (D53) |
| `riffgen/dataset/` | parsing, unrolling, preprocessing, store, sampler (FIM, augmentation), features, loader, similar songs, preview |
| `riffgen/analysis/` | riff/section detection, simple-riff filter |
| `riffgen/model/` | the three models, LoRA |
| `riffgen/train/` | trainer, pretraining, fine-tuning, checkpoints |
| `riffgen/generate/` | forced rules (constrained decoding), sampling, the generate command |
| `riffgen/check/` | plagiarism and playability checks, folder check |
| `riffgen/eval/` | evaluation and blind listening sets |
| `third_party/dadagp/` | the DadaGP encoder, MIT licence, Dadabots (vendored, unchanged) |
| `configs/` | model sizes, training stages, albums, filters, playability, plagiarism |
| `tests/` | `python -m pytest -q` |
