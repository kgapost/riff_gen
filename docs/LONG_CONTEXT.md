# Seeing the whole song: research on extreme context

> Goal: the model looks at the **whole song**. The surprises people love are often only
> understood from far away: an intro riff that returns at the end, a verse → post-verse →
> pre-chorus → chorus → post-chorus cycle, a long unorthodox chord progression that sounds
> "normal" in 1-2 bar windows and only unfolds into something wonderful over 21-22 bars.

## 1. Three different things hide behind "look at the whole song"

Your three examples live at three different *distances* and *resolutions*:

| What you hear | Distance | Lives at the level of | As tab tokens | At its own level |
|---|---|---|---|---|
| Intro riff returns at the end | 100-300 bars | the exact notes of 1-8 bars | 10,000-30,000 tokens apart | "bar 190 = bar 3", one pointer |
| Verse → post-verse → pre-chorus → chorus → post-chorus cycle | the whole song | sections | 10,000-30,000 tokens | 10-30 sections |
| A 21-22 bar unorthodox chord progression | 20-40 bars | harmony per beat or bar | 700-2,000 tokens | 22-88 harmony steps |

So "look at the whole song" means three requirements, and a longer window only solves the
first:

1. **See it.** The information must reach the model at all (context length or memory).
2. **See it at the right resolution.** A 22-bar progression is 2,000 tokens of strings,
   frets and waits, but only 22-88 steps of harmony. A model that has to find the harmony
   by attending across 2,000 tab tokens will mostly see "normal" 1-2 bar fragments; a model
   that sees the 88 harmony steps side by side can see the pattern.
3. **Be rewarded for using it.** Almost every next token is predictable from the last few
   bars. Long-range dependencies are a tiny part of the training loss, so models learn to
   ignore distant context unless training *forces* them to use it. Language models show
   the same ("lost in the middle"), Perceiver AR's long-context music becomes repetitive
   (noted by the PerceiverS authors), and the 2026 CAST paper reports that autoregressive
   models struggle to keep initial motifs over hundreds of steps.

## 2. How long is a whole song, and what do our models see today?

Estimates from DadaGP's published statistics (all instruments: median 3,558 tokens per
song, 90th percentile 9,530, 99th 16,584, max 51,998) × ~0.6 for guitars only × ~1.15 for
unrolled repeats. **Preprocessing now measures the real numbers** (`stats.json` →
`long_context`; also printed in the log).

| | median | 90 % | 99 % | max |
|---|---|---|---|---|
| DadaGP song, guitars only (estimate) | ~2,500 | ~6,700 | ~11,600 | ~36,000 tokens |
| Your 8-10 minute prog songs, 2-3 dense guitars (estimate) | | | | 15,000-30,000 tokens |

| Model | Whole song visible? | Blind spot |
|---|---|---|
| baseline (4,096 tokens) | ~70 % of DadaGP songs | your long songs: no |
| big (8,192 tokens) | ~95 % of DadaGP songs | your long songs: no. The context is filled **closest bars first**, so for a gap at bar 180 the intro is the first thing cut away. |
| huge (8,192 tokens + song memory) | all bars, but **one vector per bar**, at most 256 bars, at most 96 tokens per bar (dense 3-guitar bars get cut) | exact notes of far bars; harmony inside a bar; form above bar level |

Two consequences: for DadaGP, **seeing** is mostly solved at 8k, and **resolution and
incentive** are the bigger problems. For your albums, the songs are long *and* the
fine-tuning data is small, so we need all three.

## 3. The options

| Approach | How it works | Evidence in music | Recalls a far riff exactly? | Cost for a 40k-token song | Verdict for us |
|---|---|---|---|---|---|
| **Longer dense window** (FlashAttention + RoPE scaling + short continued training on long songs) | Same model, 32k tokens | Long-context LLM recipes (YaRN; Fu et al. 2024: short continued pretraining on length-upsampled data is enough) | Yes | Attention dominates: big at 32k costs ~2× per token vs 8k. KV cache for 32k: big 2.6 GB (fp16) | **Yes, as one stage**: simple, exact, but it does not fix resolution or incentive |
| **Structured sparse attention** (Museformer) | Each bar attends in full to a few *structure-related* bars (fixed distances like 1, 2, 4, 8… bars back) and to one summary token for every other bar | NeurIPS 2022: 3-4× longer sequences than full attention, better structure | Only for the chosen bars | Near-linear | **The idea yes**, but choose the related bars by *content* (our riff analysis), not fixed distances: the intro returns 180 bars later, not 8 |
| **Learned block-sparse** (NSA, DeepSeek 2025; MoBA, Moonshot 2025) | Keys in blocks; per query: a compressed view of all blocks + full attention on the top-k blocks a router picks + a local window | Large LLMs at 64k-1M tokens; no music paper yet. The MoBA follow-up shows everything depends on the router choosing well | Yes, if the router picks the right block | ~2-3k keys per query instead of 40k | **Later stage**: the most "extreme" token-level option. Bar = block is a natural fit. Needs custom kernels (FlexAttention on Linux) |
| **Latent bottleneck** (Perceiver AR) | Cross-attention from a short window to a very long input through a few latents | Symbolic piano music up to 32k-65k tokens (Google Magenta) | Lossy | Linear in input | **Not first**: lossy for exact riffs; reported repetitiveness |
| **Hierarchy / bar patches** (MEGABYTE; Tunesformer/NotaGen; whole-song cascaded diffusion; PhraseVAE; CAST; our huge model) | Compress bars (or phrases) into vectors; a model over bars sees the whole song cheaply; a local decoder writes the notes | **The strongest evidence for whole-song structure.** NotaGen (2025) generates whole classical pieces with bar patches; cascaded diffusion generates up to 256 measures with an explicit form level; PhraseVAE/PhraseLDM (Dec 2025) and CAST (2026) plan at phrase/skeleton level first | Structure yes, exact notes no (needs a path back to the tokens) | Tiny: 300 bars = 300 positions | **The core of the plan** (we already have version 1 in "huge") |
| **Recurrence / memory** (Transformer-XL, as in the original DadaGP paper; Compressive, Memorizing, Infini-attention, Titans) | Read the song in chunks, carry a memory forward | Transformer-XL on DadaGP | Weak to medium | Linear | **No**: built for streaming left to right. We fill gaps with the whole song known on both sides, so random access matters more |
| **State-space / linear models** (Mamba, RWKV; MIDI-RWKV for infilling) | Fixed-size state instead of attention | MIDI-RWKV infilling (its authors note 2,048 tokens limit full songs) | **Poor**: "Repeat After Me" (ICML 2024) shows such models are much worse than transformers at copying from context, and "the intro riff returns" *is* a copy task | Linear | **No** (maybe as hybrid layers much later) |
| **Retrieval / structure-aware context** (Theme Transformer conditions on a theme that must recur) | Put the *relevant* far bars into the window: earlier occurrences of the same riff, the same section's other occurrences, intro/outro | Theme Transformer: much more thematic repetition than prompting | Yes, for what is retrieved | Same as today | **Yes, first**: cheap, uses our riff analysis, works with all three models |
| **Structure-aware positions** (Agarwal et al., ICASSP 2024; our musical-time rotary positions) | Positions that know bars, phrases and sections | Better structural consistency reported | – | Free | **Yes**: add "bar inside its phrase/section" |

## 4. Recommended plan for RiffGen

Four stages. Each one is useful alone, and each is measured before the next is built (§5).

### Stage 1: use what we already compute (days, no new architecture) — DONE

Implemented and tested (decisions D63-D66 in the README). What was built, with the exact rules:

- **Context assembly** (`riffgen/dataset/sampler.py`, used for training *and* generation, so the
  model sees the same thing in both). If the whole song fits into the window it is shown
  completely. Otherwise: local bars (closest first, alternating sides) take 50 % of the token
  budget; then related bars take up to 35 %, in this order: the repeat of the nearest bar
  before the gap *and what followed it*, the intro (first 8 bars), the same for the nearest bar
  after the gap (the bars that led into its repeat), the outro (last 8 bars), up to 4 other
  occurrences of the gap's section, then the remaining repeats of the 4 bars before and after
  the gap. A chunk may be taken partly (the end nearest to the repeat first). Whatever budget
  is left goes back to local bars. Skipped bars are `gap:N`; the hidden bars are never shown.
- **Training tasks**: reprise gaps 25 %, bookend gaps 5 %, hidden local bars 10 % (4-8 bars on
  both sides); logged per example in `sampler_log.jsonl` (`mode`, `hide`, `far_in_context`).
- **Phrase position**: bar 1-16 inside its riff/section, an embedding for every token and for
  every bar of huge's song memory.
- **Huge**: 512 bars, 192 tokens per bar.
- **Measurement** (`stats.json` → `long_context` after preprocessing; `val_far/loss` in every
  training log; `python -m riffgen.eval --long-context N`): loss on gaps whose bars repeat a
  far bar, with the full window vs. only ±16 bars. The difference is the real use of far context.

Measured on synthetic 160-bar songs (not music: it only shows the mechanism works), 60 gaps
that repeat a bar ≥ 32 bars away, window 2,048 tokens: the far repeat is inside the window in
**88 %** of the gaps with the new assembly and **7 %** with "closest bars first", at the same
token count. What the trained model does with it is not measured yet: that needs DadaGP.

Original plan (for reference):

1. **Structure-aware context assembly** (sampler and generator). Instead of "closest bars
   first" until the window is full, fill the token budget in three parts:
   - ~50 % local: the closest bars on both sides (as today);
   - ~35 % related: bars that are repeat partners of the bars around the gap (our similarity
     analysis), the other occurrences of the gap's section, and the song's first and last
     8 bars (intro and outro);
   - the rest of the song as `gap:N` placeholders.

   Our format already supports non-contiguous chunks: `gap:N` marks what is skipped, and the
   musical-time rotary positions tell the model the true distance of every chunk. A 30,000-
   token song then fits into 8,192 tokens *with the bars that matter*. This is the
   retrieval idea, driven by our own riff detection.
2. **Training tasks that need far context**, so the model learns to use it:
   - *reprise gaps*: ~25 % of infill gaps are placed on bars that have a repeat partner
     ≥ 32 bars away (the answer exists, but far away);
   - *local dropout*: in ~10 % of examples the closest 4-8 bars around the gap are hidden as
     `gap:N`, so the model must use distant information;
   - *bookend gaps*: gaps in the last 16 bars of songs whose intro returns there.
3. **Phrase position**: a new embedding "bar k of the current segment" (0-15), from our
   segment borders. Metal and prog phrases have a 4/8-bar hypermeter; this gives it to the
   model directly (cf. structure-informed positional encoding).
4. **Fix huge's limits**: 512 bars instead of 256; bar tokens up to 192 instead of 96.

### Stage 2: longer dense window for "big" (one continued-training run)

Continue pretraining big from 8,192 to **32,768 tokens** on long songs (upsampled), with
RoPE scaling on the token axis and the musical-time axis extended to 1,024 bars. About
5-10 hours on the 4090. Then every DadaGP song and most of your songs fit completely, at
full detail.

Generation at 32k: fine on the 4090. On the GTX 1650 the cache of a 32k prompt is 2.6 GB
for big (fp16): it needs an 8-bit cache, or use Stage 1 assembly (8k) there.

### Stage 3: "huge v2", a three-level picture of the song (the main research step)

This is where the model gets the right **resolution** for your three examples:

```
tab tokens (local detail) ──────────────────────────────► token decoder (writes the gap)
                                                              ▲   ▲   ▲
beat level:  harmony stream  (bass pitch + pitch-class set    │   │   │
             per beat; ~1,200 steps for 300 bars) ────────────┘   │   │  "the 22-bar progression"
bar level:   bar memory (several vectors per bar) ────────────────┘   │  "this bar = bar 3"
section level: one vector per segment (10-30), form model ────────────┘  "verse → post-verse → …"
```

- **Harmony stream (long progressions).** Per beat: the lowest sounding pitch class and the
  set of sounding pitch classes, computed from the tab (we have the pitches). Also the
  root movement from the previous beat, which does not change when a pattern is
  transposed: a 2-bar figure moving up a major third each time becomes a visibly
  *repeating* pattern of intervals. A small transformer over the whole song's harmony
  stream (1,200 steps for 300 bars: full attention is cheap) sees a 22-bar progression as
  88 adjacent steps. It predicts the gap's harmony (a plan), and the token decoder
  realises it in tab. This is the skeleton-then-texture idea of CAST and the
  cascaded-diffusion work, applied to riffs.
- **Bar memory with a pointer (the returning intro).** Several vectors per bar (per half bar
  or per beat) instead of one. An extra head predicts, for each bar, *which other bar it
  repeats* (trained from our repeat partners). At generation, the bars it points to are
  copied into the token window in full detail (Stage 1 retrieval, but learned): the exact
  notes of bar 3 are available when writing bar 190.
- **Section level (the form cycle).** Our segments become one vector each; a small encoder
  over 10-30 sections learns the form: which section type follows which, which section
  repeats which. Heads predict each section's label and its "partner" section. The gap's
  section gets a plan vector from this level too.
- **Bonus: control.** The harmony plan and the pointer are readable and settable. Later:
  `--progression "…"` or `--theme 1-4` ("bring back the intro riff, varied"), the way the
  Theme Transformer conditions on a theme.

### Stage 4 (only if needed): token-level sparse attention over whole songs

If the evaluation shows that exact far detail is still missing: replace the decoder's
attention by an NSA-style bar-block attention: a local window (~1,024 tokens) + one
compressed token per bar for every bar of the song + full attention to the top-k bars a
router selects (initialised from the Stage 3 pointer). A 40k-token song then costs about
as much as a 3-4k dense window. This needs custom kernels (FlexAttention, Linux/WSL), so it
comes last.

### What we will not do

- Pure state-space/linear-attention models: bad at copying, and copying a riff from far
  away is exactly the job.
- Transformer-XL-style recurrence: made for streaming; our gaps need both sides.
- Very long dense contexts on the GTX 1650.

## 5. How we will know the model "looks at the whole song"

Without these tests we cannot tell whether a bigger window helped or just cost time.

| Test | What it measures |
|---|---|
| **Long-context benefit** | Loss on *reprise gaps* (bars with a partner ≥ 32 bars away) with the full context vs. context cut to ±16 bars. The difference = how much the model really uses far context. Should grow from Stage 1 to Stage 3. |
| **Reprise accuracy** | For those gaps: similarity of the generated bars to their far partner (not to the neighbours). |
| **Bookend test** | Gaps in the last 16 bars of songs whose intro returns: does the candidate bring the intro material back? |
| **Long-progression probe** (synthetic) | Songs with a 20-24 bar progression that is a transposed sequence (e.g. a 2-bar figure moving up a major third). The gap's harmony is only predictable after seeing ≥ 3 instances. Pitch-class accuracy of the plan, short vs. long context. |
| **Needle riff** (synthetic) | A unique riff in bars 1-4, hidden where it returns at bar 150+. |
| **Blind listening** | `python -m riffgen.eval --listen`, rating gaps near the end of songs separately. |

The data check comes first. When DadaGP is preprocessed, `stats.json` → `long_context` tells
us how many songs have riffs returning ≥ 32 bars later and how many have intro/outro
bookends. If these are rare in DadaGP but common in your albums, the long-range tasks
(Stage 1.2) and album fine-tuning carry most of the weight.

## 6. Cost overview

| | Extra training | Generation 4090 | Generation GTX 1650 | Code effort |
|---|---|---|---|---|
| Stage 1 | none (applies to all runs) | same | same | small: sampler, generator, one embedding |
| Stage 2 | +5-10 h continued pretraining (big) | fine | needs 8-bit cache or 8k window | small |
| Stage 3 | huge v2 pretraining (~1.3-1.5× huge) | ~same as huge | fine (memory is tiny) | large: harmony stream, pointer, section level |
| Stage 4 | retraining with sparse attention | faster than dense at 32k+ | Linux kernels; check support | large |

## 7. Sources

- Museformer, fine- and coarse-grained attention (NeurIPS 2022): https://arxiv.org/abs/2210.10349
- NotaGen (2025), bar-stream patching: https://arxiv.org/abs/2502.18008; tokenization study (ISMIR 2024): https://arxiv.org/abs/2410.17584
- Native Sparse Attention (DeepSeek, 2025): https://arxiv.org/abs/2502.11089; NVIDIA cuDNN NSA docs: https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/nsa.html
- MoBA (Moonshot, 2025): https://arxiv.org/abs/2502.13189; Optimizing MoBA / FlashMoBA (2025): https://arxiv.org/abs/2511.11571
- Perceiver AR (ICML 2022): https://proceedings.mlr.press/v162/hawthorne22a.html; Magenta: https://magenta.tensorflow.org/perceiver-ar; PerceiverS: https://arxiv.org/abs/2411.08307
- Repeat After Me: Transformers are Better than State Space Models at Copying (ICML 2024): https://proceedings.mlr.press/v235/jelassi24a.html
- Whole-song hierarchical generation with cascaded diffusion (ICLR 2024): https://arxiv.org/abs/2405.09901
- PhraseVAE and PhraseLDM (2025): https://arxiv.org/abs/2512.11348
- CAST, skeleton-to-texture (Scientific Reports, 2026): https://www.nature.com/articles/s41598-026-46750-0
- MuPT (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/file/73f6f8897896f7bda86ea7d1ebc1dc4f-Paper-Conference.pdf
- TOMI, full-song structure (2025): https://arxiv.org/abs/2506.23094
- Bar-level VAE features for long-term structure (APSIPA 2025): https://www.apsipa.org/proceedings/2025/papers/APSIPA2025_P391.pdf
- Structure-informed positional encoding (ICASSP 2024): https://arxiv.org/abs/2402.13301
- Theme Transformer: https://arxiv.org/abs/2111.04093
- Motifs, phrases and beyond (review, 2024): https://arxiv.org/abs/2403.07995
- MIDI-RWKV, long-context infilling: https://www.researchgate.net/publication/392736110_Personalizable_Long-Context_Symbolic_Music_Infilling_with_MIDI-RWKV
- FlexAttention (PyTorch): https://pytorch.org/blog/flexattention/
- Lost in the middle (2023): https://arxiv.org/abs/2307.03172; YaRN (2023): https://arxiv.org/abs/2309.00071; long-context data engineering (Fu et al., 2024): https://arxiv.org/abs/2402.10171
- DadaGP statistics: `third_party/dadagp/_DadaGP_stats.json`
