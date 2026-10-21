# The three models: baseline, big, huge

All three are trained the same way (pretrain on DadaGP, fine-tune per album, D7/D8), read
the same token format (D21) and generate under the same forced rules (D33). They differ in
**what the model is told about music** and, as a consequence, in size.

| | baseline | big | huge |
|---|---|---|---|
| Idea | tab language model | music-aware tab model | structure-aware model (song memory) |
| Parameters (measured) | 87M | 261M | 384M (330M decoder, 50M song memory) |
| Layers / width | 12 / 768 | 20 / 1024 | 24 / 1024 + bar & song encoders |
| Context | 4096 tokens (~60-100 bars) | 8192 tokens (~130-200 bars) | 8192 tokens **+ every bar of the song** |
| Input per token | token id | token id + 10 musical features | same as big |
| Positions | token index | token index + musical time + position in bar | same as big |
| Extra training signals | – | – | plan of the gap (pitch classes, rhythm), sections, riff identity |
| Album fine-tuning | full | LoRA (+ embeddings) | LoRA (+ embeddings) |
| Pretraining on the 4090 (estimate) | ~4 h | ~12 h | ~16 h |
| Generating on the GTX 1650 | yes (fp16, batch 32) | yes (fp16, batch 8) | yes (fp16, batch 8, slower) |
| Config | `configs/model/baseline.yaml` | `configs/model/big.yaml` | `configs/model/huge.yaml` |

Training times are estimates for the default step counts (we will measure tokens/s, the
trainer logs it). Train the baseline first: it is fast and proves the whole chain.

---

## 1. Why "bigger" is not the main axis

DadaGP has ~117M tokens with all instruments; with guitars only and repeats
unrolled it is roughly **70-90M tokens** (we measure it in `stats.json` after
preprocessing). That is small for a language model:

- A common rule of thumb is ~20 training tokens per parameter for one pass over new data.
  We can repeat the data (about 4 passes are nearly as good as new data, more passes help
  less and less), and our augmentation makes every pass different (new gaps, guitar swap and
  drop). That makes models of ~100-400M parameters reasonable.
- Beyond that, extra parameters mostly **memorize**. For us memorizing is worse than
  useless: it produces other people's riffs (that is why the plagiarism check exists).
- The fine-tuning sets (~200 tabs, a few million tokens per album) make this even sharper.
  The big and huge models therefore fine-tune with **LoRA** by default: small trainable
  adapters instead of moving all weights.

So the three models do not mainly add width. They add **inductive bias**: things about music
we know for sure and can compute exactly, so the model does not have to guess them from
very little data.

---

## 2. Baseline: the tab language model (README D30)

A decoder-only Transformer (pre-norm RMSNorm, SwiGLU, rotary positions, fused attention)
over DadaGP-format tokens. This is essentially what ProgGP did, and it is known to produce
convincing guitar parts.

What it has to work out by itself:

| Problem | Why it is hard from tokens alone |
|---|---|
| **Pitch** | `s7:f0` sounds different in every tuning. Each tuning is a different "dialect" the model must learn separately; 7-string notes are only 0.1 % of DadaGP's notes. |
| **Where am I in the bar?** | Only `wait:N` tokens tell time. The position is the *sum* of all waits since `new_measure`, and transformers are bad at summing long lists. Odd meters make it worse. |
| **Distance in FIM order** | In an infill example the middle comes *after* the suffix in the token sequence. Token positions say "the suffix is right behind me", although musically it comes after the gap. |
| **Riff identity and form** | Only what fits into 4096 tokens is visible. |

It still works, because transformers are good at learning such things given enough data. It
is the reference the other two have to beat.

---

## 3. Big: the music-aware model

Same decoder idea, larger, plus two things computed exactly by the code for every token
(`riffgen/dataset/serialize.py`, the same code at training and at generation time).

### 3.1 Musical embeddings

Every token gets its token embedding **plus** embeddings of what it means:

| Feature | Value | Why it helps |
|---|---|---|
| pitch | MIDI pitch = tuning[string] + fret | The *sound*. The same note on different strings or in different tunings shares a representation, so DadaGP's E-standard and drop-D songs teach Drop A. |
| pitch class, octave | pitch mod 12, pitch div 12 | Harmony (key, chords) and register separately. |
| fret | 0-29 | The hand position and the shape. |
| string from bottom | 1 = lowest string | Drop D's low string (s6 on a 6-string) and Drop A's low string (s7) are both "the low string". The drop-D riff idiom (one-finger power chords on the three lowest strings) carries over to Drop A, although the string numbers differ. |
| guitar | distorted0-2, clean0-1 | Also given to the note's effect tokens (palm mute belongs to this guitar). |
| position in bar | 1/24 quarter steps | Where the onset is, computed exactly from the waits. |
| beat strength | downbeat / beat / 8th / 16th / triplet / finer | The metrical grid (compound meters such as 6/8 and 12/8 use the dotted quarter as the beat). |
| bar relative to the gap | -64 … +63 | "3 bars before the gap", "2 bars after it". |
| segment | header / prefix / suffix / middle | Which part of the example the token is in. |

A useful side effect: notes the model has rarely seen in pretraining (high frets on the 7th
string) still get a meaningful vector from their pitch, fret and string features.

### 3.2 Multi-axis rotary positions

Rotary positions are split between three axes inside every attention head
(`riffgen/model/layers.py`):

| Axis | Position value | What attention can do with it |
|---|---|---|
| token | index in the sequence | Normal language-model behaviour. |
| musical time | 16th notes from the gap start | Sees the *true* musical distance in FIM order: the suffix is after the middle. "Same place 2 bars ago" is a fixed distance, which makes riff loops easy to follow. |
| position in bar | 16th notes inside the bar | Two notes on beat 3 of different bars have distance 0: a head can look at "what happened on this beat in the other bars", the core of riff repetition and of polymetric patterns. |

These only change q and k, so the fast fused attention kernels still work.

### 3.3 Whole songs

8192 tokens hold most songs completely (we check how many bars fit in `stats.json`).

---

## 4. Huge: the structure-aware model — how it "understands" the music

A neural network does not understand music the way a musician does. What we *can* do is
make it build and use explicit representations of the things a musician uses to fill a gap,
and check that it does:

| A musician filling a gap thinks about… | What the huge model has for it |
|---|---|
| the song's form: verse, chorus, which riffs repeat and how often, what comes after the gap | a **song memory**: one vector per bar of the *whole* song, related to each other by a bidirectional song encoder; a **section head** that must name every bar's section |
| a plan for the gap: harmony and rhythm first, notes second | **planning heads** that must predict the hidden bars' pitch classes and onset grid from the context, before a single note is written; the decoder reads that plan for the bar it writes |
| which bars are "the same riff" (also with small variations) | a **contrastive loss** that pulls repeated bars together in the bar encoder |
| the beat, the meter, polyrhythms | the big model's beat features and position-in-bar attention |
| what is playable, what sounds like the style | string/fret/pitch features, album fine-tuning, the playability check |

### 4.1 Architecture

```
the whole song (every bar; the gap and other unknown bars are MASKED)
   │
   ├─► bar encoder (3 layers, 384 wide, bidirectional, per bar) ── one vector per visible bar
   │        └─ contrastive loss: repeated bars (riffs) get similar vectors
   │
   ├─► song encoder (6 layers, 768 wide, bidirectional over bars) ── memory M: one vector per bar
   │        ├─ planning heads on the masked bars: pitch classes (harmony), onset grid (rhythm)
   │        └─ section head on every bar: intro / verse / chorus / bridge / ...
   │
token window (8192 tokens, as in "big")
   └─► music-aware decoder (24 layers)
            + "plan" vector of the token's own bar (from M) added to its input
            + gated cross-attention to all of M every 4 layers
            ─► next token
```

### 4.2 Why this is more than a bigger model

- **It sees the whole song**, whatever its length. The token window shows the details close
  to the gap; the memory shows everything else at bar level: every verse, every chorus,
  every repeat. A 200-bar prog song costs 200 memory vectors, not 15,000 tokens.
- **It plans before it writes.** The song encoder must say what harmony and rhythm the
  hidden bars probably have. This is trained directly (the true bars are known in training),
  so the plan is not a vague hidden state: it is forced to contain harmony and rhythm. The
  decoder then writes notes that fit the plan.
- **It knows what a riff is.** The README defines a riff as "a group of bars that repeats".
  The contrastive loss turns exactly that definition into a training signal for the bar
  encoder.
- **It knows the form.** The section head makes the song encoder track verse/chorus/bridge
  over the whole song, also for bars without a marker.

### 4.3 What keeps it honest

- **No leakage:** the bars the model must write are masked in the memory (for song-piece
  examples: all bars of the window). The planning targets are computed from the hidden bars
  but only used as training targets.
- **Starts as "big":** the cross-attention gates and the plan projection start at zero, so
  training begins exactly like the big model and opens the new paths only if they help.
- **Checkable:** the planning losses (`aux_pc`, `aux_onset`), the section loss and the
  contrastive loss are logged. If they do not go down, the structure is not being learned.

### 4.4 Costs and risks

- About 1.3× the compute of big; more code, more knobs (`aux_weights`).
- With little data the song encoder may underfit. We compare huge against big on the same
  held-out gaps and by blind listening (§6.4); we keep huge only if it is better.

### 4.5 What it enables later

- Choosing the plan: predicted pitch classes and onset grids are interpretable. A next step
  is letting you *set* them ("this harmony, this rhythm") and generating from that plan.
- Guessing the section of a gap when you give no `--section` (the section head already
  predicts it).
- Better similar-song selection and plagiarism check: the riff vectors of the bar encoder
  are a learned "riff similarity" (§3.9 option B uses model embeddings already).

---

## 5. Things that are possible but not built yet

How to make the model see the *whole* song (returning intro riffs, form cycles, long chord
progressions) is researched separately, with a staged plan: [LONG_CONTEXT.md](LONG_CONTEXT.md).


| Idea | Why not yet |
|---|---|
| Learn from your ratings (`ratings.jsonl`): a ranking model, later preference tuning (DPO) | Needs a few hundred ratings first. Ratings are already saved with seed and scores. |
| "Virtual 7-string" augmentation: shift drop-D 6-string riffs one string down onto Drop A | Moves riffs to another pitch, against the spirit of D12. An experiment only. |
| Initialise huge's decoder from a trained big model | Saves pretraining time; needs a weight-mapping step. |
| Shorter token window for huge on the GTX 1650 | Huge keeps the whole song in its memory, so it may lose little with 4096 tokens. To be measured. |
