"""Build training examples on the fly (D28, D43-D46).

Two kinds of example:

    task:song    <bos> task:song (track:X tuning:Y)* tempo:T [start] bars ... [end]
    task:infill  <bos> task:infill (track:X tuning:Y)* tempo:T bars:N [section:S]
                 <fim_prefix> [start] bars before <fim_suffix> bars after [end]
                 <fim_middle> the hidden N bars <eos>

Loss weights: infill trains the middle (+<eos>); prefix/suffix get ``context_loss_weight``
(0 = README default for fine-tuning; 1.0 = "FIM for free", cheaper pretraining, see docs).
``gap:N`` tokens and bars of simple riffs (context filter, D17) never carry loss.

Augmentation (all in here, nothing is written to disk, D45):
* new gap position and length every time (D43), 80 % on a riff/section border (D29);
* guitar swap within a kind and guitar drop, for the whole example (D44);
* an extra ``gap:N`` placeholder in 20 % of infill examples (§4.3);
* section-token dropout, so the model also works without ``--section``.

Long-range context (docs/LONG_CONTEXT.md, stage 1). When a song does not fit into the window, the
context is not "the closest bars until full" any more but three parts:

* ~50 % local: the closest bars on both sides of the gap;
* ~35 % related: for the bars next to the gap, the bars that *repeat* them elsewhere in the song and
  what followed there ("last time this riff was followed by ..."), the intro and the outro, and
  other occurrences of the gap's section;
* the rest of the song is skipped and marked as ``gap:N``; the musical-time positions keep the true
  distances, so the model knows how far away every chunk is.

Training makes the model use it: some gaps are chosen where the answer repeats a bar >= 32 bars away
("reprise"), some where the intro returns at the end ("bookend"), and in 10 % of the examples the
closest 4-8 bars on both sides are hidden, so distant bars are the only help.

For the hierarchical (huge) model each example also carries the *whole song* as one entry
per bar ("song memory"), with the target bars masked, plus planning targets for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from riffgen.dataset.serialize import (SEG_HEADER, SEG_MIDDLE, SEG_PREFIX, SEG_SONG, SEG_SUFFIX,
                                       Featurizer, TrackPlan, header_tokens, make_track_plan, phrase_positions,
                                       section_in_effect, serialize_bar, triplet_token_ids)
from riffgen.dataset.store import SongView
from riffgen.tokens import sections as sections_mod
from riffgen.tokens.vocab import K, Vocab

ONSET_SLOTS = 48


@dataclass
class SamplerConfig:
    max_len: int = 4096
    task_mix: dict = field(default_factory=lambda: {"song": 0.7, "infill": 0.3})
    gap_min: int = 4
    gap_max: int = 16
    gap_length: dict = field(default_factory=lambda: {4: 0.35, 8: 0.35, "other": 0.30})
    segment_length_prob: float = 0.3
    border_shift_prob: float = 0.2
    border_shift_max: int = 2
    extra_gap_placeholder: float = 0.2
    song_gap_placeholder: float = 0.1
    guitar_swap: float = 0.5
    guitar_drop: float = 0.2
    section_dropout: float = 0.3
    max_middle_frac: float = 0.5
    context_loss_weight: float = 0.0
    min_gap_notes_frac: float = 0.5
    hierarchical: bool = False
    max_song_bars: int = 512
    max_bar_tokens: int = 192
    # structure-aware context and long-range training tasks (stage 1 of docs/LONG_CONTEXT.md)
    related: bool = True
    local_fraction: float = 0.5
    related_fraction: float = 0.35
    anchor_bars: int = 4
    edge_bars: int = 8
    far_bars: int = 32
    reprise_prob: float = 0.25
    bookend_prob: float = 0.05
    local_dropout_prob: float = 0.10
    local_dropout_min: int = 4
    local_dropout_max: int = 8

    @classmethod
    def from_cfg(cls, data_cfg: dict, max_len: int, hierarchical: bool) -> "SamplerConfig":
        aug = dict(data_cfg.get("augment", {}) or {})
        c = cls(max_len=max_len, hierarchical=hierarchical)
        c.task_mix = dict(data_cfg.get("task_mix", c.task_mix))
        for k in ("context_loss_weight", "max_middle_frac", "section_dropout", "max_song_bars", "max_bar_tokens",
                  "min_gap_notes_frac"):
            if k in data_cfg:
                setattr(c, k, data_cfg[k])
        if "gap_length" in aug:
            c.gap_length = {(int(k) if str(k).isdigit() else k): float(v) for k, v in aug["gap_length"].items()}
        bs = aug.get("border_shift", {}) or {}
        c.border_shift_prob = float(bs.get("prob", c.border_shift_prob))
        c.border_shift_max = int(bs.get("max_bars", c.border_shift_max))
        for k in ("extra_gap_placeholder", "song_gap_placeholder", "guitar_swap", "guitar_drop",
                  "segment_length_prob", "gap_min", "gap_max", "reprise_prob", "bookend_prob", "far_bars"):
            if k in aug:
                setattr(c, k, type(getattr(c, k))(aug[k]))
        ld = aug.get("local_dropout", {}) or {}
        c.local_dropout_prob = float(ld.get("prob", c.local_dropout_prob))
        c.local_dropout_min = int(ld.get("min_bars", c.local_dropout_min))
        c.local_dropout_max = int(ld.get("max_bars", c.local_dropout_max))
        ctx = data_cfg.get("context", {}) or {}
        for k in ("related", "local_fraction", "related_fraction", "anchor_bars", "edge_bars"):
            if k in ctx:
                setattr(c, k, type(getattr(c, k))(ctx[k]))
        return c


class ExampleBuilder:
    def __init__(self, vocab: Vocab, cfg: SamplerConfig):
        self.v = vocab
        self.cfg = cfg
        self.triplets = triplet_token_ids(vocab)

    # ------------------------------------------------------------------ helpers
    def _bar(self, view, b, plan, *, section: bool, strip_tempo: bool) -> list[int]:
        return serialize_bar(view, b, self.v, plan, with_ts=False, with_section=section, strip_tempo=strip_tempo,
                             triplet_ids=self.triplets)

    def _with_ts(self, toks: list[int], view: SongView, b: int) -> list[int]:
        k = 2 if len(toks) > 1 and toks[1] in self.triplets.values() else 1
        bar = view.bars[b]
        return toks[:k] + [self.v.ts_token(int(bar["ts_num"]), int(bar["ts_den"]))] + toks[k:]

    def _has_notes_after_plan(self, view: SongView, b: int, plan: TrackPlan) -> bool:
        for ev in view.bar_events(b):
            tr = int(ev["track"])
            if tr < 0 or tr not in plan.slot_map:
                continue
            if np.any(self.v.kind[view.event_tokens(ev)] == K.NOTE):
                return True
        return False

    def _unknown_bars(self, view: SongView) -> np.ndarray:
        """Bars that become gap:N in the context: song gaps and bars removed by a filter (D55)."""
        return view.bars["gap"] | view.bars["drop"]

    def eligible_targets(self, view: SongView) -> np.ndarray:
        bars = view.bars
        return ~bars["gap"] & ~bars["ctx"] & ~bars["drop"]

    def _pick_length(self, rng) -> int:
        c = self.cfg
        keys = list(c.gap_length)
        probs = np.array([c.gap_length[k] for k in keys], float)
        k = keys[rng.choice(len(keys), p=probs / probs.sum())]
        if k == "other":
            others = [x for x in range(c.gap_min, c.gap_max + 1) if x not in keys]
            return int(rng.choice(others)) if others else c.gap_min
        return int(k)

    def _far_bars(self, view: SongView) -> np.ndarray:
        """Bars that repeat a bar at least ``far_bars`` bars away (the answer exists, but far away)."""
        p = np.asarray(view.bars["partners"])
        idx = np.arange(view.n_bars)[:, None]
        return ((p >= 0) & (np.abs(p - idx) >= self.cfg.far_bars)).any(1)

    def _bookend_bars(self, view: SongView, edge: int = 16) -> np.ndarray:
        """Bars of the last ``edge`` bars that repeat a bar of the first ``edge`` bars (the intro returns)."""
        n = view.n_bars
        out = np.zeros(n, bool)
        if n >= 3 * edge:
            p = np.asarray(view.bars["partners"])[n - edge:]
            out[n - edge:] = ((p >= 0) & (p < edge)).any(1)
        return out

    def _phrase(self, view: SongView, s: int | None = None, L: int = 0) -> np.ndarray:
        return phrase_positions(view.meta.get("borders", [0]), view.n_bars, (s, L) if s is not None else None)

    def choose_gap(self, view: SongView, rng, tries: int = 24) -> tuple[int, int] | None:
        g = self.choose_gap_ex(view, rng, tries=tries)
        return (g[0], g[1]) if g else None

    def choose_gap_ex(self, view: SongView, rng, mode: str | None = None, strict: bool = False,
                      tries: int = 24) -> tuple[int, int, str] | None:
        """Pick a hidden part. mode: normal | reprise | bookend (None = drawn from the configured shares).

        reprise: most of the bars repeat a bar >= far_bars away. bookend: the gap is in the last 16
        bars and repeats the first 16 (the intro returns). Falls back to a normal gap unless ``strict``.
        """
        c = self.cfg
        n = view.n_bars
        ok = self.eligible_targets(view)
        if n < c.gap_min or ok.sum() < c.gap_min:
            return None
        if mode is None:
            r = rng.random()
            mode = "bookend" if r < c.bookend_prob else "reprise" if r < c.bookend_prob + c.reprise_prob else "normal"
        borders = [b for b in view.meta.get("borders", [0]) if b < n] or [0]
        seg_len = {g["start"]: g["end"] - g["start"] for g in view.meta.get("segments", []) if g["kind"] == "music"}
        for m in ([mode] if (mode == "normal" or strict) else [mode, "normal"]):
            wanted, need = None, 0.0
            if m == "reprise":
                wanted, need = self._far_bars(view), 0.5
            elif m == "bookend":
                wanted, need = self._bookend_bars(view), 0.25
            if wanted is not None and not wanted.any():
                continue
            for _ in range(tries):
                s = int(borders[rng.integers(len(borders))])
                if rng.random() < c.segment_length_prob and c.gap_min <= seg_len.get(s, 0) <= c.gap_max:
                    L = seg_len[s]
                else:
                    L = self._pick_length(rng)
                if rng.random() < c.border_shift_prob:
                    s += int(rng.choice([-1, 1])) * int(rng.integers(1, c.border_shift_max + 1))
                L = min(L, n)
                if s < 0 or s + L > n or not ok[s:s + L].all():
                    continue
                if wanted is not None and wanted[s:s + L].mean() < need:
                    continue
                with_notes = sum(1 for b in range(s, s + L) if view.bar_has_notes(b, self.v))
                if with_notes < c.min_gap_notes_frac * L:
                    continue
                return s, L, m
        return None

    def _next_unit(self, view, b: int, step: int, unknown, plan) -> tuple[tuple, int, int]:
        """The context unit starting at bar b going in direction ``step``: (unit, token cost, next bar).

        A run of unknown bars collapses into one ``gap`` unit; a normal bar costs its tokens
        plus one for a possible ``ts:`` token.
        """
        n = view.n_bars
        if unknown[b]:
            j = b
            while 0 <= j < n and unknown[j]:
                j += step
            run = sorted(range(b, j, step))
            return ("gap", run), len(self.v.gap_tokens(len(run))), j
        toks = self._bar(view, b, plan, section=True, strip_tempo=False)
        return ("bar", b, toks), len(toks) + 1, b + step

    # ------------------------------------------------------------------ context assembly (long range)
    def _walk(self, included, usable, cost, pi: int, si: int, n: int, budget: int, pre_used: int = 0,
              suf_used: int = 0):
        """Take bars closest-first on both sides, alternating by token count, until the budget is used."""
        pre_closed, suf_closed = pi < 0, si >= n
        used = 0
        while not (pre_closed and suf_closed):
            take_pre = not pre_closed and (suf_closed or pre_used <= suf_used)
            b = pi if take_pre else si
            c = 0 if included[b] else cost(b)
            if not usable[b] or c > budget - used:
                if take_pre:
                    pre_closed = True
                else:
                    suf_closed = True
                continue
            included[b] = True
            used += c
            if take_pre:
                pre_used += c
                pi -= 1
                pre_closed = pi < 0
            else:
                suf_used += c
                si += 1
                suf_closed = si >= n
        return used, pi, si, pre_used, suf_used

    def _related_chunks(self, view: SongView, s: int, L: int, section: int) -> list[tuple[int, int, bool]]:
        """Bar ranges worth showing although they are far away, most useful first: (lo, hi, ascending).

        * for a bar d bars before the gap that repeats bar q: the range [q, q+d+1+L), i.e. the repeat
          *and what followed it* (the gap's counterpart); same idea mirrored for bars after the gap
          (the bars that led into the repeat). ``ascending`` says which end to keep if only part of
          the range fits: the end nearest to the repeat;
        * the intro and the outro; other occurrences of the gap's section.
        """
        c = self.cfg
        n = view.n_bars
        partners = np.asarray(view.bars["partners"])

        def avoid(q: int, lo: int, hi: int, asc: bool = True):
            lo, hi = max(0, lo), min(n, hi)
            if not (hi <= s or lo >= s + L):         # never show the hidden bars themselves
                if q < s:
                    hi = min(hi, s)
                else:
                    lo = max(lo, s + L)
            return (lo, hi, asc) if hi > lo else None

        pre, post = [], []
        for d in range(c.anchor_bars):
            a, b = s - 1 - d, s + L + d
            if a >= 0:
                for q in partners[a]:
                    q = int(q)
                    if q >= 0 and not (s <= q < s + L):
                        ch = avoid(q, q, q + d + 1 + L)
                        if ch:
                            pre.append(ch)
            if b < n:
                for q in partners[b]:
                    q = int(q)
                    if q >= 0 and not (s <= q < s + L):
                        ch = avoid(q, q - d - L, q + 1, asc=False)
                        if ch:
                            post.append(ch)
        order = pre[:1] + post[:1]
        if s >= 1:
            order.insert(1 if pre else 0, (0, min(c.edge_bars, s), True))          # intro
        if s + L < n:
            order.append((max(n - c.edge_bars, s + L), n, True))                   # outro
        if section >= 0:
            label = sections_mod.labels()[section]
            same = [g for g in view.meta.get("segments", []) if g["kind"] == "music" and g.get("section") == label]
            same.sort(key=lambda g: abs(g["start"] - s))
            for g in same[:4]:
                lo, hi = g["start"], min(g["end"], g["start"] + L)
                if hi <= s or lo >= s + L:
                    order.append((lo, hi, True))
        rest_pre, rest_post = pre[1:], post[1:]
        for k in range(max(len(rest_pre), len(rest_post))):
            if k < len(rest_pre):
                order.append(rest_pre[k])
            if k < len(rest_post):
                order.append(rest_post[k])
        seen, out = set(), []
        for ch in order:
            if ch not in seen:
                seen.add(ch)
                out.append(ch)
        return out

    def _units_from_mask(self, view, lo: int, hi: int, included, unknown, plan) -> list:
        """Bars lo..hi-1 in song order: included known bars, and one ``gap`` unit per skipped run."""
        units, b = [], lo
        while b < hi:
            if included[b] and not unknown[b]:
                units.append(("bar", b, self._bar(view, b, plan, section=True, strip_tempo=False)))
                b += 1
            else:
                j = b
                while j < hi and not (included[j] and not unknown[j]):
                    j += 1
                units.append(("gap", list(range(b, j))))
                b = j
        return units

    def _context_once(self, view, s, L, unknown, plan, budget, hide, section, local_only) -> dict:
        c = self.cfg
        n = view.n_bars
        blocked = np.zeros(n, bool)
        blocked[s:s + L] = True
        if hide > 0:
            blocked[max(0, s - hide):s] = True
            blocked[s + L:min(n, s + L + hide)] = True
        if local_only:
            blocked[:max(0, s - local_only)] = True
            blocked[min(n, s + L + local_only):] = True
        usable = ~blocked
        included = np.zeros(n, bool)
        cache: dict[int, int] = {}

        def cost(b: int) -> int:
            if unknown[b]:
                return 0
            if b not in cache:
                cache[b] = len(self._bar(view, b, plan, section=True, strip_tempo=False)) + 1
            return cache[b]

        upper = len(view.tokens) + len(view.events) + 3 * n      # upper bound for the tokens of all bars
        if upper <= budget:
            included[usable] = True
        else:
            use_related = c.related and not local_only
            pi, si = s - 1 - hide, s + L + hide
            used, pi, si, pu, su = self._walk(included, usable, cost, pi, si, n,
                                              int(c.local_fraction * budget) if use_related else budget)
            if use_related:
                rel_budget, used_rel = int(c.related_fraction * budget), 0
                for lo, hi, asc in self._related_chunks(view, s, L, section):
                    for b in (range(lo, hi) if asc else range(hi - 1, lo - 1, -1)):
                        if not usable[b] or included[b]:
                            continue
                        if used_rel + cost(b) > rel_budget:
                            break                      # part of a chunk is fine: the end that matters first
                        included[b] = True
                        used_rel += cost(b)
                self._walk(included, usable, cost, pi, si, n, budget - used - used_rel, pu, su)
        idx = np.flatnonzero(included)
        pre_idx, suf_idx = idx[idx < s], idx[idx >= s + L]
        pre_lo = int(pre_idx.min()) if len(pre_idx) else s
        suf_hi = int(suf_idx.max()) + 1 if len(suf_idx) else s + L
        pre_units = self._units_from_mask(view, pre_lo, s, included, unknown, plan)
        suf_units = self._units_from_mask(view, s + L, suf_hi, included, unknown, plan)
        prefix, pre_bar_of = self._emit(view, pre_units)
        suffix, suf_bar_of = self._emit(view, suf_units)
        partners = np.asarray(view.bars["partners"][s:s + L])
        far_in_context = any(int(q) >= 0 and abs(int(q) - (s + k)) >= c.far_bars and included[int(q)]
                             for k in range(L) for q in partners[k])
        return {"pre_units": pre_units, "suf_units": suf_units, "prefix": prefix, "suffix": suffix,
                "pre_bar_of": pre_bar_of, "suf_bar_of": suf_bar_of, "first_pre": pre_lo, "suf_hi": suf_hi,
                "far_in_context": bool(far_in_context), "context_bars": int(included.sum())}

    def _context(self, view, s, L, unknown, plan, budget, *, hide: int = 0, section: int = -1,
                 local_only: int = 0) -> dict:
        """Context around the gap within ``budget`` tokens (prefix + suffix, without the markers)."""
        res = None
        for frac in (1.0, 0.92, 0.84, 0.7, 0.5, 0.3, 0.1):       # gap:N tokens are not in the estimate
            res = self._context_once(view, s, L, unknown, plan, int(budget * frac), hide, section, local_only)
            if len(res["prefix"]) + len(res["suffix"]) <= budget:
                break
        return res

    def _emit(self, view, units, *, first_ts: bool = True) -> tuple[list[int], list[int]]:
        """Units in song order -> tokens (+ the song bar of each token, -1 for gaps)."""
        out, bar_of = [], []
        prev_ts = None
        after_gap = first_ts
        for u in units:
            if u[0] == "gap":
                g = self.v.gap_tokens(len(u[1]))
                out += g
                bar_of += [-1] * len(g)
                after_gap = True
                continue
            _, b, toks = u
            ts = (int(view.bars[b]["ts_num"]), int(view.bars[b]["ts_den"]))
            if after_gap or ts != prev_ts:
                toks = self._with_ts(toks, view, b)
            prev_ts, after_gap = ts, False
            out += toks
            bar_of += [b] * len(toks)
        return out, bar_of

    def _tempo_before(self, view: SongView, s: int) -> float:
        b = s - 1 if s > 0 else min(s + 0, view.n_bars - 1)
        return float(view.bars[b]["tempo"])

    def _extra_placeholder(self, view, unknown, protect: tuple[int, int], rng, lo: int, hi: int) -> tuple[int, int] | None:
        """Hide another 4-16 bar region of the context (not touching the protected range)."""
        c = self.cfg
        for _ in range(8):
            L = int(rng.integers(c.gap_min, c.gap_max + 1))
            if hi - lo < L + 2:
                return None
            a = int(rng.integers(lo, hi - L + 1))
            if a + L >= protect[0] - 1 and a <= protect[1]:
                continue
            if unknown[a:a + L].any():
                continue
            return a, a + L
        return None

    # ------------------------------------------------------------------ public
    def build(self, view: SongView, task: str, rng: np.random.Generator, spec: dict | None = None,
              augment: bool = True) -> dict | None:
        if task == "infill":
            ex = self.build_infill(view, rng, spec, augment)
            if ex is not None:
                return ex
            if spec is not None:
                return None
        return self.build_song(view, rng, spec if spec and spec.get("task") == "song" else None, augment)

    def _plan(self, view, rng, augment, keep=None) -> TrackPlan:
        if not augment:
            return TrackPlan.identity(view)
        return make_track_plan(view, self.v, rng, self.cfg.guitar_swap, self.cfg.guitar_drop, keep)

    def build_infill(self, view: SongView, rng, spec: dict | None = None, augment: bool = True) -> dict | None:
        c, v = self.cfg, self.v
        n = view.n_bars
        if spec is not None:
            s, L, mode = int(spec["start"]), int(spec["len"]), "spec"
        else:
            g = self.choose_gap_ex(view, rng)
            if g is None:
                return None
            s, L, mode = g
        hide = 0
        if augment and spec is None and c.related and rng.random() < c.local_dropout_prob:
            hide = int(rng.integers(c.local_dropout_min, c.local_dropout_max + 1))
        plan = self._plan(view, rng, augment)
        if not any(self._has_notes_after_plan(view, b, plan) for b in range(s, s + L)):
            plan = TrackPlan.identity(view)
        unknown = self._unknown_bars(view)
        extra = None
        if augment and spec is None and rng.random() < c.extra_gap_placeholder:
            lo, hi = max(0, s - 48), min(n, s + L + 48)
            extra = self._extra_placeholder(view, unknown, (s, s + L), rng, lo, hi)
            if extra:
                unknown = unknown.copy()
                unknown[extra[0]:extra[1]] = True
        # the middle (target)
        max_middle = int(c.max_len * c.max_middle_frac)
        while True:
            mid_units = [("bar", b, self._bar(view, b, plan, section=False, strip_tempo=True)) for b in range(s, s + L)]
            middle, mid_bar_of = self._emit(view, mid_units)
            if len(middle) + 1 <= max_middle or L <= c.gap_min or spec is not None:
                break
            L -= 1
        if len(middle) + 1 > max_middle:
            return None
        section = int(spec.get("section", section_in_effect(view, s))) if spec else section_in_effect(view, s)
        if augment and spec is None and rng.random() < c.section_dropout:
            section = -1
        header = header_tokens(v, "infill", plan, self._tempo_before(view, s), n_bars=L, section=section)
        phrase = self._phrase(view, s, L)
        # at generation time the middle is not known yet: reserve room for it instead
        reserve = max(len(middle), int(spec.get("reserve", 0))) if spec else len(middle)
        budget = c.max_len - len(header) - 3 - reserve - 1 - 2
        ctx = self._context(view, s, L, unknown, plan, budget, hide=hide, section=section,
                            local_only=int(spec.get("local_only", 0)) if spec else 0)
        prefix, suffix, pre_bar_of, suf_bar_of = ctx["prefix"], ctx["suffix"], ctx["pre_bar_of"], ctx["suf_bar_of"]
        pre_units, suf_units, first_pre = ctx["pre_units"], ctx["suf_units"], ctx["first_pre"]
        at_start = first_pre == 0
        at_end = ctx["suf_hi"] >= n
        ids = list(header) + [v.fim_prefix_id] + ([v.start_id] if at_start else []) + prefix + [v.fim_suffix_id] \
            + suffix + ([v.end_id] if at_end else []) + [v.fim_middle_id] + middle + [v.eos_id]
        loss = np.zeros(len(ids), np.float32)
        p0 = len(header) + 1 + (1 if at_start else 0)
        cw = c.context_loss_weight
        if cw > 0:
            loss[p0:p0 + len(prefix)] = self._context_weights(view, prefix, pre_bar_of, cw)
            q0 = p0 + len(prefix) + 1
            loss[q0:q0 + len(suffix)] = self._context_weights(view, suffix, suf_bar_of, cw)
            if at_end:
                loss[q0 + len(suffix)] = cw
        m0 = len(ids) - len(middle) - 1
        loss[m0:] = 1.0
        feat = Featurizer(v, view.bars["ticks"], np.stack([view.bars["ts_num"], view.bars["ts_den"]], 1), plan.tunings,
                          {SEG_PREFIX: first_pre, SEG_SUFFIX: s + L, SEG_MIDDLE: s}, ref_bar=s,
                          bar_phrase=phrase).run(ids)
        ex = {"ids": np.array(ids, np.int32), "loss": loss, **feat,
              "meta": {"song": view.meta["id"], "task": "infill", "start": s, "len": L, "section": section,
                       "swap": plan.swapped, "drop": plan.dropped, "extra_gap": extra, "mode": mode, "hide": hide,
                       "first_pre": first_pre, "context_bars": ctx["context_bars"],
                       "far_in_context": ctx["far_in_context"],
                       "prefix_bars": sum(u[0] == "bar" for u in pre_units),
                       "suffix_bars": sum(u[0] == "bar" for u in suf_units)}}
        if c.hierarchical:
            ex["song"] = self.song_memory(view, plan, target=(s, s + L), unknown=unknown, extra=extra,
                                          tok_bar=feat["bar"], seg=feat["seg"], ref=s, bar_phrase=phrase)
        return ex

    def _context_weights(self, view, toks, bar_of, cw) -> np.ndarray:
        w = np.full(len(toks), cw, np.float32)
        kinds = self.v.kind[np.array(toks, np.int64)] if toks else np.zeros(0)
        for i, b in enumerate(bar_of):
            if b < 0 or kinds[i] == K.GAP or view.bars[b]["ctx"]:
                w[i] = 0.0
        return w

    def build_song(self, view: SongView, rng, spec: dict | None = None, augment: bool = True) -> dict | None:
        c, v = self.cfg, self.v
        n = view.n_bars
        if n == 0:
            return None
        plan = self._plan(view, rng, augment)
        unknown = self._unknown_bars(view)
        if spec is not None:
            a = int(spec["start"])
        else:
            a = 0 if rng.random() < 0.25 else int(rng.integers(0, max(1, n - 4)))
        extra = None
        if augment and spec is None and rng.random() < c.song_gap_placeholder and n - a > 24:
            extra = self._extra_placeholder(view, unknown, (a, a + 8), rng, a + 8, min(n, a + 96))
            if extra:
                unknown = unknown.copy()
                unknown[extra[0]:extra[1]] = True
        header = header_tokens(v, "song", plan, float(view.bars[a]["tempo"]))
        budget = c.max_len - len(header) - 2
        units, b = [], a
        while b < n:
            unit, cost, nxt = self._next_unit(view, b, 1, unknown, plan)
            if cost > budget:
                break
            budget -= cost
            units.append(unit)
            b = nxt
        if not units:
            return None
        body, bar_of = self._emit(view, units)
        last = b - 1
        at_end = b >= n
        ids = list(header) + ([v.start_id] if a == 0 else []) + body + ([v.end_id] if at_end else [])
        loss = np.zeros(len(ids), np.float32)
        b0 = len(header) + (1 if a == 0 else 0)
        loss[b0:b0 + len(body)] = self._context_weights(view, body, bar_of, 1.0)
        if at_end:
            loss[-1] = 1.0
        phrase = self._phrase(view)
        feat = Featurizer(v, view.bars["ticks"], np.stack([view.bars["ts_num"], view.bars["ts_den"]], 1), plan.tunings,
                          {SEG_SONG: a}, ref_bar=a, bar_phrase=phrase).run(ids)
        ex = {"ids": np.array(ids, np.int32), "loss": loss, **feat,
              "meta": {"song": view.meta["id"], "task": "song", "start": a, "len": len(units),
                       "swap": plan.swapped, "drop": plan.dropped, "extra_gap": extra}}
        if c.hierarchical:
            ex["song"] = self.song_memory(view, plan, target=(a, last + 1), unknown=unknown, extra=extra,
                                          tok_bar=feat["bar"], seg=feat["seg"], ref=a, bar_phrase=phrase)
        return ex

    # ------------------------------------------------------------------ song memory (huge model)
    def song_memory(self, view: SongView, plan: TrackPlan, target: tuple[int, int], unknown: np.ndarray,
                    extra, tok_bar: np.ndarray, seg: np.ndarray, ref: int, bar_phrase: np.ndarray) -> dict:
        """One entry per bar of the whole song. Target bars are masked; planning targets for them."""
        c, v = self.cfg, self.v
        n = view.n_bars
        center = (target[0] + target[1]) // 2
        lo = max(0, min(center - c.max_song_bars // 2, n - c.max_song_bars))
        hi = min(n, lo + c.max_song_bars)
        nb = hi - lo
        Lb = c.max_bar_tokens
        bar_ids = np.zeros((nb, Lb), np.int32)
        bar_feat = {k: np.zeros((nb, Lb), np.int16) for k in ("inst", "pitch", "fret", "sfb", "beat", "pos")}
        bar_len = np.zeros(nb, np.int16)
        mask_kind = np.zeros(nb, np.int8)          # 0 visible, 1 hidden with target, 2 unknown
        aux_pc = np.zeros((nb, 12), np.float32)
        aux_onset = np.zeros((nb, ONSET_SLOTS), np.float32)
        section = np.full(nb, -1, np.int16)
        has_markers = bool((view.bars["section"] >= 0).any())
        sec = -1
        for b in range(0, hi):
            if view.bars[b]["section"] >= 0:
                sec = int(view.bars[b]["section"])
            if b >= lo and has_markers:
                section[b - lo] = sec
        ts_arr = np.stack([view.bars["ts_num"], view.bars["ts_den"]], 1)
        for k in range(nb):
            b = lo + k
            hidden = target[0] <= b < target[1] or (extra is not None and extra[0] <= b < extra[1])
            if hidden:
                mask_kind[k] = 1
                self._aux_targets(view, b, plan, aux_pc[k], aux_onset[k])
                continue
            if unknown[b]:
                mask_kind[k] = 2
                continue
            toks = self._with_ts(self._bar(view, b, plan, section=True, strip_tempo=True), view, b)[:Lb]
            f = Featurizer(v, view.bars["ticks"], ts_arr, plan.tunings, {SEG_SONG: b}, ref_bar=b)
            f._enter(SEG_SONG)
            rows = [f.step(t) for t in toks]
            bar_ids[k, :len(toks)] = toks
            bar_len[k] = len(toks)
            for j, r in enumerate(rows):
                bar_feat["inst"][k, j] = r[2] + 1
                bar_feat["pitch"][k, j] = r[3]
                bar_feat["fret"][k, j] = r[4]
                bar_feat["sfb"][k, j] = r[5]
                bar_feat["beat"][k, j] = r[6]
                bar_feat["pos"][k, j] = r[8]
        rel = np.clip(np.arange(lo, hi) - ref, -64, 63) + 64
        tempo = np.clip(view.bars["tempo"][lo:hi] / 10, 0, 40).astype(np.int16)
        ts_num = np.clip(ts_arr[lo:hi, 0], 0, 32).astype(np.int16)
        ts_den = np.clip(np.log2(np.maximum(ts_arr[lo:hi, 1], 1)).round(), 0, 5).astype(np.int16)
        local = np.where((tok_bar >= lo) & (tok_bar < hi) & (seg != SEG_HEADER), tok_bar - lo, -1).astype(np.int32)
        pairs = []
        partners = view.bars["partners"][lo:hi]
        for k in range(nb):
            if mask_kind[k] != 0:
                continue
            for j in partners[k]:
                j = int(j)
                if lo <= j < hi and mask_kind[j - lo] == 0 and j - lo != k:
                    pairs.append((k, j - lo))
        return {
            "bar_ids": bar_ids, "bar_len": bar_len, **{f"bar_{k}": a for k, a in bar_feat.items()},
            "mask_kind": mask_kind, "rel": rel.astype(np.int16), "tempo": tempo, "ts_num": ts_num, "ts_den": ts_den,
            "section": section, "phrase": (bar_phrase[lo:hi] + 1).astype(np.int16), "aux_pc": aux_pc, "aux_onset": aux_onset,
            "aux_mask": (mask_kind == 1) & (aux_pc.sum(1) > 0),
            "pairs": np.array(pairs, np.int32).reshape(-1, 2), "tok_bar": local, "bar_lo": lo,
        }

    def _aux_targets(self, view, b, plan, pc, onset) -> None:
        bar = view.bars[b]
        ticks = max(int(bar["ticks"]), 1)
        for ev in view.bar_events(b):
            tr = int(ev["track"])
            if tr < 0 or tr not in plan.slot_map:
                continue
            tuning = plan.tunings[plan.slot_map[tr]]
            hit = False
            for t in view.event_tokens(ev):
                if v_kind(self.v, t) == K.NOTE:
                    s, f = int(self.v.string[t]), int(self.v.fret[t])
                    if 1 <= s <= len(tuning):
                        pc[(tuning[s - 1] + f) % 12] = 1.0
                        hit = True
            if hit:
                onset[min(ONSET_SLOTS - 1, int(ev["tick"]) * ONSET_SLOTS // ticks)] = 1.0


def v_kind(vocab: Vocab, t) -> int:
    return int(vocab.kind[int(t)])


# ---------------------------------------------------------------------------
# fixed validation examples (D43: validation gaps are fixed)

def make_val_specs(views: list[tuple[int, SongView]], builder: ExampleBuilder, seed: int, n_infill: int,
                   n_song: int) -> list[dict]:
    from riffgen.utils.seed import derive
    rng = derive(seed, "val_specs")
    specs = []
    if not views:
        return specs
    order = rng.permutation(len(views))
    k = 0
    tries = 0
    while len([s for s in specs if s["task"] == "infill"]) < n_infill and tries < n_infill * 5:
        idx, view = views[order[k % len(views)]]
        k += 1
        tries += 1
        g = builder.choose_gap(view, derive(seed, "val_gap", idx, k))
        if g:
            specs.append({"store_song": idx, "song": view.meta["id"], "task": "infill", "start": g[0], "len": g[1]})
    for j in range(n_song):
        idx, view = views[order[(k + j) % len(views)]]
        a = int(derive(seed, "val_song", idx, j).integers(0, max(1, view.n_bars - 4)))
        specs.append({"store_song": idx, "song": view.meta["id"], "task": "song", "start": a})
    return specs


def make_far_specs(views: list[tuple[int, SongView]], builder: ExampleBuilder, seed: int, n: int) -> list[dict]:
    """Fixed gaps whose bars repeat a bar >= far_bars away: they measure whether far context helps."""
    from riffgen.utils.seed import derive
    rng = derive(seed, "far_specs")
    specs: list[dict] = []
    if not views:
        return specs
    order = rng.permutation(len(views))
    k = tries = 0
    while len(specs) < n and tries < n * 8:
        idx, view = views[order[k % len(views)]]
        k += 1
        tries += 1
        g = builder.choose_gap_ex(view, derive(seed, "far_gap", idx, k), mode="reprise", strict=True)
        if g:
            specs.append({"store_song": idx, "song": view.meta["id"], "task": "infill", "start": g[0], "len": g[1]})
    return specs


def holdout_specs(views: list[tuple[int, SongView]], builder: ExampleBuilder, lengths=(4, 8), limit: int = 200) -> list[dict]:
    """Every border x every length of the held-out songs (fine-tuning early stopping)."""
    specs = []
    for idx, view in views:
        ok = builder.eligible_targets(view)
        for b in view.meta.get("borders", [0]):
            for L in lengths:
                if b + L <= view.n_bars and ok[b:b + L].all():
                    specs.append({"store_song": idx, "song": view.meta["id"], "task": "infill", "start": int(b), "len": L})
    if len(specs) > limit:
        step = len(specs) / limit
        specs = [specs[int(i * step)] for i in range(limit)]
    return specs
