"""Stage 1 of docs/LONG_CONTEXT.md: structure-aware context, far-repeat tasks, phrase positions."""

import dataclasses

import numpy as np
import pytest

from riffgen.dataset.sampler import ExampleBuilder, SamplerConfig, make_far_specs
from riffgen.dataset.serialize import phrase_positions
from riffgen.dataset.store import Store
from riffgen.generate.pipeline import Gap, build_prompt
from riffgen.tokens.vocab import K
from riffgen.tools.synth import NEEDLE, write_long_corpus
from riffgen.utils.seed import derive

MAX_LEN = 2048


@pytest.fixture(scope="module")
def long_store(tmp_path_factory):
    from riffgen.dataset import preprocess
    root = tmp_path_factory.mktemp("long")
    write_long_corpus(root / "dadagp", n_songs=4, n_bars=160)
    out = root / "store"
    preprocess.main(["dadagp", "--src", str(root / "dadagp"), "--out", str(out), "--workers", "1",
                     "--val-fraction", "0"])
    return Store(out)


def _builder(store, **kw):
    cfg = dataclasses.replace(SamplerConfig(max_len=MAX_LEN, max_middle_frac=0.5), **kw)
    return ExampleBuilder(store.load_vocab(), cfg)


def _needle_in_prefix(ex, vocab) -> bool:
    ids = ex["ids"]
    pre = ids[int(np.flatnonzero(ids == vocab.fim_prefix_id)[0]):int(np.flatnonzero(ids == vocab.fim_suffix_id)[0])]
    return vocab.id(NEEDLE[0]) in set(pre.tolist())


def test_the_song_is_too_long_for_the_window(long_store):
    from riffgen.dataset.serialize import TrackPlan
    b = _builder(long_store)
    view = long_store.view(0)
    total = sum(len(b._bar(view, k, TrackPlan.identity(view), section=True, strip_tempo=False)) for k in range(view.n_bars))
    assert view.n_bars >= 150 and total > 2 * MAX_LEN


def test_gap_at_the_end_sees_the_intro_riff(long_store):
    """The intro riff (unique note) returns in the last 4 bars; the window must contain the intro."""
    vocab = long_store.load_vocab()
    for i in range(len(long_store)):
        view = long_store.view(i)
        n = view.n_bars
        spec = {"task": "infill", "start": n - 4, "len": 4}
        with_ctx = _builder(long_store).build_infill(view, np.random.default_rng(0), spec=spec, augment=False)
        without = _builder(long_store, related=False).build_infill(view, np.random.default_rng(0), spec=spec,
                                                                    augment=False)
        assert len(with_ctx["ids"]) <= MAX_LEN
        assert _needle_in_prefix(with_ctx, vocab), "intro missing from the structure-aware window"
        assert not _needle_in_prefix(without, vocab), "control: closest-first window should not reach the intro"
        # the far bars are marked, not silently glued together
        assert (vocab.kind[with_ctx["ids"]] == K.GAP).any()


def test_window_never_contains_the_hidden_bars(long_store):
    """The answer must not leak into the context (the needle is only in the hidden bars and the intro)."""
    vocab = long_store.load_vocab()
    view = long_store.view(0)
    n = view.n_bars
    spec = {"task": "infill", "start": n - 4, "len": 4}
    ex = _builder(long_store).build_infill(view, np.random.default_rng(0), spec=spec, augment=False)
    ids = ex["ids"]
    bars = ex["bar"]
    seg = ex["seg"]
    ctx = (seg == 1) | (seg == 2)           # prefix, suffix
    assert not np.any(ctx & (bars >= n - 4) & (bars < n) & (vocab.kind[ids] == K.NOTE))


def test_far_repeats_are_found_and_in_the_window(long_store):
    vocab = long_store.load_vocab()
    b = _builder(long_store)
    views = [(i, long_store.view(i)) for i in range(len(long_store))]
    specs = make_far_specs(views, b, seed=3, n=12)
    assert len(specs) >= 6

    def rate(builder):
        hits = [builder.build_infill(long_store.view(sp["store_song"]), np.random.default_rng(0), spec=sp,
                                     augment=False) for sp in specs]
        assert all(len(h["ids"]) <= MAX_LEN for h in hits)
        return float(np.mean([h["meta"]["far_in_context"] for h in hits]))

    new, old = rate(b), rate(_builder(long_store, related=False))
    assert new >= old + 0.25, (new, old)       # measured on a larger set: ~0.88 vs ~0.07
    assert vocab is not None


def test_training_draws_the_long_range_tasks(long_store):
    b = _builder(long_store, reprise_prob=0.5, bookend_prob=0.2)
    modes = []
    for k in range(60):
        ex = b.build_infill(long_store.view(k % len(long_store)), derive(1, "mode", k))
        modes.append(ex["meta"]["mode"])
    assert {"normal", "reprise", "bookend"} <= set(modes), modes


def test_local_dropout_hides_the_closest_bars(long_store):
    vocab = long_store.load_vocab()
    b = _builder(long_store, local_dropout_prob=1.0, extra_gap_placeholder=0.0)
    ex = b.build_infill(long_store.view(0), derive(2, "hide"))
    assert 4 <= ex["meta"]["hide"] <= 8
    ids = ex["ids"]
    i_suf, i_mid = (int(np.flatnonzero(ids == x)[0]) for x in (vocab.fim_suffix_id, vocab.fim_middle_id))
    assert vocab.kind[ids[i_suf - 1]] == K.GAP and vocab.kind[ids[i_suf + 1]] == K.GAP
    assert i_mid > i_suf


def test_local_only_window_is_a_plain_neighbourhood(long_store):
    vocab = long_store.load_vocab()
    view = long_store.view(0)
    n = view.n_bars
    spec = {"task": "infill", "start": n - 4, "len": 4, "local_only": 16}
    ex = _builder(long_store).build_infill(view, np.random.default_rng(0), spec=spec, augment=False)
    assert not _needle_in_prefix(ex, vocab)
    assert ex["meta"]["context_bars"] <= 16 + 1 + 4


def test_phrase_positions():
    ph = phrase_positions([0, 8, 16], 24)
    assert list(ph[:10]) == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1] and ph[16] == 0
    # the gap edges are borders; borders inside the hidden part are ignored
    ph = phrase_positions([0, 8, 12, 16], 24, gap=(10, 6))
    assert ph[10] == 0 and ph[11] == 1 and ph[12] == 2 and ph[16] == 0 and ph[9] == 1
    assert phrase_positions([0], 40)[39] == 15       # capped at 16 bars


def test_training_and_generation_compute_the_same_features(long_store):
    """The generator's featurizer, stepped through the prompt and then the true middle, must reproduce
    the features of a training example for the same gap."""
    vocab = long_store.load_vocab()
    view = long_store.view(1)
    n = view.n_bars
    gap = Gap(n - 20, 8, list(range(n - 20, n - 12)))
    sc = SamplerConfig(max_len=MAX_LEN, max_middle_frac=0.5)
    prompt, feat = build_prompt(view, vocab, sc, gap, None)
    spec = {"task": "infill", "start": gap.start, "len": gap.length,
            "reserve": min(int(sc.max_len * sc.max_middle_frac), 110 * gap.length)}
    ex = ExampleBuilder(vocab, sc).build_infill(view, np.random.default_rng(0), spec=spec, augment=False)
    cut = len(prompt["ids"])
    assert np.array_equal(prompt["ids"], ex["ids"][:cut])
    rows = [feat.step(int(t)) for t in ex["ids"][cut:]]
    got = {"bar": [r[0] for r in rows], "pitch": [r[3] for r in rows], "phrase": [r[11] for r in rows],
           "time": [r[9] for r in rows]}
    assert np.array_equal(got["bar"], ex["bar"][cut:])
    assert np.array_equal(got["phrase"], ex["phrase"][cut:])
    assert np.allclose(got["time"], ex["time"][cut:])
    assert ex["phrase"][cut] == 1                      # the hidden part starts a phrase
    assert np.array_equal(got["pitch"], ex["pitch"][cut:])


def test_short_songs_are_unchanged_by_the_new_assembly(dadagp_store):
    """When everything fits, the window is simply the whole song (nothing skipped)."""
    st = Store(dadagp_store)
    v = st.load_vocab()
    b = ExampleBuilder(v, SamplerConfig(max_len=4096))
    checked = 0
    for i in range(len(st)):
        if st.view(i).n_bars > 40:
            continue
        ex = b.build_infill(st.view(i), derive(0, i), augment=False)
        if ex is None:
            continue
        checked += 1
        assert ex["meta"]["context_bars"] >= st.view(i).n_bars - ex["meta"]["len"] - 1
    assert checked >= 10
