import numpy as np

from riffgen.dataset.sampler import ExampleBuilder, SamplerConfig
from riffgen.dataset.store import Store, song_from_view
from riffgen.generate.constraints import GapConstraint, GapSpec
from riffgen.generate.pipeline import Gap, last_notes_before
from riffgen.tokens.vocab import K
from riffgen.utils.seed import derive


def test_examples_are_deterministic_and_well_formed(dadagp_store):
    st = Store(dadagp_store)
    v = st.load_vocab()
    b = ExampleBuilder(v, SamplerConfig(max_len=1024))
    for i in range(len(st)):
        for task in ("infill", "song"):
            e1 = b.build(st.view(i), task, derive(7, i, task))
            e2 = b.build(st.view(i), task, derive(7, i, task))
            if e1 is None:
                continue
            assert np.array_equal(e1["ids"], e2["ids"])
            assert len(e1["ids"]) <= 1024
            assert e1["ids"][0] == v.bos_id
            kinds = v.kind[e1["ids"]]
            assert not np.any(e1["loss"][kinds == K.GAP] > 0)          # gap:N is never trained
            if e1["meta"]["task"] == "infill":
                m0 = int(np.flatnonzero(e1["ids"] == v.fim_middle_id)[0])
                assert e1["loss"][:m0 + 1].sum() == 0                  # context weight 0 by default
                assert np.all(e1["loss"][m0 + 1:] == 1)
                assert e1["ids"][-1] == v.eos_id


def test_constraints_accept_real_middles(dadagp_store, album_store):
    """The grammar must never block real music (except ties onto a note that is not sounding)."""
    total = accepted = 0
    for path in (dadagp_store, album_store):
        st = Store(path)
        v = st.load_vocab()
        b = ExampleBuilder(v, SamplerConfig(max_len=2048, max_middle_frac=0.9))
        for i in range(len(st)):
            view = st.view(i)
            song = song_from_view(view, v)
            ex = b.build_infill(view, derive(0, i), augment=False)
            if ex is None:
                continue
            ids = ex["ids"]
            m0 = int(np.flatnonzero(ids == v.fim_middle_id)[0]) + 1
            s, L = ex["meta"]["start"], ex["meta"]["len"]
            spec = GapSpec(L, [int(view.bars[x]["ticks"]) for x in range(s, s + L)],
                           [(int(view.bars[x]["ts_num"]), int(view.bars[x]["ts_den"])) for x in range(s, s + L)],
                           {int(v.inst[t]): 7 for t in ids[:m0] if v.kind[t] == K.TRACK}, force_ts_each_bar=True,
                           max_fret=29, prev_notes=last_notes_before(song, Gap(s, L, [])))
            c = GapConstraint(v, spec)
            total += 1
            ok = True
            for t in ids[m0:]:
                if not c.allowed()[t]:
                    ok = False
                    break
                c.advance(int(t))
            accepted += ok and c.done
    assert total > 10 and accepted >= total - 2


def test_random_tokens_under_constraints_fill_bars_exactly():
    from riffgen.tokens.vocab import Vocab
    v = Vocab.build()
    spec = GapSpec(3, [3840, 3360, 3360], [(4, 4), (7, 8), (7, 8)], {0: 7, 1: 7}, force_ts_each_bar=True)
    rng = np.random.default_rng(0)
    for _ in range(5):
        c = GapConstraint(v, spec)
        toks = []
        while not c.done and len(toks) < 5000:
            allowed = np.flatnonzero(c.allowed())
            t = int(rng.choice(allowed))
            toks.append(t)
            c.advance(t)
        assert c.done
        words = v.decode(toks)
        assert words.count("new_measure") == 3 and words[-1] == "<eos>"
        bars = []
        for w in words:
            if w == "new_measure":
                bars.append(0)
            elif w.startswith("wait:"):
                bars[-1] += int(w.split(":")[1])
        assert bars == [3840, 3360, 3360]
        assert not any(w.startswith(("bass", "drums", "bfx:tempo", "gap:", "section:")) for w in words)
