import numpy as np
import pytest
import torch

from riffgen.dataset.loader import collate
from riffgen.dataset.sampler import ExampleBuilder, SamplerConfig
from riffgen.dataset.store import Store
from riffgen.model.lora import apply_lora, merge_lora
from riffgen.model.model import RiffGenModel
from riffgen.train.common import load_model_yaml, model_config
from riffgen.utils.seed import derive


def _model(name, vocab):
    cfg = model_config(load_model_yaml(name), vocab)
    if name in ("baseline", "big", "huge"):
        cfg.n_layer, cfg.d_model, cfg.n_head, cfg.d_ff = 4, 64, 4, 128
        cfg.bar_encoder.d_model, cfg.bar_encoder.n_head, cfg.bar_encoder.d_ff = 32, 2, 64
        cfg.song_encoder.d_model, cfg.song_encoder.n_head, cfg.song_encoder.d_ff = 48, 4, 96
        cfg.cross_attn_every = 2
    cfg.dropout = 0.0
    torch.manual_seed(0)
    m = RiffGenModel(cfg).eval()
    if cfg.hierarchical:
        with torch.no_grad():
            for blk in m.blocks:
                if blk.cross is not None:
                    blk.cross.gate.fill_(0.5)
            m.plan_proj.weight.normal_(0, 0.02)
    return m


def _example(store, vocab, cfg, i=3):
    b = ExampleBuilder(vocab, SamplerConfig(max_len=cfg.max_seq_len, hierarchical=cfg.hierarchical,
                                            max_song_bars=cfg.max_song_bars, max_bar_tokens=cfg.max_bar_tokens))
    return b.build_infill(store.view(i), derive(1, 2), augment=True)


@pytest.mark.parametrize("name", ["baseline", "big", "huge"])
def test_forward_backward(dadagp_store, name):
    st = Store(dadagp_store)
    v = st.load_vocab()
    m = _model(name, v).train()
    ex = _example(st, v, m.cfg)
    out = m(collate([ex, ex], v.pad_id))
    loss = out["loss_sum"] / out["weight_sum"] + sum(out.get("aux", {}).values())
    loss.backward()
    assert torch.isfinite(loss)
    assert m.emb.tok.weight.grad is not None


@pytest.mark.parametrize("name", ["baseline", "big", "huge"])
def test_kv_cache_matches_full_forward(dadagp_store, name):
    st = Store(dadagp_store)
    v = st.load_vocab()
    m = _model(name, v)
    ex = _example(st, v, m.cfg)
    ids = ex["ids"]
    T = len(ids)
    cut = int(np.flatnonzero(ids == v.fim_middle_id)[0]) + 1
    with torch.no_grad():
        ref = m.lm_head(m.hidden(collate([ex], v.pad_id, 1)))[0].float()
    pr = {k: (x[:cut] if isinstance(x, np.ndarray) and x.shape[:1] == ids.shape[:1] else x) for k, x in ex.items()}
    if "song" in ex:
        pr["song"] = dict(ex["song"])
        pr["song"]["tok_bar"] = ex["song"]["tok_bar"][:cut]
    state = m.prefill(collate([pr], v.pad_id, 1), 1, 1)
    B = 2
    for c in state["caches"]:
        H, hd = c["pk"].shape[1], c["pk"].shape[3]
        c["gk"], c["gv"], c["n"] = torch.zeros(B, H, T, hd), torch.zeros(B, H, T, hd), 0
    diff = (m.prompt_logits(state)[0] - ref[cut - 1]).abs().max().item()
    keys = ["ids", "inst", "pitch", "fret", "sfb", "beat", "seg", "pos", "bar_rel", "time", "phase"]
    for t in range(cut, min(T, cut + 30)):
        f = {k: torch.as_tensor(np.asarray(ex[k][t:t + 1])).reshape(1, 1).expand(B, 1) for k in keys}
        f["abs_bar"] = torch.tensor([[int(ex["bar"][t])]]).expand(B, 1)
        out = m.decode_step(state, f)
        diff = max(diff, (out[1] - ref[t]).abs().max().item())
    assert diff < 1e-4


def test_lora_merge_is_exact(dadagp_store):
    st = Store(dadagp_store)
    v = st.load_vocab()
    m = _model("big", v)
    apply_lora(m, rank=4, alpha=8, dropout=0.0)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith(".B"):
                p.normal_(0, 0.02)
    ex = collate([_example(st, v, m.cfg)], v.pad_id)
    with torch.no_grad():
        a = m(ex)["loss_sum"].item()
        merge_lora(m)
        b = m(ex)["loss_sum"].item()
    assert abs(a - b) < 1e-3 * max(1.0, abs(a))
