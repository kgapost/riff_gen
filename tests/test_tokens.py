from riffgen.tokens import tunings as T
from riffgen.tokens.vocab import K, Vocab, split_wait, swap_table


def test_vocab_deterministic_and_dadagp_first():
    v1, v2 = Vocab.build(), Vocab.build()
    assert v1.hash == v2.hash
    assert v1.tokens[0] == "wait:240"          # first entry of DadaGP's frequency list keeps id 0
    assert v1.kind[v1.id("distorted0:note:s7:f24")] == K.NOTE
    assert v1.id("distorted0:note:s7:f24") != v1.unk_id   # every Drop A note exists


def test_split_wait_matches_dadagp():
    assert split_wait(121) == [64, 32, 16, 8, 1]
    assert split_wait(480) == [480]
    assert split_wait(20080) == [16384, 2048, 1024, 512, 64, 32, 16]
    assert sum(split_wait(1234)) == 1234


def test_tunings():
    assert T.name_of(T.DROP_A7) == "drop_a7"
    assert T.classify(T.DROP_A7) == ("drop", 7, 0)
    assert T.name_of(T.make("drop", 6, 0)) == "drop_d6"
    assert T.name_of(T.make("dropdbl", 7, 0)) == "dropdbl_a7"   # DadaGP's "g7_drop"
    assert T.dadagp_drop_strings(T.make("drop", 6, -2)) == {6}
    assert T.dadagp_drop_strings(T.DROP_A7) == frozenset()     # not a DadaGP tuning at all
    assert len(set(T.all_names())) == len(T.all_names())


def test_swap_table():
    v = Vocab.build()
    tab = swap_table(v, {0: 1, 1: 0})
    assert v.tokens[tab[v.id("distorted0:note:s6:f3")]] == "distorted1:note:s6:f3"
    assert v.tokens[tab[v.id("distorted1:rest")]] == "distorted0:rest"
    assert tab[v.id("nfx:palm_mute")] == v.id("nfx:palm_mute")
