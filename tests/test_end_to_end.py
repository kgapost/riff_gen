"""The whole pipeline on synthetic data: preprocess, pretrain, resume, finetune (+DadaGP mix,
plagiarism index), generate (constrained, spliced into the original file), check, eval."""

import json

import guitarpro as gp
import pytest

from riffgen.tools import smoke


@pytest.mark.parametrize("huge", [False, True])
def test_smoke(tmp_path, huge):
    root = tmp_path / "smoke"
    smoke.main(["--keep", str(root), "--songs", "24"] + (["--huge"] if huge else []))
    out = root / "outputs"
    rep = json.loads((out / "report.json").read_text())
    assert rep["kept"] >= 1
    orig = gp.parse(str(next((root / "album" / "songs").glob("*.gp5"))))
    cand = gp.parse(str(out / "cand_01.gp5"))
    assert len(cand.measureHeaders) == len(orig.measureHeaders)
    for t in cand.tracks:
        for m in t.measures:
            beats = m.voices[0].beats
            if beats:
                assert sum(b.duration.time for b in beats) == m.header.length
    assert (root / "listen" / "key.json").is_file()
