import shutil
from pathlib import Path

import pytest

from riffgen.tools.synth import write_album, write_token_corpus

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth")
    write_token_corpus(root / "tokens", 30)
    write_album(root / "album")
    # the real the corpus example song, as a corpus-style folder
    d = root / "tokens" / "P" / "Progmetal"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "progmetal.gp3", d / "progmetal.gp3")
    shutil.copy(FIXTURES / "progmetal.gp3.tokens.txt", d / "progmetal.gp3.tokens.txt")
    return root


@pytest.fixture(scope="session")
def pretrain_store(synth_root) -> Path:
    from riffgen.dataset import preprocess
    out = synth_root / "store"
    preprocess.main(["tokens", "--src", str(synth_root / "tokens"), "--out", str(out), "--workers", "1",
                     "--val-fraction", "0.2"])
    return out


@pytest.fixture(scope="session")
def album_store(synth_root) -> Path:
    from riffgen.dataset import preprocess
    out = synth_root / "album_store"
    preprocess.main(["album", str(synth_root / "album"), "model=test_tiny", f"reports_dir={synth_root / 'reports'}",
                     "data.others=[favourites]", "data.holdout=[own/track03.gp5]",
                     "--out", str(out), "--workers", "1"])
    return out
