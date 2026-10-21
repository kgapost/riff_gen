import shutil
from pathlib import Path

import pytest

from riffgen.tools.synth import write_album, write_dadagp_corpus

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth")
    write_dadagp_corpus(root / "dadagp", 30)
    write_album(root / "album")
    # the real DadaGP example song, as a DadaGP-style folder
    d = root / "dadagp" / "P" / "Progmetal"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "progmetal.gp3", d / "progmetal.gp3")
    shutil.copy(FIXTURES / "progmetal.gp3.tokens.txt", d / "progmetal.gp3.tokens.txt")
    return root


@pytest.fixture(scope="session")
def dadagp_store(synth_root) -> Path:
    from riffgen.dataset import preprocess
    out = synth_root / "store"
    preprocess.main(["dadagp", "--src", str(synth_root / "dadagp"), "--out", str(out), "--workers", "1",
                     "--val-fraction", "0.2"])
    return out


@pytest.fixture(scope="session")
def album_config(synth_root) -> Path:
    cfg = synth_root / "album.yaml"
    cfg.write_text(f"""inherit: train/finetune.yaml
album_id: test_album
model: test_tiny
reports_dir: {synth_root / 'reports'}
data:
  root: {synth_root / 'album'}
  store: {synth_root / 'album_store'}
  own: own
  others: [favourites]
  holdout: [own/track03.gp5]
""")
    return cfg


@pytest.fixture(scope="session")
def album_store(synth_root, album_config) -> Path:
    from riffgen.dataset import preprocess
    out = synth_root / "album_store"
    preprocess.main(["album", str(album_config), "--out", str(out), "--workers", "1"])
    return out
