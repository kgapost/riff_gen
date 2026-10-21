"""RiffGen: fill gaps in Guitar Pro songs with 7-string Drop A guitar riffs."""

from pathlib import Path

__version__ = "0.1.0"

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
CONFIG_DIR = REPO_DIR / "configs"
THIRD_PARTY_DIR = REPO_DIR / "third_party"
