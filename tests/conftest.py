import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import build_fixtures  # noqa: E402


@pytest.fixture(scope="session")
def fixture_dataset() -> Path:
    """The committed mini XMIDI fixture dataset (rebuilt if missing)."""
    d = build_fixtures.DEFAULT_OUT
    if not d.exists() or len(list(d.rglob("*"))) < len(build_fixtures.SPECS):
        build_fixtures.build(d)
    return d


@pytest.fixture
def tonal_piece():
    return build_fixtures.tonal_piece
