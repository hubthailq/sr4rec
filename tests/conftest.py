import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # tests.fixtures
EXAMPLES = ROOT / "examples"


@pytest.fixture
def toy(tmp_path):
    from tests.fixtures.synthetic import make_dataset

    return make_dataset(tmp_path / "data" / "toy")


def write_cfg(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
