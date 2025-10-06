import shutil
from pathlib import Path
import pytest

@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]

    # Copia train.py
    shutil.copy2(repo / "train.py", tmp_path / "train.py")

    # Dataset
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    src_csv = repo / "data" / "drug200.csv"
    assert src_csv.exists(), "Manca data/drug200.csv nel repo"
    shutil.copy2(src_csv, tmp_path / "data" / "drug200.csv")

    # Cartelle output che train.py si aspetta
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "model").mkdir(parents=True, exist_ok=True)

    return tmp_path