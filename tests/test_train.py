import re
import sys
import subprocess
from pathlib import Path

def _run_train(workdir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "train.py"],
        cwd=workdir,
        text=True,
        capture_output=True,
        timeout=180,
    )

def test_train_generates_artifacts(tmp_repo: Path):
    proc = _run_train(tmp_repo)
    assert proc.returncode == 0, f"train.py fallito.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    results = tmp_repo / "results"
    model = tmp_repo / "model"

    metrics = results / "metrics.txt"
    png = results / "model_results.png"
    skops = model / "drug_pipeline.skops"

    assert metrics.exists() and metrics.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0
    assert skops.exists() and skops.stat().st_size > 0

    content = metrics.read_text(encoding="utf-8")
    vals = [float(v) for v in re.findall(r":\s*([0-9]*\.?[0-9]+)", content)[:2]]
    assert len(vals) == 2, f"Formato inatteso per metrics.txt: {content!r}"
    for v in vals:
        assert 0.0 <= v <= 1.0

def test_train_is_idempotent(tmp_repo: Path):
    p1 = _run_train(tmp_repo); assert p1.returncode == 0, p1.stderr
    p2 = _run_train(tmp_repo); assert p2.returncode == 0, p2.stderr

    assert (tmp_repo / "results" / "metrics.txt").exists()
    assert (tmp_repo / "results" / "model_results.png").exists()
    assert (tmp_repo / "model" / "drug_pipeline.skops").exists()