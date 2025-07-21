from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.skipif(pytest.importorskip("pandas") is None, reason="requires pandas")
def test_write_leaderboard(tmp_path: Path) -> None:
    from anml_exp.benchmarks.evaluator import run_benchmark
    from anml_exp.benchmarks.leaderboard import write_leaderboard
    pytest.importorskip("tabulate")
    result = run_benchmark(
        dataset="toy-blobs",
        model_name="isolation_forest",
        seed=0,
        hardware={
            "device_type": "CPU",
            "vendor": "test",
            "model": "generic",
            "driver": "N/A",
            "num_devices": 1,
            "notes": "test",
        },
        output=tmp_path / "res.json",
        n_estimators=10,
    )
    write_leaderboard([result], tmp_path)
    assert (tmp_path / "leaderboard.json").exists()
    assert (tmp_path / "leaderboard.md").exists()

