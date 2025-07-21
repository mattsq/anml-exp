from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("stumpy")
from anml_exp import cli


def test_cli_benchmark(tmp_path: Path) -> None:
    out = tmp_path / "res.json"
    cli.main([
        "benchmark",
        "--dataset",
        "toy-blobs",
        "--model",
        "isolation_forest",
        "--output",
        str(out),
    ])
    assert out.exists()
