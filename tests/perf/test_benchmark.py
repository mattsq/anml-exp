from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from pathlib import Path as _Path

import jsonschema
import pytest

sys.path.append(str(_Path(__file__).resolve().parents[2]))
from evaluator import run_benchmark


@pytest.mark.skipif(os.getenv("RUN_PERF") != "1", reason="Perf test")
def test_benchmark_smoke(tmp_path: Path) -> None:
    """Run a lightweight benchmark and validate schema."""
    out = tmp_path / "result.json"
    result = run_benchmark(
        dataset="toy-blobs",
        model_name="isolation_forest",
        seed=0,
        hardware="test",
        output=out,
        n_estimators=10,
    )
    schema = json.loads(Path("results/results-schema.json").read_text())
    jsonschema.validate(result, schema)
