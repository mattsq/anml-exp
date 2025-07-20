from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path

import jsonschema
import pytest

from anml_exp.benchmarks.evaluator import run_benchmark


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
    schema = json.loads(
        files("anml_exp.resources").joinpath("results-schema.json").read_text()
    )
    jsonschema.validate(result, schema)
