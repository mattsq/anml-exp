from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path

import jsonschema
import pytest

from anml_exp.benchmarks.evaluator import run_benchmark


@pytest.mark.skipif(os.getenv("RUN_PERF") != "1", reason="Perf test")
def test_matrix_profile_perf(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    result = run_benchmark(
        dataset="nab-twitter-aapl",
        model_name="matrix_profile",
        seed=0,
        hardware="test",
        output=out,
    )
    schema = json.loads(
        files("anml_exp.resources").joinpath("results-schema.json").read_text()
    )
    jsonschema.validate(result, schema)
