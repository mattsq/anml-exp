from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")


@pytest.mark.skipif(os.getenv("RUN_PERF") != "1", reason="Perf test")
def test_benchmark_smoke(tmp_path: Path) -> None:
    """Run a lightweight benchmark and validate schema."""
    out = tmp_path / "result.json"
    from anml_exp.benchmarks.evaluator import run_benchmark
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
        output=out,
        n_estimators=10,
    )
    schema = json.loads(
        files("anml_exp.resources").joinpath("results-schema.json").read_text()
    )
    jsonschema.validate(result, schema)
