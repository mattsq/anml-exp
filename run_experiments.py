"""Execute benchmark experiments defined in YAML configs."""
from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from anml_exp.benchmarks.evaluator import run_benchmark

SCHEMA_PATH = files("anml_exp.resources").joinpath("results-schema.json")


def _validate(result: dict[str, Any]) -> None:
    """Validate ``result`` using ``results/results-schema.json``."""
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(result, schema)


def _run(config: Path) -> None:
    data = yaml.safe_load(config.read_text())

    dataset = data["dataset"]
    seed = int(data.get("seed", 42))
    hardware = str(data.get("hardware", "unknown"))
    output_dir = Path(data.get("output_dir", f"results/{config.stem}"))

    models: dict[str, Any] = data["models"]

    for model_name, params in models.items():
        params = params or {}
        out_file = output_dir / f"{model_name}.json"
        result = run_benchmark(
            dataset=dataset,
            model_name=model_name,
            seed=seed,
            hardware=hardware,
            output=out_file,
            **params,
        )
        _validate(result)
        print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment YAML")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    args = parser.parse_args()
    _run(args.config)


if __name__ == "__main__":
    main()
