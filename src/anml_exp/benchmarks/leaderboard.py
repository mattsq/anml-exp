"""Benchmark predefined datasets and produce a leaderboard."""
from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd  # type: ignore[import-untyped]

from .evaluator import _normalize_hardware, run_benchmark

DATASETS: Sequence[str] = ["breast-cancer", "wine", "digits"]

MODELS: Sequence[str] = [
    "isolation_forest",
    "local_outlier_factor",
    "one_class_svm",
    "pca",
]

SCHEMA_PATH = files("anml_exp.resources").joinpath("results-schema.json")


def _validate(result: dict[str, Any]) -> None:
    """Validate ``result`` against ``results-schema.json``."""
    import jsonschema  # type: ignore[import-untyped]

    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(result, schema)


def run_all(
    *,
    datasets: Sequence[str] = DATASETS,
    models: Sequence[str] = MODELS,
    seed: int = 42,
    hardware: str | Mapping[str, Any] = "unknown",
) -> list[dict[str, Any]]:
    """Run benchmarks for all ``datasets`` and ``models``."""
    hw = _normalize_hardware(hardware)
    results = []
    for dataset in datasets:
        for model_name in models:
            out_file = Path(f"results/{dataset}/{model_name}.json")
            result = run_benchmark(
                dataset=dataset,
                model_name=model_name,
                seed=seed,
                hardware=hw,
                output=out_file,
            )
            _validate(result)
            results.append(result)
    return results


def write_leaderboard(results: Sequence[dict[str, Any]], out_dir: Path) -> None:
    """Write aggregated ``results`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "leaderboard.json"
    md_path = out_dir / "leaderboard.md"

    with json_path.open("w") as f:
        json.dump(list(results), f, indent=2)

    df = (
        pd.DataFrame(results)[["dataset", "model", "roc_auc", "pr_auc", "f1"]]
        .sort_values(["dataset", "model"])
        .reset_index(drop=True)
    )
    md_path.write_text(df.to_markdown(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full benchmark suite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardware", default="unknown")
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results/leaderboard"),
    )
    args = parser.parse_args()

    results = run_all(seed=args.seed, hardware=args.hardware)
    write_leaderboard(results, args.output_dir)


if __name__ == "__main__":
    main()

