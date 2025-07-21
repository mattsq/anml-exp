from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmarks.evaluator import MODEL_REGISTRY, run_benchmark
from .benchmarks.leaderboard import run_all, write_leaderboard


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="anml-exp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("benchmark", help="Run a single benchmark")
    bench.add_argument("--dataset", required=True)
    bench.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--hardware", default="unknown")
    bench.add_argument("--output", type=Path, required=True)

    leader = subparsers.add_parser("leaderboard", help="Run benchmark suite")
    leader.add_argument("--seed", type=int, default=42)
    leader.add_argument("--hardware", default="unknown")
    leader.add_argument("--output_dir", type=Path, default=Path("results/leaderboard"))

    args, unknown = parser.parse_known_args(argv)

    if args.command == "benchmark":
        model_params: dict[str, Any] = {
            k: eval(v) for k, v in (kv.split("=", 1) for kv in unknown if "=" in kv)
        }
        hw = (
            json.loads(args.hardware)
            if args.hardware.strip().startswith("{")
            else args.hardware
        )
        result = run_benchmark(
            dataset=args.dataset,
            model_name=args.model,
            seed=args.seed,
            hardware=hw,
            output=args.output,
            **model_params,
        )
        print(result)
    elif args.command == "leaderboard":
        hw = (
            json.loads(args.hardware)
            if args.hardware.strip().startswith("{")
            else args.hardware
        )
        results = run_all(seed=args.seed, hardware=hw)
        write_leaderboard(results, args.output_dir)


if __name__ == "__main__":
    main()
