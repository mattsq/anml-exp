"""Benchmark runner producing results JSON."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from anml_exp.data import load_dataset
from anml_exp.models import (
    BaseAnomalyModel,
    DeepSVDDModel,
    IsolationForestModel,
    LocalOutlierFactorModel,
    MatrixProfileModel,
    OneClassSVMModel,
    PCAAnomalyModel,
    USADModel,
)

# Registry of available models
MODEL_REGISTRY: dict[str, type[BaseAnomalyModel]] = {
    "isolation_forest": IsolationForestModel,
    "local_outlier_factor": LocalOutlierFactorModel,
    "one_class_svm": OneClassSVMModel,
    "pca": PCAAnomalyModel,
    "deep_svdd": DeepSVDDModel,
    "usad": USADModel,
    "matrix_profile": MatrixProfileModel,
}


def run_benchmark(
    dataset: str,
    model_name: str,
    seed: int,
    hardware: str,
    output: Path,
    **model_params: Any,
) -> dict[str, Any]:
    """Run a benchmark and write results to ``output``."""
    X_train, y_train = load_dataset(dataset, split="train")
    X_test, y_test = load_dataset(dataset, split="test")

    model_cls = MODEL_REGISTRY[model_name]
    start = time.perf_counter()
    model = model_cls(**model_params)
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    scores = model.score_samples(X_test)
    score_time = time.perf_counter() - start

    roc_auc = float(roc_auc_score(y_test, scores))
    pr_auc = float(average_precision_score(y_test, scores))
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    idx = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[idx])
    preds = (scores >= threshold).astype(int)
    f1 = float(f1_score(y_test, preds))

    params = model.model.get_params() if hasattr(model, "model") else {}

    result = {
        "dataset": dataset,
        "model": model_name,
        "n_samples": int(len(X_test)),
        "seed": seed,
        "hardware": hardware,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "threshold": threshold,
        "fit_time": float(fit_time),
        "score_time": float(score_time),
        "params": params,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmark")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardware", default="unknown")
    parser.add_argument("--output", type=Path, required=True)
    args, unknown = parser.parse_known_args()
    model_params = {
        k: eval(v) for k, v in (kv.split("=", 1) for kv in unknown if "=" in kv)
    }
    result = run_benchmark(
        dataset=args.dataset,
        model_name=args.model,
        seed=args.seed,
        hardware=args.hardware,
        output=args.output,
        **model_params,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
