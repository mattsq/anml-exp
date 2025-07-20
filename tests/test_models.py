from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_blobs  # type: ignore[import-untyped]

from anomaly_models import IsolationForestModel, LocalOutlierFactorModel


def _toy_data() -> NDArray[np.float64]:
    X, _ = make_blobs(n_samples=30, centers=1, cluster_std=0.5, random_state=0)
    return cast(NDArray[np.float64], X.astype(np.float64))


def _check_model(model: Any) -> None:
    X = _toy_data()
    fitted = model.fit(X)
    assert fitted is model
    scores = model.score_samples(X)
    assert scores.shape == (X.shape[0],)
    model.set_threshold(float(np.percentile(scores, 90)))
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_isolation_forest() -> None:
    _check_model(IsolationForestModel())


def test_local_outlier_factor() -> None:
    _check_model(LocalOutlierFactorModel())
