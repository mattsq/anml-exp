from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.datasets import make_blobs  # type: ignore[import-untyped]

from anml_exp.models.autoencoder import AutoEncoderModel
from anml_exp.models.deep_svdd import DeepSVDDModel
from anml_exp.models.isolation_forest import IsolationForestModel
from anml_exp.models.local_outlier_factor import LocalOutlierFactorModel
from anml_exp.models.one_class_svm import OneClassSVMModel
from anml_exp.models.pca_detector import PCAAnomalyModel
from anml_exp.models.usad import USADModel


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


def test_one_class_svm() -> None:
    _check_model(OneClassSVMModel())


def test_autoencoder() -> None:
    _check_model(AutoEncoderModel(n_epochs=2))


def test_pca_anomaly() -> None:
    _check_model(PCAAnomalyModel())


def test_deep_svdd() -> None:
    _check_model(DeepSVDDModel(n_epochs=2))


def test_usad_model() -> None:
    _check_model(USADModel(n_epochs=2))


def test_matrix_profile_model() -> None:
    pytest.importorskip("stumpy")
    from anml_exp.models import MatrixProfileModel
    series = np.linspace(0, 1, 50)
    window = 5
    windows = np.lib.stride_tricks.sliding_window_view(series, window)
    model = MatrixProfileModel(window_size=window).fit(windows[:-10])
    scores = model.score_samples(windows[-10:])
    assert scores.shape == (10,)
