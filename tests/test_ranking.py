from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from anomaly_models import IsolationForestModel
from datasets.registry import load_dataset


@given(
    seed=st.integers(0, 1000),
    scale=st.floats(0.5, 3.0, allow_nan=False, allow_infinity=False),
    dx=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
    dy=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=10)
def test_ranking_invariance(seed: int, scale: float, dx: float, dy: float) -> None:
    """Anomaly score ordering should be invariant to affine transforms."""
    X_train, _ = load_dataset("toy-blobs", split="train", seed=seed)
    X_test, _ = load_dataset("toy-blobs", split="test", seed=seed)

    base = IsolationForestModel(random_state=seed, n_estimators=50).fit(X_train)
    scores = base.score_samples(X_test)
    order = np.argsort(scores)

    shift = np.array([dx, dy], dtype=np.float64)
    Xt = X_train * scale + shift
    Xtt = X_test * scale + shift
    transformed = IsolationForestModel(random_state=seed, n_estimators=50).fit(Xt)
    scores_t = transformed.score_samples(Xtt)

    np.testing.assert_array_equal(order, np.argsort(scores_t))
