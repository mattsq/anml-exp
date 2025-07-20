"""STOMP-based matrix profile anomaly detector."""
from __future__ import annotations

from typing import Optional

import numpy as np
import stumpy  # type: ignore[import-untyped]

from .base import ArrayLike, BaseAnomalyModel, NDArrayF


class MatrixProfileModel(BaseAnomalyModel):
    """Matrix profile detector using :func:`stumpy.stump`.

    Parameters
    ----------
    window_size:
        Length of the sliding window (``m`` in matrix profile terms).
    """

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self._threshold: float | None = None

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] | None = None
    ) -> "MatrixProfileModel":
        self._train_ts = np.asarray(X, dtype=np.float64).ravel()
        return self

    def score_samples(self, X: ArrayLike) -> NDArrayF:
        ts = np.asarray(X, dtype=np.float64).ravel()
        profile = stumpy.stump(ts, m=self.window_size)
        scores: NDArrayF = profile[:, 0]
        return scores

    @property
    def decision_threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Model has no threshold set")
        return self._threshold

    def set_threshold(self, value: float) -> None:
        self._threshold = value
