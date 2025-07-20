"""Matrix profile anomaly detector using STOMP."""
from __future__ import annotations

from typing import Optional, cast

import numpy as np
import stumpy  # type: ignore[import-not-found]

from .base import ArrayLike, BaseAnomalyModel, NDArray


def _reconstruct(windows: NDArray) -> NDArray:
    """Reconstruct original series from windows."""
    return np.concatenate([windows[0], windows[1:, -1]])


class MatrixProfileModel(BaseAnomalyModel):
    """STOMP-based matrix profile detector."""

    def __init__(self, window_size: int = 24) -> None:
        self.window_size = window_size
        self.series_: NDArray | None = None
        self._threshold: float | None = None

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] | None = None
    ) -> "MatrixProfileModel":
        self.series_ = _reconstruct(np.asarray(X, dtype=np.float64))
        return self

    def score_samples(self, X: ArrayLike) -> NDArray:
        if self.series_ is None:
            raise RuntimeError("Model has not been fitted")

        test_series = _reconstruct(np.asarray(X, dtype=np.float64))
        full_series = np.concatenate([self.series_, test_series])
        profile = cast(
            NDArray,
            stumpy.stump(full_series, m=self.window_size)[:, 0],
        ).astype(np.float64)
        start = len(self.series_) - self.window_size + 1
        return profile[start : start + len(X)]

    @property
    def decision_threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Model has no threshold set")
        return self._threshold

    def set_threshold(self, value: float) -> None:
        self._threshold = value
