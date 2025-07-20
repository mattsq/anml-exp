"""Local Outlier Factor model wrapper."""
from __future__ import annotations

from typing import Optional

from sklearn.neighbors import LocalOutlierFactor  # type: ignore[import-untyped]

from .base import BaseAnomalyModel, ArrayLike, NDArray


class LocalOutlierFactorModel(BaseAnomalyModel):
    """Wrapper around :class:`~sklearn.neighbors.LocalOutlierFactor`."""

    def __init__(self, **params: Optional[int]):
        # Force novelty=True so we can call ``score_samples`` on new data
        self.model = LocalOutlierFactor(novelty=True, **params)
        self._threshold: float | None = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] | None = None) -> "LocalOutlierFactorModel":
        self.model.fit(X)
        return self

    def score_samples(self, X: ArrayLike) -> NDArray:
        scores: NDArray = -self.model.score_samples(X)
        return scores

    @property
    def decision_threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Model has no threshold set")
        return self._threshold

    def set_threshold(self, value: float) -> None:
        self._threshold = value
