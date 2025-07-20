"""Isolation Forest model wrapper."""
from __future__ import annotations

from typing import Optional

from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]

from .base import ArrayLike, BaseAnomalyModel, NDArrayF


class IsolationForestModel(BaseAnomalyModel):
    """Wrapper around :class:`~sklearn.ensemble.IsolationForest`."""

    def __init__(self, **params: Optional[int]):
        self.model = IsolationForest(**params)
        self._threshold: float | None = None

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] | None = None
    ) -> "IsolationForestModel":
        self.model.fit(X)
        return self

    def score_samples(self, X: ArrayLike) -> NDArrayF:
        scores: NDArrayF = -self.model.decision_function(X)
        return scores

    @property
    def decision_threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Model has no threshold set")
        return self._threshold

    def set_threshold(self, value: float) -> None:
        self._threshold = value
