"""Isolation Forest model wrapper."""
from __future__ import annotations

from typing import Optional

from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from .base import ArrayLike, BaseAnomalyModel, NDArray


class IsolationForestModel(BaseAnomalyModel):
    """Wrapper around :class:`~sklearn.ensemble.IsolationForest`."""

    def __init__(self, **params: Optional[int]):
        self.scaler = StandardScaler()
        self.model = IsolationForest(**params)
        self._threshold: float | None = None

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] | None = None
    ) -> "IsolationForestModel":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        return self

    def score_samples(self, X: ArrayLike) -> NDArray:
        Xs = self.scaler.transform(X)
        scores: NDArray = -self.model.decision_function(Xs)
        return scores


