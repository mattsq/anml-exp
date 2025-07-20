"""One-Class SVM model wrapper."""
from __future__ import annotations

from typing import Optional

from sklearn.svm import OneClassSVM  # type: ignore[import-untyped]

from .base import ArrayLike, BaseAnomalyModel, NDArrayF


class OneClassSVMModel(BaseAnomalyModel):
    """Wrapper around :class:`~sklearn.svm.OneClassSVM`."""

    def __init__(self, **params: Optional[int]):
        self.model = OneClassSVM(**params)
        self._threshold: float | None = None

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] | None = None
    ) -> "OneClassSVMModel":
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
