"""PCA-based anomaly detection via reconstruction error."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA  # type: ignore[import-untyped]

from .base import ArrayLike, BaseAnomalyModel, NDArray


class PCAAnomalyModel(BaseAnomalyModel):
    """Principal Component Analysis reconstruction error detector."""

    def __init__(self, n_components: int | None = None, **params: Optional[int]):
        self.model = PCA(n_components=n_components, **params)
        self._threshold: float | None = None

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] | None = None
    ) -> "PCAAnomalyModel":
        self.model.fit(X)
        return self

    def score_samples(self, X: ArrayLike) -> NDArray:
        transformed = self.model.transform(X)
        recon = self.model.inverse_transform(transformed)
        errors: NDArray = np.mean((X - recon) ** 2, axis=1)
        return errors

    @property
    def decision_threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Model has no threshold set")
        return self._threshold

    def set_threshold(self, value: float) -> None:
        self._threshold = value

