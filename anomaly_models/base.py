"""Base classes for anomaly detection models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

ArrayLike = np.ndarray
NDArray = np.ndarray


class BaseAnomalyModel(ABC):
    """Abstract base class for anomaly detection models."""

    @abstractmethod
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] | None = None) -> "BaseAnomalyModel":
        """Fit the model."""
        raise NotImplementedError

    @abstractmethod
    def score_samples(self, X: ArrayLike) -> NDArray:
        """Compute anomaly scores for ``X``."""
        raise NotImplementedError

    def predict(self, X: ArrayLike, *, threshold: float | None = None) -> NDArray:
        """Predict binary labels using ``threshold`` or ``self.decision_threshold``."""
        if threshold is None:
            threshold = self.decision_threshold
        scores = self.score_samples(X)
        return (scores >= threshold).astype(int)

    @property
    @abstractmethod
    def decision_threshold(self) -> float:
        """Threshold used for ``predict``."""
        raise NotImplementedError
