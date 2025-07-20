"""Base classes for anomaly detection models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Self

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from anml_exp.registry import Registry

import numpy as np
from numpy.typing import NDArray as _NDArray

ArrayLike = _NDArray[Any]
NDArray = _NDArray[np.float64]


class BaseAnomalyModel(ABC):
    """Abstract base class for anomaly detection models."""

    _threshold: float | None

    def __init__(self) -> None:
        # ``_threshold`` is initialised to ``None`` so subclasses are not
        # forced to call ``super().__init__``.
        self._threshold = None

    @abstractmethod
    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] | None = None
    ) -> Self:
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
    def decision_threshold(self) -> float:
        """Threshold used for :meth:`predict`."""
        if self._threshold is None:
            raise RuntimeError("Model has no threshold set")
        return self._threshold

    def set_threshold(self, value: float) -> None:
        """Set the :attr:`decision_threshold`."""
        self._threshold = value

    # ------------------------------------------------------------------
    # Optional artefact persistence helpers
    # ------------------------------------------------------------------
    def save(self, registry: "Registry", name: str, version: str) -> str:
        """Save ``self`` to ``registry`` and return its digest."""
        return registry.save(self, name, version)

    @classmethod
    def load(cls: type[Self], registry: "Registry", name: str, version: str) -> Self:
        """Load an instance from ``registry``."""
        obj = registry.load(name, version)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded artefact is not a {cls.__name__}")
        return obj
