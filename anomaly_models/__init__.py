"""Anomaly detection models."""
from .base import BaseAnomalyModel
from .isolation_forest import IsolationForestModel

__all__ = [
    "BaseAnomalyModel",
    "IsolationForestModel",
]
