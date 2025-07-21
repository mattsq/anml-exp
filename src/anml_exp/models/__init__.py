"""Anomaly detection models."""
from .autoencoder import AutoEncoderModel
from .base import BaseAnomalyModel
from .deep_svdd import DeepSVDDModel
from .isolation_forest import IsolationForestModel
from .local_outlier_factor import LocalOutlierFactorModel

try:
    from .matrix_profile import MatrixProfileModel
except Exception:  # pragma: no cover - optional dependency missing
    MatrixProfileModel = None  # type: ignore[misc,assignment]
from .one_class_svm import OneClassSVMModel
from .pca_detector import PCAAnomalyModel
from .usad import USADModel

__all__ = [
    "BaseAnomalyModel",
    "IsolationForestModel",
    "LocalOutlierFactorModel",
    "OneClassSVMModel",
    "AutoEncoderModel",
    "PCAAnomalyModel",
    "DeepSVDDModel",
    "USADModel",
]

if MatrixProfileModel is not None:
    __all__.append("MatrixProfileModel")
