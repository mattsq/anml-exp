"""Anomaly detection models."""
from .base import BaseAnomalyModel
from .isolation_forest import IsolationForestModel
from .local_outlier_factor import LocalOutlierFactorModel

try:
    from .autoencoder import AutoEncoderModel
    from .deep_svdd import DeepSVDDModel
    from .usad import USADModel
except Exception:  # pragma: no cover - optional dependency missing
    AutoEncoderModel = None  # type: ignore[misc,assignment]
    DeepSVDDModel = None  # type: ignore[misc,assignment]
    USADModel = None  # type: ignore[misc,assignment]

try:
    from .matrix_profile import MatrixProfileModel
except Exception:  # pragma: no cover - optional dependency missing
    MatrixProfileModel = None  # type: ignore[misc,assignment]

from .diffusion.ddad_tabular import DDADTabularModel
from .one_class_svm import OneClassSVMModel
from .pca_detector import PCAAnomalyModel
from .score_matching.gnsm import GNSMModel

__all__ = [
    "BaseAnomalyModel",
    "IsolationForestModel",
    "LocalOutlierFactorModel",
    "OneClassSVMModel",
    "PCAAnomalyModel",
    "GNSMModel",
    "DDADTabularModel",
]

if AutoEncoderModel is not None:
    __all__.append("AutoEncoderModel")
if DeepSVDDModel is not None:
    __all__.append("DeepSVDDModel")
if USADModel is not None:
    __all__.append("USADModel")

if MatrixProfileModel is not None:
    __all__.append("MatrixProfileModel")

_models: dict[str, type[BaseAnomalyModel]] = {
    "ddad_tab": DDADTabularModel,
}
