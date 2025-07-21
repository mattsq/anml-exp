"""anml-exp package."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("anml-exp")
except PackageNotFoundError:  # pragma: no cover - fallback during dev
    __version__ = "0.0.1"

from .registry import Registry

__all__ = ["Registry"]
